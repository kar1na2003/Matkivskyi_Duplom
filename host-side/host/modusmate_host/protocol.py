"""ModusMate UART wire protocol - mirror of firmware comm/comm_proto.h.

Frame: [SOF=0xA5][TYPE u8][SEQ u8][LEN u16 LE][PAYLOAD][CRC16-CCITT u16 LE]
CRC is computed over [TYPE | SEQ | LEN_LO | LEN_HI | PAYLOAD].

Reliability:
- Each host -> board command carries a sequence number (SEQ).
- Board ACK payload echoes the SEQ so the host can match the right ACK and
  drop stale duplicates.
- Board de-duplicates incoming commands by (TYPE, SEQ); a re-sent command
  after a lost ACK is re-ACKed but not re-executed.
- BoardLink retransmits a command up to MAX_RETRIES times when no matching
  ACK arrives within ACK_TIMEOUT_MS.
- Board -> host telemetry events (EVT_FPS / EVT_DETECTION / EVT_FRAME_*) are
  fire-and-forget and use SEQ=0 in the frame header.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Generator, Optional

PROTOCOL_VERSION = 2

# --- constants (must match firmware) ---
SOF = 0xA5
MAX_PAYLOAD = 1280

# Reliability tunables (host-side).
ACK_TIMEOUT_MS = 300
MAX_RETRIES = 3

# Host -> board commands
CMD_PING        = 0x01
CMD_GET_INFO    = 0x02
CMD_SET_ALGO    = 0x10
CMD_SET_LCD     = 0x11
CMD_SET_STREAM  = 0x12
CMD_SET_BENCH   = 0x13
CMD_GET_FPS     = 0x20
CMD_SET_BAUDRATE = 0x21   # payload: u32 baudrate LE
CMD_BENCH_BEGIN = 0x30
CMD_BENCH_CHUNK = 0x31
CMD_BENCH_END   = 0x32

# Board -> host events
EVT_ACK           = 0x80
EVT_LOG           = 0x81
EVT_INFO          = 0x82
EVT_FPS           = 0x90
EVT_DETECTION     = 0x91
EVT_BENCH_RESULT  = 0x92
EVT_FRAME_BEGIN   = 0xA0
EVT_FRAME_CHUNK   = 0xA1
EVT_FRAME_END     = 0xA2

# ACK status codes
ACK_OK         = 0x00
ACK_BAD_LEN    = 0x01
ACK_BAD_PARAM  = 0x02
ACK_UNKNOWN    = 0xFF
ACK_BAD_CRC    = 0xFE

FRAME_GRAY8 = 0x01

PREVIEW_W = 320
PREVIEW_H = 2400


def crc16_ccitt(data: bytes) -> int:
    """CRC16-CCITT, poly 0x1021, init 0xFFFF, no reflection, no xor-out."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc


def encode_frame(type_id: int, payload: bytes = b"", seq: int = 0) -> bytes:
    """Encode one frame. SEQ defaults to 0 for events; commands pass an
    incrementing sequence number for retransmission/dedup."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too large: {len(payload)} > {MAX_PAYLOAD}")
    body = struct.pack("<BBH", type_id & 0xFF, seq & 0xFF, len(payload)) + payload
    crc = crc16_ccitt(body)
    return bytes([SOF]) + body + struct.pack("<H", crc)


@dataclass
class Frame:
    type: int
    seq: int
    payload: bytes


class FrameDecoder:
    """Byte-oriented incremental decoder. Yields Frame on each complete frame.

    Tolerates non-protocol bytes (e.g. printf debug output) between frames -
    such bytes are discarded silently until SOF is seen.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> Generator[Frame, None, None]:
        self._buf.extend(chunk)
        while True:
            f = self._try_parse_one()
            if f is None:
                return
            yield f

    def _try_parse_one(self) -> Optional[Frame]:
        # Find SOF
        idx = self._buf.find(SOF.to_bytes(1, "little"))
        if idx < 0:
            self._buf.clear()
            return None
        if idx > 0:
            del self._buf[:idx]

        # Need SOF + TYPE + SEQ + LEN(2) = 5 bytes
        if len(self._buf) < 5:
            return None

        ftype = self._buf[1]
        fseq = self._buf[2]
        flen = self._buf[3] | (self._buf[4] << 8)
        if flen > MAX_PAYLOAD:
            # bogus - discard SOF byte and resync
            del self._buf[0]
            return self._try_parse_one()

        total = 1 + 1 + 1 + 2 + flen + 2  # SOF + TYPE + SEQ + LEN + PAYLOAD + CRC
        if len(self._buf) < total:
            return None

        body = bytes(self._buf[1:1 + 4 + flen])  # TYPE..PAYLOAD (4-byte header before payload)
        rx_crc = self._buf[1 + 4 + flen] | (self._buf[1 + 4 + flen + 1] << 8)
        expected = crc16_ccitt(body)
        if rx_crc != expected:
            # bad CRC - drop SOF and try again
            del self._buf[0]
            return self._try_parse_one()

        payload = bytes(self._buf[5:5 + flen])
        del self._buf[:total]
        return Frame(type=ftype, seq=fseq, payload=payload)
