"""High-level board link: serial transport + framer + per-event subscribers."""
from __future__ import annotations

import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import serial  # type: ignore
import serial.tools.list_ports  # type: ignore

from . import protocol as P


@dataclass
class Detection:
    class_id: int
    conf: float
    x: int
    y: int
    w: int
    h: int


@dataclass
class FpsReport:
    fps: float
    algo_us: int
    infer_us: int


@dataclass
class BenchResult:
    class_id: int
    conf: float
    algo_us: int
    infer_us: int


@dataclass
class AlgoInfo:
    algo_id: int
    family: int
    name: str


def list_serial_ports() -> List[Tuple[str, str]]:
    """Return list of (device, description). Filters down to KitProg3-friendly ports."""
    out = []
    for p in serial.tools.list_ports.comports():
        out.append((p.device, p.description or ""))
    return out


class BoardLink:
    """Owns the serial port and the read thread. Thread-safe public methods."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.05):
        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self._decoder = P.FrameDecoder()
        self._stop = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._tx_lock = threading.Lock()

        self._ack_q: "queue.Queue[Tuple[int,int,int]]" = queue.Queue(maxsize=64)
        self._info_q: "queue.Queue[List[AlgoInfo]]" = queue.Queue(maxsize=4)
        self._bench_q: "queue.Queue[BenchResult]" = queue.Queue(maxsize=8)

        self._on_fps: Optional[Callable[[FpsReport], None]] = None
        self._on_detections: Optional[Callable[[List[Detection]], None]] = None
        self._on_log: Optional[Callable[[str], None]] = None
        self._on_preview: Optional[Callable[[bytes, int, int], None]] = None

        # preview frame assembly
        self._frame_buf: Optional[bytearray] = None
        self._frame_w = 0
        self._frame_h = 0

        # tx sequence counter (1..255 wrapping; 0 reserved for telemetry)
        self._seq = 0
        self._seq_lock = threading.Lock()

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq = (self._seq % 255) + 1   # 1..255 wrap
            return self._seq

    # ---------- subscriptions ----------
    def on_fps(self, cb: Callable[[FpsReport], None]) -> None: self._on_fps = cb
    def on_detections(self, cb: Callable[[List[Detection]], None]) -> None: self._on_detections = cb
    def on_log(self, cb: Callable[[str], None]) -> None: self._on_log = cb
    def on_preview(self, cb: Callable[[bytes, int, int], None]) -> None: self._on_preview = cb

    # ---------- lifecycle ----------
    def start(self) -> None:
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="modusmate-reader")
        self._reader.start()

    def close(self) -> None:
        self._stop.set()
        try:
            self._ser.close()
        except Exception:
            pass
        if self._reader:
            self._reader.join(timeout=1.0)

    def __enter__(self) -> "BoardLink":
        self.start()
        return self

    def __exit__(self, *a) -> None:
        self.close()

    # ---------- low-level send ----------
    def _send_raw(self, type_id: int, payload: bytes, seq: int) -> None:
        frame = P.encode_frame(type_id, payload, seq=seq)
        with self._tx_lock:
            self._ser.write(frame)
            self._ser.flush()

    def _send(self, type_id: int, payload: bytes = b"") -> None:
        """Best-effort send (no ack, no retry). Used for fire-and-forget."""
        self._send_raw(type_id, payload, seq=0)

    def _send_and_wait_ack(self, type_id: int, payload: bytes = b"",
                           timeout: float = 1.0) -> bool:
        """Send a command and wait for a matching ACK.

        Implements TCP-like retransmission: if no ACK with the right (type,seq)
        arrives within ACK_TIMEOUT_MS, the frame is re-sent up to MAX_RETRIES
        times. The board de-duplicates by (type,seq), so a re-sent frame after
        a lost ACK simply re-ACKs without re-executing.
        """
        seq = self._next_seq()
        per_try_timeout = min(timeout, P.ACK_TIMEOUT_MS / 1000.0)
        deadline = time.monotonic() + timeout

        for attempt in range(P.MAX_RETRIES + 1):
            self._send_raw(type_id, payload, seq=seq)

            # Wait for *the* ack matching (type,seq); ignore stale ones.
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    cmd, ack_seq, status = self._ack_q.get(
                        timeout=min(per_try_timeout, remaining))
                except queue.Empty:
                    break
                if cmd == type_id and ack_seq == seq:
                    return status == P.ACK_OK
                # else: stale ack from a previous retry/cmd; drop and keep waiting
            # no matching ack within per-try window: retry the same seq
            if time.monotonic() >= deadline:
                break
        return False

    # ---------- public commands ----------
    def ping(self, timeout: float = 1.0) -> bool:
        return self._send_and_wait_ack(P.CMD_PING, timeout=timeout)

    def get_info(self, timeout: float = 2.0) -> List[AlgoInfo]:
        try:
            while True: self._info_q.get_nowait()
        except queue.Empty:
            pass
        if not self._send_and_wait_ack(P.CMD_GET_INFO, timeout=timeout):
            return []
        try:
            return self._info_q.get(timeout=timeout)
        except queue.Empty:
            return []

    def set_algo(self, algo_id: int) -> bool:
        return self._send_and_wait_ack(P.CMD_SET_ALGO, bytes([algo_id & 0xFF]))

    def set_lcd(self, enabled: bool) -> bool:
        return self._send_and_wait_ack(P.CMD_SET_LCD, bytes([1 if enabled else 0]))

    def set_stream(self, enabled: bool) -> bool:
        return self._send_and_wait_ack(P.CMD_SET_STREAM, bytes([1 if enabled else 0]))

    def set_bench(self, enabled: bool) -> bool:
        return self._send_and_wait_ack(P.CMD_SET_BENCH, bytes([1 if enabled else 0]))

    def set_baudrate(self, baudrate: int, verify: bool = True,
                     settle_s: float = 0.10) -> bool:
        """Negotiate a new UART baud rate with the board.

        Flow: send CMD_SET_BAUDRATE at the current baud, wait for the ACK
        (which the board emits at the OLD baud), then reconfigure pyserial
        to the new baud and optionally ping the board to verify.  If the
        ping fails we revert to the original baud so the link stays usable.

        Returns True if the switch is verified (or if ``verify`` is False
        and the ACK was OK).
        """
        if baudrate < 9600 or baudrate > 3_000_000:
            raise ValueError("baudrate out of sane range")
        old_baud = self._ser.baudrate
        if baudrate == old_baud:
            return True

        # Board ACKs at the OLD baud, then reconfigures its SCB.
        if not self._send_and_wait_ack(P.CMD_SET_BAUDRATE,
                                       struct.pack("<I", baudrate),
                                       timeout=1.0):
            return False

        # Give the firmware a moment to finish shifting out the ACK and
        # reconfigure its SCB, then switch our end.
        time.sleep(settle_s)
        try:
            self._ser.baudrate = baudrate
        except Exception:
            return False
        # pyserial buffers may hold garbage bytes captured during the switch.
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass

        if not verify:
            return True

        # Verify the new baud works.  If it doesn't, revert so the caller
        # isn't stranded.
        if self.ping(timeout=1.0):
            return True
        try:
            self._ser.baudrate = old_baud
            self._ser.reset_input_buffer()
        except Exception:
            pass
        return False

    def push_bench_image(self, rgb888: bytes, width: int, height: int,
                         chunk_size: int = 240, timeout: float = 5.0,
                         on_chunk: Optional[Callable[[int, int], None]] = None,
                         ack_chunks: bool = False) -> Optional[BenchResult]:
        """Push a 320x240 RGB888 image and wait for the bench result.

        ``on_chunk`` (sent_bytes, total_bytes) is called after every chunk so a
        UI can show progress.  When ``ack_chunks`` is False (default) chunks
        are streamed fire-and-forget — the ``BENCH_END`` ACK reports
        ``BAD_PARAM`` if any chunk was dropped, and the host returns ``None``
        for that image.  This is ~10x faster than per-chunk stop-and-wait at
        115200 baud while remaining safe (CRC catches corruption).
        """
        if len(rgb888) != width * height * 3:
            raise ValueError("rgb888 size mismatch")
        # drain any old bench results / acks left over from a previous push
        for q in (self._bench_q, self._ack_q):
            try:
                while True: q.get_nowait()
            except queue.Empty:
                pass

        if not self._send_and_wait_ack(P.CMD_BENCH_BEGIN,
                                       struct.pack("<HH", width, height)):
            return None
        total = len(rgb888)
        for off in range(0, total, chunk_size):
            payload = rgb888[off:off + chunk_size]
            if ack_chunks:
                if not self._send_and_wait_ack(P.CMD_BENCH_CHUNK, payload,
                                               timeout=2.0):
                    return None
            else:
                # Fire-and-forget: do NOT wait for per-chunk ACKs.  Each
                # chunk still needs a unique (type,seq) so the board's
                # dedup ring doesn't mistake consecutive chunks for a
                # retransmission of the first one.  With seq cycling
                # 1..255 and a firmware dedup window of 8, re-use of a
                # seq happens 255 frames later — no collision.
                self._send_raw(P.CMD_BENCH_CHUNK, payload,
                               seq=self._next_seq())
            if on_chunk is not None:
                try:
                    on_chunk(min(off + len(payload), total), total)
                except Exception:
                    pass
        # Drain stray chunk-ACKs that the firmware emitted while we streamed.
        try:
            while True: self._ack_q.get_nowait()
        except queue.Empty:
            pass
        if not self._send_and_wait_ack(P.CMD_BENCH_END, timeout=timeout):
            return None
        try:
            return self._bench_q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------- read loop ----------
    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._ser.read(256)
            except serial.SerialException:
                break
            if not data:
                continue
            for f in self._decoder.feed(data):
                self._dispatch(f)

    def _dispatch(self, f: P.Frame) -> None:
        if f.type == P.EVT_ACK and len(f.payload) >= 3:
            try: self._ack_q.put_nowait((f.payload[0], f.payload[1], f.payload[2]))
            except queue.Full: pass
        elif f.type == P.EVT_LOG:
            if self._on_log:
                try: self._on_log(f.payload.decode("ascii", errors="replace"))
                except Exception: pass
        elif f.type == P.EVT_INFO:
            self._info_q.put_nowait(self._parse_info(f.payload))
        elif f.type == P.EVT_FPS and len(f.payload) >= 12:
            fps_x100, algo_us, infer_us = struct.unpack("<III", f.payload[:12])
            if self._on_fps:
                self._on_fps(FpsReport(fps=fps_x100/100.0, algo_us=algo_us, infer_us=infer_us))
        elif f.type == P.EVT_DETECTION and len(f.payload) >= 1:
            self._dispatch_detections(f.payload)
        elif f.type == P.EVT_BENCH_RESULT and len(f.payload) >= 10:
            cls, conf_x100, algo_us, infer_us = struct.unpack("<BBII", f.payload[:10])
            self._bench_q.put_nowait(BenchResult(class_id=cls, conf=conf_x100/100.0,
                                                  algo_us=algo_us, infer_us=infer_us))
        elif f.type == P.EVT_FRAME_BEGIN and len(f.payload) >= 5:
            w, h, fmt = struct.unpack("<HHB", f.payload[:5])
            self._frame_w, self._frame_h = w, h
            self._frame_buf = bytearray()
        elif f.type == P.EVT_FRAME_CHUNK and self._frame_buf is not None:
            self._frame_buf.extend(f.payload)
        elif f.type == P.EVT_FRAME_END and self._frame_buf is not None:
            if self._on_preview:
                try: self._on_preview(bytes(self._frame_buf), self._frame_w, self._frame_h)
                except Exception: pass
            self._frame_buf = None

    @staticmethod
    def _parse_info(p: bytes) -> List[AlgoInfo]:
        out: List[AlgoInfo] = []
        if not p: return out
        n = p[0]
        i = 1
        for _ in range(n):
            if i + 3 > len(p): break
            algo_id = p[i]; family = p[i+1]; nlen = p[i+2]; i += 3
            name = p[i:i+nlen].decode("ascii", errors="replace"); i += nlen
            out.append(AlgoInfo(algo_id=algo_id, family=family, name=name))
        return out

    def _dispatch_detections(self, p: bytes) -> None:
        if not self._on_detections:
            return
        n = p[0]
        i = 1
        det: List[Detection] = []
        for _ in range(n):
            if i + 10 > len(p): break
            cls = p[i]; conf = p[i+1]
            x, y, w, h = struct.unpack("<hhhh", p[i+2:i+10])
            det.append(Detection(class_id=cls, conf=conf/100.0, x=x, y=y, w=w, h=h))
            i += 10
        self._on_detections(det)
