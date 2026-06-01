"""Pytest unit tests for the wire protocol - CRC + framer round-trip."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modusmate_host import protocol as P


def test_crc_known_vector():
    # CRC-CCITT(0xFFFF) of "123456789" is 0x29B1
    assert P.crc16_ccitt(b"123456789") == 0x29B1


def test_crc_empty_is_init():
    assert P.crc16_ccitt(b"") == 0xFFFF


def test_encode_frame_shape():
    f = P.encode_frame(P.CMD_PING, b"")
    assert f[0] == P.SOF
    assert f[1] == P.CMD_PING
    assert f[2] == 0          # default seq
    assert f[3] == 0 and f[4] == 0  # len=0
    assert len(f) == 1 + 1 + 1 + 2 + 0 + 2


def test_encode_frame_with_seq():
    f = P.encode_frame(P.CMD_PING, b"", seq=42)
    assert f[2] == 42
    dec = P.FrameDecoder()
    [fr] = list(dec.feed(f))
    assert fr.type == P.CMD_PING
    assert fr.seq == 42
    assert fr.payload == b""


def test_roundtrip_single_frame():
    dec = P.FrameDecoder()
    enc = P.encode_frame(P.CMD_SET_ALGO, bytes([7]))
    frames = list(dec.feed(enc))
    assert len(frames) == 1
    assert frames[0].type == P.CMD_SET_ALGO
    assert frames[0].payload == bytes([7])


def test_roundtrip_multiple_frames_byte_dribble():
    payloads = [(P.CMD_PING, b""),
                (P.EVT_FPS, b"\x10\x20\x30\x40\x00\x00\x00\x00\x00\x00\x00\x00"),
                (P.EVT_ACK, b"\x01\x05\x00")]   # cmd, seq, status
    stream = b"".join(P.encode_frame(t, p) for t, p in payloads)
    dec = P.FrameDecoder()
    got = []
    # dribble one byte at a time
    for b in stream:
        got.extend(dec.feed(bytes([b])))
    assert [(f.type, f.payload) for f in got] == payloads


def test_framer_tolerates_garbage_prefix():
    stream = b"hello world\n\r" + P.encode_frame(P.CMD_PING) + b"trailing"
    dec = P.FrameDecoder()
    got = list(dec.feed(stream))
    assert len(got) == 1
    assert got[0].type == P.CMD_PING


def test_framer_drops_bad_crc():
    ok = P.encode_frame(P.CMD_PING)
    bad = bytearray(P.encode_frame(P.CMD_SET_ALGO, bytes([3])))
    bad[-1] ^= 0xFF  # corrupt CRC
    dec = P.FrameDecoder()
    got = list(dec.feed(bytes(bad) + ok))
    # bad frame dropped, good frame still decoded
    assert len(got) == 1
    assert got[0].type == P.CMD_PING


def test_oversized_len_is_rejected():
    dec = P.FrameDecoder()
    # SOF, type, seq, len=0xFFFF
    junk = bytes([P.SOF, 0x99, 0x00, 0xFF, 0xFF])
    list(dec.feed(junk))  # must not raise
    # feed a valid frame after - it should decode
    good = P.encode_frame(P.CMD_PING)
    got = list(dec.feed(good))
    assert any(f.type == P.CMD_PING for f in got)
