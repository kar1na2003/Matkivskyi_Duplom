"""Functional tests for modusmate_host.link.BoardLink using a fake serial loopback."""
from __future__ import annotations

import struct
import threading
import time

import pytest

from modusmate_host import protocol as P
from modusmate_host.link import BoardLink


# ---------- helpers ----------

def _ack_handler(*expected_cmds: int):
    """Build a tx_handler that auto-ACKs the given command IDs (status=0).

    The ACK echoes the command's SEQ as required by protocol v2.
    """
    dec = P.FrameDecoder()

    def handle(data: bytes, fs):
        for f in dec.feed(data):
            if f.type in expected_cmds:
                ack = P.encode_frame(P.EVT_ACK, bytes([f.type, f.seq, 0]))
                fs.push_rx(ack)
    return handle


# ---------- tests ----------

def test_ping_roundtrip(fake_serial):
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]
    fs.set_tx_handler(_ack_handler(P.CMD_PING))
    link.start()
    try:
        assert link.ping(timeout=1.0) is True
    finally:
        link.close()


def test_ping_times_out_when_no_ack(fake_serial):
    link = BoardLink("FAKE")
    # no tx handler -> nothing pushed back
    link.start()
    try:
        assert link.ping(timeout=0.2) is False
    finally:
        link.close()


def test_set_algo_lcd_stream_bench_send_correct_payloads(fake_serial):
    """Verify the on-wire payload matches each setter and is ACKed."""
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]

    captured = []
    dec = P.FrameDecoder()

    def handle(data: bytes, f):
        for fr in dec.feed(data):
            captured.append((fr.type, fr.payload))
            f.push_rx(P.encode_frame(P.EVT_ACK, bytes([fr.type, fr.seq, 0])))

    fs.set_tx_handler(handle)
    link.start()
    try:
        assert link.set_algo(15)
        assert link.set_lcd(True)
        assert link.set_stream(False)
        assert link.set_bench(True)
    finally:
        link.close()

    assert (P.CMD_SET_ALGO, bytes([15])) in captured
    assert (P.CMD_SET_LCD, bytes([1])) in captured
    assert (P.CMD_SET_STREAM, bytes([0])) in captured
    assert (P.CMD_SET_BENCH, bytes([1])) in captured


def test_get_info_parses_algo_list(fake_serial):
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]
    dec = P.FrameDecoder()

    # build EVT_INFO payload: count (u8) | for each: id (u8) | family (u8) | nlen (u8) | name
    entries = [(0, 0, "passthrough"), (9, 1, "sobel"), (24, 4, "otsu")]
    buf = bytearray([len(entries)])
    for aid, fam, name in entries:
        nb = name.encode("ascii")
        buf += bytes([aid, fam, len(nb)]) + nb

    def handle(data: bytes, fs_):
        for f in dec.feed(data):
            if f.type == P.CMD_GET_INFO:
                fs_.push_rx(P.encode_frame(P.EVT_INFO, bytes(buf)))
                fs_.push_rx(P.encode_frame(P.EVT_ACK,
                                           bytes([f.type, f.seq, 0])))

    fs.set_tx_handler(handle)
    link.start()
    try:
        infos = link.get_info(timeout=1.0)
    finally:
        link.close()

    assert len(infos) == 3
    assert infos[0].algo_id == 0 and infos[0].name == "passthrough"
    assert infos[1].algo_id == 9 and infos[1].family == 1 and infos[1].name == "sobel"
    assert infos[2].name == "otsu"


def test_fps_event_dispatches_to_callback(fake_serial):
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]

    seen = []
    ev = threading.Event()

    def on_fps(r):
        seen.append(r)
        ev.set()

    link.on_fps(on_fps)
    link.start()
    try:
        # fps_x100=1234 (12.34 fps), algo=4500 us, infer=18000 us
        payload = struct.pack("<III", 1234, 4500, 18000)
        fs.push_rx(P.encode_frame(P.EVT_FPS, payload))
        assert ev.wait(timeout=1.0)
    finally:
        link.close()

    assert len(seen) == 1
    assert abs(seen[0].fps - 12.34) < 1e-6
    assert seen[0].algo_us == 4500
    assert seen[0].infer_us == 18000


def test_detection_event_dispatches_with_correct_fields(fake_serial):
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]

    seen = []
    ev = threading.Event()

    def on_det(d):
        seen.append(d)
        ev.set()

    link.on_detections(on_det)
    link.start()
    try:
        # 2 detections: rock @ (10,20,80,80) conf=0.85 ; paper @ (-3,5,40,40) conf=0.40
        body = bytearray([2])
        body += bytes([0, 85]) + struct.pack("<hhhh", 10, 20, 80, 80)
        body += bytes([1, 40]) + struct.pack("<hhhh", -3, 5, 40, 40)
        fs.push_rx(P.encode_frame(P.EVT_DETECTION, bytes(body)))
        assert ev.wait(timeout=1.0)
    finally:
        link.close()

    dets = seen[0]
    assert len(dets) == 2
    assert dets[0].class_id == 0 and abs(dets[0].conf - 0.85) < 1e-6
    assert dets[0].x == 10 and dets[0].y == 20 and dets[0].w == 80 and dets[0].h == 80
    assert dets[1].class_id == 1 and dets[1].x == -3


def test_log_event_dispatches_text(fake_serial):
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]

    seen = []
    ev = threading.Event()

    def on_log(s):
        seen.append(s)
        ev.set()

    link.on_log(on_log)
    link.start()
    try:
        fs.push_rx(P.encode_frame(P.EVT_LOG, b"hello world"))
        assert ev.wait(timeout=1.0)
    finally:
        link.close()

    assert seen == ["hello world"]


def test_preview_frame_assembly(fake_serial):
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]

    seen = []
    ev = threading.Event()

    def on_prev(buf, w, h):
        seen.append((buf, w, h))
        ev.set()

    link.on_preview(on_prev)
    link.start()
    try:
        w, h = 80, 60
        # FRAME_BEGIN: u16 w, u16 h, u8 fmt
        fs.push_rx(P.encode_frame(P.EVT_FRAME_BEGIN, struct.pack("<HHB", w, h, P.FRAME_GRAY8)))
        body = bytes(((i * 7) & 0xFF) for i in range(w * h))  # 4800 bytes
        # split into chunks below MAX_PAYLOAD
        chunk_sz = 240
        for i in range(0, len(body), chunk_sz):
            fs.push_rx(P.encode_frame(P.EVT_FRAME_CHUNK, body[i:i + chunk_sz]))
        fs.push_rx(P.encode_frame(P.EVT_FRAME_END, b""))
        assert ev.wait(timeout=1.0)
    finally:
        link.close()

    buf, gw, gh = seen[0]
    assert gw == 80 and gh == 60
    assert buf == body


def test_push_bench_image_full_sequence(fake_serial):
    """Exercise the BEGIN/CHUNK*N/END flow and synthesised EVT_BENCH_RESULT."""
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]
    dec = P.FrameDecoder()

    state = {"chunks": 0, "begun": False, "ended": False}

    def handle(data: bytes, fs_):
        for f in dec.feed(data):
            if f.type == P.CMD_BENCH_BEGIN:
                state["begun"] = True
                fs_.push_rx(P.encode_frame(P.EVT_ACK, bytes([f.type, f.seq, 0])))
            elif f.type == P.CMD_BENCH_CHUNK:
                state["chunks"] += 1
                fs_.push_rx(P.encode_frame(P.EVT_ACK, bytes([f.type, f.seq, 0])))
            elif f.type == P.CMD_BENCH_END:
                state["ended"] = True
                fs_.push_rx(P.encode_frame(P.EVT_ACK, bytes([f.type, f.seq, 0])))
                # then the result: rock @ 92%, algo=2_000us, infer=15_000us
                result = struct.pack("<BBII", 0, 92, 2000, 15000)
                fs_.push_rx(P.encode_frame(P.EVT_BENCH_RESULT, result))

    fs.set_tx_handler(handle)
    link.start()
    try:
        # tiny image so test is fast: 16x16
        img = bytes(range(256)) * 3  # 768 bytes = 16*16*3
        result = link.push_bench_image(img, 16, 16, chunk_size=120, timeout=2.0)
    finally:
        link.close()

    assert state["begun"] and state["ended"]
    assert state["chunks"] == (768 + 119) // 120  # ceil(768/120) = 7
    assert result is not None
    assert result.class_id == 0
    assert abs(result.conf - 0.92) < 1e-6
    assert result.algo_us == 2000
    assert result.infer_us == 15000


def test_push_bench_image_size_mismatch_raises(fake_serial):
    link = BoardLink("FAKE")
    link.start()
    try:
        with pytest.raises(ValueError):
            link.push_bench_image(b"\x00" * 10, 16, 16)  # wrong size
    finally:
        link.close()


def test_decoder_recovers_from_garbage_then_valid_frame(fake_serial):
    """Stuff random bytes in front of an EVT_LOG and confirm it still arrives."""
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]

    seen = []
    ev = threading.Event()

    def on_log(s):
        seen.append(s)
        ev.set()

    link.on_log(on_log)
    link.start()
    try:
        garbage = b"random printf debug noise without SOF\n\r"
        good = P.encode_frame(P.EVT_LOG, b"valid")
        fs.push_rx(garbage + good)
        assert ev.wait(timeout=1.0)
    finally:
        link.close()

    assert seen == ["valid"]
