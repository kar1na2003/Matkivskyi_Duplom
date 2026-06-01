"""Tests that exercise the TCP-style reliability layer: retransmission on
lost ACK, dedup of duplicate frames, and seq-mismatch rejection."""
from __future__ import annotations

import threading

import pytest

from modusmate_host import protocol as P
from modusmate_host.link import BoardLink


def test_retransmits_when_first_ack_is_lost(fake_serial, monkeypatch):
    """Drop the first attempt's ACK; the host must re-send and succeed."""
    # speed the test up
    monkeypatch.setattr(P, "ACK_TIMEOUT_MS", 50)

    link = BoardLink("FAKE")
    fs = fake_serial["fs"]
    dec = P.FrameDecoder()

    state = {"sends_seen": 0, "seqs_seen": []}

    def handle(data: bytes, fs_):
        for f in dec.feed(data):
            if f.type != P.CMD_PING:
                continue
            state["sends_seen"] += 1
            state["seqs_seen"].append(f.seq)
            # ack only on the second attempt
            if state["sends_seen"] >= 2:
                fs_.push_rx(P.encode_frame(P.EVT_ACK,
                                           bytes([f.type, f.seq, 0])))

    fs.set_tx_handler(handle)
    link.start()
    try:
        assert link.ping(timeout=2.0) is True
    finally:
        link.close()

    # host should have re-sent the *same* seq
    assert state["sends_seen"] >= 2
    assert len(set(state["seqs_seen"])) == 1   # same seq across retries


def test_stale_ack_for_old_seq_is_ignored(fake_serial, monkeypatch):
    """If the board sends an ACK with the wrong seq, the host must NOT
    accept it — it should keep waiting for the correct seq."""
    monkeypatch.setattr(P, "ACK_TIMEOUT_MS", 80)
    monkeypatch.setattr(P, "MAX_RETRIES", 0)  # one shot, no retry

    link = BoardLink("FAKE")
    fs = fake_serial["fs"]
    dec = P.FrameDecoder()

    def handle(data: bytes, fs_):
        for f in dec.feed(data):
            if f.type == P.CMD_PING:
                # send back an ack with a *bogus* seq value (seq+99)
                bad_seq = (f.seq + 99) & 0xFF
                fs_.push_rx(P.encode_frame(P.EVT_ACK,
                                           bytes([f.type, bad_seq, 0])))

    fs.set_tx_handler(handle)
    link.start()
    try:
        # ping must FAIL because no ack with the matching seq ever arrives
        assert link.ping(timeout=0.4) is False
    finally:
        link.close()


def test_seq_increments_across_calls(fake_serial):
    """Two consecutive commands must use distinct sequence numbers."""
    link = BoardLink("FAKE")
    fs = fake_serial["fs"]
    dec = P.FrameDecoder()

    seqs = []

    def handle(data: bytes, fs_):
        for f in dec.feed(data):
            if f.type == P.CMD_PING:
                seqs.append(f.seq)
                fs_.push_rx(P.encode_frame(P.EVT_ACK,
                                           bytes([f.type, f.seq, 0])))

    fs.set_tx_handler(handle)
    link.start()
    try:
        assert link.ping(timeout=1.0)
        assert link.ping(timeout=1.0)
        assert link.ping(timeout=1.0)
    finally:
        link.close()

    assert len(seqs) == 3
    assert seqs[0] != seqs[1] and seqs[1] != seqs[2]


def test_seq_zero_reserved_for_telemetry():
    """Frame-level SEQ defaults to 0 for fire-and-forget events."""
    f = P.encode_frame(P.EVT_FPS, b"\x00" * 12)
    # SOF, type, seq=0, len=12 LE
    assert f[2] == 0


def test_ack_status_codes_match_firmware():
    """Sanity-check the ACK status code constants stay aligned."""
    assert P.ACK_OK == 0x00
    assert P.ACK_BAD_LEN == 0x01
    assert P.ACK_BAD_PARAM == 0x02
    assert P.ACK_UNKNOWN == 0xFF
    assert P.ACK_BAD_CRC == 0xFE
