"""Pytest fixtures: a fake serial.Serial that does in-process loopback.

Lets us drive BoardLink without a real KitProg3 attached.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import pytest
import serial  # type: ignore

from modusmate_host import link as link_mod
from modusmate_host import protocol as P


class FakeSerial:
    """Minimal stand-in for serial.Serial.

    Tests register an on-write callback that simulates the firmware: it
    decodes the host's command and pushes one or more EVT_* frames into
    the RX buffer for BoardLink's reader thread to consume.
    """

    def __init__(self, port: str = "FAKE", baudrate: int = 115200,
                 timeout: float = 0.05, **_: object) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._rx = bytearray()
        self._rx_cv = threading.Condition()
        self._closed = False
        self._tx_handler: Optional[Callable[[bytes, "FakeSerial"], None]] = None

    # -- test-side helpers --
    def set_tx_handler(self, fn: Callable[[bytes, "FakeSerial"], None]) -> None:
        self._tx_handler = fn

    def push_rx(self, data: bytes) -> None:
        with self._rx_cv:
            self._rx.extend(data)
            self._rx_cv.notify_all()

    # -- pyserial-compatible API used by BoardLink --
    def write(self, data: bytes) -> int:
        if self._closed:
            raise serial.SerialException("port closed")
        if self._tx_handler is not None:
            self._tx_handler(bytes(data), self)
        return len(data)

    def flush(self) -> None:
        return None

    def read(self, n: int = 1) -> bytes:
        with self._rx_cv:
            if self._closed:
                raise serial.SerialException("port closed")
            if not self._rx:
                self._rx_cv.wait(timeout=self.timeout)
            if not self._rx:
                return b""
            out = bytes(self._rx[:n])
            del self._rx[:n]
            return out

    def close(self) -> None:
        with self._rx_cv:
            self._closed = True
            self._rx_cv.notify_all()


@pytest.fixture
def fake_serial(monkeypatch):
    """Replace serial.Serial inside the link module with FakeSerial."""
    holder: dict = {}

    def factory(*args, **kwargs):
        fs = FakeSerial(*args, **kwargs)
        holder["fs"] = fs
        return fs

    monkeypatch.setattr(link_mod.serial, "Serial", factory)
    yield holder
