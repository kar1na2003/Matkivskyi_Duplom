#!/usr/bin/env python3
"""End-to-end smoke test for a flashed ModusMate board.

Best run after installing the deterministic stump model so the bench
assertions are stable:

    python -m modusmate_host.models flash stump_const --port /dev/cu.usbmodem3102
    python host/scripts/smoke_test_board.py --port /dev/cu.usbmodem3102

Or in one shot:

    python host/scripts/smoke_test_board.py --port /dev/cu.usbmodem3102 --flash-stump

Phases:
    E1  Link sanity      - ping + get_info()
    E2  Stump bench      - push synthetic 320x240 RGB888, assert
                           class_id==0 && conf_x100==100
    E3  Algo sweep       - cycle all 43 firmware algos, log
                           algo_us / infer_us per algo
    E4  Preview frames   - stream 3 frames, assert dims and non-degeneracy
    E5  Stress           - 50 back-to-back bench pushes, drop count == 0

Exit code 0 on full PASS, 1 on any failure.
"""
from __future__ import annotations

import argparse
import os
import statistics
import struct
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Allow ``python host/scripts/smoke_test_board.py`` without install.
_HERE = Path(__file__).resolve().parent
_HOST = _HERE.parent
if str(_HOST) not in sys.path:
    sys.path.insert(0, str(_HOST))

from modusmate_host.algos import ALGO_NAMES, FIRMWARE_ALGO_COUNT  # type: ignore
from modusmate_host.link import BoardLink                          # type: ignore
from modusmate_host import models as _models                       # type: ignore


# ---- coloured pass/fail printing -------------------------------------------

class _Tag:
    PASS = "[PASS]"
    FAIL = "[FAIL]"
    INFO = "[ ..  ]"


def _print(tag: str, msg: str) -> None:
    print(f"{tag} {msg}")


# ---- synthetic test image --------------------------------------------------

def _synth_image(width: int = 320, height: int = 240) -> bytes:
    """Deterministic 320x240 RGB888 gradient with a checkerboard XOR.

    Not a real photo but has enough structure to be visually useful and
    enough bit changes to validate that no chunk has been silently zeroed."""
    buf = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            r = (x * 255) // (width - 1)
            g = (y * 255) // (height - 1)
            b = ((x + y) * 255) // (width + height - 2)
            if ((x // 16) ^ (y // 16)) & 1:
                r ^= 0x40
                g ^= 0x40
                b ^= 0x40
            o = (y * width + x) * 3
            buf[o + 0] = r
            buf[o + 1] = g
            buf[o + 2] = b
    return bytes(buf)


# ---- phases ----------------------------------------------------------------

def _phase_link_sanity(link: BoardLink) -> Tuple[bool, str]:
    if not link.ping(timeout=1.5):
        return False, "no PING ACK"
    info = link.get_info(timeout=2.0)
    if not info:
        return False, "GET_INFO returned no algos"
    n = len(info)
    return True, f"PING ok, GET_INFO returned {n} algos"


def _phase_stump_bench(link: BoardLink, expected: dict,
                       timeout: float) -> Tuple[bool, str]:
    img = _synth_image()
    # Make sure bench mode is active.
    if not link.set_bench(True):
        return False, "SET_BENCH(true) ACK missing"
    if not link.set_algo(0):
        return False, "SET_ALGO(passthrough) ACK missing"
    res = link.push_bench_image(img, 320, 240, timeout=timeout)
    if res is None:
        return False, "no BENCH_RESULT"
    exp_cls = int(expected.get("class_id", 0))
    exp_conf = int(expected.get("conf_x100", 100))
    got_conf = int(round(res.conf * 100))
    detail = (f"got class={res.class_id} conf={got_conf} "
              f"algo_us={res.algo_us} infer_us={res.infer_us}")
    if res.class_id != exp_cls:
        return False, f"class mismatch (want {exp_cls}); {detail}"
    if got_conf != exp_conf:
        return False, f"confidence mismatch (want {exp_conf}); {detail}"
    if res.algo_us == 0 or res.infer_us == 0:
        return False, f"timing zero ({detail})"
    return True, detail


def _phase_algo_sweep(link: BoardLink,
                      timeout: float,
                      algo_ids: List[int]) -> Tuple[bool, str]:
    img = _synth_image()
    rows: List[Tuple[int, str, int, int, int]] = []  # id,name,algo_us,inf_us,cls
    misses: List[str] = []
    for aid in algo_ids:
        name = ALGO_NAMES[aid] if aid < len(ALGO_NAMES) else f"id{aid}"
        if not link.set_algo(aid):
            misses.append(f"{name}(SET_ALGO no ACK)")
            continue
        res = link.push_bench_image(img, 320, 240, timeout=timeout)
        if res is None:
            misses.append(f"{name}(no result)")
            continue
        rows.append((aid, name, res.algo_us, res.infer_us, res.class_id))
    print(f"  algo sweep: {len(rows)}/{len(algo_ids)} returned a result")
    print(f"  {'id':>3}  {'name':<22} {'algo_us':>8} {'infer_us':>8}  cls")
    for aid, name, a_us, i_us, cls in rows:
        print(f"  {aid:>3}  {name:<22} {a_us:>8} {i_us:>8}  {cls}")
    if misses:
        return False, f"missed: {', '.join(misses)}"
    return True, f"{len(rows)} algos exercised cleanly"


def _phase_preview(link: BoardLink, frames: int = 3,
                   timeout: float = 5.0) -> Tuple[bool, str]:
    captured: List[Tuple[bytes, int, int]] = []
    done = threading.Event()

    def _cb(data: bytes, w: int, h: int) -> None:
        captured.append((data, w, h))
        if len(captured) >= frames:
            done.set()

    # Stop bench mode so the firmware allocates time to stream previews.
    link.set_bench(False)
    link.on_preview(_cb)
    if not link.set_stream(True):
        return False, "SET_STREAM(true) ACK missing"
    try:
        ok = done.wait(timeout=timeout)
    finally:
        link.set_stream(False)
        link.on_preview(lambda *_: None)
    if not ok:
        return False, f"only {len(captured)}/{frames} preview frames in {timeout:.1f}s"
    for i, (data, w, h) in enumerate(captured[:frames]):
        if w * h != len(data) or w != 80 or h != 60:
            return False, f"frame {i} bad shape {w}x{h} len={len(data)}"
        if all(b == data[0] for b in data):
            return False, f"frame {i} fully constant ({data[0]})"
    return True, f"{frames} frames @ 80x60, distinct pixel values"


def _phase_stress(link: BoardLink, n: int, timeout: float
                  ) -> Tuple[bool, str]:
    if not link.set_bench(True):
        return False, "SET_BENCH(true) ACK missing"
    if not link.set_algo(0):
        return False, "SET_ALGO(passthrough) ACK missing"
    img = _synth_image()
    rtts: List[float] = []
    drops = 0
    for i in range(n):
        t0 = time.monotonic()
        res = link.push_bench_image(img, 320, 240, timeout=timeout)
        dt = time.monotonic() - t0
        if res is None:
            drops += 1
            continue
        rtts.append(dt)
        time.sleep(0.05)
    if drops:
        return False, f"{drops}/{n} dropped"
    mean = statistics.mean(rtts) * 1000.0
    p95 = sorted(rtts)[int(0.95 * (len(rtts) - 1))] * 1000.0
    return True, f"{n}/{n} ok, mean={mean:.0f}ms p95={p95:.0f}ms"


# ---- main ------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True,
                   help="serial port (e.g. /dev/cu.usbmodem3102)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--flash-stump", action="store_true",
                   help="flash models/stump_const before the test")
    p.add_argument("--fw-dir", default=None,
                   help="firmware project dir override (only with --flash-stump)")
    p.add_argument("--skip-stress", action="store_true")
    p.add_argument("--bench-timeout", type=float, default=30.0,
                   help="per-image timeout for bench pushes (seconds). "
                        "At 115200 baud, 320x240 RGB takes ~20s on the wire "
                        "so the default leaves ~10s of headroom.")
    p.add_argument("--stress-count", type=int, default=5,
                   help="E5 bench-push count (each takes ~20s)")
    p.add_argument("--full-sweep", action="store_true",
                   help="E3 cycles all 43 firmware algos (~15 min). "
                        "Default is a 6-algo subset (~2 min).")
    p.add_argument("--algo-subset", default="0,4,9,11,13,38",
                   help="comma-separated algo IDs for E3 when --full-sweep "
                        "is not set (default covers passthrough, gaussian, "
                        "sobel, prewitt, kirsch, erode)")
    p.add_argument("--model", default="stump_const",
                   help="manifest name to load expected_smoke_output from")
    args = p.parse_args(argv)

    started = time.monotonic()

    # Optional: flash the stump first.
    if args.flash_stump:
        fw = Path(args.fw_dir) if args.fw_dir else _models.DEFAULT_FW_DIR
        try:
            _models.flash(args.model, fw_dir=fw,
                          port=args.port, baud=args.baud, verify=True)
        except _models.ModelInstallError as e:
            _print(_Tag.FAIL, f"flash {args.model}: {e}")
            return 1

    # Pull expected smoke output from the manifest (if any).
    try:
        manifest = _models.load_manifest(args.model)
    except FileNotFoundError as e:
        _print(_Tag.FAIL, str(e))
        return 1
    expected = manifest.expected_smoke_output or {}

    if not expected:
        _print(_Tag.INFO,
               f"model '{args.model}' has no expected_smoke_output; "
               "E2 will only check that a result arrives, not its value")

    results: List[Tuple[str, bool, str]] = []

    with BoardLink(args.port, baudrate=args.baud) as link:
        # E1 — link sanity
        ok, msg = _phase_link_sanity(link)
        _print(_Tag.PASS if ok else _Tag.FAIL, f"E1 link sanity: {msg}")
        results.append(("E1", ok, msg))
        if not ok:
            return 1

        # E2 — stump bench
        if expected:
            ok, msg = _phase_stump_bench(link, expected, args.bench_timeout)
        else:
            res = link.push_bench_image(_synth_image(), 320, 240,
                                        timeout=args.bench_timeout)
            ok = res is not None
            msg = ("got result" if ok else "no BENCH_RESULT")
        _print(_Tag.PASS if ok else _Tag.FAIL, f"E2 stump bench: {msg}")
        results.append(("E2", ok, msg))

        # E3 — algo sweep
        if args.full_sweep:
            algo_ids = list(range(FIRMWARE_ALGO_COUNT))
        else:
            algo_ids = [int(x) for x in args.algo_subset.split(",") if x.strip()]
        ok, msg = _phase_algo_sweep(link, args.bench_timeout, algo_ids)
        _print(_Tag.PASS if ok else _Tag.FAIL, f"E3 algo sweep: {msg}")
        results.append(("E3", ok, msg))

        # E4 — preview frames
        ok, msg = _phase_preview(link)
        _print(_Tag.PASS if ok else _Tag.FAIL, f"E4 preview frames: {msg}")
        results.append(("E4", ok, msg))

        # E5 — stress
        if args.skip_stress:
            _print(_Tag.INFO, "E5 stress: skipped")
            results.append(("E5", True, "skipped"))
        else:
            ok, msg = _phase_stress(link, args.stress_count, args.bench_timeout)
            _print(_Tag.PASS if ok else _Tag.FAIL, f"E5 stress: {msg}")
            results.append(("E5", ok, msg))

    elapsed = time.monotonic() - started
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print()
    print(f"summary: {n_pass} pass, {n_fail} fail in {elapsed:.1f}s")
    for name, ok, msg in results:
        print(f"  {name:>3}: {'PASS' if ok else 'FAIL'}  {msg}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
