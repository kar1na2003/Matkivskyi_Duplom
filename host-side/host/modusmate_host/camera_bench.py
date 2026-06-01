"""Camera benchmark: display dataset images fullscreen on the host PC monitor,
let the board's USB camera capture them, and collect classification results
over UART via the existing EVT_DETECTION / EVT_FPS telemetry.

No firmware changes required — uses the live camera pipeline (bench_mode=False).

Usage (CLI):
    modusmate-bench --port /dev/cu.usbmodem3102 --algo passthrough --limit 10 --mode camera

Usage (programmatic):
    from modusmate_host.camera_bench import run_camera_benchmark
    summary = run_camera_benchmark(port="/dev/cu.usbmodem3102", ...)
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .algos import ALGO_NAMES, FIRMWARE_ALGO_COUNT
from .dataset import load_samples
from .link import BoardLink, Detection, FpsReport

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# macOS sleep inhibitor
# ---------------------------------------------------------------------------

class _SleepInhibitor:
    """Prevents macOS from sleeping the display while the benchmark runs."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None

    def acquire(self) -> None:
        if sys.platform == "darwin":
            try:
                self._proc = subprocess.Popen(
                    ["caffeinate", "-d", "-i"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass  # caffeinate not available — ignore

    def release(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=2)
            self._proc = None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CamBenchProgress:
    """Progress update emitted by run_camera_benchmark."""
    kind: str           # "start" | "algo_begin" | "image" | "algo_end" |
                        # "done" | "error"
    message: str = ""
    algo_id: int = -1
    algo_name: str = ""
    image_index: int = 0
    image_total: int = 0
    algo_index: int = 0
    algo_total: int = 0
    correct: int = 0
    seen: int = 0
    no_detect: int = 0
    accuracy: float = 0.0
    mean_algo_us: int = 0
    mean_infer_us: int = 0


@dataclass
class CamBenchSummary:
    """Final result returned by run_camera_benchmark."""
    rows: List[Dict[str, object]] = field(default_factory=list)
    per_algo: Dict[int, Dict[str, float]] = field(default_factory=dict)
    samples_per_algo: int = 0
    cancelled: bool = False


ProgressCb = Callable[[CamBenchProgress], None]


def _emit(cb: Optional[ProgressCb], p: CamBenchProgress) -> None:
    if cb is not None:
        try:
            cb(p)
        except Exception:
            pass


def _resolve_algo_ids(spec: str) -> List[int]:
    if spec.lower() == "all":
        return list(range(FIRMWARE_ALGO_COUNT))
    out: List[int] = []
    for token in spec.split(","):
        t = token.strip()
        if not t:
            continue
        if t.isdigit():
            out.append(int(t))
        elif t in ALGO_NAMES:
            out.append(ALGO_NAMES.index(t))
        else:
            raise SystemExit(f"unknown algo: {t}")
    return out


# ---------------------------------------------------------------------------
# Fullscreen display window (OpenCV-based, works on macOS without Tk issues)
# ---------------------------------------------------------------------------

_CV_WINDOW = "ModusMate Camera Benchmark"


class _FullscreenDisplay:
    """Manages a fullscreen OpenCV window that shows dataset images."""

    def __init__(self) -> None:
        self._closed = False

    def create(self) -> None:
        """Create the fullscreen window."""
        cv2.namedWindow(_CV_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(_CV_WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        # Show initial black frame
        self.show_black()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def show_image(self, img_path: Path) -> None:
        """Display an image scaled to fill the screen (maintains aspect ratio)."""
        if self._closed:
            return
        img = cv2.imread(str(img_path))
        if img is None:
            return
        cv2.imshow(_CV_WINDOW, img)
        # Process events so the image is actually drawn
        key = cv2.waitKey(1)
        if key == 27:  # Escape
            self._closed = True

    def show_black(self) -> None:
        """Show a black screen."""
        if self._closed:
            return
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow(_CV_WINDOW, black)
        cv2.waitKey(1)

    def close(self) -> None:
        """Close the fullscreen window."""
        if not self._closed:
            self._closed = True
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    def pump_events(self) -> None:
        """Call periodically to keep the window responsive."""
        if not self._closed:
            key = cv2.waitKey(1)
            if key == 27:
                self._closed = True


# ---------------------------------------------------------------------------
# Core camera benchmark
# ---------------------------------------------------------------------------

def _cv_wait(display: _FullscreenDisplay, ms: int) -> None:
    """Wait for *ms* milliseconds while pumping OpenCV events."""
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        remaining = int((end - time.monotonic()) * 1000)
        wait_ms = min(max(remaining, 1), 100)
        key = cv2.waitKey(wait_ms)
        if key == 27:  # Escape
            display._closed = True
            return


def run_camera_benchmark(*,
                         port: str,
                         baud: int = 115200,
                         algo_ids: Optional[List[int]] = None,
                         algo_spec: str = "all",
                         limit: int = 50,
                         dataset_dir: Optional[Path] = None,
                         seed: int = 42,
                         stabilize_ms: int = 500,
                         capture_ms: int = 2000,
                         confidence_threshold: float = 0.30,
                         nn: Optional[str] = None,
                         fw_dir: Optional[Path] = None,
                         progress: Optional[ProgressCb] = None,
                         stop_event: Optional[threading.Event] = None,
                         display: Optional[_FullscreenDisplay] = None,
                         ) -> CamBenchSummary:
    """Run the camera benchmark.

    Creates a fullscreen OpenCV window (if no external display provided),
    displays dataset images, and collects detections from the board.

    When ``nn="auto"``, for each algorithm the matching trained model
    (with ``prep_algo == algo_name``) is flashed before benchmarking.
    When ``nn`` is a specific model name, that single model is flashed
    once before running all algorithms.
    """
    own_display = display is None
    if own_display:
        display = _FullscreenDisplay()
        display.create()

    inhibitor = _SleepInhibitor()
    inhibitor.acquire()

    try:
        return _do_camera_benchmark(
            display=display, port=port, baud=baud,
            algo_ids=algo_ids, algo_spec=algo_spec, limit=limit,
            dataset_dir=dataset_dir, seed=seed,
            stabilize_ms=stabilize_ms, capture_ms=capture_ms,
            confidence_threshold=confidence_threshold,
            nn=nn, fw_dir=fw_dir,
            progress=progress, stop_event=stop_event)
    finally:
        inhibitor.release()
        if own_display:
            display.close()


def _do_camera_benchmark(*,
                         port: str,
                         baud: int = 115200,
                         algo_ids: Optional[List[int]] = None,
                         algo_spec: str = "all",
                         limit: int = 50,
                         dataset_dir: Optional[Path] = None,
                         seed: int = 42,
                         stabilize_ms: int = 500,
                         capture_ms: int = 2000,
                         confidence_threshold: float = 0.30,
                         nn: Optional[str] = None,
                         fw_dir: Optional[Path] = None,
                         progress: Optional[ProgressCb] = None,
                         stop_event: Optional[threading.Event] = None,
                         display: _FullscreenDisplay,
                         ) -> CamBenchSummary:
    """Internal: run the benchmark loop (assumes display is already open)."""
    if algo_ids is None:
        algo_ids = _resolve_algo_ids(algo_spec)

    samples = load_samples(limit=limit, seed=seed, dataset_dir=dataset_dir)
    if not samples:
        _emit(progress, CamBenchProgress(kind="error",
              message="no samples found; check dataset"))
        return CamBenchSummary()

    # Build algo→model mapping when nn="auto"
    algo_to_model: Dict[str, str] = {}
    if nn == "auto":
        from . import models as _models
        all_models = _models.list_models()
        for m in all_models:
            if m.prep_algo:
                # Keep latest (last alphabetically) if multiple exist
                algo_to_model[m.prep_algo] = m.path.name
    elif nn is not None:
        # Flash a single model once before starting
        from . import models as _models
        _emit(progress, CamBenchProgress(kind="start",
              message=f"flashing model '{nn}'..."))
        try:
            _models.flash(nn,
                          fw_dir=fw_dir or _models.DEFAULT_FW_DIR,
                          port=port, baud=baud, verify=True)
        except _models.ModelInstallError as e:
            _emit(progress, CamBenchProgress(kind="error",
                  message=f"flash {nn} failed: {e}"))
            return CamBenchSummary()

    id_to_label = {s.label_id: s.label for s in samples}
    summary = CamBenchSummary(samples_per_algo=len(samples))

    _emit(progress, CamBenchProgress(
        kind="start",
        message=f"camera bench: {len(algo_ids)} algo(s), {len(samples)} images, "
                f"stabilize={stabilize_ms}ms, capture={capture_ms}ms",
        algo_total=len(algo_ids), image_total=len(samples)))

    # Open board link
    link = BoardLink(port, baudrate=baud)
    link.start()
    if not link.ping(timeout=2.0):
        _emit(progress, CamBenchProgress(kind="error",
              message=f"no response from {port}"))
        link.close()
        return summary

    # Ensure bench mode is OFF (we want live camera pipeline)
    link.set_bench(False)
    link.set_lcd(True)
    link.set_stream(False)

    # Detection collector
    det_lock = threading.Lock()
    det_buffer: List[Detection] = []
    fps_buffer: List[FpsReport] = []

    def _on_det(dets: List[Detection]) -> None:
        with det_lock:
            det_buffer.extend(dets)

    def _on_fps(r: FpsReport) -> None:
        with det_lock:
            fps_buffer.append(r)

    link.on_detections(_on_det)
    link.on_fps(_on_fps)

    # Per-algo accumulators
    per_total: Dict[int, int] = {}
    per_correct: Dict[int, int] = {}
    per_no_det: Dict[int, int] = {}
    per_algo_us: Dict[int, int] = {}
    per_infer_us: Dict[int, int] = {}

    cancelled = False
    last_flashed: Optional[str] = None
    try:
        for a_idx, aid in enumerate(algo_ids):
            if stop_event and stop_event.is_set():
                cancelled = True
                break

            name = ALGO_NAMES[aid] if aid < len(ALGO_NAMES) else f"algo{aid}"

            # Auto-flash the matching trained model for this algorithm
            if nn == "auto" and name in algo_to_model:
                model_name = algo_to_model[name]
                if model_name != last_flashed:
                    _emit(progress, CamBenchProgress(
                        kind="start",
                        message=f"flashing model '{model_name}' for {name}..."))
                    link.close()
                    try:
                        from . import models as _models
                        _models.flash(model_name,
                                      fw_dir=fw_dir or _models.DEFAULT_FW_DIR,
                                      port=port, baud=baud, verify=True)
                    except Exception as e:
                        _emit(progress, CamBenchProgress(
                            kind="error",
                            message=f"flash {model_name} failed: {e}; skipping {name}"))
                        # Reopen link for next algo
                        link = BoardLink(port, baudrate=baud)
                        link.start()
                        link.on_detections(_on_det)
                        link.on_fps(_on_fps)
                        continue
                    last_flashed = model_name
                    # Reopen link after flash
                    link = BoardLink(port, baudrate=baud)
                    link.start()
                    if not link.ping(timeout=3.0):
                        _emit(progress, CamBenchProgress(
                            kind="error",
                            message=f"board not responding after flash; aborting"))
                        break
                    link.set_bench(False)
                    link.set_lcd(True)
                    link.set_stream(False)
                    link.on_detections(_on_det)
                    link.on_fps(_on_fps)
            elif nn == "auto" and name not in algo_to_model:
                _emit(progress, CamBenchProgress(
                    kind="error",
                    message=f"no trained model for {name}; skipping"))
                continue

            if not link.set_algo(aid):
                _emit(progress, CamBenchProgress(
                    kind="error", message=f"could not select {name}; skipping"))
                continue

            per_total[aid] = 0
            per_correct[aid] = 0
            per_no_det[aid] = 0
            per_algo_us[aid] = 0
            per_infer_us[aid] = 0

            _emit(progress, CamBenchProgress(
                kind="algo_begin", algo_id=aid, algo_name=name,
                algo_index=a_idx, algo_total=len(algo_ids),
                image_total=len(samples)))

            for s_idx, s in enumerate(samples):
                if stop_event and stop_event.is_set():
                    cancelled = True
                    break

                # Show image on screen
                display.show_image(s.path)

                # Wait for stabilization (auto-exposure, camera latency)
                # Use cv2.waitKey to keep the window responsive
                _cv_wait(display, stabilize_ms)

                # Clear detection buffer and collect for capture_ms
                with det_lock:
                    det_buffer.clear()
                    fps_buffer.clear()

                _cv_wait(display, capture_ms)

                # Harvest detections
                with det_lock:
                    dets = list(det_buffer)
                    fps_reports = list(fps_buffer)

                # Pick best detection above threshold
                best_cls = 0xFF
                best_conf = 0.0
                for d in dets:
                    if d.conf >= confidence_threshold and d.conf > best_conf:
                        best_cls = d.class_id
                        best_conf = d.conf

                # Get timing from FPS reports
                algo_us = 0
                infer_us = 0
                if fps_reports:
                    algo_us = fps_reports[-1].algo_us
                    infer_us = fps_reports[-1].infer_us

                # Record result
                per_total[aid] += 1
                if best_cls == 0xFF:
                    per_no_det[aid] += 1
                    pred_label = ""
                else:
                    pred_label = id_to_label.get(best_cls, "")
                    if best_cls == s.label_id:
                        per_correct[aid] += 1
                per_algo_us[aid] += algo_us
                per_infer_us[aid] += infer_us

                summary.rows.append({
                    "algo": name, "algo_id": aid,
                    "image": str(s.path), "label": s.label,
                    "label_id": s.label_id,
                    "pred_id": best_cls if best_cls != 0xFF else -1,
                    "pred_label": pred_label,
                    "conf": round(best_conf, 4),
                    "algo_us": algo_us, "infer_us": infer_us,
                })

                tot = per_total[aid] or 1
                _emit(progress, CamBenchProgress(
                    kind="image", algo_id=aid, algo_name=name,
                    algo_index=a_idx, algo_total=len(algo_ids),
                    image_index=s_idx + 1, image_total=len(samples),
                    seen=per_total[aid], correct=per_correct[aid],
                    no_detect=per_no_det[aid],
                    accuracy=per_correct[aid] / tot,
                    mean_algo_us=per_algo_us[aid] // tot,
                    mean_infer_us=per_infer_us[aid] // tot))

            # End of algo
            if aid in per_total and per_total[aid] > 0:
                tot = per_total[aid]
                acc = per_correct[aid] / tot
                summary.per_algo[aid] = {
                    "name": name,
                    "seen": tot,
                    "correct": per_correct[aid],
                    "accuracy": acc,
                    "no_detect": per_no_det[aid],
                    "mean_algo_us": per_algo_us[aid] // tot,
                    "mean_infer_us": per_infer_us[aid] // tot,
                }
                _emit(progress, CamBenchProgress(
                    kind="algo_end", algo_id=aid, algo_name=name,
                    algo_index=a_idx, algo_total=len(algo_ids),
                    seen=tot, correct=per_correct[aid],
                    no_detect=per_no_det[aid], accuracy=acc,
                    mean_algo_us=per_algo_us[aid] // tot,
                    mean_infer_us=per_infer_us[aid] // tot,
                    message=f"acc={acc*100:.1f}%  algoUs={per_algo_us[aid]//tot}  "
                            f"inferUs={per_infer_us[aid]//tot}"))

    finally:
        display.show_black()
        try:
            link.close()
        except Exception:
            pass

    summary.cancelled = cancelled
    _emit(progress, CamBenchProgress(kind="done",
          message=f"{len(summary.rows)} total samples processed"))
    return summary
