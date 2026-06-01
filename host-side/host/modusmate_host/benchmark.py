"""Run the Kaggle Rock-Paper-Scissors benchmark across one or all algorithms.

For each (algorithm, image):
  1. SET_ALGO, SET_BENCH(true).
  2. Push the resized 320x240 RGB888 image (BENCH_BEGIN/CHUNK/END).
  3. Wait for the BENCH_RESULT event from the board.
  4. Record predicted class, confidence, algo_us, infer_us.

Output: results.csv (per-image rows) + summary.md (per-algo accuracy / mean times).
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import serial  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .algos import ALGO_NAMES, FIRMWARE_ALGO_COUNT
from .dataset import ID_TO_LABEL, load_image_320x240_rgb888, load_samples
from .link import BoardLink, BenchResult


def _resolve_algo_ids(spec: str) -> List[int]:
    if spec.lower() == "all":
        # Only enumerate algos the firmware actually implements; the trailing
        # entries in ALGO_NAMES are host-known stubs (e.g. mser, sift, surf)
        # that the board would time out on.
        return list(range(FIRMWARE_ALGO_COUNT))
    out: List[int] = []
    for token in spec.split(","):
        t = token.strip()
        if not t: continue
        if t.isdigit():
            out.append(int(t))
        elif t in ALGO_NAMES:
            out.append(ALGO_NAMES.index(t))
        else:
            raise SystemExit(f"unknown algo: {t}")
    return out


def _open_link(port: str, baud: int, max_attempts: int = 5,
               upgrade_baud: Optional[int] = None) -> Optional[BoardLink]:
    """Open a BoardLink, retrying for a few seconds while macOS re-enumerates
    a KitProg3 USB endpoint.  If ``upgrade_baud`` is given (and != ``baud``)
    we attempt to negotiate the higher rate after the initial ping; on any
    failure we keep the original baud.

    If the initial ping at ``baud`` fails we also try ``upgrade_baud`` (if
    given) as a fallback - this recovers from a previous run that left the
    firmware at a non-default rate without rebooting."""
    delay = 0.5
    last_err: Optional[Exception] = None
    candidates = [baud]
    if upgrade_baud and upgrade_baud != baud:
        candidates.append(upgrade_baud)
    for _ in range(max_attempts):
        for cand in candidates:
            try:
                link = BoardLink(port, baudrate=cand)
                link.start()
                if link.ping(timeout=2.0):
                    # Always end up at upgrade_baud (if requested) so the
                    # caller sees the negotiated rate regardless of which
                    # candidate succeeded.
                    target = upgrade_baud if upgrade_baud else baud
                    if cand != target:
                        link.set_baudrate(target)
                    return link
                link.close()
            except (serial.SerialException, OSError) as e:
                last_err = e
        time.sleep(delay)
        delay = min(delay * 2, 4.0)
    if last_err is not None:
        print(f"[bench] could not (re)open {port}: {last_err}")
    return None


# --------------------------------------------------------------- programmatic API
@dataclass
class BenchProgress:
    """Single update emitted by ``run_benchmark`` for the GUI / CLI."""
    kind: str                 # "start" | "algo_begin" | "image_loading" |
                              # "image_begin" | "image_processed" | "chunk" |
                              # "image" | "algo_end" | "done" | "error"
    message: str = ""
    algo_id: int = -1
    algo_name: str = ""
    image_index: int = 0      # 0-based within current algo
    image_total: int = 0      # samples per algo
    algo_index: int = 0       # 0-based across selected algos
    algo_total: int = 0
    correct: int = 0
    seen: int = 0
    no_detect: int = 0
    mean_algo_us: int = 0
    mean_infer_us: int = 0
    mean_conf: float = 0.0
    accuracy: float = 0.0
    # Live preview payloads (large; consumed-then-dropped by the GUI).
    image_rgb: Optional[bytes] = None       # 320x240 RGB888 host input
    preview_gray: Optional[bytes] = None    # board's post-algo preview
    preview_w: int = 0
    preview_h: int = 0


@dataclass
class BenchSummary:
    """Final result returned by ``run_benchmark``."""
    rows: List[Dict[str, object]] = field(default_factory=list)
    per_algo: Dict[int, Dict[str, float]] = field(default_factory=dict)
    samples_per_algo: int = 0
    cancelled: bool = False
    nn_name: Optional[str] = None


ProgressCb = Callable[[BenchProgress], None]


def _emit(cb: Optional[ProgressCb], p: BenchProgress) -> None:
    if cb is not None:
        try:
            cb(p)
        except Exception:
            pass


def run_benchmark(*,
                  port: str,
                  baud: int = 115200,
                  algo_ids: Optional[List[int]] = None,
                  algo_spec: str = "all",
                  limit: int = 200,
                  dataset_dir: Optional[Path] = None,
                  seed: int = 42,
                  per_image_timeout: float = 30.0,
                  max_reconnects: int = 20,
                  csv_out: Optional[Path] = None,
                  summary_out: Optional[Path] = None,
                  progress: Optional[ProgressCb] = None,
                  stop_event: Optional[threading.Event] = None,
                  nn: Optional[str] = None,
                  fw_dir: Optional[Path] = None,
                  upgrade_baud: Optional[int] = None,
                  enable_preview: bool = True,
                  save_processed_dir: Optional[Path] = None,
                  ) -> BenchSummary:
    """Run the RPS benchmark and return aggregated results.

    ``algo_ids`` overrides ``algo_spec`` if provided. ``progress`` is invoked
    from the calling thread on every state change. ``stop_event`` lets a GUI
    abort cleanly between images.  When ``nn`` is set we install + flash the
    matching model from ``models/<nn>/`` before opening the link.

    When ``save_processed_dir`` is provided each (input, board-post-algo)
    image pair is written to ``<dir>/<algo_name>/<idx>_{in,out}.png`` so the
    user can inspect what each algorithm produced. Streaming is implicitly
    enabled in this mode regardless of ``enable_preview`` because the
    "out" image *is* the preview frame.
    """
    if nn is not None:
        from . import models as _models                  # local import
        try:
            _emit(progress, BenchProgress(kind="start",
                  message=f"flashing model '{nn}' before benchmark…"))
            _models.flash(nn,
                          fw_dir=fw_dir or _models.DEFAULT_FW_DIR,
                          port=port, baud=baud, verify=True)
        except _models.ModelInstallError as e:
            _emit(progress, BenchProgress(kind="error",
                                          message=f"flash {nn} failed: {e}"))
            return BenchSummary(samples_per_algo=0, nn_name=nn)

    if algo_ids is None:
        algo_ids = _resolve_algo_ids(algo_spec)
    samples = load_samples(limit=limit, seed=seed, dataset_dir=dataset_dir)
    if not samples:
        _emit(progress, BenchProgress(kind="error",
                                      message="no samples found; check Kaggle "
                                              "credentials or --dataset-dir"))
        return BenchSummary(samples_per_algo=0, nn_name=nn)

    # When using a generic class-folder dataset (e.g. one written by
    # ``algo_train --save-test-dir``) the labels are arbitrary, so the
    # built-in ID_TO_LABEL (rock/paper/scissors) doesn't apply. Build a
    # local mapping from whatever labels the loader produced.
    id_to_label = {s.label_id: s.label for s in samples}

    summary = BenchSummary(samples_per_algo=len(samples), nn_name=nn)
    _emit(progress, BenchProgress(kind="start",
                                  message=f"{len(algo_ids)} algorithm(s), "
                                          f"{len(samples)} samples each",
                                  algo_total=len(algo_ids),
                                  image_total=len(samples)))

    per_total: Dict[int, int] = defaultdict(int)
    per_correct: Dict[int, int] = defaultdict(int)
    per_a_us: Dict[int, int] = defaultdict(int)
    per_i_us: Dict[int, int] = defaultdict(int)
    per_no_det: Dict[int, int] = defaultdict(int)
    per_conf: Dict[int, float] = defaultdict(float)

    link = _open_link(port, baud, upgrade_baud=upgrade_baud)
    if link is None:
        _emit(progress, BenchProgress(kind="error",
                                      message=f"no response from {port}"))
        return summary
    if upgrade_baud and upgrade_baud != baud:
        _emit(progress, BenchProgress(
            kind="start",
            message=f"link baud upgraded {baud} -> {upgrade_baud}"))
    reconnects = 0

    def configure(l: BoardLink) -> bool:
        return (l.set_lcd(False) and l.set_stream(False) and l.set_bench(True))

    if not configure(link):
        _emit(progress, BenchProgress(kind="error",
                                      message="failed to set initial state"))
        try: link.close()
        except Exception: pass
        return summary

    # Forward every board preview frame onto the progress channel and
    # turn streaming on so the firmware emits one preview per processed
    # bench frame.  The GUI displays it next to the host-side input
    # image, giving a live "what the board sees" view.
    cur_algo: Dict[str, object] = {"id": -1, "name": ""}

    # Latest preview frame captured from the board for the current
    # bench image. Filled by `_fwd_preview` and consumed (then cleared)
    # right after each BENCH_RESULT, so the (input, output) pair we
    # write to disk really corresponds to the same image.
    latest_preview: Dict[str, object] = {"data": None, "w": 0, "h": 0}

    def _fwd_preview(data: bytes, w: int, h: int) -> None:
        latest_preview["data"] = data
        latest_preview["w"] = w
        latest_preview["h"] = h
        _emit(progress, BenchProgress(
            kind="image_processed",
            algo_id=int(cur_algo["id"]),
            algo_name=str(cur_algo["name"]),
            preview_gray=data, preview_w=w, preview_h=h))

    def _enable_stream(l: BoardLink) -> None:
        l.on_preview(_fwd_preview)
        try:
            l.set_stream(True)
        except Exception:
            pass
    # If the user asked for processed-image dumps, force-enable preview
    # streaming (we need the board's post-algo frame to write the "out"
    # PNG) regardless of the GUI's "show preview" checkbox.
    if enable_preview or save_processed_dir is not None:
        _enable_stream(link)

    # Lazy PIL import so the CLI import path stays cheap even when
    # save_processed_dir is unused.
    _PIL = None
    if save_processed_dir is not None:
        try:
            from PIL import Image as _PIL_Image  # type: ignore
            _PIL = _PIL_Image
        except Exception as e:
            _emit(progress, BenchProgress(
                kind="error",
                message=f"--save-processed needs Pillow ({e}); "
                        "skipping image dump"))
            save_processed_dir = None
        else:
            save_processed_dir.mkdir(parents=True, exist_ok=True)

    cancelled = False
    try:
        for a_idx, aid in enumerate(algo_ids):
            if stop_event is not None and stop_event.is_set():
                cancelled = True
                break
            name = ALGO_NAMES[aid] if aid < len(ALGO_NAMES) else f"algo{aid}"
            if not link.set_algo(aid):
                _emit(progress, BenchProgress(
                    kind="error", message=f"could not select {name}; skipping",
                    algo_id=aid, algo_name=name,
                    algo_index=a_idx, algo_total=len(algo_ids)))
                continue
            cur_algo["id"] = aid
            cur_algo["name"] = name
            _emit(progress, BenchProgress(
                kind="algo_begin", algo_id=aid, algo_name=name,
                algo_index=a_idx, algo_total=len(algo_ids),
                image_total=len(samples)))
            t0 = time.time()
            for s_idx, s in enumerate(samples):
                if stop_event is not None and stop_event.is_set():
                    cancelled = True
                    break
                # Stage 1: load + decode the image from disk.
                _emit(progress, BenchProgress(
                    kind="image_loading", algo_id=aid, algo_name=name,
                    algo_index=a_idx, algo_total=len(algo_ids),
                    image_index=s_idx + 1, image_total=len(samples),
                    message=s.path.name))
                try:
                    rgb = load_image_320x240_rgb888(s.path)
                except Exception as e:
                    _emit(progress, BenchProgress(
                        kind="error",
                        message=f"skip {s.path.name}: {e}",
                        algo_id=aid, algo_name=name))
                    continue

                # Surface the input frame to the GUI before we push it.
                _emit(progress, BenchProgress(
                    kind="image_begin", algo_id=aid, algo_name=name,
                    algo_index=a_idx, algo_total=len(algo_ids),
                    image_index=s_idx + 1, image_total=len(samples),
                    image_rgb=rgb))

                # Clear the previous frame's preview so we never write a
                # stale "out" image if streaming is interrupted on this
                # bench image.
                latest_preview["data"] = None

                res: Optional[BenchResult] = None
                try:
                    def _on_chunk(sent: int, total: int,
                                  _aid=aid, _name=name, _aidx=a_idx,
                                  _sidx=s_idx) -> None:
                        _emit(progress, BenchProgress(
                            kind="chunk", algo_id=_aid, algo_name=_name,
                            algo_index=_aidx, algo_total=len(algo_ids),
                            image_index=_sidx + 1, image_total=len(samples),
                            seen=sent, mean_algo_us=total))
                    res = link.push_bench_image(rgb, 320, 240,
                                                timeout=per_image_timeout,
                                                on_chunk=_on_chunk)
                except (serial.SerialException, OSError) as e:
                    _emit(progress, BenchProgress(
                        kind="error",
                        message=f"serial error: {e}; reconnecting...",
                        algo_id=aid, algo_name=name))
                    try: link.close()
                    except Exception: pass
                    reconnects += 1
                    if reconnects > max_reconnects:
                        _emit(progress, BenchProgress(
                            kind="error",
                            message="too many reconnects; aborting"))
                        raise
                    new_link = _open_link(port, baud, upgrade_baud=upgrade_baud)
                    if new_link is None:
                        _emit(progress, BenchProgress(
                            kind="error",
                            message="reconnect failed; aborting"))
                        raise
                    link = new_link
                    if not configure(link) or not link.set_algo(aid):
                        raise serial.SerialException("re-configure failed")
                    if enable_preview:
                        _enable_stream(link)
                    res = None  # this image counts as no-result

                if res is None:
                    summary.rows.append({"algo": name, "algo_id": aid,
                                         "image": str(s.path), "label": s.label,
                                         "label_id": s.label_id, "pred_id": -1,
                                         "pred_label": "", "conf": 0.0,
                                         "algo_us": 0, "infer_us": 0})
                    per_total[aid] += 1
                    per_no_det[aid] += 1
                else:
                    pred_label = id_to_label.get(res.class_id, "")
                    summary.rows.append({"algo": name, "algo_id": aid,
                                         "image": str(s.path), "label": s.label,
                                         "label_id": s.label_id,
                                         "pred_id": res.class_id,
                                         "pred_label": pred_label,
                                         "conf": round(res.conf, 4),
                                         "algo_us": res.algo_us,
                                         "infer_us": res.infer_us})
                    per_total[aid] += 1
                    if res.class_id == s.label_id:
                        per_correct[aid] += 1
                    if res.class_id == 0xFF:
                        per_no_det[aid] += 1
                    per_a_us[aid] += res.algo_us
                    per_i_us[aid] += res.infer_us
                    per_conf[aid] += res.conf

                # Save the (input, processed) pair for this image if
                # requested.  We do this whether or not the firmware
                # produced a usable detection, because the user is
                # interested in the *image* output of the algorithm.
                if save_processed_dir is not None and _PIL is not None:
                    try:
                        out_dir = save_processed_dir / name
                        out_dir.mkdir(parents=True, exist_ok=True)
                        idx_str = f"{s_idx + 1:04d}"
                        in_img = _PIL.frombytes(
                            "RGB", (320, 240), bytes(rgb))
                        in_img.save(out_dir / f"{idx_str}_in.png")
                        pv = latest_preview.get("data")
                        if pv is not None:
                            pw = int(latest_preview["w"])
                            ph = int(latest_preview["h"])
                            if pw > 0 and ph > 0 and pw * ph == len(pv):
                                out_img = _PIL.frombytes(
                                    "L", (pw, ph), bytes(pv))
                                out_img.save(
                                    out_dir / f"{idx_str}_out.png")
                    except Exception as _save_e:
                        _emit(progress, BenchProgress(
                            kind="error",
                            message=f"save image failed: {_save_e}"))

                tot = per_total[aid] or 1
                _emit(progress, BenchProgress(
                    kind="image", algo_id=aid, algo_name=name,
                    algo_index=a_idx, algo_total=len(algo_ids),
                    image_index=s_idx + 1, image_total=len(samples),
                    seen=tot, correct=per_correct[aid],
                    no_detect=per_no_det[aid],
                    mean_algo_us=per_a_us[aid] // tot,
                    mean_infer_us=per_i_us[aid] // tot,
                    mean_conf=per_conf[aid] / tot,
                    accuracy=per_correct[aid] / tot))

            elapsed = time.time() - t0
            tot = per_total[aid] or 1
            acc = per_correct[aid] / tot
            mean_a = per_a_us[aid] // tot
            mean_i = per_i_us[aid] // tot
            summary.per_algo[aid] = {
                "name": name,
                "seen": tot,
                "correct": per_correct[aid],
                "accuracy": acc,
                "no_detect": per_no_det[aid],
                "mean_conf": per_conf[aid] / tot,
                "mean_algo_us": mean_a,
                "mean_infer_us": mean_i,
                "elapsed_s": elapsed,
            }
            _emit(progress, BenchProgress(
                kind="algo_end", algo_id=aid, algo_name=name,
                algo_index=a_idx, algo_total=len(algo_ids),
                seen=tot, correct=per_correct[aid],
                no_detect=per_no_det[aid],
                mean_algo_us=mean_a, mean_infer_us=mean_i,
                mean_conf=per_conf[aid] / tot, accuracy=acc,
                message=f"acc={acc*100:.1f}%  algoUs={mean_a}  "
                        f"inferUs={mean_i}  ({elapsed:.1f}s)"))
    finally:
        # Best-effort: reset the firmware baud back to the boot/default rate
        # so a subsequent run (or any other tool) doesn't have to guess where
        # the link was left.  Done before close so the ACK can drain.
        if upgrade_baud and upgrade_baud != baud:
            try: link.set_baudrate(baud)
            except Exception: pass
        try: link.set_stream(False)
        except Exception: pass
        try: link.set_bench(False)
        except Exception: pass
        try: link.close()
        except Exception: pass

    summary.cancelled = cancelled

    if csv_out is not None:
        fieldnames = ["algo", "algo_id", "image", "label", "label_id",
                      "pred_id", "pred_label", "conf", "algo_us", "infer_us"]
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary.rows)

    if summary_out is not None:
        lines = ["# ModusMate benchmark summary",
                 f"- samples per algorithm: {summary.samples_per_algo}",
                 "",
                 "| Algorithm | Accuracy | Correct | No-detect | Mean conf "
                 "| Mean algo µs | Mean infer µs |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for aid in algo_ids:
            row = summary.per_algo.get(aid)
            if row is None:
                continue
            lines.append(f"| {row['name']} | {row['accuracy']*100:5.1f}% | "
                         f"{row['correct']}/{row['seen']} | "
                         f"{row['no_detect']} | {row['mean_conf']:.2f} | "
                         f"{row['mean_algo_us']} | {row['mean_infer_us']} |")
        summary_out.write_text("\n".join(lines), encoding="utf-8")

    _emit(progress, BenchProgress(kind="done",
                                  message="cancelled" if cancelled else "done"))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True, help="Serial port (e.g. /dev/cu.usbmodem... or COM4)")
    ap.add_argument("--baud", type=int, default=115200,
                    help="initial/handshake baud; firmware always boots at 115200")
    ap.add_argument("--mode", choices=["uart", "camera"], default="uart",
                    help="'uart' pushes images over serial (default); "
                         "'camera' displays images fullscreen on the PC and "
                         "lets the board's camera capture them")
    ap.add_argument("--upgrade-baud", type=int, default=1000000,
                    help="after first ping, send CMD_SET_BAUDRATE to switch "
                         "the link to this rate (default 1000000). KitProg3 "
                         "VCOM is reliable up to ~3 Mbaud. Cuts the 230 KB "
                         "image push from ~20 s to ~1 s. Pass 115200 to disable.")
    ap.add_argument("--algo", default="all", help="'all' or comma-separated names/ids")
    ap.add_argument("--limit", type=int, default=200, help="max images per algorithm (0 = all)")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--summary", default="summary.md")
    ap.add_argument("--dataset-dir", default=None,
                    help="path to a pre-downloaded RPS dataset; if omitted, kagglehub fetches it")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-image-timeout", type=float, default=30.0,
                    help="seconds; must exceed UART transfer time "
                         "(~20s for 320x240 RGB at 115200 baud)")
    ap.add_argument("--max-reconnects", type=int, default=20,
                    help="abort after this many serial-disconnect events")
    ap.add_argument("--nn", default=None,
                    help="name of a model under models/ to flash before "
                         "running the benchmark (e.g. stump_const)")
    ap.add_argument("--fw-dir", default=None,
                    help="firmware project dir override (only used with --nn)")
    ap.add_argument("--save-processed", default=None, metavar="DIR",
                    help="directory to dump (input, post-algo) PNG pairs "
                         "per image. Useful when running without --nn to "
                         "inspect what each algorithm produces. Files are "
                         "written as <DIR>/<algo>/<idx>_in.png + _out.png")
    # Camera-mode specific options
    ap.add_argument("--stabilize-ms", type=int, default=500,
                    help="(camera mode) ms to wait after displaying image "
                         "for camera auto-exposure to settle")
    ap.add_argument("--capture-ms", type=int, default=2000,
                    help="(camera mode) ms to collect detections per image")
    ap.add_argument("--confidence-threshold", type=float, default=0.30,
                    help="(camera mode) minimum confidence to accept a detection")
    args = ap.parse_args()

    algo_ids = _resolve_algo_ids(args.algo)

    # --- Camera mode ---
    if args.mode == "camera":
        from .camera_bench import run_camera_benchmark, CamBenchProgress
        print(f"[bench] camera mode: {len(algo_ids)} algorithm(s); loading dataset...")

        def _on_cam_progress(p: CamBenchProgress) -> None:
            if p.kind == "start":
                print(f"[bench] {p.message}")
            elif p.kind == "algo_begin":
                print(f"[bench] {p.algo_name}: ", end="", flush=True)
            elif p.kind == "image":
                if p.image_index % 5 == 0:
                    print(".", end="", flush=True)
            elif p.kind == "algo_end":
                print(f" acc={p.accuracy*100:5.1f}% "
                      f"({p.correct}/{p.seen}) "
                      f"meanAlgo={p.mean_algo_us} µs "
                      f"meanInf={p.mean_infer_us} µs")
            elif p.kind == "error":
                print(f"\n[bench] ERROR: {p.message}")
            elif p.kind == "done":
                print(f"[bench] {p.message}")

        cam_summary = run_camera_benchmark(
            port=args.port,
            baud=args.baud,
            algo_ids=algo_ids,
            limit=args.limit,
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            seed=args.seed,
            stabilize_ms=args.stabilize_ms,
            capture_ms=args.capture_ms,
            confidence_threshold=args.confidence_threshold,
            nn=args.nn,
            fw_dir=Path(args.fw_dir) if args.fw_dir else None,
            progress=_on_cam_progress,
        )
        if not cam_summary.rows:
            return 2
        # Write CSV
        import csv as _csv
        csv_p = Path(args.out)
        fieldnames = ["algo", "algo_id", "image", "label", "label_id",
                      "pred_id", "pred_label", "conf", "algo_us", "infer_us"]
        with csv_p.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(cam_summary.rows)
        # Write summary
        md_p = Path(args.summary)
        lines = ["# ModusMate camera benchmark summary",
                 f"- samples per algorithm: {cam_summary.samples_per_algo}",
                 f"- stabilize: {args.stabilize_ms} ms, capture: {args.capture_ms} ms",
                 "",
                 "| Algorithm | Accuracy | Correct | No-detect "
                 "| Mean algo µs | Mean infer µs |",
                 "|---|---:|---:|---:|---:|---:|"]
        for aid in algo_ids:
            row = cam_summary.per_algo.get(aid)
            if row is None:
                continue
            lines.append(f"| {row['name']} | {row['accuracy']*100:5.1f}% | "
                         f"{row['correct']}/{row['seen']} | "
                         f"{row['no_detect']} | "
                         f"{row['mean_algo_us']} | {row['mean_infer_us']} |")
        md_p.write_text("\n".join(lines), encoding="utf-8")
        print(f"[bench] wrote {args.out}")
        print(f"[bench] wrote {args.summary}")
        return 0

    # --- UART mode (default) ---
    print(f"[bench] {len(algo_ids)} algorithm(s); loading dataset...")

    def _on_progress(p: BenchProgress) -> None:
        if p.kind == "start":
            print(f"[bench] {p.message}")
        elif p.kind == "algo_begin":
            print(f"[bench] {p.algo_name}: ", end="", flush=True)
        elif p.kind == "image":
            if p.image_index % 20 == 0:
                print(".", end="", flush=True)
        elif p.kind == "algo_end":
            print(f" acc={p.accuracy*100:5.1f}% "
                  f"({p.correct}/{p.seen}) "
                  f"meanAlgo={p.mean_algo_us} µs "
                  f"meanInf={p.mean_infer_us} µs")
        elif p.kind == "error":
            print(f"\n[bench] {p.message}")
        elif p.kind == "done":
            print(f"[bench] {p.message}")

    summary = run_benchmark(
        port=args.port,
        baud=args.baud,
        algo_ids=algo_ids,
        limit=args.limit,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        seed=args.seed,
        per_image_timeout=args.per_image_timeout,
        max_reconnects=args.max_reconnects,
        csv_out=Path(args.out),
        summary_out=Path(args.summary),
        progress=_on_progress,
        nn=args.nn,
        fw_dir=Path(args.fw_dir) if args.fw_dir else None,
        upgrade_baud=args.upgrade_baud,
        save_processed_dir=Path(args.save_processed) if args.save_processed else None,
    )
    if not summary.rows:
        return 2
    print(f"[bench] wrote {args.out}")
    print(f"[bench] wrote {args.summary}")
    if args.save_processed:
        print(f"[bench] saved per-image PNGs under {args.save_processed}/<algo>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
