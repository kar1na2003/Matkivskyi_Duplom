# ModusMate host tools

Python host-side controller, live GUI and benchmark runner for the
ModusMate PSoC Edge EPC2 image-processing firmware.

## Install

```bash
cd host
pip install -e .[dev]
```

Dependencies: `pyserial`, `Pillow`, `numpy`, `kagglehub` (auto-installed).

## Run

### Live GUI

```bash
python -m modusmate_host.gui      # or: modusmate-gui
```

Workflow: pick the KitProg3 COM port (macOS `/dev/cu.usbmodem...`,
Linux `/dev/ttyACM*`, Windows `COMx`), **Connect**, select an algorithm
from the dropdown, toggle **LCD on board** and **Stream preview to PC**
as desired. FPS, latest detection, and an 80×60 grayscale preview
update live.

### Benchmark

Downloads the Kaggle Rock-Paper-Scissors dataset (`drgfreeman/rockpaperscissors`)
on first run via `kagglehub` and caches it in `~/.modusmate/datasets/rps/`.
For every algorithm × image pair it sends the image to the board, runs
the on-device image-processing stage, then the AI, and records the
prediction + timings.

```bash
# single algorithm, 50 images, quick sanity check
modusmate-bench --port /dev/cu.usbmodem... --algo passthrough --limit 50

# full sweep
modusmate-bench --port /dev/cu.usbmodem... --algo all --limit 200
```

Outputs:
- `results.csv` — one row per (algorithm, image) with predicted class, confidence and timings.
- `summary.md` — per-algorithm accuracy, mean `algo_us`, mean `infer_us`.

### Tests

```bash
cd host
pytest
```

## Neural-network swap & flash

The board has room for exactly one model in Flash + SoCMEM, so the host
treats the on-device NN as a swappable artefact: pick a model from
`models/<name>/`, copy its `model.h` + `model.c` into the firmware tree,
rebuild, re-flash. Each entry under `models/` carries a `manifest.json`
describing input/output shapes and (optionally) a deterministic
`expected_smoke_output` for the smoke test.

Two models ship with the repo:

- `object_detect_rps` — the original Imagimob DeepCraft RPS detector
  (320×320 RGB → 8×5 float, 1.7 MB Flash).
- `stump_const` — a minimal stub with the same Imagimob ABI that always
  returns `class=0, conf=1.0`. Use it to validate the bench protocol and
  the reflash pipeline end-to-end without depending on a real classifier.

```bash
# list models
python -m modusmate_host.models list

# install one (copies into <fw>/proj_cm55/model/, no build/flash)
python -m modusmate_host.models install stump_const --fw ~/mtw-cli/...

# build + flash + verify with a ping
python -m modusmate_host.models flash stump_const --port /dev/cu.usbmodem3102

# inside a benchmark run
modusmate-bench --port /dev/cu.usbmodem3102 --algo all --nn stump_const
```

The firmware path defaults to `$MODUSMATE_FW_DIR` (or
`~/mtw-cli/ws-camera-imgproc-usb/camera-imgproc-usb` if unset).

## End-to-end smoke test

Once a model is flashed, `host/scripts/smoke_test_board.py` exercises
the full link → bench → preview path and prints PASS/FAIL per phase.
Use the stump for stable assertions:

```bash
python scripts/smoke_test_board.py \
    --port /dev/cu.usbmodem3102 --flash-stump
```

Phases: link sanity (ping + get_info), stump bench round-trip,
sweep all 43 firmware algos, capture preview frames, 50× bench stress.

## Protocol

All constants mirror `proj_cm55/source/comm/comm_proto.h`. Frame layout:

```
[SOF=0xA5][TYPE u8][LEN u16 LE][PAYLOAD][CRC16-CCITT u16 LE]
```

CRC covers `TYPE..PAYLOAD`. UART is 115200 8N1 on the KitProg3 debug
COM port — the same port that prints firmware boot banner text. The
Python framer discards any non-frame bytes silently so printf and
protocol frames can share the link.

## Dataset location

If you already have the RPS dataset locally, skip the download with:

```bash
modusmate-bench --port ... --dataset-dir /path/to/rps
```

The loader auto-detects class subdirectories named `rock/`, `paper/`, `scissors/`.
