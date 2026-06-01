# ModusMate — Global Agent Guide

This file is the entry point for any AI agent (Copilot, Copilot CLI, or
subagents) working in this repository. It gives orientation, the repo map,
how the two halves of the system interact, the canonical dev workflows,
and the rules that must not be broken.

Detailed agent playbooks (Project Creator workflow, Autonomous Debug
loop, NN model registry deep-dive, fault-register reference) live in
[host-side/AGENTS.md](host-side/AGENTS.md). When the user asks to
**create**, **debug**, **flash**, or work with **models**, jump there.

---

## 1. What this project is

**ModusMate** is an end-to-end image-processing + tiny-ML demo running
on an Infineon **PSoC Edge (PSE84) EPC2** kit:

- A **USB UVC camera** streams 320×240 frames to the board.
- The board runs **43+ classical image-processing algorithms**
  (Gaussian, Canny, Sobel, AKAZE, LBP, Otsu, watershed, …) on the
  CM55 core, then feeds the result into a **swappable Imagimob
  DeepCraft neural network** for classification.
- A **Python host** (CLI + Tk GUI) drives the board over UART:
  selects the algorithm, streams a preview, runs benchmarks against
  a Kaggle Rock-Paper-Scissors dataset, trains per-algorithm NNs,
  and swap-flashes the chosen model.

Two AI-agent surfaces live in this repo on top of that demo:

1. **Project Creator agent** — replaces the ModusToolbox Project
   Creator GUI. Takes natural-language prompts and scaffolds a new
   MTB project from a Code Example. See
   [host-side/AGENTS.md](host-side/AGENTS.md#project-creation-workflow).
2. **Autonomous Debug agent** — build → flash → UART capture →
   GDB backtrace → diagnose loop for runtime issues. See
   [host-side/AGENTS.md](host-side/AGENTS.md#autonomous-debug-agent).

---

## 2. Repo map

```
modusmate-bundle/
├── README.txt                       # Top-level quick start
├── AGENTS.md                        # ← you are here (global)
│
├── board-side/
│   └── ws-camera-imgproc-usb/       # ModusToolbox 3.x workspace
│       ├── camera-imgproc-usb/      # Multi-core MTB app
│       │   ├── Makefile             # TARGET=KIT_PSE84_EVAL_EPC2
│       │   ├── proj_cm33_s/         # CM33 secure
│       │   ├── proj_cm33_ns/        # CM33 non-secure (USB, UART)
│       │   ├── proj_cm55/           # CM55 (imgproc + NN inference)
│       │   │   ├── source/
│       │   │   │   ├── comm/        # UART framing (comm_proto.h)
│       │   │   │   ├── imgproc/     # 43 algorithm kernels
│       │   │   │   ├── inference_task.c/.h
│       │   │   │   ├── lcd_task.c/.h
│       │   │   │   ├── usb_camera_task.c/.h
│       │   │   │   ├── pipeline_config.c/.h
│       │   │   │   └── main.c
│       │   │   └── model/           # Current NN (model.c/.h) — swapped
│       │   ├── bsps/                # BSP — DO NOT MODIFY
│       │   └── configs/
│       └── mtb_shared/              # MTB libs — READ ONLY (deps)
│
├── host-side/
│   ├── AGENTS.md                    # Detailed agent playbooks (deep dive)
│   ├── README.md                    # Host setup & usage
│   ├── host/
│   │   ├── pyproject.toml           # name: modusmate-host
│   │   ├── modusmate_host/          # Python package
│   │   │   ├── protocol.py          # UART wire protocol (mirrors comm_proto.h)
│   │   │   ├── link.py              # BoardLink — serial + retries
│   │   │   ├── gui.py               # Tk live preview GUI
│   │   │   ├── benchmark.py         # modusmate-bench entry point
│   │   │   ├── sweep.py             # Multi-algo sweep
│   │   │   ├── algos.py             # Firmware algo registry mirror
│   │   │   ├── algo_lib.py          # NumPy reference impls
│   │   │   ├── algo_train.py        # Per-algo NN training
│   │   │   ├── algo_metrics.py
│   │   │   ├── c_export.py          # Imagimob ABI exporter
│   │   │   ├── dataset.py           # Kaggle RPS loader (kagglehub)
│   │   │   ├── full_pipeline.py
│   │   │   ├── models.py            # NN registry + install/flash CLI
│   │   │   └── plot_results.py
│   │   ├── scripts/
│   │   │   └── smoke_test_board.py  # E1–E5 end-to-end PASS/FAIL
│   │   └── tests/                   # pytest — protocol, algos, registry…
│   │
│   ├── models/                      # NN registry (~50 dirs)
│   │   ├── stump_const/             # Deterministic stub (class=0, conf=1.0)
│   │   ├── object_detect_rps/       # Original Imagimob RPS detector
│   │   └── <algo>_rps_YYYY-MM-DD/   # Per-algorithm trained models
│   │       ├── model.h
│   │       ├── model.c
│   │       └── manifest.json
│   │
│   └── project_creator/             # Project Creator agent CLI
│       ├── project_creator.py       # Main entry (6 subcommands)
│       ├── cli_wrapper.py           # Wraps project-creator-cli.exe
│       ├── build_wrapper.py         # make build/qprogram via modus-shell
│       ├── manifest_parser.py
│       ├── config.py
│       └── debug/
│           ├── gdb_default.gdb      # Generic Cortex-M fault dump
│           └── gdb_explorer.gdb     # PSE84-specific (NS core, HW BP)
```

---

## 3. How host-side and board-side interact

### Wire protocol (UART, 115200 8N1, KitProg3 debug COM)

Frame layout, mirrored in
[host-side/host/modusmate_host/protocol.py](host-side/host/modusmate_host/protocol.py)
and `board-side/.../proj_cm55/source/comm/comm_proto.h`:

```
[SOF=0xA5][TYPE u8][SEQ u8][LEN u16 LE][PAYLOAD][CRC16-CCITT u16 LE]
```

- CRC covers `TYPE..PAYLOAD`.
- Host → board commands carry an incrementing **SEQ**; board ACK echoes
  the SEQ. Board dedupes by `(TYPE, SEQ)`.
- Host retransmits on missing ACK (timeout 300 ms, up to 3 retries).
- Board → host telemetry (`EVT_FPS`, `EVT_DETECTION`, `EVT_FRAME_*`)
  is fire-and-forget, `SEQ=0`.
- The framer ignores non-frame bytes, so printf banner text and
  protocol frames share the same UART.

`PROTOCOL_VERSION = 2`. If you touch one side, touch the other.

### NN swap-and-flash

The board has room for **exactly one** model in Flash + SoCMEM. The
flow:

1. Host copies `models/<name>/model.{h,c}` into
   `board-side/.../camera-imgproc-usb/proj_cm55/model/`.
2. `make build -j` (via modus-shell).
3. `make qprogram` to flash.
4. Ping the board to verify the new ABI is live.

All models export the same Imagimob ABI
(`IMAI_init` / `IMAI_compute` / `IMAI_finalize`, fixed
`IMAI_DATAIN_SHAPE` / `IMAI_DATAOUT_SHAPE`) so the firmware needs
zero conditional code per model.

### Firmware path resolution

The host tools find the firmware tree via:

1. `--fw <DIR>` CLI flag
2. `$MODUSMATE_FW_DIR`
3. Default: `~/mtw-cli/ws-camera-imgproc-usb/camera-imgproc-usb`

In **this bundle** the firmware lives at
`board-side/ws-camera-imgproc-usb/camera-imgproc-usb` — pass it
explicitly, or set `MODUSMATE_FW_DIR` to that absolute path.

---

## 4. Common dev workflows

### One-time setup

```bash
# Host
cd host-side/host
python3 -m venv .venv
.venv/bin/pip install -e .[dev]
.venv/bin/pip install kagglehub

# Firmware deps (ModusToolbox 3.x must be installed)
cd ../../board-side/ws-camera-imgproc-usb/camera-imgproc-usb
make getlibs   # only if mtb_shared isn't already populated
```

### Build & flash the firmware

```bash
cd board-side/ws-camera-imgproc-usb/camera-imgproc-usb
make build -j
make qprogram        # flash only (after a build)
make program -j      # build + flash
```

On Windows the same commands must run inside **modus-shell** (Cygwin
bash) — the `project_creator/project_creator.py build|program|qprogram`
helpers handle that automatically.

### Swap & flash a NN model

```bash
cd host-side
export MODUSMATE_FW_DIR="$PWD/../board-side/ws-camera-imgproc-usb/camera-imgproc-usb"

python -m modusmate_host.models list
python -m modusmate_host.models flash stump_const --port /dev/cu.usbmodem<N>
```

### GUI

```bash
cd host-side/host
./run_gui.sh                 # or: modusmate-gui
```

Pick KitProg3 COM port → **Connect** → choose algorithm → toggle
**LCD on board** / **Stream preview to PC**.

### Benchmark & sweep

```bash
# Quick sanity check
modusmate-bench --port /dev/cu.usbmodem<N> --algo passthrough --limit 50

# Full sweep across all 43 firmware algorithms
modusmate-bench --port /dev/cu.usbmodem<N> --algo all --limit 200
# Outputs: results.csv, summary.md
```

### Smoke test (E1–E5)

```bash
cd host-side/host
python scripts/smoke_test_board.py --port /dev/cu.usbmodem<N> --flash-stump
```

Phases: link sanity → stump bench → 43-algo sweep → preview frames
→ 50× stress. Non-zero exit on any failure.

### Python tests

```bash
cd host-side/host
pytest
```

---

## 5. Agent workflows (pointers)

These workflows are fully specified in
[host-side/AGENTS.md](host-side/AGENTS.md). Use them when the user's
intent matches:

| User intent | Agent | Where |
|---|---|---|
| "create / scaffold / new project / blinky / wifi MQTT app" | **Project Creator** | host-side/AGENTS.md §Project Creation Workflow (Steps 1–9) |
| "debug / crash / HardFault / assert / hang / wrong output" | **Autonomous Debug** | host-side/AGENTS.md §Autonomous Debug Agent (Phases 0–8) |
| "list / install / flash / add a NN model" | **NN Registry** | host-side/AGENTS.md §NN Model Registry |
| "decode this fault register / CFSR / BFAR / MMFAR" | **Fault reference** | host-side/AGENTS.md §Fault Register Quick Reference |

The Project Creator CLI exposes **only** these six commands — do not
invent others:

```
project_creator/project_creator.py list-boards
project_creator/project_creator.py list-apps <BOARD_ID>
project_creator/project_creator.py create -b <BOARD_ID> -a <CE_ID> [-n NAME] [-d DIR]
project_creator/project_creator.py build    <PROJECT_DIR>
project_creator/project_creator.py program  <PROJECT_DIR>
project_creator/project_creator.py qprogram <PROJECT_DIR>
```

Plus the debug helpers: `debug`, `capture-uart`.

---

## 6. Rules an agent must follow

### Do NOT modify

- `board-side/**/bsps/` — BSP files (board-support package).
- `board-side/**/TARGET_*/` — generated board configs.
- `board-side/**/mtb_shared/` — third-party / Infineon libraries.
- `board-side/**/libs/`, `**/deps/` — resolved deps.
- The build system structure (`Makefile`, `common.mk`, `common_app.mk`)
  — don't rename them or change `MTB_TYPE`.
- Anything under `host-side/host/modusmate_host.egg-info/` — generated.

### Read-only references (you MAY look, NOT edit)

- Library headers under `mtb_shared/<lib>/*/include/*.h` are valid
  documentation for API signatures, struct fields and enums.
- Library READMEs under `mtb_shared/<lib>/*/README.md` for usage.

### Protocol invariants

- `PROTOCOL_VERSION = 2`. Any change to `comm_proto.h` requires the
  same change to `modusmate_host/protocol.py` in the same commit.
- All NN models must keep the Imagimob ABI
  (`IMAI_init/compute/finalize`, fixed `DATAIN`/`DATAOUT` shapes) so
  the firmware needs no per-model conditional code.
- The host UART decoder must remain tolerant of non-frame bytes
  (printf banner text shares the link).

### Tooling rules

- **Never grep / filter** Project Creator CLI output — read it fully
  and reason about it.
- **Use only the six documented Project Creator subcommands** plus
  `debug` / `capture-uart`. Do not invent flags.
- `build` / `program` / `qprogram` run inside modus-shell on Windows —
  always go through the Python wrapper, never call `make` directly
  from a regular Windows shell.
- Commands take 5–60 s (manifest download, git clone, build) —
  that is normal, wait for them.

### Code-modification etiquette

- Make **targeted** edits — don't refactor surrounding code, don't
  add docstrings/comments to code you didn't change, don't introduce
  helpers for one-time operations.
- When applying user configuration (WiFi SSID, MQTT broker, certs,
  …), **replace placeholder `#define`s in-place** — don't add new
  config layers.
- If a build fails, **read the full error, fix, rebuild**. Don't
  give up after one failure; max 5 iterations before reporting back.

---

## 7. Quick links

- Top-level quick start: [README.txt](README.txt)
- Host README (install, GUI, benchmark, models, protocol):
  [host-side/host/README.md](host-side/host/README.md)
- Detailed agent playbooks (Project Creator + Debug + NN registry +
  fault refs): [host-side/AGENTS.md](host-side/AGENTS.md)
- Wire protocol (canonical Python side):
  [host-side/host/modusmate_host/protocol.py](host-side/host/modusmate_host/protocol.py)
- Firmware entry: `board-side/ws-camera-imgproc-usb/camera-imgproc-usb/proj_cm55/source/main.c`
