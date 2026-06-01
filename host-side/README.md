# Project Creator Agent

An AI-powered replacement for the ModusToolbox Project Creator tool. Takes natural language prompts and creates customized embedded projects by selecting the best Code Example (CE) and modifying it to match your requirements.

## Architecture

- **`project_creator/`** — Python CLI tool that wraps `project-creator-cli.exe`
- **`AGENTS.md`** — Instructions for Copilot CLI to act as the intelligent project creator

## Prerequisites

- **Python 3.8+** (no external packages needed)
- **ModusToolbox** installed (provides `project-creator-cli.exe`)
- **Copilot CLI** for the AI-powered workflow

## Quick Start

### 1. Verify setup

```bash
# Check that project-creator-cli is detected
python project_creator/project_creator.py list-boards
```

If the CLI is not auto-detected, set the path explicitly:
```bash
set PROJECT_CREATOR_CLI=C:\Users\YourName\ModusToolbox\tools_3.6\project-creator\project-creator-cli.exe
```

### 2. Use with Copilot CLI

Just ask naturally:
```
> Create a PSoC Edge blinky app where the LED blinks every 10 seconds
> I need a WiFi MQTT client for the PSE84 board that publishes sensor data
> Create an empty app for the CY8CKIT-062-WIFI-BT board
```

Copilot CLI reads `AGENTS.md` and uses the Python tool to:
1. Determine the board from your prompt
2. Fetch available Code Examples for that board
3. Select the best CE as a base
4. Clone it into `~/mtw-cli/ws-<app-name>/`
5. Modify the code to match your requirements
6. Build the project and fix any compilation errors

### 3. Use the Python tool directly

```bash
# List all available boards
python project_creator/project_creator.py list-boards

# List CEs for a specific board
python project_creator/project_creator.py list-apps KIT_PSE84_EVAL_EPC2

# Clone a CE
python project_creator/project_creator.py create \
  -b KIT_PSE84_EVAL_EPC2 \
  -a mtb-example-psoc-edge-hello-world \
  -n my-blinky-app

# Build the project (runs inside modus-shell automatically)
python project_creator/project_creator.py build <PROJECT_DIR>

# Build + flash (make build + make qprogram)
python project_creator/project_creator.py program <PROJECT_DIR>

# Flash only (make qprogram)
python project_creator/project_creator.py qprogram <PROJECT_DIR>
```

## Configuration

| Setting | Env Var | Default |
|---|---|---|
| CLI path | `PROJECT_CREATOR_CLI` | Auto-detected from ModusToolbox install |
| Modus-shell bash | `MODUS_SHELL_BASH` | Auto-detected from ModusToolbox install |
| Workspace root | `MTW_CLI_WORKSPACE` | `~/mtw-cli` |

The tool auto-detects `project-creator-cli.exe` by searching:
1. `PROJECT_CREATOR_CLI` env var
2. `CY_TOOLS_DIR` env var + `/project-creator/project-creator-cli.exe`
3. `~/ModusToolbox/tools_*/project-creator/project-creator-cli.exe` (latest version)

## Project Structure

```
project-creator-agent/
├── AGENTS.md                              # Copilot CLI instructions
├── README.md                              # This file
├── project_creator/
│   ├── __init__.py
│   ├── project_creator.py                 # Main CLI entry point
│   ├── cli_wrapper.py                     # project-creator-cli.exe wrapper
│   ├── build_wrapper.py                   # Build/flash via modus-shell
│   ├── manifest_parser.py                 # Output parser (strips noise)
│   ├── config.py                          # Path detection & defaults
│   └── requirements.txt                   # Dependencies (stdlib only)
└── mm_agents/                             # Existing agent framework (unchanged)
```
