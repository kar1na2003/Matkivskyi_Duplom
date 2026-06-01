#!/bin/bash
# Run the ModusMate GUI.
#
# Prefers ./.venv (brew Python 3.12 + python-tk@3.12) when present,
# otherwise falls back to /usr/bin/python3 (Apple system Python with Tk 8.5).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python -m modusmate_host.gui "$@"
fi

echo "[run_gui] no .venv found, falling back to /usr/bin/python3"
exec /usr/bin/python3 -m modusmate_host.gui "$@"
