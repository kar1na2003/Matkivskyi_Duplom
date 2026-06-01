#!/usr/bin/env python3
"""Project Creator Agent — CLI entry point.

A thin CLI wrapper around project-creator-cli.exe that provides clean,
parseable output for Copilot CLI to consume.

Usage:
    python project_creator.py list-boards
    python project_creator.py list-apps <board-id>
    python project_creator.py create -b <board-id> -a <app-id> [-d <dir>] [-n <name>]
    python project_creator.py build <project-dir>
    python project_creator.py program <project-dir>
    python project_creator.py qprogram <project-dir>
    python project_creator.py debug <project-dir> [--port COM4] [--duration 15]
    python project_creator.py capture-uart --port COM4 [--duration 10]
"""

import argparse
import sys
import os

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project_creator.cli_wrapper import ProjectCreatorCLI, ProjectCreatorError
from project_creator.build_wrapper import build, qprogram, program, BuildError
from project_creator import config


def cmd_list_boards(args):
    """List all available boards (BSPs)."""
    cli = ProjectCreatorCLI()
    boards = cli.list_boards()

    for board in boards:
        print(board)

    return 0


def cmd_list_apps(args):
    """List all available CEs for a given board."""
    cli = ProjectCreatorCLI()
    apps = cli.list_apps(args.board_id)

    for app in apps:
        print(app)

    return 0


def cmd_create(args):
    """Clone a CE for a given board and optionally rename it."""
    cli = ProjectCreatorCLI()

    target_dir = args.target_dir
    if target_dir is None:
        effective_name = args.name or args.app_id
        target_dir = config.get_workspace_dir(effective_name)

    print(f"Board:      {args.board_id}")
    print(f"CE:         {args.app_id}")
    print(f"Target dir: {target_dir}")
    if args.name:
        print(f"App name:   {args.name}")
    print()
    print("Cloning project...")

    project_path = cli.create_project(
        board_id=args.board_id,
        app_id=args.app_id,
        target_dir=target_dir,
        user_app_name=args.name,
    )

    print(f"\nProject created at: {project_path}")
    return 0


def cmd_build(args):
    """Build the project using make build inside modus-shell."""
    project_dir = os.path.abspath(args.project_dir)
    print(f"Building: {project_dir}")
    print()

    try:
        output = build(project_dir)
        print(output)
        print("Build SUCCEEDED")
        return 0
    except BuildError as e:
        print(e.output, file=sys.stderr)
        print(f"\nBuild FAILED: {e}", file=sys.stderr)
        return e.returncode


def cmd_program(args):
    """Build and flash the project (make build + make qprogram)."""
    project_dir = os.path.abspath(args.project_dir)
    print(f"Building and flashing: {project_dir}")
    print()

    try:
        output = program(project_dir)
        print(output)
        print("Build + Flash SUCCEEDED")
        return 0
    except BuildError as e:
        print(e.output, file=sys.stderr)
        print(f"\nFAILED: {e}", file=sys.stderr)
        return e.returncode


def cmd_qprogram(args):
    """Flash the project without rebuilding (make qprogram)."""
    project_dir = os.path.abspath(args.project_dir)
    print(f"Flashing: {project_dir}")
    print()

    try:
        output = qprogram(project_dir)
        print(output)
        print("Flash SUCCEEDED")
        return 0
    except BuildError as e:
        print(e.output, file=sys.stderr)
        print(f"\nFlash FAILED: {e}", file=sys.stderr)
        return e.returncode


def cmd_debug(args):
    """Run the autonomous debug loop (build → flash → UART → analyze)."""
    from project_creator.debug.debug_loop import run_debug_session

    issues = run_debug_session(
        project_dir=args.project_dir,
        port=args.port,
        baud=args.baud,
        uart_duration=args.uart_duration,
        uart_until=args.uart_until,
        gdb_script=args.gdb_script,
        do_build=args.do_build,
        do_flash=args.do_flash,
        do_gdb=args.do_gdb,
        iterations=args.iterations,
        output_dir=args.output_dir,
        target=args.target,
    )
    # Return non-zero if any real failures detected
    failure_keywords = ["FAILED", "FAIL", "HardFault", "Fault", "overflow", "Error"]
    for issue in issues:
        if any(kw in issue for kw in failure_keywords):
            return 1
    return 0


def cmd_capture_uart(args):
    """Capture UART output or list/discover COM ports."""
    try:
        from project_creator.debug.capture_uart import (
            capture, discover_kitprog3_ports, list_ports,
        )
    except SystemExit:
        print("ERROR: pyserial is required for UART capture.", file=sys.stderr)
        print("  Fix: pip install pyserial", file=sys.stderr)
        return 1

    if args.list_all:
        list_ports()
        return 0

    if args.discover:
        ports = discover_kitprog3_ports()
        if ports:
            print("KitProg3 UART ports:")
            for p in ports:
                print(f"  {p.device:10} — {p.description}")
        else:
            print("No KitProg3 UART ports found.")
        return 0

    if not args.port:
        # Try auto-discover
        ports = discover_kitprog3_ports()
        if len(ports) == 1:
            args.port = ports[0].device
            print(f"[auto] Using: {args.port} ({ports[0].description})")
        else:
            print("ERROR: --port required (or use --discover to find ports)", file=sys.stderr)
            return 1

    found, content = capture(
        port=args.port,
        baud=args.baud,
        duration=args.duration,
        until_pattern=args.until,
        output_file=args.output,
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="project_creator",
        description="Project Creator Agent — Create ModusToolbox projects from natural language prompts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- list-boards ---
    sub_boards = subparsers.add_parser(
        "list-boards",
        help="List all available boards (BSPs)",
    )
    sub_boards.set_defaults(func=cmd_list_boards)

    # --- list-apps ---
    sub_apps = subparsers.add_parser(
        "list-apps",
        help="List all available CEs (code examples) for a board",
    )
    sub_apps.add_argument(
        "board_id",
        help="The board (BSP) ID, e.g. KIT_PSE84_EVAL_EPC2",
    )
    sub_apps.set_defaults(func=cmd_list_apps)

    # --- create ---
    sub_create = subparsers.add_parser(
        "create",
        help="Clone a CE for a board into a workspace directory",
    )
    sub_create.add_argument(
        "-b", "--board-id", required=True,
        help="The board (BSP) ID, e.g. KIT_PSE84_EVAL_EPC2",
    )
    sub_create.add_argument(
        "-a", "--app-id", required=True,
        help="The CE (code example) ID, e.g. mtb-example-psoc-edge-hello-world",
    )
    sub_create.add_argument(
        "-d", "--target-dir", default=None,
        help="Target directory (default: ~/mtw-cli/ws-<app-name>/)",
    )
    sub_create.add_argument(
        "-n", "--name", default=None,
        help="Custom project name (default: CE template name)",
    )
    sub_create.set_defaults(func=cmd_create)

    # --- build ---
    sub_build = subparsers.add_parser(
        "build",
        help="Build the project (make build) inside modus-shell",
    )
    sub_build.add_argument(
        "project_dir",
        help="Path to the project root directory",
    )
    sub_build.set_defaults(func=cmd_build)

    # --- program ---
    sub_program = subparsers.add_parser(
        "program",
        help="Build and flash the project (make build + make qprogram)",
    )
    sub_program.add_argument(
        "project_dir",
        help="Path to the project root directory",
    )
    sub_program.set_defaults(func=cmd_program)

    # --- qprogram ---
    sub_qprogram = subparsers.add_parser(
        "qprogram",
        help="Flash the project without rebuilding (make qprogram)",
    )
    sub_qprogram.add_argument(
        "project_dir",
        help="Path to the project root directory",
    )
    sub_qprogram.set_defaults(func=cmd_qprogram)

    # --- debug ---
    sub_debug = subparsers.add_parser(
        "debug",
        help="Autonomous debug loop: build → flash → UART capture → analyze",
    )
    sub_debug.add_argument(
        "project_dir",
        help="Path to the project root directory",
    )
    sub_debug.add_argument("--port", help="UART COM port (auto-discover if omitted)")
    sub_debug.add_argument("--baud", type=int, default=115200)
    sub_debug.add_argument("--uart-duration", type=float, default=15,
                           help="UART capture duration in seconds")
    sub_debug.add_argument("--uart-until", default="PASS|FAIL|assert|HardFault|Error",
                           help="Stop UART capture when this regex matches")
    sub_debug.add_argument("--gdb-script", help="Path to GDB batch script")
    sub_debug.add_argument("--no-build", dest="do_build", action="store_false", default=True)
    sub_debug.add_argument("--no-flash", dest="do_flash", action="store_false", default=True)
    sub_debug.add_argument("--gdb", dest="do_gdb", action="store_true", default=False,
                           help="Run GDB backtrace (starts OpenOCD → connects GDB → captures registers)")
    sub_debug.add_argument("--target", help="Board TARGET name (auto-detect from Makefile if omitted)")
    sub_debug.add_argument("--iterations", type=int, default=1,
                           help="Number of build-flash-capture iterations")
    sub_debug.add_argument("--output-dir", help="Directory for debug session logs")
    sub_debug.set_defaults(func=cmd_debug)

    # --- capture-uart ---
    sub_uart = subparsers.add_parser(
        "capture-uart",
        help="Capture UART output from a board, or discover COM ports",
    )
    sub_uart.add_argument("--port", help="COM port (auto-discover if omitted)")
    sub_uart.add_argument("--baud", type=int, default=115200)
    sub_uart.add_argument("--duration", type=float, default=10,
                          help="Capture duration in seconds")
    sub_uart.add_argument("--until", default=None,
                          help="Regex pattern — stop capture when matched")
    sub_uart.add_argument("--output", "-o", default=None,
                          help="Save captured output to file")
    sub_uart.add_argument("--list-all", action="store_true",
                          help="List all available COM ports")
    sub_uart.add_argument("--discover", action="store_true",
                          help="Discover KitProg3 UART ports")
    sub_uart.set_defaults(func=cmd_capture_uart)

    args = parser.parse_args()

    try:
        return args.func(args)
    except (ProjectCreatorError, BuildError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
