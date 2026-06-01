#!/usr/bin/env python3
"""Autonomous debug loop for ModusToolbox projects.

Orchestrates: build → flash → capture UART → GDB backtrace → analyze.
Adapted from pdl_agent/debug_loop.py to use the shared modus-shell wrapper.

Usage (standalone):
  python -m project_creator.debug.debug_loop --project-dir <DIR> [options]

Usage (via project_creator.py):
  python project_creator.py debug <PROJECT_DIR> [--port COM4] [--duration 15]
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path for imports when run standalone
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from project_creator.build_wrapper import _run_in_modus_shell, _get_bash, BuildError
from project_creator import config

try:
    from project_creator.debug.capture_uart import (
        capture, discover_kitprog3_ports, list_ports,
    )
    HAS_SERIAL = True
except SystemExit:
    HAS_SERIAL = False


GDB_DIR = Path(__file__).parent


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def step_build(project_dir):
    """Build the project using make build."""
    print("\n" + "=" * 60)
    print("STEP: Build")
    print("=" * 60)
    rc, output = _run_in_modus_shell(
        "make build -j", project_dir,
        timeout=config.CLI_TIMEOUT_BUILD,
    )
    if rc != 0:
        print(f"Build FAILED (rc={rc})")
        print(output[-3000:] if len(output) > 3000 else output)
        return False, output
    print("Build: PASS")
    return True, output


def step_flash(project_dir):
    """Flash the project using make qprogram."""
    print("\n" + "=" * 60)
    print("STEP: Flash")
    print("=" * 60)
    rc, output = _run_in_modus_shell(
        "make qprogram", project_dir,
        timeout=config.CLI_TIMEOUT_PROGRAM,
    )
    if rc != 0:
        print(f"Flash FAILED (rc={rc})")
        print(output[-2000:] if len(output) > 2000 else output)
        return False, output
    print("Flash: PASS")
    return True, output


def step_capture_uart(port, baud, duration, until_pattern, log_path):
    """Capture UART output from the board."""
    print("\n" + "=" * 60)
    print("STEP: UART Capture")
    print("=" * 60)
    if not HAS_SERIAL:
        print("WARNING: pyserial not installed — skipping UART capture.")
        print("  Fix: pip install pyserial")
        return False, ""

    found, content = capture(
        port=port, baud=baud,
        duration=duration,
        until_pattern=until_pattern,
        output_file=str(log_path),
    )
    return True, content


def _find_elf(project_dir):
    """Find the ELF file in the project's build output directory."""
    build_dir = Path(project_dir) / "build"
    if not build_dir.is_dir():
        return None
    # Search for .elf files recursively
    elfs = list(build_dir.rglob("*.elf"))
    if not elfs:
        return None
    # Prefer Debug builds, then most recently modified
    debug_elfs = [e for e in elfs if "Debug" in str(e)]
    if debug_elfs:
        return str(max(debug_elfs, key=lambda p: p.stat().st_mtime))
    return str(max(elfs, key=lambda p: p.stat().st_mtime))


def _detect_target(project_dir):
    """Read the Makefile to determine the TARGET board."""
    makefile = Path(project_dir) / "Makefile"
    if not makefile.is_file():
        return None
    content = makefile.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^\s*TARGET\s*[?:]*=\s*(\S+)", content, re.MULTILINE)
    return match.group(1) if match else None


def _is_pse84_target(target):
    """Check if the target is a PSE84/Explorer board requiring ProgTools OpenOCD."""
    if not target:
        return False
    pse84_markers = ["PSE84", "EXPLORER", "pse84", "explorer"]
    return any(m in target for m in pse84_markers)


def _find_bsp_openocd_config(project_dir, target):
    """Find OpenOCD board config files from the project's BSP.

    CE projects store BSP configs at:
      bsps/TARGET_<BOARD>/config/GeneratedSource/
    or in mtb_shared/TARGET_<BOARD>/...
    """
    search_dirs = [
        Path(project_dir) / "bsps" / f"TARGET_{target}" / "config" / "GeneratedSource",
        Path(project_dir) / "bsps" / f"TARGET_{target}" / "config",
    ]
    # Also check mtb_shared for BSP
    mtb_shared = Path(project_dir) / "mtb_shared"
    if mtb_shared.is_dir():
        for bsp_dir in mtb_shared.rglob(f"TARGET_{target}"):
            gs = bsp_dir / "config" / "GeneratedSource"
            if gs.is_dir():
                search_dirs.append(gs)

    for d in search_dirs:
        if d.is_dir():
            return str(d)
    return None


def _build_openocd_command(project_dir, target, port):
    """Build the OpenOCD command line for the given target.

    Returns (openocd_exe, [args]) or (None, None) if OpenOCD not found.
    """
    is_pse84 = _is_pse84_target(target)

    # Pick the right OpenOCD binary
    if is_pse84 and config.OPENOCD.get("progtools"):
        ocd = config.OPENOCD["progtools"]
    elif config.OPENOCD.get("standard"):
        ocd = config.OPENOCD["standard"]
    elif config.OPENOCD.get("progtools"):
        ocd = config.OPENOCD["progtools"]
    else:
        return None, None

    ocd_bin = ocd["bin"]
    ocd_scripts = ocd["scripts"]

    args = []
    if ocd_scripts:
        args += ["-s", ocd_scripts]

    # Add BSP-specific config search path (for qspi_config.cfg etc.)
    bsp_cfg_dir = _find_bsp_openocd_config(project_dir, target)
    if bsp_cfg_dir:
        args += ["-s", bsp_cfg_dir]

    if is_pse84:
        # PSE84/Explorer: pse84xgxs2.cfg + QSPI + hardware breakpoints
        args += [
            "-c", "set DEVICE b0",
        ]
        # Check for QSPI flash loader in BSP
        if bsp_cfg_dir:
            flm_files = list(Path(bsp_cfg_dir).glob("*.FLM"))
            if flm_files:
                args += ["-c", f"set QSPI_FLASHLOADER {flm_files[0]}"]
            qspi_cfg = Path(bsp_cfg_dir) / "qspi_config.cfg"
            if qspi_cfg.is_file():
                args += ["-c", "source [find qspi_config.cfg]"]

        args += [
            "-c", "source [find interface/kitprog3.cfg]",
            "-c", "transport select swd",
            "-c", "source [find target/infineon/pse84xgxs2.cfg]",
            "-c", f"gdb_port {port}",
            "-c", "gdb_breakpoint_override hard",
            "-c", "init",
        ]
    else:
        # Standard targets: KitProg3 + auto-detect target config
        # Try to infer the correct target config from the TARGET name
        target_cfg = _infer_openocd_target_cfg(target)
        args += [
            "-f", "interface/kitprog3.cfg",
            "-c", "transport select swd",
            "-f", target_cfg,
            "-c", f"gdb_port {port}",
            "-c", "init",
            "-c", "reset halt",
        ]

    return ocd_bin, args


def _infer_openocd_target_cfg(target):
    """Infer the OpenOCD target config file from the board TARGET name."""
    t = target.upper()
    if "PSC3" in t:
        return "target/infineon/psc3x8.cfg"
    if "PSB3" in t or "ATOMIC" in t:
        return "target/infineon/psb3000_2.cfg"
    if "XMC7" in t:
        return "target/infineon/xmc7xxx.cfg"
    if "CY8C" in t or "062" in t:
        return "target/infineon/psoc6_2m.cfg"
    if "PMG1" in t:
        return "target/infineon/psoc4.cfg"
    if "T2G" in t or "TRAVEO" in t:
        return "target/infineon/traveo2_6m.cfg"
    # Default fallback
    return "target/infineon/psoc6.cfg"


def step_gdb_backtrace(project_dir, gdb_script, log_path, target=None, port=None):
    """Run a GDB batch session: start OpenOCD → connect GDB → capture output → kill OpenOCD.

    Flow:
    1. Detect target board from Makefile (if not provided)
    2. Find the ELF file in build output
    3. Start OpenOCD as a background process (TCP port 3333)
    4. Wait for OpenOCD to be ready
    5. Run GDB batch with: target remote :3333, file <elf>, source <gdb_script>
    6. Capture GDB output
    7. Kill OpenOCD
    """
    print("\n" + "=" * 60)
    print("STEP: GDB Backtrace (OpenOCD → GDB)")
    print("=" * 60)

    port = port or config.DEFAULT_GDB_PORT

    # Resolve GDB script path
    script_path = Path(gdb_script)
    if not script_path.is_file():
        script_path = GDB_DIR / gdb_script
    if not script_path.is_file():
        print(f"WARNING: GDB script not found: {gdb_script}")
        return False, ""

    # Detect target board
    if not target:
        target = _detect_target(project_dir)
    if not target:
        print("WARNING: Could not determine TARGET from Makefile.")
        print("  Specify --target or add TARGET= to the project Makefile.")
        return False, ""
    print(f"  Target board: {target}")

    # Find ELF
    elf_path = _find_elf(project_dir)
    if not elf_path:
        print("WARNING: No ELF file found in build/. Build the project first.")
        return False, ""
    print(f"  ELF: {elf_path}")

    # Check GDB is available
    gdb_exe = config.GDB
    if not gdb_exe:
        print("WARNING: arm-none-eabi-gdb not found.")
        print("  Set CY_GDB_CUSTOM_PATH env var or install GCC toolchain.")
        return False, ""

    # Build OpenOCD command
    ocd_bin, ocd_args = _build_openocd_command(project_dir, target, port)
    if not ocd_bin:
        print("WARNING: OpenOCD not found.")
        print("  Install ModusToolbox or set CY_OPENOCD_CUSTOM_PATH.")
        return False, ""
    print(f"  OpenOCD: {ocd_bin}")
    print(f"  GDB: {gdb_exe}")
    print(f"  Port: {port}")

    # --- Start OpenOCD as background process ---
    import subprocess
    import socket

    print("\n  Starting OpenOCD...")
    ocd_proc = subprocess.Popen(
        [ocd_bin] + ocd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

    # Wait for OpenOCD to be ready (poll TCP port)
    is_pse84 = _is_pse84_target(target)
    wait_time = 8 if is_pse84 else 4  # PSE84 boot ROM takes longer
    ready = False
    for i in range(wait_time * 2):
        time.sleep(0.5)
        if ocd_proc.poll() is not None:
            # OpenOCD exited prematurely
            stderr = ocd_proc.stderr.read().decode("utf-8", errors="replace")
            print(f"  ERROR: OpenOCD exited with code {ocd_proc.returncode}")
            print(stderr[:2000])
            return False, stderr
        try:
            sock = socket.create_connection(("localhost", port), timeout=1)
            sock.close()
            ready = True
            print(f"  OpenOCD ready (port {port})")
            break
        except (ConnectionRefusedError, socket.timeout, OSError):
            continue

    if not ready:
        print(f"  WARNING: OpenOCD may not be ready after {wait_time}s, proceeding anyway...")

    # --- Run GDB batch ---
    try:
        print("  Running GDB batch...")

        # Convert paths to Cygwin format for modus-shell
        from project_creator.build_wrapper import _to_cygwin_path
        cyg_elf = _to_cygwin_path(elf_path)
        cyg_script = _to_cygwin_path(str(script_path.resolve()))
        cyg_gdb = _to_cygwin_path(gdb_exe)

        gdb_cmd = (
            f"'{cyg_gdb}' --batch "
            f"-ex 'target remote :{port}' "
            f"-ex 'file {cyg_elf}' "
            f"-x '{cyg_script}'"
        )

        rc, gdb_output = _run_in_modus_shell(
            gdb_cmd, project_dir,
            timeout=config.CLI_TIMEOUT_GDB,
        )

        if log_path:
            Path(log_path).write_text(gdb_output, encoding="utf-8")

        print(gdb_output[:3000] if len(gdb_output) > 3000 else gdb_output)
        return rc == 0, gdb_output

    finally:
        # --- Kill OpenOCD ---
        if ocd_proc.poll() is None:
            print("  Stopping OpenOCD...")
            ocd_proc.terminate()
            try:
                ocd_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ocd_proc.kill()
                ocd_proc.wait()


def analyze(uart_log, gdb_log=""):
    """Heuristic analysis of UART + GDB output for common failure patterns."""
    issues = []

    # UART patterns
    uart_checks = [
        (r"PASS|PASSED", "Test PASSED ✅"),
        (r"FAIL|FAILED", "Test FAILED"),
        (r"HardFault", "HardFault detected"),
        (r"UsageFault", "UsageFault"),
        (r"MemManage", "MemManage fault"),
        (r"BusFault", "BusFault"),
        (r"assert|ASSERT|CY_ASSERT", "Assertion failure"),
        (r"Stack.*overflow|stack_overflow", "Stack overflow"),
        (r"Error|ERROR|error:", "Error reported"),
    ]
    for pattern, label in uart_checks:
        if re.search(pattern, uart_log):
            issues.append(f"[UART] {label}")

    # GDB patterns
    gdb_checks = [
        (r"CFSR.*=.*0x[1-9a-fA-F]", "Non-zero CFSR (fault status)"),
        (r"HFSR.*=.*0x[1-9a-fA-F]", "Non-zero HFSR (hard fault)"),
        (r"#0.*in.*\(", "Backtrace captured"),
        (r"Cannot access memory", "Memory access error in GDB"),
    ]
    for pattern, label in gdb_checks:
        if re.search(pattern, gdb_log):
            issues.append(f"[GDB]  {label}")

    return issues


def run_debug_session(project_dir, port=None, baud=115200,
                      uart_duration=15, uart_until="PASS|FAIL|assert|HardFault|Error",
                      gdb_script=None, do_build=True, do_flash=True,
                      do_gdb=False, iterations=1, output_dir=None, target=None):
    """Run a full autonomous debug session.

    Args:
        project_dir: Path to the project root
        port: UART COM port (None = auto-discover)
        baud: UART baud rate
        uart_duration: Seconds to capture UART
        uart_until: Regex pattern to stop UART capture
        gdb_script: Path to GDB batch script (None = auto-select based on target)
        do_build: Whether to build before flashing
        do_flash: Whether to flash before capturing
        do_gdb: Whether to run GDB backtrace
        iterations: Number of debug iterations
        output_dir: Directory for session logs
        target: Board TARGET name (auto-detected from Makefile if None)

    Returns:
        List of detected issues from the last iteration
    """
    project_dir = os.path.abspath(project_dir)
    if not os.path.isdir(project_dir):
        print(f"ERROR: Project directory not found: {project_dir}")
        return ["Project directory not found"]

    # Auto-discover UART port if needed
    if port is None and HAS_SERIAL:
        found = discover_kitprog3_ports()
        if len(found) == 1:
            port = found[0].device
            print(f"[auto-discover] Using UART port: {port} ({found[0].description})")
        elif len(found) > 1:
            print("Multiple KitProg3 ports found — specify --port:")
            for p in found:
                print(f"  {p.device} — {p.description}")
        else:
            print("WARNING: No KitProg3 UART port found. UART capture will be skipped.")

    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(project_dir, "debug_output")
    session_dir = os.path.join(output_dir, timestamp())
    os.makedirs(session_dir, exist_ok=True)
    print(f"Session output: {session_dir}")

    all_issues = []

    for iteration in range(1, iterations + 1):
        print(f"\n{'#' * 60}")
        print(f"# Iteration {iteration}/{iterations}")
        print(f"{'#' * 60}")

        # Build
        if do_build:
            ok, build_output = step_build(project_dir)
            if not ok:
                print("Build failed — stopping.")
                return ["Build failed"]

        # Flash
        if do_flash:
            ok, flash_output = step_flash(project_dir)
            if not ok:
                print("Flash failed — stopping.")
                return ["Flash failed"]
            time.sleep(1)  # Let device reset

        # UART capture
        uart_content = ""
        if port:
            uart_log_path = os.path.join(session_dir, f"uart_iter{iteration}.txt")
            uart_ok, uart_content = step_capture_uart(
                port, baud, uart_duration, uart_until, uart_log_path,
            )

        # GDB backtrace
        gdb_content = ""
        if do_gdb:
            # Auto-select GDB script if not specified
            effective_gdb_script = gdb_script
            if not effective_gdb_script:
                detected_target = target or _detect_target(project_dir)
                if _is_pse84_target(detected_target):
                    effective_gdb_script = str(GDB_DIR / "gdb_explorer.gdb")
                else:
                    effective_gdb_script = str(GDB_DIR / "gdb_default.gdb")

            gdb_log_path = os.path.join(session_dir, f"gdb_iter{iteration}.txt")
            gdb_ok, gdb_content = step_gdb_backtrace(
                project_dir, effective_gdb_script, gdb_log_path,
                target=target,
            )

        # Analyze
        print("\n" + "=" * 60)
        print(f"ANALYSIS — Iteration {iteration}")
        print("=" * 60)
        issues = analyze(uart_content, gdb_content)
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  No issues detected in this iteration.")
        print(f"  Logs: {session_dir}")
        all_issues = issues

    print(f"\n[debug_loop] Complete. Session logs in: {session_dir}")
    return all_issues


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous debug loop for ModusToolbox projects.",
    )
    parser.add_argument("project_dir", help="Path to the project root directory")
    parser.add_argument("--port", help="UART COM port (auto-discover if omitted)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--uart-duration", type=float, default=15)
    parser.add_argument("--uart-until", default="PASS|FAIL|assert|HardFault|Error")
    parser.add_argument("--gdb-script", help="Path to GDB batch script")
    parser.add_argument("--no-build", dest="do_build", action="store_false", default=True)
    parser.add_argument("--no-flash", dest="do_flash", action="store_false", default=True)
    parser.add_argument("--gdb", dest="do_gdb", action="store_true", default=False,
                        help="Run GDB backtrace (requires board + OpenOCD)")
    parser.add_argument("--target", help="Board TARGET name (auto-detect from Makefile if omitted)")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output-dir", help="Directory for session logs")
    parser.add_argument("--discover", action="store_true",
                        help="Discover KitProg3 UART ports and exit")
    parser.add_argument("--list-ports", action="store_true",
                        help="List all COM ports and exit")
    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        return

    if args.discover:
        ports = discover_kitprog3_ports()
        if ports:
            print("KitProg3 UART ports found:")
            for p in ports:
                print(f"  {p.device:10} — {p.description}")
        else:
            print("No KitProg3 UART ports found.")
        return

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


if __name__ == "__main__":
    main()
