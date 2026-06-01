#!/usr/bin/env python3
"""Capture UART output from a KitProg3 / Infineon virtual COM port.

Adapted from pdl_agent/capture_uart.py for use in the project creator agent.

Usage (standalone):
  python -m project_creator.debug.capture_uart --port COM4
  python -m project_creator.debug.capture_uart --list

Usage (as module):
  from project_creator.debug.capture_uart import capture, list_ports, discover_kitprog3_ports
"""

import argparse
import re
import sys
import time
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


def _check_pyserial():
    if serial is None:
        sys.exit(
            "ERROR: pyserial not installed.\n"
            "  Fix: pip install pyserial"
        )


def list_ports():
    """Print all available COM ports."""
    _check_pyserial()
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No COM ports found.")
        return []
    print(f"{'Port':<12} {'Device':<30} {'Description'}")
    print("-" * 70)
    for p in sorted(ports):
        print(f"{p.device:<12} {p.name:<30} {p.description}")
    return ports


def discover_kitprog3_ports():
    """Return list of COM ports that look like KitProg3 / Infineon UART."""
    _check_pyserial()
    keywords = ["KitProg3", "KitProg", "Infineon", "CMSIS-DAP", "USB Serial"]
    found = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "") + " " + (p.manufacturer or "")
        if any(kw.lower() in desc.lower() for kw in keywords):
            found.append(p)
    return found


def capture(port, baud=115200, timeout_idle=0, duration=0,
            output_file=None, until_pattern=None, quiet=False):
    """Capture UART output.

    Args:
        port: COM port name (e.g., "COM4")
        baud: Baud rate (default: 115200)
        timeout_idle: Stop after N seconds of no data (0 = disabled)
        duration: Stop after N total seconds (0 = unlimited)
        output_file: Write output to this file path (None = stdout only)
        until_pattern: Stop when this regex matches (None = no pattern)
        quiet: If True, suppress stdout echo

    Returns:
        (found_pattern: bool, captured_text: str)
    """
    _check_pyserial()
    pattern = re.compile(until_pattern) if until_pattern else None
    outfile = open(output_file, "w", encoding="utf-8") if output_file else None
    found_pattern = False

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[capture_uart] Opened {port} @ {baud} baud", flush=True)
        if until_pattern:
            print(f"[capture_uart] Will stop on pattern: {until_pattern}", flush=True)
        if duration:
            print(f"[capture_uart] Will stop after {duration}s", flush=True)
    except serial.SerialException as e:
        sys.exit(f"ERROR opening {port}: {e}")

    start_time = time.time()
    last_data_time = time.time()
    buf = ""

    try:
        while True:
            now = time.time()

            if duration and (now - start_time) >= duration:
                print(f"\n[capture_uart] Duration {duration}s reached, stopping.", flush=True)
                break

            if timeout_idle and (now - last_data_time) >= timeout_idle:
                print(f"\n[capture_uart] Idle for {timeout_idle}s, stopping.", flush=True)
                break

            raw = ser.read(256)
            if not raw:
                continue

            last_data_time = now
            text = raw.decode("utf-8", errors="replace")
            buf += text

            for line in text.splitlines(keepends=True):
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                formatted = f"[{ts}] {line}"
                if not quiet:
                    print(formatted, end="", flush=True)
                if outfile:
                    outfile.write(formatted)
                    outfile.flush()

            if pattern and pattern.search(buf):
                print(f"\n[capture_uart] Pattern '{until_pattern}' found, stopping.", flush=True)
                found_pattern = True
                break

    except KeyboardInterrupt:
        print("\n[capture_uart] Interrupted by user.", flush=True)
    finally:
        ser.close()
        if outfile:
            outfile.close()
            print(f"[capture_uart] Output saved to: {output_file}", flush=True)

    return found_pattern, buf


def main():
    parser = argparse.ArgumentParser(
        description="Capture UART output from a KitProg3 virtual COM port.",
    )
    parser.add_argument("--port", help="COM port (e.g. COM4)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0,
                        help="Stop after N seconds of idle (0 = disabled)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Stop after N total seconds (0 = unlimited)")
    parser.add_argument("--output", help="Write output to this file")
    parser.add_argument("--until", help="Stop when this regex pattern appears")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list", action="store_true", help="List COM ports and exit")
    parser.add_argument("--discover", action="store_true",
                        help="Discover KitProg3 COM ports and exit")
    args = parser.parse_args()

    if args.list:
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
            list_ports()
        return

    if not args.port:
        print("ERROR: --port is required (use --list or --discover)")
        sys.exit(1)

    found, _ = capture(
        port=args.port, baud=args.baud,
        timeout_idle=args.timeout, duration=args.duration,
        output_file=args.output, until_pattern=args.until,
        quiet=args.quiet,
    )
    sys.exit(0 if (found or not args.until) else 1)


if __name__ == "__main__":
    main()
