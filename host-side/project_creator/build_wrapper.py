"""Build and program wrapper for ModusToolbox projects.

Runs `make build` and `make qprogram` inside the modus-shell (Cygwin bash)
which provides the required toolchain environment (make, gcc, etc.).
"""

import subprocess
import os
import sys

from . import config


class BuildError(Exception):
    """Raised when a build or program command fails."""

    def __init__(self, message: str, output: str = "", returncode: int = 1):
        super().__init__(message)
        self.output = output
        self.returncode = returncode


def _get_bash():
    """Get the modus-shell bash path. Raises if not found."""
    bash = config.MODUS_SHELL_BASH
    if not bash:
        raise BuildError(
            "modus-shell bash not found. Set MODUS_SHELL_BASH env var "
            "or install ModusToolbox."
        )
    if not os.path.isfile(bash):
        raise BuildError(f"modus-shell bash not found at: {bash}")
    return bash


def _to_cygwin_path(win_path: str) -> str:
    """Convert a Windows path to a Cygwin-compatible path.

    e.g. C:\\Users\\Anup\\project → /cygdrive/c/Users/Anup/project
    """
    win_path = os.path.abspath(win_path)
    # Replace backslashes
    posix = win_path.replace("\\", "/")
    # Convert drive letter: C:/... → /cygdrive/c/...
    if len(posix) >= 2 and posix[1] == ":":
        drive = posix[0].lower()
        posix = f"/cygdrive/{drive}{posix[2:]}"
    return posix


def _run_in_modus_shell(command: str, project_dir: str,
                         timeout: int = None) -> tuple:
    """Run a shell command inside modus-shell bash.

    Args:
        command: The shell command to run (e.g., "make build")
        project_dir: The project root directory (Windows path)
        timeout: Subprocess timeout in seconds

    Returns:
        (returncode, output) tuple
    """
    bash = _get_bash()
    timeout = timeout or config.CLI_TIMEOUT_BUILD

    if not os.path.isdir(project_dir):
        raise BuildError(f"Project directory not found: {project_dir}")

    cyg_dir = _to_cygwin_path(project_dir)

    # Run: cd to project dir, then execute the command
    # --login loads the environment, -c runs non-interactively
    shell_cmd = f"cd '{cyg_dir}' && {command}"

    try:
        result = subprocess.run(
            [bash, "--login", "-c", shell_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or "") + "\n" + (e.stderr or "")
        raise BuildError(
            f"Command timed out after {timeout}s: {command}",
            output=output,
        ) from e

    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return result.returncode, output


def build(project_dir: str) -> str:
    """Run `make build` in the project directory.

    Args:
        project_dir: Full path to the project root (Windows path)

    Returns:
        Build output (stdout + stderr)

    Raises:
        BuildError with full output on failure
    """
    returncode, output = _run_in_modus_shell(
        "make build -j",
        project_dir,
        timeout=config.CLI_TIMEOUT_BUILD,
    )

    if returncode != 0:
        raise BuildError(
            f"Build failed (exit code {returncode})",
            output=output,
            returncode=returncode,
        )

    return output


def qprogram(project_dir: str) -> str:
    """Run `make qprogram` in the project directory (flash only, no build).

    Args:
        project_dir: Full path to the project root (Windows path)

    Returns:
        Flash output (stdout + stderr)

    Raises:
        BuildError with full output on failure
    """
    returncode, output = _run_in_modus_shell(
        "make qprogram",
        project_dir,
        timeout=config.CLI_TIMEOUT_PROGRAM,
    )

    if returncode != 0:
        raise BuildError(
            f"Flash (qprogram) failed (exit code {returncode})",
            output=output,
            returncode=returncode,
        )

    return output


def program(project_dir: str) -> str:
    """Run `make build` then `make qprogram` (build + flash).

    Args:
        project_dir: Full path to the project root (Windows path)

    Returns:
        Combined output from build and flash

    Raises:
        BuildError with full output on failure (at either step)
    """
    # Step 1: Build
    build_output = build(project_dir)

    # Step 2: Flash
    flash_output = qprogram(project_dir)

    return build_output + "\n" + flash_output
