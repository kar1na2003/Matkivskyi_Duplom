"""Configuration module for the Project Creator Agent.

Handles auto-detection of project-creator-cli path, workspace defaults,
and environment variable overrides.
"""

import os
import sys
import pathlib


def _get_mtb_root() -> pathlib.Path:
    """Get the ModusToolbox root directory."""
    if sys.platform == "win32":
        return pathlib.Path.home() / "ModusToolbox"
    elif sys.platform == "darwin":
        return pathlib.Path("/Applications/ModusToolbox")
    else:
        return pathlib.Path.home() / "ModusToolbox"


def _find_modus_shell_bash() -> str:
    """Auto-detect the modus-shell bash executable path.

    Search order:
    1. MODUS_SHELL_BASH env var (explicit override)
    2. CY_TOOLS_DIR env var + /modus-shell/bin/bash.exe
    3. Common ModusToolbox install locations
    """
    explicit = os.environ.get("MODUS_SHELL_BASH")
    if explicit and os.path.isfile(explicit):
        return explicit

    tools_dir = os.environ.get("CY_TOOLS_DIR")
    if tools_dir:
        candidate = os.path.join(tools_dir, "modus-shell", "bin", "bash.exe")
        if os.path.isfile(candidate):
            return candidate

    mtb_root = _get_mtb_root()

    if mtb_root.is_dir():
        tools_dirs = sorted(mtb_root.glob("tools_*"), reverse=True)
        for td in tools_dirs:
            if sys.platform == "win32":
                candidate = td / "modus-shell" / "bin" / "bash.exe"
            else:
                candidate = td / "modus-shell" / "bin" / "bash"
            if candidate.is_file():
                return str(candidate)

    # On Linux/macOS, system bash may work if make and toolchain are in PATH
    if sys.platform != "win32":
        return "/bin/bash"

    return None


def _find_project_creator_cli() -> str:
    """Auto-detect the project-creator-cli executable path.

    Search order:
    1. PROJECT_CREATOR_CLI env var (explicit override)
    2. CY_TOOLS_DIR env var + /project-creator/project-creator-cli.exe
    3. Common ModusToolbox install locations
    """
    # 1. Explicit override
    explicit = os.environ.get("PROJECT_CREATOR_CLI")
    if explicit and os.path.isfile(explicit):
        return explicit

    # 2. CY_TOOLS_DIR env var
    tools_dir = os.environ.get("CY_TOOLS_DIR")
    if tools_dir:
        candidate = os.path.join(tools_dir, "project-creator", "project-creator-cli.exe")
        if os.path.isfile(candidate):
            return candidate

    # 3. Scan common ModusToolbox install locations
    mtb_root = _get_mtb_root()

    if mtb_root.is_dir():
        # Find the latest tools_X.Y directory
        tools_dirs = sorted(mtb_root.glob("tools_*"), reverse=True)
        for td in tools_dirs:
            if sys.platform == "win32":
                candidate = td / "project-creator" / "project-creator-cli.exe"
            else:
                candidate = td / "project-creator" / "project-creator-cli"
            if candidate.is_file():
                return str(candidate)

    return None


# --- Resolved configuration ---

PROJECT_CREATOR_CLI = _find_project_creator_cli()
MODUS_SHELL_BASH = _find_modus_shell_bash()

DEFAULT_WORKSPACE_ROOT = os.environ.get(
    "MTW_CLI_WORKSPACE",
    str(pathlib.Path.home() / "mtw-cli"),
)

WORKSPACE_PATTERN = "ws-{app_name}"

# Subprocess timeout for project-creator-cli commands (seconds)
CLI_TIMEOUT_LIST = 120      # list-boards / list-apps (manifest download can be slow)
CLI_TIMEOUT_CREATE = 600    # create project (git clone can be slow)
CLI_TIMEOUT_BUILD = 600     # make build (compilation can take a while)
CLI_TIMEOUT_PROGRAM = 120   # make qprogram (flashing)
CLI_TIMEOUT_GDB = 60        # GDB batch session


def _find_openocd() -> dict:
    """Auto-detect OpenOCD executable and scripts directory.

    Returns dict with:
        'standard':  {'bin': path, 'scripts': path}  — standard MTB OpenOCD
        'progtools': {'bin': path, 'scripts': path}  — ModusToolboxProgtools (for PSE84)
    Either or both may be None.
    """
    result = {"standard": None, "progtools": None}

    # --- Standard MTB OpenOCD (tools_X.Y/openocd/) ---
    mtb_root = _get_mtb_root()
    if mtb_root and mtb_root.is_dir():
        for td in sorted(mtb_root.glob("tools_*"), reverse=True):
            ocd_bin = td / "openocd" / "bin" / "openocd.exe"
            ocd_scripts = td / "openocd" / "scripts"
            if not ocd_bin.is_file() and sys.platform != "win32":
                ocd_bin = td / "openocd" / "bin" / "openocd"
            if ocd_bin.is_file():
                result["standard"] = {
                    "bin": str(ocd_bin),
                    "scripts": str(ocd_scripts) if ocd_scripts.is_dir() else None,
                }
                break

    # --- ModusToolboxProgtools OpenOCD (for PSE84/Explorer) ---
    # Check C:\Infineon\Tools\ModusToolboxProgtools-*
    progtools_roots = []
    if sys.platform == "win32":
        progtools_roots.append(pathlib.Path("C:/Infineon/Tools"))
    if mtb_root:
        progtools_roots.append(mtb_root)

    for root in progtools_roots:
        if not root.is_dir():
            continue
        for pt_dir in sorted(root.glob("ModusToolboxProgtools*"), reverse=True):
            ocd_bin = pt_dir / "openocd" / "bin" / "openocd.exe"
            ocd_scripts = pt_dir / "openocd" / "scripts"
            if not ocd_bin.is_file() and sys.platform != "win32":
                ocd_bin = pt_dir / "openocd" / "bin" / "openocd"
            if ocd_bin.is_file():
                result["progtools"] = {
                    "bin": str(ocd_bin),
                    "scripts": str(ocd_scripts) if ocd_scripts.is_dir() else None,
                }
                break
        if result["progtools"]:
            break

    # Env var overrides
    env_ocd = os.environ.get("CY_OPENOCD_CUSTOM_PATH")
    if env_ocd:
        ocd_bin_path = pathlib.Path(env_ocd)
        if ocd_bin_path.is_dir():
            # Path points to bin/ directory
            exe = ocd_bin_path / "openocd.exe"
            if not exe.is_file():
                exe = ocd_bin_path / "openocd"
            scripts = ocd_bin_path.parent / "scripts"
            if exe.is_file():
                result["progtools"] = {
                    "bin": str(exe),
                    "scripts": str(scripts) if scripts.is_dir() else None,
                }
        elif ocd_bin_path.is_file():
            scripts = ocd_bin_path.parent.parent / "scripts"
            result["progtools"] = {
                "bin": str(ocd_bin_path),
                "scripts": str(scripts) if scripts.is_dir() else None,
            }

    return result


def _find_gdb() -> str:
    """Auto-detect arm-none-eabi-gdb executable path.

    Search order:
    1. CY_GDB_CUSTOM_PATH env var
    2. GCC toolchain inside ModusToolbox (tools_X.Y/gcc/)
    3. Infineon Tools (C:\\Users\\<user>\\Infineon\\Tools\\mtb-gcc-arm-eabi\\*\\gcc\\bin)
    """
    env_gdb = os.environ.get("CY_GDB_CUSTOM_PATH")
    if env_gdb and os.path.isfile(env_gdb):
        return env_gdb

    exe_name = "arm-none-eabi-gdb.exe" if sys.platform == "win32" else "arm-none-eabi-gdb"

    # Check ModusToolbox tools_X.Y/gcc/bin/
    mtb_root = _get_mtb_root()
    if mtb_root and mtb_root.is_dir():
        for td in sorted(mtb_root.glob("tools_*"), reverse=True):
            for gcc_dir in sorted(td.glob("gcc*"), reverse=True):
                gdb_exe = gcc_dir / "bin" / exe_name
                if gdb_exe.is_file():
                    return str(gdb_exe)

    # Check Infineon Tools directory (mtb-gcc-arm-eabi/<version>/gcc/bin/)
    infineon_roots = []
    if sys.platform == "win32":
        infineon_roots.append(pathlib.Path.home() / "Infineon" / "Tools")
        infineon_roots.append(pathlib.Path("C:/Infineon/Tools"))
    for inf_root in infineon_roots:
        if not inf_root.is_dir():
            continue
        for gcc_pkg in sorted(inf_root.glob("mtb-gcc-arm-eabi*"), reverse=True):
            # Versioned subdirs: mtb-gcc-arm-eabi/14.2.1/gcc/bin/
            for ver_dir in sorted(gcc_pkg.iterdir(), reverse=True):
                gdb_exe = ver_dir / "gcc" / "bin" / exe_name
                if gdb_exe.is_file():
                    return str(gdb_exe)

    return None


# --- Debug tool paths ---
OPENOCD = _find_openocd()
GDB = _find_gdb()
DEFAULT_GDB_PORT = 3333


def get_workspace_dir(app_name: str, workspace_root: str = None) -> str:
    """Build the default workspace directory for a given app name.

    Returns e.g. ~/mtw-cli/ws-my-blinky-app
    """
    root = workspace_root or DEFAULT_WORKSPACE_ROOT
    folder = WORKSPACE_PATTERN.format(app_name=app_name)
    return str(pathlib.Path(root) / folder)


def validate():
    """Validate that the project-creator-cli is available. Raises if not found."""
    if not PROJECT_CREATOR_CLI:
        raise FileNotFoundError(
            "project-creator-cli not found. Set PROJECT_CREATOR_CLI env var "
            "or install ModusToolbox."
        )
    if not os.path.isfile(PROJECT_CREATOR_CLI):
        raise FileNotFoundError(
            f"project-creator-cli not found at: {PROJECT_CREATOR_CLI}"
        )
