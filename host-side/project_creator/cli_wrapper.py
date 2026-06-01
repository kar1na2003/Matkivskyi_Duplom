"""Wrapper around the ModusToolbox project-creator-cli executable.

Provides a clean Python interface for listing boards, listing apps,
and creating projects — hiding the subprocess and output-parsing details.
"""

import subprocess
import os
import sys

from . import config
from . import manifest_parser


class ProjectCreatorError(Exception):
    """Raised when project-creator-cli returns an error."""


class ProjectCreatorCLI:
    """Python wrapper around project-creator-cli.exe."""

    def __init__(self, cli_path: str = None):
        self._cli = cli_path or config.PROJECT_CREATOR_CLI
        if not self._cli:
            raise ProjectCreatorError(
                "project-creator-cli not found. Set PROJECT_CREATOR_CLI env var "
                "or install ModusToolbox."
            )
        if not os.path.isfile(self._cli):
            raise ProjectCreatorError(
                f"project-creator-cli not found at: {self._cli}"
            )

    def _run(self, args: list, timeout: int = None) -> str:
        """Execute project-creator-cli with the given arguments.

        Returns the combined stdout+stderr as a string.
        Raises ProjectCreatorError on non-zero exit code.
        """
        cmd = [self._cli] + args
        timeout = timeout or config.CLI_TIMEOUT_LIST

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as e:
            raise ProjectCreatorError(
                f"project-creator-cli timed out after {timeout}s. "
                f"Command: {' '.join(cmd)}"
            ) from e
        except FileNotFoundError as e:
            raise ProjectCreatorError(
                f"project-creator-cli not found at: {self._cli}"
            ) from e

        # Combine stdout and stderr (CLI mixes data with logs across both)
        output = (result.stdout or "") + "\n" + (result.stderr or "")

        if result.returncode != 0:
            raise ProjectCreatorError(
                f"project-creator-cli exited with code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{output}"
            )

        return output

    def list_boards(self) -> list:
        """Fetch the list of available board (BSP) IDs.

        Returns a sorted list of board ID strings.
        Always fetches fresh from the manifest.
        """
        raw = self._run(["--list-boards", "--verbose", "0"])
        boards = manifest_parser.parse_board_list(raw)
        return sorted(boards)

    def list_apps(self, board_id: str) -> list:
        """Fetch the list of available CE (app) IDs for a given board.

        Args:
            board_id: The BSP ID (e.g., "KIT_PSE84_EVAL_EPC2")

        Returns a sorted list of CE ID strings.
        Always fetches fresh from the manifest.
        """
        if not board_id:
            raise ProjectCreatorError("board_id is required")

        raw = self._run(["--list-apps", board_id, "--verbose", "0"])
        apps = manifest_parser.parse_app_list(raw)
        return sorted(apps)

    def create_project(
        self,
        board_id: str,
        app_id: str,
        target_dir: str = None,
        user_app_name: str = None,
    ) -> str:
        """Clone a CE for the given board using project-creator-cli.

        Args:
            board_id: The BSP ID (e.g., "KIT_PSE84_EVAL_EPC2")
            app_id: The CE ID (e.g., "mtb-example-psoc-edge-hello-world")
            target_dir: Where to create the project. If None, uses default workspace.
            user_app_name: Custom project name. If None, uses CE template name.

        Returns the path to the created project directory.
        """
        if not board_id:
            raise ProjectCreatorError("board_id is required")
        if not app_id:
            raise ProjectCreatorError("app_id is required")

        # Determine output directory
        effective_name = user_app_name or app_id
        if target_dir is None:
            target_dir = config.get_workspace_dir(effective_name)

        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Build CLI arguments
        args = [
            "--board-id", board_id,
            "--app-id", app_id,
            "--target-dir", target_dir,
            "--verbose", "1",
        ]
        if user_app_name:
            args.extend(["--user-app-name", user_app_name])

        self._run(args, timeout=config.CLI_TIMEOUT_CREATE)

        # The project is created inside target_dir with the app name
        project_dir = os.path.join(target_dir, user_app_name or app_id)
        if os.path.isdir(project_dir):
            return project_dir

        # Fallback: if the CLI created directly in target_dir
        return target_dir
