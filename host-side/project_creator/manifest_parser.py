"""Parser for project-creator-cli output.

The CLI outputs verbose manifest download/parsing logs before the actual data.
This module extracts the clean board and CE lists from that noisy output.
"""

import re


# Lines matching these patterns are manifest noise to skip
_NOISE_PATTERNS = [
    re.compile(r"^(Loading|Finished loading|Found environment|Online Content|"
               r"Processing|Downloading|Getting manifests|Starting to parse|"
               r"Finished (parsing|download)|No SDK)"),
    re.compile(r"^\s*$"),  # blank lines
]


def _is_noise(line: str) -> bool:
    """Return True if the line is manifest download/parsing noise."""
    stripped = line.strip()
    if not stripped:
        return True
    for pattern in _NOISE_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


def parse_board_list(raw_output: str) -> list:
    """Extract board (BSP) IDs from --list-boards output.

    The CLI prints something like:
        ... (manifest noise) ...
        List of BSPs:
        CY8CEVAL-062S2
        CY8CKIT-040T
        ...

    Or all on one line after the header.
    """
    lines = raw_output.splitlines()
    boards = []
    data_started = False

    for line in lines:
        stripped = line.strip()

        # Detect the header that marks start of data
        if re.match(r"^List of BSPs:", stripped, re.IGNORECASE):
            # Check if IDs are on the same line after the colon
            after_colon = stripped.split(":", 1)[1].strip()
            if after_colon:
                boards.extend(after_colon.split())
            data_started = True
            continue

        if data_started:
            if _is_noise(line):
                continue
            # Each non-empty line is a board ID (or multiple space-separated)
            tokens = stripped.split()
            boards.extend(tokens)

    return [b for b in boards if b]


def parse_app_list(raw_output: str) -> list:
    """Extract app (CE) IDs from --list-apps output.

    The CLI prints something like:
        ... (manifest noise) ...
        List of available applications for <board>:
        mtb-example-psoc-edge-hello-world
        mtb-example-psoc-edge-empty-app
        ...

    Or all on one line after the header.
    """
    lines = raw_output.splitlines()
    apps = []
    data_started = False

    for line in lines:
        stripped = line.strip()

        # Detect the header — matches multiple formats
        if re.match(r"^List of (available )?(template )?applications", stripped, re.IGNORECASE):
            after_colon = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if after_colon:
                apps.extend(after_colon.split())
            data_started = True
            continue

        if data_started:
            if _is_noise(line):
                continue
            tokens = stripped.split()
            apps.extend(tokens)

    return [a for a in apps if a]
