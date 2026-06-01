"""Sanity test: host algo enum must mirror firmware imgproc.h."""
from __future__ import annotations

from modusmate_host import algos


def test_algo_count_matches_firmware():
    """First 51 entries must mirror firmware imgproc.h (PASSTHROUGH..FAST12).

    The host list is allowed to advertise *additional* host-known names
    (SIFT/SURF/...) for forward-compat; ``get_info()`` constrains the GUI to
    what the firmware really implements.
    """
    assert algos.FIRMWARE_ALGO_COUNT == 51
    assert len(algos.ALGO_NAMES) >= 51
    assert algos.ALGO_NAMES[42] == "watershed"
    assert algos.ALGO_NAMES[43] == "sharpen"
    assert algos.ALGO_NAMES[44] == "emboss"
    assert algos.ALGO_NAMES[45] == "mser"
    assert algos.ALGO_NAMES[46] == "agast"
    assert algos.ALGO_NAMES[47] == "brief"
    assert algos.ALGO_NAMES[48] == "akaze"
    assert algos.ALGO_NAMES[49] == "blob_log_multiscale"
    assert algos.ALGO_NAMES[50] == "fast12"


def test_algo_names_are_unique_lowercase():
    assert len(set(algos.ALGO_NAMES)) == len(algos.ALGO_NAMES)
    for n in algos.ALGO_NAMES:
        assert n == n.lower()
        assert " " not in n


def test_known_algo_indices():
    """A few key indices must match firmware imgproc_algo_t enum order."""
    assert algos.ALGO_NAMES[0] == "passthrough"
    assert algos.ALGO_NAMES[1] == "grayscale"
    assert algos.ALGO_NAMES[9] == "sobel"
    assert algos.ALGO_NAMES[15] == "canny"
    assert algos.ALGO_NAMES[24] == "otsu"
    assert algos.ALGO_NAMES[42] == "watershed"
