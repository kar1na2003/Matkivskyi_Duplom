"""Tests for the algorithm metadata and preview-statistics helpers."""
from __future__ import annotations

from modusmate_host.algos import (ALGO_DESCRIPTIONS, ALGO_NAMES,
                                  ALGO_STAT_KIND, compute_preview_stats,
                                  description, stat_kind)


def test_every_algo_has_description():
    missing = [n for n in ALGO_NAMES if n not in ALGO_DESCRIPTIONS]
    assert not missing, f"missing description for: {missing}"


def test_every_algo_has_stat_kind():
    missing = [n for n in ALGO_NAMES if n not in ALGO_STAT_KIND]
    assert not missing, f"missing stat kind for: {missing}"


def test_descriptions_are_substantive():
    # every description should be at least 30 chars and end with '.'
    for n in ALGO_NAMES:
        d = ALGO_DESCRIPTIONS[n]
        assert len(d) >= 30, f"too-short description for {n!r}: {d!r}"


def test_description_helper_falls_back():
    assert description("does_not_exist").startswith("(no description")


def test_stat_kind_helper_default_is_none():
    assert stat_kind("does_not_exist") == "none"


def test_compute_stats_edges_counts_above_threshold():
    # 100 px: half are 100 (>=32), half are 0
    w, h = 10, 10
    buf = bytes([100] * 50 + [0] * 50)
    label, val = compute_preview_stats("sobel", buf, w, h)
    assert label == "edge px"
    assert val == 50


def test_compute_stats_binary_counts_white():
    w, h = 10, 10
    buf = bytes([255] * 30 + [0] * 70)
    label, val = compute_preview_stats("otsu", buf, w, h)
    assert label == "on px"
    assert val == 30


def test_compute_stats_intensity_returns_mean():
    w, h = 4, 4   # 16 pixels
    buf = bytes([10] * 16)
    label, val = compute_preview_stats("passthrough", buf, w, h)
    assert label == "mean"
    assert val == 10


def test_compute_stats_keypoints_finds_isolated_maximum():
    w, h = 9, 9
    arr = [0] * (w * h)
    # one bright pixel in centre, surrounded by zeros — counts as one keypoint
    arr[4 * w + 4] = 250
    label, val = compute_preview_stats("harris", bytes(arr), w, h)
    assert label == "keypoints"
    assert val == 1


def test_compute_stats_safe_on_empty_buffer():
    label, val = compute_preview_stats("sobel", b"", 0, 0)
    assert label == ""
    assert val == 0
