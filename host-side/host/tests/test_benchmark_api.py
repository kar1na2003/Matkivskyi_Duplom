"""Smoke test for the programmatic benchmark API exported for the GUI."""
from __future__ import annotations

from modusmate_host import benchmark as bench


def test_progress_dataclass_has_expected_fields():
    p = bench.BenchProgress(kind="image", algo_name="sobel",
                            image_index=3, image_total=10,
                            accuracy=0.5, mean_algo_us=42, mean_infer_us=7)
    assert p.kind == "image"
    assert p.algo_name == "sobel"
    assert p.image_index == 3
    assert p.image_total == 10
    assert p.mean_algo_us == 42


def test_summary_defaults():
    s = bench.BenchSummary()
    assert s.rows == []
    assert s.per_algo == {}
    assert s.cancelled is False
    assert s.samples_per_algo == 0


def test_run_benchmark_is_callable():
    # programmatic API must be importable and callable (we don't actually
    # run it — that needs hardware).
    assert callable(bench.run_benchmark)
