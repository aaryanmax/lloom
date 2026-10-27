"""Unit tests for E5 No-Free-Lunch evaluation module (offline, 0 API calls)."""

from lloom_lab.experiments.e5 import compute_metrics, format_e5_markdown


def test_compute_metrics_perfect_match():
    preds = {"doc_1": {"A", "B"}, "doc_2": {"C"}}
    gt = {"doc_1": {"A", "B"}, "doc_2": {"C"}}
    m = compute_metrics(preds, gt)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_compute_metrics_disjoint():
    preds = {"doc_1": {"X"}, "doc_2": {"Y"}}
    gt = {"doc_1": {"A"}, "doc_2": {"B"}}
    m = compute_metrics(preds, gt)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_compute_metrics_partial_overlap():
    preds = {"doc_1": {"A", "B"}}
    gt = {"doc_1": {"A", "C"}}
    m = compute_metrics(preds, gt)
    # tp=1 (A), fp=1 (B), fn=1 (C)
    assert abs(m["precision"] - 0.5) < 1e-6
    assert abs(m["recall"] - 0.5) < 1e-6
    assert abs(m["f1"] - 0.5) < 1e-6


def test_format_e5_markdown():
    results = {
        "zeroshot_test": {
            "10": {
                "true_f1_mean": 0.5,
                "true_f1_std": 0.05,
                "nfl_f1_mean": 0.05,
                "nfl_f1_std": 0.01,
                "collapse_pct": 90.0,
            }
        }
    }
    md = format_e5_markdown(results)
    assert "No-Free-Lunch" in md
    assert "zeroshot_test" in md
    assert "90.0%" in md
