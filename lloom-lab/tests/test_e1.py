"""E1 grid: zero-shot coverage, cell orchestration, and aggregation (offline)."""

import json

from conftest import FakeClient
from lloom_lab.experiments import e1

DOCS = [
    {"doc_id": f"d{i}", "text": f"doc {i} text",
     "concepts": {"generic": ["Economy"], "specific": ["Taxes"]}}
    for i in range(6)
]


def test_zeroshot_coverage_counts_used_ground_truths():
    assignments = {"d0": ["Taxes", "Immigration"], "d1": ["Economy"]}
    gt = {"generic": ["Economy", "Healthcare"], "specific": ["Taxes"]}
    split = e1.zeroshot_coverage(assignments, gt)
    assert split == {"generic": 0.5, "specific": 1.0}


def test_zeroshot_coverage_empty_assignments_is_zero():
    split = e1.zeroshot_coverage({}, {"generic": ["Economy"],
                                      "specific": ["Taxes"]})
    assert split == {"generic": 0.0, "specific": 0.0}


def test_zeroshot_assign_drops_hallucinated_names(monkeypatch):
    monkeypatch.setattr(e1, "FULL_TAXONOMY", ["Taxes", "Economy"])
    # one batch, canned response with an invented concept + invented doc id
    fake = FakeClient({"assignments": [
        {"doc_id": "d0", "concepts": ["Taxes", "Made Up Concept"]},
        {"doc_id": "dX", "concepts": ["Taxes"]},
        {"doc_id": "d1", "concepts": []},
    ]})
    got = e1.zeroshot_assign(fake, DOCS, model="gemini")
    # invented doc id dropped, invented concept name dropped, empty row kept
    assert got == {"d0": ["Taxes"], "d1": []}


def test_bertopic_concepts_falls_back_on_tiny_input():
    # n=2: BERTopic cannot form meaningful topics — degrade, never raise
    concepts, info = e1.bertopic_concepts(DOCS[:2])
    assert len(concepts) <= 1


def test_run_cell_zeroshot_scores_without_matcher(tmp_path, monkeypatch):
    monkeypatch.setattr(e1, "FULL_TAXONOMY", ["Economy", "Taxes", "Healthcare"])
    fake = FakeClient({"assignments": [{"doc_id": "d0", "concepts": ["Taxes"]}]})
    record = e1.run_cell(fake, DOCS[:1], "zeroshot_gemini", n=1, trial=0,
                         out_dir=tmp_path)
    assert record["status"] == "complete"
    assert record["coverage_split"] == {"generic": 0.0, "specific": 1.0}
    assert "match_rows" not in record  # zero-shot needs no matcher judgment
    # cell record persisted and re-loaded as complete (resume skips it)
    reloaded = json.loads(
        e1.run_path("zeroshot_gemini", 1, 0, tmp_path).read_text(encoding="utf-8"))
    assert reloaded["coverage_split"]["specific"] == 1.0


def test_collect_aggregates_per_method(tmp_path):
    for method in ("lloom_gemma", "bertopic"):
        for trial in range(2):
            path = e1.run_path(method, 10, trial, tmp_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "status": "complete",
                "coverage_split": {"generic": 0.5, "specific": 0.25},
                "coverage": 0.375, "n_generated": 4, "errors": {},
            }), encoding="utf-8")
    result = e1.collect(["lloom_gemma", "bertopic"], ns=[10, 20], n_trials=2,
                        out_dir=tmp_path)
    e10 = result["per_method"]["lloom_gemma"]["per_n"][0]
    assert e10["n_completed"] == 2
    assert e10["specific"]["mean"] == 0.25
    assert e10["specific"]["values"] == [0.25, 0.25]
    # n=20 has no completed cells
    assert result["per_method"]["lloom_gemma"]["per_n"][1]["n_completed"] == 0


def test_write_summary_renders_tables(tmp_path):
    result = e1.collect(["lloom_gemma"], ns=[10], n_trials=1, out_dir=tmp_path)
    jpath, mpath = e1.write_summary(result, summary_dir=tmp_path)
    text = mpath.read_text(encoding="utf-8")
    assert "## lloom_gemma" in text and "| 10 |" in text
    assert jpath.exists()
