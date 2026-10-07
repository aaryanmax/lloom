"""Deterministic trial sampling + ground-truth assembly."""

import pytest

from lloom_lab.experiments import sampling


def _docs(n=20):
    return [
        {"doc_id": f"d{i}", "doc_length": 5, "prevalence": 0.2,
         "concepts": {"specific": [f"s{i % 4}"], "generic": [f"g{i % 4}"]}}
        for i in range(n)
    ]


def test_subsample_deterministic():
    docs = _docs()
    a = sampling.subsample_docs(docs, 5, trial=0)
    b = sampling.subsample_docs(docs, 5, trial=0)
    assert [d["doc_id"] for d in a] == [d["doc_id"] for d in b]
    assert len(a) == 5


def test_cell_filter():
    docs = _docs() + [
        {"doc_id": "long", "doc_length": 10, "prevalence": 0.4,
         "concepts": {"specific": ["s9"], "generic": ["g9"]}},
    ]
    out = sampling.subsample_docs(docs, 1, cell=(10, 0.4))
    assert out[0]["doc_id"] == "long"


def test_n_too_large_raises():
    with pytest.raises(ValueError):
        sampling.subsample_docs(_docs(3), 5)


def test_ground_truth_union():
    gt = sampling.ground_truth_for(_docs())
    assert gt["specific"] == ["s0", "s1", "s2", "s3"]
    assert gt["generic"] == ["g0", "g1", "g2", "g3"]
