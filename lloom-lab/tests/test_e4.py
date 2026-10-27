"""Unit tests for E4 generalization gap module (offline, 0 API calls)."""

from unittest.mock import patch

from lloom_lab.experiments.e4 import gap_for_record


def mock_scorer(client, concept_name, criteria_prompt, docs_dict, model="gemini", batch_size=40):
    # Deterministic mock score: 1.0 if doc_id starts with "train", 0.5 otherwise
    return {
        doc_id: {"score": 1.0 if "train" in doc_id else 0.5, "rationale": "mock"}
        for doc_id in docs_dict
    }


def test_gap_for_record_computes_correct_gap():
    record = {
        "meta": {"n": 2, "trial": 0},
        "doc_ids": ["doc_train_1", "doc_train_2"],
        "concepts": [
            {"name": "Concept A", "criteria_prompt": "Prompt A"},
            {"name": "Concept B", "criteria_prompt": "Prompt B"},
        ],
    }
    corpus = [
        {"doc_id": "doc_train_1", "text": "Train doc 1"},
        {"doc_id": "doc_train_2", "text": "Train doc 2"},
        {"doc_id": "doc_held_1", "text": "Held doc 1"},
        {"doc_id": "doc_held_2", "text": "Held doc 2"},
    ]

    with patch("lloom_lab.experiments.e4.score_examples", side_effect=mock_scorer):
        result = gap_for_record(None, record, corpus, model="gemini")

    assert "meta" in result
    assert "gaps" in result
    assert len(result["gaps"]) == 2
    # Concept A: empirical = 1.0, held = 0.5 -> gap = 0.5
    for g in result["gaps"]:
        assert abs(g["empirical"] - 1.0) < 1e-6
        assert abs(g["held_out"] - 0.5) < 1e-6
        assert abs(g["gap"] - 0.5) < 1e-6
    assert abs(result["mean_gap"] - 0.5) < 1e-6


def test_gap_for_record_respects_caps():
    record = {
        "meta": {"n": 5, "trial": 0},
        "doc_ids": [f"doc_train_{i}" for i in range(5)],
        "concepts": [
            {"name": f"Concept {i}", "criteria_prompt": f"Prompt {i}"}
            for i in range(5)
        ],
    }
    corpus = [{"doc_id": f"doc_train_{i}", "text": "t"} for i in range(5)] + [
        {"doc_id": f"doc_held_{i}", "text": "h"} for i in range(5)
    ]

    with patch("lloom_lab.experiments.e4.score_examples", side_effect=mock_scorer):
        result = gap_for_record(
            None, record, corpus, concept_cap=2, train_cap=2, held_cap=2
        )

    assert len(result["gaps"]) == 2
