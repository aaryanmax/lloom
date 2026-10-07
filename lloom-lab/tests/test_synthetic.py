"""Generator correctness: paper §5.2 invariants on a small corpus."""

import pytest

from lloom_lab.data import synthetic
from lloom_lab.data.concepts import ALL_SPECIFIC, parent_of


def test_seed_count_mapping():
    assert synthetic.seed_count(5, 0.20) == 1
    assert synthetic.seed_count(5, 0.40) == 2
    assert synthetic.seed_count(10, 0.20) == 2
    assert synthetic.seed_count(10, 0.40) == 4


def test_small_corpus_passes_verification():
    docs = synthetic.build_corpus(seed=7, docs_per_cell=3)
    report = synthetic.verify_corpus(docs, require_all_concepts=False)
    assert report["n_docs"] == 12
    assert report["problems"] == []


def test_corpus_deterministic():
    a = [d.to_dict() for d in synthetic.build_corpus(seed=7, docs_per_cell=2)]
    b = [d.to_dict() for d in synthetic.build_corpus(seed=7, docs_per_cell=2)]
    assert a == b


def test_every_concept_appears_in_full_corpus():
    docs = synthetic.build_corpus(seed=42, docs_per_cell=40)
    used = {s for d in docs for s in d.specifics}
    assert used == set(ALL_SPECIFIC)


def test_prevalence_exact_by_construction():
    docs = synthetic.build_corpus(seed=42, docs_per_cell=5)
    for d in docs:
        k = sum(len(v) for v in d.seed_sentences.values())
        assert k / d.doc_length == d.prevalence


def test_generics_are_parents_of_specifics():
    docs = synthetic.build_corpus(seed=42, docs_per_cell=3)
    for d in docs:
        assert d.generics == sorted({parent_of(s) for s in d.specifics})


def test_verify_rejects_corrupted_doc():
    docs = synthetic.build_corpus(seed=1, docs_per_cell=2)
    docs[0].doc_length = 99  # corrupt
    with pytest.raises(AssertionError):
        synthetic.verify_corpus(docs)
