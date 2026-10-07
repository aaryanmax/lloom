"""Synthetic dataset generator — reproduces LLooM paper §5.2 with full control.

Paper spec (§5.2.1 + Appendix C.4):
- Domain: "politics"
- Ground truth: 10 Generic concepts, each with 4 Specific children (40 Specific)
- Document length: 5 or 10 sentences
- Within-document concept prevalence: 20% or 40% of sentences are seed sentences
- Volume: 40 documents per length x prevalence combination (4 combos = 160 docs)
- Verification: programmatic (lengths + seed-sentence counts) + manual spot-check

Deviation from the paper (documented): the paper generated documents with GPT-4;
we compose documents from hand-written sentence pools so that (a) zero API calls
are needed to build the corpus, (b) ground-truth labels are exact by construction,
and (c) text-label mappings are guaranteed absent from any LLM pre-training data.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from lloom_lab.data.concepts import (
    ALL_GENERIC,
    ALL_SPECIFIC,
    parent_of,
)
from lloom_lab.data.sentences import FILLERS, SEED_SENTENCES

# (doc_length, prevalence) -> exact number of seed sentences.
# 5 sents: 20% -> 1, 40% -> 2.  10 sents: 20% -> 2, 40% -> 4.
LENGTHS = (5, 10)
PREVALENCES = (0.20, 0.40)


def seed_count(doc_length: int, prevalence: float) -> int:
    k = round(doc_length * prevalence)
    assert 0 < k < doc_length, (doc_length, prevalence)
    return k


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return _SENT_SPLIT.split(text.strip())


@dataclass
class SyntheticDoc:
    doc_id: str
    text: str
    doc_length: int
    prevalence: float
    specifics: list[str] = field(default_factory=list)
    seed_sentences: dict[str, list[str]] = field(default_factory=dict)

    @property
    def generics(self) -> list[str]:
        return sorted({parent_of(s) for s in self.specifics})

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "doc_length": self.doc_length,
            "prevalence": self.prevalence,
            "concepts": {"specific": self.specifics, "generic": self.generics},
            "seed_sentences": self.seed_sentences,
        }


class _SentenceBank:
    """Shuffles each pool and deals without repeats; reshuffles when exhausted
    so a sentence never repeats within a pool cycle (and never within a doc)."""

    def __init__(self, pool: list[str], rng: random.Random):
        self._pool = pool
        self._rng = rng
        self._queue: list[str] = []

    def draw(self) -> str:
        if not self._queue:
            self._queue = list(self._pool)
            self._rng.shuffle(self._queue)
        return self._queue.pop()


def build_corpus(
    seed: int = 42,
    docs_per_cell: int = 40,
    lengths: tuple[int, ...] = LENGTHS,
    prevalences: tuple[float, ...] = PREVALENCES,
) -> list[SyntheticDoc]:
    """Generate the full synthetic corpus.

    Concept assignment uses a round-robin over a reshuffled deck of all 40
    Specific concepts so that every concept appears across the corpus with
    roughly equal frequency regardless of cell.
    """
    rng = random.Random(seed)
    seed_banks = {s: _SentenceBank(SEED_SENTENCES[s], rng) for s in ALL_SPECIFIC}
    filler_bank = _SentenceBank(FILLERS, rng)
    deck: list[str] = []

    def next_concepts(n: int) -> list[str]:
        nonlocal deck
        if len(deck) < n:
            deck = list(ALL_SPECIFIC)
            rng.shuffle(deck)
        out = [deck.pop() for _ in range(n)]
        rng.shuffle(out)
        return out

    docs: list[SyntheticDoc] = []
    for length in lengths:
        for prevalence in prevalences:
            k = seed_count(length, prevalence)
            for i in range(docs_per_cell):
                # How many distinct Specific concepts this doc seeds: with
                # 1 or 2 seed sentences -> 1-2 concepts; with 4 -> 2-4. This
                # makes Generic concepts span multiple Specifics per doc,
                # mirroring the paper's "implicit theme" design.
                if k == 1:
                    n_concepts = 1
                elif k == 2:
                    n_concepts = rng.choice([1, 2])
                else:
                    n_concepts = rng.choice([2, 4])
                n_concepts = min(n_concepts, k)
                picks = next_concepts(n_concepts)
                # Distribute the k seed sentences over the picked concepts.
                allocation = [k // n_concepts] * n_concepts
                for j in range(k % n_concepts):
                    allocation[j] += 1

                seed_sents: list[str] = []
                per_concept: dict[str, list[str]] = {}
                for concept, n_sents in zip(picks, allocation, strict=True):
                    for _ in range(n_sents):
                        s = seed_banks[concept].draw()
                        seed_sents.append(s)
                        per_concept.setdefault(concept, []).append(s)
                fillers = [filler_bank.draw() for _ in range(length - k)]

                sentences = seed_sents + fillers
                rng.shuffle(sentences)
                doc_id = f"{length}sent_{int(prevalence * 100)}prev_{i:03d}"
                docs.append(SyntheticDoc(
                    doc_id=doc_id,
                    text=" ".join(sentences),
                    doc_length=length,
                    prevalence=prevalence,
                    specifics=sorted(per_concept),
                    seed_sentences=per_concept,
                ))
    return docs


def verify_corpus(docs: list[SyntheticDoc], require_all_concepts: bool = True) -> dict:
    """Programmatic verification (paper §5.2.1): lengths, seed-sentence counts,
    label consistency, and corpus balance. Raises AssertionError on violation.

    `require_all_concepts=False` relaxes the every-concept-present check for
    small test corpora (the real 40-docs-per-cell corpus covers all 40).
    """
    problems: list[str] = []
    lengths_ok = prevalence_ok = labels_ok = True
    per_concept_docs: dict[str, int] = {s: 0 for s in ALL_SPECIFIC}
    seen_sentences: dict[str, list[str]] = {}

    for d in docs:
        n = len(split_sentences(d.text))
        if n != d.doc_length:
            lengths_ok = False
            problems.append(f"{d.doc_id}: {n} sentences, expected {d.doc_length}")
        k_expected = seed_count(d.doc_length, d.prevalence)
        k_actual = sum(len(v) for v in d.seed_sentences.values())
        if k_actual != k_expected:
            prevalence_ok = False
            problems.append(
                f"{d.doc_id}: {k_actual} seed sentences, expected {k_expected}")
        if sorted(d.seed_sentences) != d.specifics:
            labels_ok = False
            problems.append(f"{d.doc_id}: label/seed mismatch")
        for g in d.generics:
            if g not in ALL_GENERIC:
                labels_ok = False
                problems.append(f"{d.doc_id}: unknown generic {g!r}")
        for s, sents in d.seed_sentences.items():
            per_concept_docs[s] += 1
            for sent in sents:
                if sent not in SEED_SENTENCES[s]:
                    problems.append(f"{d.doc_id}: sentence not in pool of {s}")
                seen_sentences.setdefault(sent, []).append(d.doc_id)
        if len(set(d.text.split())) == 0:  # trivial guard
            problems.append(f"{d.doc_id}: empty text")

    dup_pairs = [(s, ids) for s, ids in seen_sentences.items() if len(ids) > 1]
    missing = [s for s, c in per_concept_docs.items() if c == 0]

    report = {
        "n_docs": len(docs),
        "sentence_counts_ok": lengths_ok,
        "prevalence_counts_ok": prevalence_ok,
        "labels_ok": labels_ok,
        "problems": problems,
        "docs_per_concept": {s: c for s, c in per_concept_docs.items()},
        "min_docs_per_concept": min(per_concept_docs.values()),
        "max_docs_per_concept": max(per_concept_docs.values()),
        "concepts_missing": missing,
        "reused_sentences": len(dup_pairs),
    }
    assert lengths_ok and prevalence_ok and labels_ok, problems
    if require_all_concepts:
        assert not missing, f"concepts missing from corpus: {missing}"
    return report
