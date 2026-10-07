"""Reproducible sampling of the synthetic corpus for trials.

Every experiment run is fully determined by (n, trial, seed): the same subset
of documents is recovered on any machine, so results JSONs stay comparable
across developers and reruns.
"""

from __future__ import annotations

import random


def subsample_docs(docs: list[dict], n: int, trial: int = 0, seed: int = 42,
                   cell: tuple[int, float] | None = None) -> list[dict]:
    """Deterministically draw n docs.

    cell=(doc_length, prevalence) restricts to one §5.2 combination (the
    paper's synthetic evaluation is per-cell); None draws from the whole corpus.
    """
    if cell is not None:
        length, prevalence = cell
        docs = [d for d in docs
                if d["doc_length"] == length and d["prevalence"] == prevalence]
    if n > len(docs):
        raise ValueError(f"requested n={n}, only {len(docs)} docs available")
    # random.Random accepts only primitive seeds; a stable string encodes all factors.
    rng = random.Random(f"{seed}|{n}|{trial}|{cell}")
    return rng.sample(docs, n)


def ground_truth_for(docs: list[dict]) -> dict[str, list[str]]:
    """Ground-truth concept sets present in a doc subset (coverage denominator).

    Returns {"generic": [...], "specific": [...]}.
    """
    generics: set[str] = set()
    specifics: set[str] = set()
    for d in docs:
        generics.update(d["concepts"]["generic"])
        specifics.update(d["concepts"]["specific"])
    return {"generic": sorted(generics), "specific": sorted(specifics)}
