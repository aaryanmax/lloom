"""No-Free-Lunch (E5) permuted-label variants.

The classic NFL construction: keep the texts fixed, permute the ground-truth
labels across documents. Any method that derives concepts from text content can
do no better than chance against a relabeled ground truth, because no text
feature carries information about the permuted assignment.

We use a *derangement* (no document keeps its own labels), which is the
strongest form of the relabeling. Each variant has a fixed seed so runs are
reproducible.
"""

from __future__ import annotations

import random


def _derangement(n: int, rng: random.Random) -> list[int]:
    """Random permutation p with p[i] != i for all i (rejection sampling)."""
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def permuted_labels(
    labels: dict[str, dict], seed: int
) -> dict[str, dict]:
    """Given {doc_id: {"specific": [...], "generic": [...]}} return a new
    mapping where every document receives another document's labels.

    A position-level derangement is not enough: two documents can share an
    identical label set, so we retry until the *label-level* mapping is also
    a derangement (no doc keeps an equal label set).
    """
    doc_ids = sorted(labels)
    while True:
        p = _derangement(len(doc_ids), random.Random(seed))
        mapping = {doc_ids[i]: labels[doc_ids[p[i]]] for i in range(len(doc_ids))}
        if all(mapping[d] != labels[d] for d in doc_ids):
            return mapping
        seed += 1  # deterministic retry chain


def build_variants(
    labels: dict[str, dict], n_variants: int = 10, base_seed: int = 2026
) -> dict[str, dict[str, dict]]:
    """Return {variant_name: {doc_id: labels}} for n_variants NFL versions."""
    out = {}
    for v in range(n_variants):
        seed = base_seed + v
        out[f"nfl_v{v:02d}_seed{seed}"] = permuted_labels(labels, seed)
    return out
