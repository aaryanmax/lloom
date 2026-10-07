"""Matcher validation (paper §5.1 style): how trustworthy is the automated
coverage metric?

Procedure:
1. Draw (ground_truth, generated_concept) pairs from completed run records
   (stratified: the matcher's own pick per ground truth, plus random non-picked
   concepts as hard negatives).
2. A human labels each pair 1/0 (exact same meaning or not) -> fixture JSON.
3. `validation_report` compares the matcher's decisions against the labels:
   per-pair agreement + per-run MAE between automated and manual coverage.

The paper validates on 16 trials (MAE = 0.07); our bar is documented in
results/pilot/matcher_validation_report.json when generated.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from lloom_lab.config import PILOT_DIR
from lloom_lab.metrics import coverage as cov

logger = logging.getLogger(__name__)

PAIRS_FILENAME = "matcher_validation_pairs.json"


def extract_pairs(runs: list[dict], n_pairs: int = 40, seed: int = 7) -> list[dict]:
    """Stratified (ground_truth, concept) pairs from run records.

    For each run: the matcher's chosen concept per ground truth is a natural
    pair; we add random unpaired concepts as hard negatives. `runs` items need
    keys: ground_truth, concepts, doc_ids (used as run id fallback).
    """
    rng = random.Random(seed)
    pairs: list[dict] = []
    per_run_budget = max(2, n_pairs // max(1, len(runs)))
    for run in runs:
        gts = run["ground_truth"]["generic"] + run["ground_truth"]["specific"]
        names = [c["name"] for c in run["concepts"]]
        by_name = {c["name"]: c for c in run["concepts"]}
        rows = cov.match_concepts(
            _CachedMatcherProxy(run), gts, cov.parse_generated(run["concepts"]))
        picked = {r["ground_truth"]: r["concept"] for r in rows}
        run_id = run.get("meta", {}).get("run_id", "")

        # Interleave positives and hard negatives so one category cannot
        # exhaust the run's budget before the other is sampled.
        positives = [(gt, picked[gt]) for gt in gts if picked.get(gt)]
        negatives = []
        for gt in gts:
            unpicked = [n for n in names if n != picked.get(gt)]
            if unpicked:
                negatives.append((gt, rng.choice(unpicked)))
        rng.shuffle(negatives)
        quota_pos = per_run_budget // 2
        chosen = positives[:quota_pos] + negatives[: per_run_budget - quota_pos]
        for gt, name in chosen:
            pairs.append(_pair(run_id, gt, name, by_name[name],
                               matcher_matched=(gt, name) in positives))
    return pairs[:n_pairs]


def _pair(run_id: str, gt: str, concept_name: str, concept: dict,
          matcher_matched: bool) -> dict:
    return {"run_id": run_id, "ground_truth": gt, "concept": concept_name,
            "concept_criteria": concept["criteria_prompt"],
            "matcher_matched": matcher_matched, "label": None, "rationale": ""}


class _CachedMatcherProxy:
    """Replays the matcher through the real client so cached responses are hit
    and no new API calls are made for pairs already scored in this run."""

    def __init__(self, run: dict):
        self._run = run

    def chat_json(self, model, prompt, **kwargs):
        from lloom_lab.config import make_client
        client = make_client()
        try:
            return client.chat_json(model, prompt, **kwargs)
        finally:
            client.close()


def manual_coverage(pairs: list[dict]) -> dict[str, float]:
    """Coverage per run using human labels only."""
    out: dict[str, float] = {}
    by_run: dict[str, list[dict]] = {}
    for p in pairs:
        if p["label"] is None:
            raise ValueError(f"unlabeled pair for {p['ground_truth']!r} — label it first")
        by_run.setdefault(p["run_id"], []).append(p)
    for run_id, ps in by_run.items():
        out[run_id] = sum(p["label"] for p in ps) / len(ps)
    return out


def validation_report(pairs: list[dict]) -> dict:
    """Agreement stats + MAE between matcher decisions and human labels."""
    labeled = [p for p in pairs if p["label"] is not None]
    if not labeled:
        raise ValueError("no labeled pairs")
    agree = sum(1 for p in labeled
                if bool(p["matcher_matched"]) == bool(p["label"]))
    tp = sum(1 for p in labeled if p["matcher_matched"] and p["label"])
    fp = sum(1 for p in labeled if p["matcher_matched"] and not p["label"])
    fn = sum(1 for p in labeled if not p["matcher_matched"] and p["label"])
    per_run = manual_coverage(labeled)
    return {
        "n_pairs": len(labeled),
        "agreement": agree / len(labeled),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "n_matches": tp + fn,
        "per_run_manual_coverage": per_run,
        "note": "labels authored by the project author (not independent annotators)",
    }


def load_labeled_pairs(path: Path | None = None) -> list[dict]:
    path = path or (PILOT_DIR / PAIRS_FILENAME)
    return json.loads(Path(path).read_text(encoding="utf-8"))
