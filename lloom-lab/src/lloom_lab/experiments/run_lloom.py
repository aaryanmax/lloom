"""Experiment driver: run the mini-LLooM pipeline on a document subset and
persist a complete, self-describing run record.

Design goals:
- Resumable: the response cache (prompt-hash keyed) makes re-running a
  partially completed run nearly free; run records are written incrementally
  after every stage so a crash never loses more than the current stage.
- Robust: per-stage try/except with logged, structured failures — one bad
  LLM response never aborts a whole experiment. Distill failures degrade to
  empty bullets; Synthesize failures skip a cluster; Score failures count as
  non-matches (handled inside score.py).
- Self-describing: every run JSON embeds its exact config, doc ids, ground
  truth, model ids, and ledger snapshot for later analysis.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from lloom_lab.config import RUNS_DIR
from lloom_lab.pipeline import cluster as cluster_mod
from lloom_lab.pipeline import distill as distill_mod
from lloom_lab.pipeline import score as score_mod
from lloom_lab.pipeline import synthesize as synth_mod

logger = logging.getLogger(__name__)

DEFAULT_RUN_CFG = {
    "distill_model": "qwen",
    # Flash-Lite chain (3.1 -> 3.5, two 500/day pools = 1000/day total) —
    # ~2 s/call vs Gemma's 50-70 s, so the lite pools are always drained
    # first; Gemma only serves if both lite pools are exhausted. Flash-Lite
    # stays on the matcher (precision-critical, low volume).
    "synth_model": "gemini",
    "score_model": "qwen",
    "min_cluster_size": 3,
    # One unifying pattern per cluster (paper §3 spirit). The pilot showed 3
    # per tiny cluster explodes into ~95 micro-concepts at n=100 that drown
    # the matcher; 1/cluster keeps the concept count ≈ cluster count.
    "n_concepts_per_cluster": 1,
    "score_batch_size": 20,
    "use_filter": False,
    "seed_phrase": "",
    # Coverage (E1/pilot) is computed by the *matcher* from concepts + criteria,
    # not from scores — skipping Score saves ~80% of local compute. Turn off
    # only for analyses that need concept scores (prevalence, E4).
    "skip_score": False,
}


def load_corpus(path: Path | None = None) -> list[dict]:
    """Load the synthetic corpus (docs.jsonl) as plain dicts."""
    path = path or (RUNS_DIR.parent.parent / "data_store" / "synthetic_v1" / "docs.jsonl")
    if not path.exists():
        alt = Path(__file__).resolve().parents[3] / "data_store" / "synthetic_v1" / "docs.jsonl"
        path = alt if alt.exists() else path
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _atomic_write(path: Path, payload: dict) -> None:
    """Write JSON atomically (tmp + replace) so a crash never leaves a torn file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def run_subset(client, docs: list[dict], run_cfg: dict | None = None,
               run_meta: dict | None = None,
               out_path: Path | None = None) -> dict:
    """Run one LLooM iteration over `docs` (list of corpus dicts).

    Returns the run record: {"status", "config", "meta", "doc_ids", "text",
    "ground_truth", "distilled", "clusters", "concepts", "scores", "errors"}.
    """
    cfg = {**DEFAULT_RUN_CFG, **(run_cfg or {})}
    docs_map = {d["doc_id"]: d["text"] for d in docs}
    record: dict[str, Any] = {
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg,
        "meta": run_meta or {},
        "doc_ids": sorted(docs_map),
        "ground_truth": {
            "specific": sorted({s for d in docs for s in d["concepts"]["specific"]}),
            "generic": sorted({g for d in docs for g in d["concepts"]["generic"]}),
        },
        "errors": {},
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(out_path, record)

    # --- 1. Distill (local model; the highest-volume step) -------------------
    try:
        distilled = []
        for i, (doc_id, text) in enumerate(sorted(docs_map.items()), 1):
            d = distill_mod.distill(client, text, seed_phrase=cfg["seed_phrase"],
                                    use_filter=cfg["use_filter"],
                                    model=cfg["distill_model"])
            d["doc_id"] = doc_id
            if not d["bullets"]:
                # A doc that distills to nothing would silently vanish from
                # clustering; keep a marker bullet so it stays in the pool.
                logger.warning("doc %s produced no bullets", doc_id)
                d["bullets"] = [f"(no content extracted from {doc_id})"]
            distilled.append(d)
            if out_path is not None and i % 10 == 0:
                record["distilled"] = distilled
                _atomic_write(out_path, record)
        record["distilled"] = distilled
    except Exception as e:  # noqa: BLE001 — persist and continue where possible
        logger.exception("distill stage failed")
        record["errors"]["distill"] = repr(e)
        record["distilled"] = locals().get("distilled", [])
    if out_path is not None:
        _atomic_write(out_path, record)

    # --- 2. Cluster (local embeddings + HDBSCAN) ------------------------------
    try:
        clusters = cluster_mod.pooled_clusters(record["distilled"],
                                               cfg["min_cluster_size"])
        record["clusters"] = {str(k): [it["doc_id"] for it in v]
                              for k, v in clusters.items()}
        if not clusters:
            raise RuntimeError("no clusters found — clustering collapsed")
    except Exception as e:
        logger.exception("cluster stage failed")
        record["errors"]["cluster"] = repr(e)
        record["status"] = "failed"
        record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if out_path is not None:
            _atomic_write(out_path, record)
        return record
    if out_path is not None:
        _atomic_write(out_path, record)

    # --- 3. Synthesize (cloud model; one call per cluster) -------------------
    concepts: list[dict] = []
    seen_names: set[str] = set()
    for label, items in sorted(clusters.items()):
        try:
            for c in synth_mod.synthesize(client, items,
                                          seed_phrase=cfg["seed_phrase"],
                                          n_concepts=cfg["n_concepts_per_cluster"],
                                          model=cfg["synth_model"]):
                key = c["name"].lower()
                if key not in seen_names:
                    seen_names.add(key)
                    c["cluster_label"] = label
                    concepts.append(c)
        except Exception as e:
            # One degenerate cluster (bad JSON, timeout) must not kill the run.
            logger.warning("synthesize failed for cluster %s: %r", label, e)
            record["errors"][f"synthesize.cluster_{label}"] = repr(e)
    record["concepts"] = concepts
    if out_path is not None:
        _atomic_write(out_path, record)

    # --- 4. Score (local model; batched multiple-choice) ---------------------
    scores: dict[str, dict] = {}
    if cfg["skip_score"]:
        record["scores"] = scores
    else:
        for c in concepts:
            try:
                scores[c["name"]] = score_mod.score_examples(
                    client, c["name"], c["criteria_prompt"], docs_map,
                    model=cfg["score_model"], batch_size=cfg["score_batch_size"])
            except Exception as e:
                logger.warning("score failed for concept %r: %r", c["name"], e)
                record["errors"][f"score.{c['name']}"] = repr(e)
            # scoring is the slowest stage — persist per concept so progress is
            # visible (and a crash loses at most one concept's work)
            record["scores"] = scores
            if out_path is not None:
                _atomic_write(out_path, record)
    record["status"] = "complete" if not record["errors"] else "complete_with_errors"
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if out_path is not None:
        _atomic_write(out_path, record)
    return record


def load_run(path: Path) -> dict | None:
    """Load a run record; None if missing or not complete."""
    if not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("corrupt run file: %s", path)
        return None
    return data if data.get("status", "").startswith("complete") else None
