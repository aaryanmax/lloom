"""E1: coverage-vs-n across methods (the headline experiment).

Grid: n ∈ {10, 20, 40, 80, 160} × methods × 5 trials on the synthetic corpus.

Methods
  lloom_gemma      mini-LLooM (Distill/Score local Qwen, Synthesize → cloud
                   Flash-Lite chain 3.1 → 3.5, auto-failover to Gemma 4 31B
                   once both 500/day lite pools are exhausted)
  lloom_qwen       mini-LLooM, all-local (Synthesize → Qwen3-8B)
  zeroshot_gemini  Flash-Lite labels docs directly against the FULL 50-concept
                   taxonomy (no induction); coverage = fraction of the subset's
                   ground-truth concepts it actually uses on ≥1 doc
  zeroshot_qwen    same, local Qwen
  bertopic         classic topic model baseline (bge-small embeddings, BERTopic
                   UMAP+HDBSCAN), topics named by top keywords and matched by
                   the same few-shot matcher as the LLM-induced concepts

Scoring: LLooM and BERTopic concepts go through the batched few-shot matcher
(Flash-Lite; the probe showed Gemma 500s on large matcher prompts). Zero-shot
coverage is computed directly from its assignments — the taxonomy names ARE the
ground truths, so no matcher judgment is needed. *Specific* coverage is the
primary metric (the matcher over-matches generic umbrella GTs, pilot finding).

Everything is resumable: completed run files are skipped, all LLM responses are
cache-first, and a QuotaExceeded pauses the grid cleanly for the next day.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from lloom_lab.config import RUNS_DIR
from lloom_lab.data.concepts import GENERIC_TO_SPECIFIC
from lloom_lab.experiments import sampling
from lloom_lab.experiments.run_lloom import load_corpus, load_run, run_subset
from lloom_lab.llm.client import QuotaExceeded
from lloom_lab.metrics import coverage as cov
from lloom_lab.pipeline.cluster import embed

logger = logging.getLogger(__name__)

NS = (10, 20, 40, 80, 160)
N_TRIALS = 5
METHODS = ("lloom_gemma", "lloom_qwen", "zeroshot_gemini", "zeroshot_qwen",
           "bertopic")
E1_DIR = RUNS_DIR / "e1"
SUMMARY_DIR = RUNS_DIR.parent / "e1"   # results/e1 (summaries live beside runs/)
SEED_PHRASE = "about politics"
FULL_TAXONOMY = sorted(GENERIC_TO_SPECIFIC) + sorted(
    s for subs in GENERIC_TO_SPECIFIC.values() for s in subs)


def run_path(method: str, n: int, trial: int,
             out_dir: Path | None = None) -> Path:
    return (out_dir or E1_DIR) / method / f"n{n:03d}_t{trial}.json"


# --- zero-shot baseline ------------------------------------------------------

_ZEROSHOT_PROMPT = """You are labeling text documents with concepts from a fixed taxonomy.

TAXONOMY (use ONLY these exact concept names):
{taxonomy}

DOCUMENTS:
{docs}

TASK: For EACH document, list every taxonomy concept that clearly applies to it.
Rules:
- Use only exact concept names copied from the taxonomy above.
- If no taxonomy concept applies to a document, use an empty list.
- Do not invent concepts and do not modify names.

Respond with JSON only:
{{"assignments": [{{"doc_id": "<id>", "concepts": ["<CONCEPT>", ...]}}]}}"""


def zeroshot_assign(client, docs: list[dict], model: str,
                    batch_size: int = 10) -> dict[str, list[str]]:
    """Label each doc with taxonomy concepts; returns {doc_id: [concept, ...]}."""
    assignments: dict[str, list[str]] = {}
    taxonomy = "\n".join(f"- {c}" for c in FULL_TAXONOMY)
    allowed = set(FULL_TAXONOMY)
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        docs_block = "\n".join(f"[{d['doc_id']}] {d['text']}" for d in batch)
        prompt = _ZEROSHOT_PROMPT.format(taxonomy=taxonomy, docs=docs_block)
        data = client.chat_json(model, prompt, kind="zeroshot")
        rows = data.get("assignments", []) if isinstance(data, dict) else []
        for row in rows:
            doc_id = str(row.get("doc_id", ""))
            if doc_id not in {d["doc_id"] for d in batch}:
                continue  # hallucinated id
            names = [c for c in row.get("concepts", []) if c in allowed]
            assignments.setdefault(doc_id, []).extend(names)
    return assignments


def zeroshot_coverage(assignments: dict[str, list[str]],
                      ground_truth: dict[str, list[str]]) -> dict[str, float]:
    """Fraction of the subset's ground-truth concepts used on >= 1 doc."""
    used = {c for cs in assignments.values() for c in cs}
    out = {}
    for kind, gts in ground_truth.items():
        out[kind] = (sum(1 for g in gts if g in used) / len(gts)) if gts else 0.0
    return out


# --- BERTopic baseline -------------------------------------------------------

def bertopic_concepts(docs: list[dict], min_topic_size: int | None = None
                      ) -> tuple[list[dict], dict]:
    """Fit BERTopic on the subset's full texts; return matcher-ready
    pseudo-concepts (topics named by their top keywords) + run info."""
    from bertopic import BERTopic

    texts = [d["text"] for d in docs]
    n = len(texts)
    mts = min_topic_size or max(3, min(10, n // 10))
    X = embed(texts)
    info: dict = {"min_topic_size": mts, "n_docs": n}
    try:
        model = BERTopic(min_topic_size=mts, calculate_probabilities=False)
        _, probs = model.fit_transform(texts, X)
    except Exception as e:  # noqa: BLE001 — BERTopic degenerates on tiny subsets
        info["error"] = repr(e)
        return [], info
    topics = model.get_topics()
    info["n_topics"] = len([t for t in topics if t != -1])
    concepts = []
    for tid, words in sorted(topics.items()):
        if tid == -1 or not words:
            continue
        kws = [w for w, _ in words]
        concepts.append({
            "name": " ".join(kws[:3]).title(),
            "criteria_prompt": "Documents about: " + ", ".join(kws[:8]),
            "topic_id": int(tid),
            "top_words": kws[:10],
        })
    return concepts, info


# --- one grid cell -----------------------------------------------------------

def run_cell(client, corpus: list[dict], method: str, n: int, trial: int,
             out_dir: Path | None = None) -> dict:
    """Run + score one (method, n, trial) cell; persists and returns the record."""
    path = run_path(method, n, trial, out_dir)
    docs = sampling.subsample_docs(corpus, n, trial=trial)
    gt = sampling.ground_truth_for(docs)
    record: dict = {
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": {"experiment": "E1", "method": method, "n": n, "trial": trial},
        "doc_ids": sorted(d["doc_id"] for d in docs),
        "ground_truth": gt,
        "errors": {},
    }

    if method.startswith("lloom"):
        # cloud-synth arm routes through the lite chain (fast, 2×500/day);
        # the client's chain appends gemma as the last-resort pool
        synth_model = "gemini" if method == "lloom_gemma" else "qwen"
        record = run_subset(
            client, docs,
            run_cfg={"synth_model": synth_model, "seed_phrase": SEED_PHRASE,
                     "skip_score": True},
            run_meta=record["meta"], out_path=path)
        record["meta"] = {**record.get("meta", {}), "experiment": "E1",
                          "method": method, "n": n, "trial": trial}
        generated = cov.parse_generated(record.get("concepts", []))
    elif method.startswith("zeroshot"):
        model = "gemini" if method == "zeroshot_gemini" else "qwen"
        assignments = zeroshot_assign(client, docs, model)
        record["assignments"] = assignments
        generated = []
    elif method == "bertopic":
        concepts, info = bertopic_concepts(docs)
        record["bertopic"] = info
        record["concepts"] = concepts
        generated = cov.parse_generated(concepts)
    else:
        raise ValueError(f"unknown method {method}")

    # --- coverage ------------------------------------------------------------
    if method.startswith("zeroshot"):
        split = zeroshot_coverage(record["assignments"], gt)
        record["coverage_split"] = split
        record["coverage"] = sum(split.values()) / len(split) if split else 0.0
        record["n_generated"] = len(FULL_TAXONOMY)
    else:
        if generated:
            types = {g: "generic" for g in gt["generic"]}
            types.update({s: "specific" for s in gt["specific"]})
            rows = cov.match_concepts(client, gt["generic"] + gt["specific"],
                                      generated, model="gemini")
            split = cov.coverage_split(rows, types)
            record["coverage_split"] = split
            record["coverage"] = cov.coverage(rows)
            record["match_rows"] = rows
        else:
            record["coverage_split"] = {"generic": 0.0, "specific": 0.0}
            record["coverage"] = 0.0
        record["n_generated"] = len(generated)

    record["status"] = "complete" if not record.get("errors") \
        else "complete_with_errors"
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, record)
    return record


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


# --- the grid ----------------------------------------------------------------

def run_e1(client, methods: list[str] | tuple = METHODS,
           ns: list[int] | tuple = NS, n_trials: int = N_TRIALS,
           corpus: list[dict] | None = None,
           out_dir: Path | None = None, jobs: int = 1,
           client_factory=None) -> dict:
    """Execute (or resume) the E1 grid. Returns a partial/full summary dict.

    A QuotaExceeded pauses the grid: everything completed so far is on disk
    and a re-invocation skips it (cache makes re-scores free).

    jobs > 1 runs cells in a thread pool. The RPM throttle is process-global,
    so parallel workers share one slot timeline per model id and cannot jointly
    breach an RPM cap. Cells stay method-major so lloom_gemma and lloom_qwen
    (which share the distill cache for the same subsets) rarely overlap.
    """
    corpus = corpus if corpus is not None else load_corpus()
    out_dir = out_dir or E1_DIR
    paused = None

    def _one_cell(cell: tuple[str, int, int], cli) -> None:
        m, n, t = cell
        record = run_cell(cli, corpus, m, n, t, out_dir)
        cs = record.get("coverage_split", {})
        logger.info("%s n=%d t=%d: generic=%.2f specific=%.2f",
                    m, n, t, cs.get("generic", 0.0), cs.get("specific", 0.0))

    if jobs <= 1:
        for method in methods:
            for n in ns:
                for trial in range(n_trials):
                    if load_run(run_path(method, n, trial, out_dir)):
                        logger.info("skip complete cell %s n=%d t=%d",
                                    method, n, trial)
                        continue
                    try:
                        _one_cell((method, n, trial), client)
                    except Exception as e:  # noqa: BLE001
                        if isinstance(e, QuotaExceeded):
                            paused = f"quota at {method} n={n} t={trial}"
                            logger.warning("quota exhausted — pausing grid (%s)", paused)
                            break
                        logger.exception("cell %s n=%d t=%d crashed", method, n, trial)
                if paused:
                    break
            if paused:
                break
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cells = [(m, n, t) for m in methods for n in ns for t in range(n_trials)
                 if not load_run(run_path(m, n, t, out_dir))]
        logger.info("parallel grid: %d pending cells, %d workers", len(cells), jobs)
        import threading
        quota_hit = threading.Event()

        def _work(cell):
            if quota_hit.is_set():
                return
            try:
                cli = client_factory() if client_factory else client
                _one_cell(cell, cli)
            except Exception as e:  # noqa: BLE001
                if isinstance(e, QuotaExceeded):
                    quota_hit.set()
                    logger.warning("quota exhausted — pausing grid (%s n=%d t=%d)", *cell)
                else:
                    logger.exception("cell %s n=%d t=%d crashed", *cell)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_work, c) for c in cells]
            for _f in as_completed(futures):
                if quota_hit.is_set():
                    break
            # a completed cell must never be re-submitted by a later resume;
            # in-flight cells that hit the quota are simply redone next time
            for f in futures:
                f.cancel()
        if quota_hit.is_set():
            done = sum(1 for c in cells if load_run(run_path(c[0], c[1], c[2], out_dir)))
            paused = f"quota — {done}/{len(cells)} pending cells completed this pass"
    return collect(methods, ns, n_trials, out_dir, paused=paused)


def collect(methods: list[str] | tuple = METHODS, ns: list[int] | tuple = NS,
            n_trials: int = N_TRIALS, out_dir: Path | None = None,
            paused: str | None = None) -> dict:
    """Aggregate completed E1 records into a per-method summary structure."""
    from lloom_lab.experiments.pilot import summarize

    out_dir = out_dir or E1_DIR
    per_method = {}
    for method in methods:
        scores = []
        for n in ns:
            for trial in range(n_trials):
                rec = load_run(run_path(method, n, trial, out_dir))
                if rec is None:
                    continue
                cs = rec.get("coverage_split", {})
                scores.append({"n": n, "trial": trial,
                               "generic": cs.get("generic", 0.0),
                               "specific": cs.get("specific", 0.0),
                               "coverage": rec.get("coverage", 0.0),
                               "n_generated": rec.get("n_generated", 0),
                               "errors": rec.get("errors", {})})
        per_method[method] = summarize(scores, list(ns), n_trials)
    return {"per_method": per_method,
            "params": {"ns": list(ns), "n_trials": n_trials,
                       "methods": list(methods)},
            "paused": paused}


def write_summary(result: dict, summary_dir: Path | None = None) -> tuple[Path, Path]:
    """Persist the summary JSON + a markdown table per method."""
    summary_dir = summary_dir or SUMMARY_DIR
    summary_dir.mkdir(parents=True, exist_ok=True)
    jpath = summary_dir / "e1_summary.json"
    jpath.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    lines = ["# E1 — coverage vs n (primary metric: specific coverage)", ""]
    for method, summary in result["per_method"].items():
        lines.append(f"## {method}")
        lines.append("| n | runs | generic mean±std | specific mean±std | CV(specific) |")
        lines.append("|---|---|---|---|---|")
        for e in summary["per_n"]:
            g, s = e["generic"], e["specific"]
            if e["n_completed"] == 0:
                lines.append(f"| {e['n']} | 0 | — | — | — |")
                continue
            cv_cell = f"{s['cv']:.2f}" if s.get("cv") is not None else "—"
            lines.append(
                f"| {e['n']} | {e['n_completed']} | "
                f"{g['mean']:.2f} ± {g['std']:.2f} | "
                f"{s['mean']:.2f} ± {s['std']:.2f} | {cv_cell} |")
        lines.append("")
    if result.get("paused"):
        lines.append(f"**Grid paused**: {result['paused']} — re-run to resume.")
    mpath = summary_dir / "e1_summary.md"
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jpath, mpath
