"""GO/NO-GO pilot (Phase 2/3 gate): coverage-vs-n on the synthetic corpus.

Runs the LLooM pipeline on n ∈ {10, 25, 50, 100} × `n_trials` document
subsets, scores coverage with the batched matcher (Gemini), and computes a
noise verdict. Everything is resumable: completed run files are skipped and
every LLM response is cache-first, so re-invocations cost ~nothing.

The GO/NO-GO rule is a documented heuristic (only 2 trials in the pilot):
GO if the coefficient of variation of *specific* coverage at the largest n
is ≤ NOISE_THRESHOLD and mean specific coverage is not near floor.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from lloom_lab.config import PILOT_DIR, RUNS_DIR
from lloom_lab.experiments import sampling
from lloom_lab.experiments.run_lloom import load_corpus, run_subset
from lloom_lab.metrics import coverage as cov

logger = logging.getLogger(__name__)

NS = (10, 25, 50, 100)
N_TRIALS = 2
NOISE_THRESHOLD = 0.25   # max acceptable CV of specific coverage at n_max
FLOOR = 0.20             # below this mean specific coverage the signals are too weak


def run_id(n: int, trial: int) -> str:
    return f"n{n:03d}_t{trial}"


def run_path(n: int, trial: int, out_dir: Path | None = None) -> Path:
    return (out_dir or (RUNS_DIR / "pilot")) / f"{run_id(n, trial)}.json"


def score_run(client, run_record: dict, matcher_model: str = "gemini",
              batch_size: int = cov.DEFAULT_BATCH_SIZE) -> dict:
    """Automated coverage of one run record via the few-shot matcher."""
    generated = cov.parse_generated(run_record["concepts"])
    ground = run_record["ground_truth"]
    gts = ground["generic"] + ground["specific"]
    if not generated:
        return {"coverage": 0.0, "coverage_split": {"generic": 0.0, "specific": 0.0},
                "match_rows": [], "n_generated": 0}
    types = {g: "generic" for g in ground["generic"]}
    types.update({s: "specific" for s in ground["specific"]})
    rows = cov.match_concepts(client, gts, generated, model=matcher_model,
                              batch_size=batch_size)
    split = cov.coverage_split(rows, types)
    return {"coverage": cov.coverage(rows), "coverage_split": split,
            "match_rows": rows, "n_generated": len(generated)}


def summarize(scores: list[dict], ns: list[int], n_trials: int) -> dict:
    """Pure aggregation: per-n mean/std of coverage (pure function, unit-tested).

    `scores` items: {"n", "trial", "generic", "specific"}.
    """
    out = {"per_n": [], "n_trials": n_trials}
    for n in ns:
        rows = [s for s in scores if s["n"] == n]
        entry = {"n": n, "n_completed": len(rows)}
        for kind in ("generic", "specific"):
            vals = [s[kind] for s in rows]
            entry[kind] = {
                "mean": statistics.fmean(vals) if vals else None,
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "values": vals,
            }
        # coefficient of variation where meaningful
        for kind in ("generic", "specific"):
            mean = entry[kind]["mean"]
            entry[kind]["cv"] = (entry[kind]["std"] / mean
                                 if mean and mean > 0 else None)
        out["per_n"].append(entry)
    return out


def decide(summary: dict, noise_threshold: float = NOISE_THRESHOLD,
           floor: float = FLOOR) -> dict:
    """GO/NO-GO verdict from a summarize() structure (pure function)."""
    last = max(summary["per_n"], key=lambda e: e["n"])
    spec = last["specific"]
    reasons: list[str] = []
    if spec["mean"] is None or last["n_completed"] < summary["n_trials"]:
        reasons.append(f"only {last['n_completed']} run(s) at n={last['n']} — incomplete")
        return {"verdict": "NO-GO", "reasons": reasons}
    if spec["mean"] < floor:
        reasons.append(
            f"mean specific coverage at n={last['n']} is {spec['mean']:.2f} "
            f"< floor {floor} — pipeline may be under-performing")
    cv = spec.get("cv")
    if cv is not None and cv > noise_threshold:
        reasons.append(
            f"specific-coverage CV at n={last['n']} is {cv:.2f} > "
            f"{noise_threshold} — too noisy for clean curves")
    if not reasons:
        reasons.append(
            f"specific coverage {spec['mean']:.2f} ± {spec['std']:.2f} at "
            f"n={last['n']} (CV {cv:.2f}) — signal looks clean enough")
        return {"verdict": "GO", "reasons": reasons}
    return {"verdict": "NO-GO", "reasons": reasons}


def run_pilot(client, ns: list[int] = NS, n_trials: int = N_TRIALS,
              corpus: list[dict] | None = None,
              seed_phrase: str = "about politics",
              synth_model: str = "gemini", matcher_model: str = "gemini",
              out_dir: Path | None = None) -> dict:
    """Execute (or resume) the pilot; returns the summary + verdict."""
    corpus = corpus if corpus is not None else load_corpus()
    out_dir = out_dir or (RUNS_DIR / "pilot")
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: list[dict] = []
    for n in ns:
        for trial in range(n_trials):
            path = run_path(n, trial, out_dir)
            record = None
            try:
                record = run_subset(
                    client, sampling.subsample_docs(corpus, n, trial=trial),
                    # skip_score: coverage comes from the matcher (concepts +
                    # criteria), not from per-doc scores — saves ~80% of the
                    # slow local compute for exactly the same coverage number.
                    run_cfg={"synth_model": synth_model,
                             "seed_phrase": seed_phrase, "skip_score": True},
                    run_meta={"pilot": True, "n": n, "trial": trial},
                    out_path=path)
            except Exception:
                # run_subset already persists stage failures; a throw here is
                # unexpected (disk, sampling) — log and continue the grid.
                logger.exception("run %s crashed", run_id(n, trial))
                continue
            if not record.get("concepts"):
                logger.warning("run %s produced no concepts — scoring as zero",
                               run_id(n, trial))
            try:
                s = score_run(client, record, matcher_model=matcher_model)
                scores.append({"n": n, "trial": trial,
                               "generic": s["coverage_split"].get("generic", 0.0),
                               "specific": s["coverage_split"].get("specific", 0.0),
                               "coverage": s["coverage"],
                               "n_generated": s["n_generated"],
                               "errors": record.get("errors", {})})
                logger.info("run %s: generic=%.2f specific=%.2f (%d concepts)",
                            run_id(n, trial), scores[-1]["generic"],
                            scores[-1]["specific"], s["n_generated"])
            except Exception:
                logger.exception("matcher failed for %s (run record saved; "
                                 "re-run to rescore from cache)", run_id(n, trial))

    summary = summarize(scores, ns, n_trials)
    verdict = decide(summary)
    result = {"summary": summary, "verdict": verdict,
              "scores": scores,
              "params": {"ns": ns, "n_trials": n_trials,
                         "synth_model": synth_model,
                         "matcher_model": matcher_model,
                         "noise_threshold": NOISE_THRESHOLD, "floor": FLOOR}}

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    (PILOT_DIR / "pilot_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["| n | runs | generic mean±std | specific mean±std | CV(specific) |",
             "|---|---|---|---|---|"]
    for e in summary["per_n"]:
        g, s = e["generic"], e["specific"]
        cv_cell = f"{s['cv']:.2f}" if s["cv"] is not None else "—"
        lines.append(
            f"| {e['n']} | {e['n_completed']} | "
            f"{g['mean']:.2f} ± {g['std']:.2f} | "
            f"{s['mean']:.2f} ± {s['std']:.2f} | {cv_cell} |")
    (PILOT_DIR / "pilot_summary.md").write_text(
        "# GO/NO-GO pilot\n\n" + "\n".join(lines) +
        f"\n\n**Verdict: {verdict['verdict']}** — " +
        " ".join(verdict["reasons"]) + "\n", encoding="utf-8")
    logger.info("pilot verdict: %s", verdict["verdict"])
    return result
