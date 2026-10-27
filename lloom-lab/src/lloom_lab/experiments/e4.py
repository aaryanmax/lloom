"""E4: empirical vs held-out Score gap as n grows (uniform convergence).

    python -m lloom_lab.experiments.e4 --methods lloom_gemma lloom_qwen --ns 40 80 160
    # (on Kaggle T4: same command; llama-server on port 8123, full offload)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import RESULTS_DIR, make_client
from lloom_lab.experiments import e1
from lloom_lab.pipeline.score import score_examples

E4_DIR = RESULTS_DIR / "e4"


def load_corpus() -> list[dict]:
    from lloom_lab.config import DATA_STORE
    p = DATA_STORE / "synthetic_v1" / "docs.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def gap_for_record(client, record: dict, corpus: list[dict], model: str = "gemini",
                   held_cap: int = 40, train_cap: int = 40, concept_cap: int = 3,
                   batch_size: int = 40, seed: int = 42) -> dict:
    """Mean empirical vs held-out score across the record's induced concepts."""
    by_id = {d["doc_id"]: d["text"] for d in corpus}
    all_train_ids = list(record.get("doc_ids", []))
    rng = random.Random(seed)
    if len(all_train_ids) <= train_cap:
        train_ids = all_train_ids
    else:
        train_ids = rng.sample(all_train_ids, train_cap)

    held_pool = sorted(set(by_id.keys()) - set(all_train_ids))
    held_ids = held_pool if len(held_pool) <= held_cap else rng.sample(held_pool, held_cap)

    concepts = record.get("concepts", [])
    valid_concepts = [c for c in concepts if c.get("name") and c.get("criteria_prompt")]
    if concept_cap and len(valid_concepts) > concept_cap:
        selected_concepts = rng.sample(valid_concepts, concept_cap)
    else:
        selected_concepts = valid_concepts

    gaps = []
    for c in selected_concepts:
        name = c.get("name", "")
        criteria = c.get("criteria_prompt", "")
        if not criteria:
            continue
        r_tr = score_examples(client, name, criteria,
                              {i: by_id[i] for i in train_ids}, model=model,
                              batch_size=batch_size)
        r_ho = score_examples(client, name, criteria,
                              {i: by_id[i] for i in held_ids}, model=model,
                              batch_size=batch_size)
        emp = sum(v["score"] for v in r_tr.values()) / max(1, len(r_tr))
        hld = sum(v["score"] for v in r_ho.values()) / max(1, len(r_ho))
        gaps.append({
            "concept": name,
            "empirical": emp,
            "held_out": hld,
            "gap": emp - hld
        })

    mg = [g["gap"] for g in gaps]
    return {
        "meta": record.get("meta", {}),
        "n_concepts": len(gaps),
        "mean_empirical": sum(g["empirical"] for g in gaps) / max(1, len(gaps)),
        "mean_held_out": sum(g["held_out"] for g in gaps) / max(1, len(gaps)),
        "mean_gap": sum(mg) / max(1, len(mg)) if mg else 0.0,
        "gaps": gaps
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["lloom_qwen"])
    ap.add_argument("--ns", nargs="+", type=int, default=[40, 80, 160])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--model", default="gemini")
    ap.add_argument("--concept-cap", type=int, default=3)
    ap.add_argument("--train-cap", type=int, default=40)
    ap.add_argument("--held-cap", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=40)
    args = ap.parse_args()

    client = make_client()
    corpus = load_corpus()
    E4_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print(
        f"Running E4: methods={args.methods}, ns={args.ns}, "
        f"trials={args.trials}, model={args.model}"
    )
    for m in args.methods:
        for n in args.ns:
            for t in range(args.trials):
                out = E4_DIR / f"{m}_n{n:03d}_t{t}.json"
                if out.exists():
                    rows.append(json.loads(out.read_text(encoding="utf-8")))
                    print(f"[cached] {m} n={n} t={t}")
                    continue
                rec = e1.load_run(e1.run_path(m, n, t))
                if rec is None or not rec.get("concepts"):
                    print(f"[skip] {m} n={n} t={t}: no concepts found")
                    continue
                try:
                    row = gap_for_record(client, rec, corpus, model=args.model,
                                         held_cap=args.held_cap, train_cap=args.train_cap,
                                         concept_cap=args.concept_cap, batch_size=args.batch_size)
                except Exception as ex:
                    print(f"{m} n={n} t={t} FAILED: {ex} — re-run to resume")
                    break
                out.write_text(json.dumps(row, indent=1), encoding="utf-8")
                rows.append(row)
                print(f"[done] {m} n={n} t={t}: mean_gap={row['mean_gap']:.4f}")

    (E4_DIR / "e4_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\n--- E4 Summary ---")
    for n in sorted(list({r["meta"]["n"] for r in rows})):
        sel = [r["mean_gap"] for r in rows if r["meta"]["n"] == n]
        print(f"n={n:3d}: mean gap = {sum(sel)/len(sel):.4f} across {len(sel)} runs")


if __name__ == "__main__":
    main()
