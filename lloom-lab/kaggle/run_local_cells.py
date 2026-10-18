"""Run LLooM cells with ALL-local models (no Gemini, no ledger) — Kaggle only.

    python kaggle/run_local_cells.py --method lloom_qwen --jobs 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import DATA_STORE, RUNS_DIR, make_client
from lloom_lab.experiments import sampling
from lloom_lab.experiments.run_lloom import run_subset


def load_corpus():
    p = DATA_STORE / "synthetic_v1" / "docs.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="lloom_qwen")
    ap.add_argument("--ns", nargs="+", type=int, default=[10, 20, 40, 80, 160])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor
    corpus = load_corpus()
    out_dir = RUNS_DIR / "e1" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    def cell(n, t):
        path = out_dir / f"n{n:03d}_t{t}.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("concepts"):
            return f"skip n{n} t{t}"
        docs = sampling.subsample_docs(corpus, n, trial=t)
        rec = {
            "status": "running",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "meta": {"experiment": "E1", "method": args.method, "n": n, "trial": t},
            "doc_ids": sorted(d["doc_id"] for d in docs),
            "errors": {}
        }
        try:
            run_subset(make_client(), docs,
                       run_cfg={"synth_model": "qwen", "skip_score": True},
                       run_meta=rec["meta"], out_path=path)
            return f"done n{n} t{t}"
        except Exception as ex:
            return f"FAIL n{n} t{t}: {ex}"

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for res in pool.map(lambda c: cell(*c), [(n, t) for n in args.ns for t in range(args.trials)]):
            print(res, flush=True)


if __name__ == "__main__":
    main()
