"""Matcher validation harness (two-step, near-zero API).

    python scripts/validate_matcher.py --extract   # sample pairs from completed runs
    # ... human-labels each pair's "label" field in the JSON (1 = same meaning) ...
    python scripts/validate_matcher.py --report    # agreement stats vs matcher

Uses the response cache: re-scoring runs costs no new Gemini calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import PILOT_DIR  # noqa: E402
from lloom_lab.experiments import matcher_validation as mv  # noqa: E402
from lloom_lab.experiments.pilot import N_TRIALS, NS, run_id, run_path  # noqa: E402
from lloom_lab.experiments.run_lloom import load_run  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _completed_runs() -> list[dict]:
    runs = []
    for n in NS:
        for t in range(N_TRIALS):
            rec = load_run(run_path(n, t))
            if rec and rec.get("concepts"):
                rec.setdefault("meta", {})["run_id"] = run_id(n, t)
                runs.append(rec)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extract", action="store_true", help="sample pairs into the pairs file")
    ap.add_argument("--report", action="store_true", help="compute agreement stats")
    ap.add_argument("--n-pairs", type=int, default=40)
    args = ap.parse_args()
    if not (args.extract or args.report):
        ap.error("choose --extract and/or --report")

    pairs_path = PILOT_DIR / mv.PAIRS_FILENAME
    if args.extract:
        runs = _completed_runs()
        if not runs:
            sys.exit("no completed pilot runs found")
        pairs = mv.extract_pairs(runs, n_pairs=args.n_pairs)
        if pairs_path.exists():  # keep existing labels for surviving pairs
            old = {(p["run_id"], p["ground_truth"], p["concept"]): p["label"]
                   for p in json.loads(pairs_path.read_text(encoding="utf-8"))}
            for p in pairs:
                p["label"] = old.get((p["run_id"], p["ground_truth"], p["concept"]))
        pairs_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        unlabeled = sum(1 for p in pairs if p["label"] is None)
        print(f"wrote {len(pairs)} pairs to {pairs_path} ({unlabeled} unlabeled — "
              "set 'label': 1/0 in the JSON, then run --report)")

    if args.report:
        pairs = mv.load_labeled_pairs(pairs_path)
        report = mv.validation_report(pairs)
        out = PILOT_DIR / "matcher_validation_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
