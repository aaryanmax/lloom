"""Run (or resume) the GO/NO-GO pilot.

    python scripts/run_pilot.py                # full grid: n ∈ {10,25,50,100} × 2 trials
    python scripts/run_pilot.py --smoke        # one n=10 run — validates the whole chain cheaply
    python scripts/run_pilot.py --ns 10 25 --trials 1

Requires llama-server (local Distill/Score) and GEMINI_API_KEY (Synthesize +
matcher). Safe to re-run: completed runs are skipped and all responses cached.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import make_client  # noqa: E402
from lloom_lab.experiments.pilot import N_TRIALS, NS, run_pilot  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--synth-model", default="gemini",
                    choices=["gemini", "qwen"])
    ap.add_argument("--matcher-model", default="gemini",
                    choices=["gemini", "qwen"])
    ap.add_argument("--smoke", action="store_true",
                    help="single n=10 trial: validate the chain, ~15-30 calls")
    args = ap.parse_args()

    ns, trials = ([10], 1) if args.smoke else (args.ns, args.trials)
    client = make_client()
    try:
        result = run_pilot(client, ns=ns, n_trials=trials,
                           synth_model=args.synth_model,
                           matcher_model=args.matcher_model)
    except KeyboardInterrupt:
        logging.warning("interrupted — run records are saved; re-run to resume")
        sys.exit(130)
    finally:
        client.close()

    for row in result["scores"]:
        print(f"n={row['n']:>3} t={row['trial']}  generic={row['generic']:.2f}  "
              f"specific={row['specific']:.2f}  concepts={row['n_generated']}  "
              f"errors={len(row['errors'])}")
    print("\nverdict:", result["verdict"]["verdict"])
    for r in result["verdict"]["reasons"]:
        print(" -", r)
    print("\nfull summary: lloom-lab/results/pilot/pilot_summary.{json,md}")


if __name__ == "__main__":
    main()
