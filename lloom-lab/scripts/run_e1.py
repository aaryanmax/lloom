"""E1 driver CLI: coverage-vs-n across methods, resumable, quota-guarded.

Requires llama-server running (Distill/Score/zero-shot-Qwen are local).

    C:/Users/Admin/Projects/.supervenv/Scripts/python.exe scripts/run_e1.py
    C:/Users/Admin/Projects/.supervenv/Scripts/python.exe scripts/run_e1.py \
        --methods lloom_gemma bertopic --trials 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import make_client  # noqa: E402
from lloom_lab.experiments import e1  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(e1.METHODS),
                        choices=e1.METHODS)
    parser.add_argument("--ns", nargs="+", type=int, default=list(e1.NS))
    parser.add_argument("--trials", type=int, default=e1.N_TRIALS)
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel cells (RPM caps stay enforced process-wide)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = make_client()
    result = e1.run_e1(client, methods=args.methods, ns=args.ns,
                       n_trials=args.trials, jobs=args.jobs,
                       client_factory=make_client)
    jpath, mpath = e1.write_summary(result)

    # quick spend report across all cloud pools (legacy label "gemini" rows
    # count toward the first pool in the chain, mirroring pick_gemini_provider)
    chain = {}
    for i, (mid, d, _) in enumerate(client.gemini_chain):
        chain[mid] = [d, client.ledger.today_count(mid)]
        if i == 0:
            chain[mid][1] += client.ledger.today_count("gemini")
    chain[client.gemma_chain[0]] = [
        client.gemma_chain[1], client.ledger.today_count(client.gemma_chain[0])]
    spend = "  ".join(f"{mid}: {used}/{lim}" for mid, (lim, used) in chain.items())
    print(f"\nE1 grid pause={result.get('paused') or 'no'} — summary:")
    print(f"  {jpath}\n  {mpath}")
    print(f"today's cloud spend: {spend}")


if __name__ == "__main__":
    main()
