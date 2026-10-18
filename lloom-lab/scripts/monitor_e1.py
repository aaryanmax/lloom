"""E1 live progress monitor with auto-refresh and quota tracking."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import make_client  # noqa: E402
from lloom_lab.experiments import e1  # noqa: E402


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def render_dashboard(client) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 64)
    print(f"  E1 GRID EXPERIMENT MONITOR — {now}")
    print("=" * 64)

    total_done = 0
    total_cells = len(e1.METHODS) * len(e1.NS) * e1.N_TRIALS

    print(f"\n{'Method':<18} | {'Progress':<10} | {'Completed Cells / Details'}")
    print("-" * 64)

    for m in e1.METHODS:
        completed = []
        for n in e1.NS:
            for t in range(e1.N_TRIALS):
                p = e1.run_path(m, n, t)
                if p.exists():
                    completed.append((n, t))

        done_count = len(completed)
        total_done += done_count
        pct = (done_count / 25) * 100

        # Bar chart
        bar_len = 10
        filled = int(bar_len * done_count / 25)
        bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"

        # Latest completed
        details = f"{bar} {done_count:2d}/25 ({pct:4.0f}%)"
        if done_count == 25:
            details += " [DONE]"
        elif completed:
            last_n, last_t = completed[-1]
            details += f" (last: n={last_n} t={last_t})"
        else:
            details += " (pending)"

        print(f"{m:<18} | {details}")

    print("-" * 64)
    total_pct = (total_done / total_cells) * 100
    overall_bar = "[" + "=" * int(20 * total_done / total_cells) + " " * (20 - int(20 * total_done / total_cells)) + "]"
    print(f"TOTAL PROGRESS     : {overall_bar} {total_done:3d}/{total_cells} ({total_pct:.1f}%)")

    # Cloud Quota usage
    print("\n" + "-" * 64)
    print("Cloud Quota Status (Today's Requests):")
    try:
        chain = {}
        for i, (mid, d, _) in enumerate(client.gemini_chain):
            used = client.ledger.today_count(mid)
            if i == 0:
                used += client.ledger.today_count("gemini")
            chain[mid] = (used, d)
        g_mid, g_lim, _ = client.gemma_chain
        chain[g_mid] = (client.ledger.today_count(g_mid), g_lim)

        for model_id, (used, limit) in chain.items():
            print(f"  * {model_id:<24}: {used:4d} / {limit:<5d}")
    except Exception as e:
        print(f"  Could not read ledger: {e}")

    print("=" * 64)
    print("Press Ctrl+C to exit monitor (runs independently in background).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60,
                        help="Refresh interval in seconds (default: 60)")
    args = parser.parse_args()

    client = make_client()
    try:
        while True:
            clear_screen()
            render_dashboard(client)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExiting monitor.")


if __name__ == "__main__":
    main()
