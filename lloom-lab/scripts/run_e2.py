"""E2 — rectangles vs the PAC bound (offline simulation)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lloom_lab.config import RESULTS_DIR
from lloom_lab.theory.rectangles import empirical_sample_complexity


def main():
    print("Running E2 PAC Bound simulation...")
    rows = empirical_sample_complexity(
        eps_values=[0.05, 0.10, 0.20],
        delta_values=[0.05, 0.10],
        m_grid=list(range(10, 620, 20)),
        trials=400,
    )
    out = RESULTS_DIR / "e2"
    out.mkdir(exist_ok=True, parents=True)
    (out / "e2_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    violations = 0
    pairs = sorted(list({(r["eps"], r["delta"]) for r in rows}))
    for eps, delta in pairs:
        sel = [r for r in rows if r["eps"] == eps and r["delta"] == delta]
        bound_m = sel[0]["bound_m"]
        below = [r["m"] for r in sel if r["failure_rate"] > delta and r["theory_ok"]]
        violations += len(below)
        print(f"eps={eps:<4} delta={delta:<4}: PAC bound m>={bound_m:<4} | violations above bound: {len(below)} (expected: 0)")

    print(f"\n[E2 COMPLETE] Total PAC bound violations: {violations} (PAC Guarantee Verified: {violations == 0})")
    print(f"Results saved to: {out / 'e2_rows.json'}")


if __name__ == "__main__":
    main()
