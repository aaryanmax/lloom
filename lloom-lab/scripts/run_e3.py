"""E3 — growth function -> effective VC dim (offline theoretical simulation)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from lloom_lab.config import RESULTS_DIR
from lloom_lab.theory import vc


def main():
    print("Running E3 VC Dimension & Growth Function simulation...")
    out = RESULTS_DIR / "e3"
    out.mkdir(exist_ok=True, parents=True)
    result = {}

    # (1) textbook anchors — must reproduce thresholds d=1, intervals d=2, rectangles d=4
    uniform01 = lambda m: np.random.default_rng(42).uniform(0, 1, size=(m, 1))
    result["threshold_vc"] = vc.effective_vc_dim(
        uniform01, vc.threshold_realizer_1d, d_max=3, trials=50
    )
    result["interval_vc"] = vc.effective_vc_dim(
        uniform01, vc.interval_realizer_1d, d_max=4, trials=50
    )
    result["rectangle_vc"] = vc.effective_vc_dim(
        lambda m: np.random.default_rng(42).uniform(0, 1, size=(m, 2)),
        vc.rectangle_realizer,
        d_max=5,
        trials=200,
    )

    # (2) growth function vs Sauer envelope for intervals (d=2)
    ms = list(range(1, 15))
    gf = vc.growth_function(
        [np.random.default_rng(m).uniform(0, 1, size=(m, 1)) for m in ms],
        vc.interval_realizer_1d,
    )
    result["growth"] = {
        "m": ms,
        "Pi_H": [int(x) for x in gf],
        "sauer": [int(vc.sauer_bound(m, 2)) for m in ms],
    }

    (out / "e3_results.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    
    print("\n[E3 COMPLETE] Effective VC Dimensions:")
    print(f"  * Thresholds 1D : VC = {result['threshold_vc']} (Expected: 1)")
    print(f"  * Intervals 1D  : VC = {result['interval_vc']} (Expected: 2)")
    print(f"  * Rectangles 2D : VC = {result['rectangle_vc']} (Expected: 4)")
    print(f"\nGrowth Function vs Sauer Bound (Intervals d=2):")
    for m, pi, sauer in zip(result["growth"]["m"][:7], result["growth"]["Pi_H"][:7], result["growth"]["sauer"][:7]):
        print(f"  m={m:2d} | Pi_H={pi:3d} | Sauer's Bound={sauer:3d} | Dichotomies Shattered: {pi == 2**m}")

    print(f"\nResults saved to: {out / 'e3_results.json'}")


if __name__ == "__main__":
    main()
