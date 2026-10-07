"""E2: axis-parallel rectangles — the classic PAC control experiment (Mitchell,
ch. 1/7). A *known* hypothesis class with a *known* sample-complexity bound,
so we can validate empirical curves against theory before trusting the LLM
curves from E1.

Concept class: rectangles R = [a,b] x [c,d] in the unit square, target
function = membership. Under the uniform distribution, true error = area of
the symmetric difference between target and hypothesis.

Sample-complexity bound (Mitchell 1997, eq. 2.9-derived for rectangles with
13 degenerate cases): a consistent learner PAC-learns rectangles with
    m >= (4/eps) * ln(13/delta)   training examples.
Empirical check: draw m labeled samples from a random target rectangle, fit
the consistent learner (tightest box around positives), and measure how often
true error exceeds eps; that failure rate must be <= delta once m satisfies
the bound.
"""

from __future__ import annotations

import math

import numpy as np


def pac_bound_rectangles(eps: float, delta: float) -> int:
    """Mitchell's sufficient sample size for PAC-learning axis-parallel
    rectangles: m >= (4/eps) * ln(13/delta)."""
    if not (0 < eps <= 1) or not (0 < delta <= 1):
        raise ValueError("eps and delta must lie in (0, 1]")
    return math.ceil((4.0 / eps) * math.log(13.0 / delta))


def fit_rectangle(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Consistent learner: the tightest axis-parallel box around the positives.

    Returns [x1_lo, x1_hi, x2_lo, x2_hi]. If there are no positives the
    hypothesis is the empty rectangle (everything negative).
    """
    pos = points[labels.astype(bool)]
    if len(pos) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0])  # degenerate: contains nothing
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    return np.array([lo[0], hi[0], lo[1], hi[1]])


def predict_rectangle(box: np.ndarray, points: np.ndarray) -> np.ndarray:
    inside_x = (points[:, 0] >= box[0]) & (points[:, 0] <= box[1])
    inside_y = (points[:, 1] >= box[2]) & (points[:, 1] <= box[3])
    return (inside_x & inside_y).astype(int)


def rectangle_error(box: np.ndarray, target: np.ndarray) -> float:
    """True error under the uniform distribution = area(target XOR hypothesis).

    `target` is [tx1, tx2, ty1, ty2]; boxes are [x1, x2, y1, y2]. Area of the
    symmetric difference = area(A) + area(B) - 2 * area(A and B).
    """

    def area(r):
        return max(0.0, r[1] - r[0]) * max(0.0, r[3] - r[2])

    def overlap(r, s):
        w = min(r[1], s[1]) - max(r[0], s[0])
        h = min(r[3], s[3]) - max(r[2], s[2])
        return max(0.0, w) * max(0.0, h)

    return area(box) + area(target) - 2.0 * overlap(box, target)


def empirical_sample_complexity(
    eps_values: list[float],
    delta_values: list[float],
    m_grid: list[int],
    trials: int = 400,
    seed: int = 42,
) -> list[dict]:
    """For each (eps, delta, m): estimate P(true error > eps) over `trials`
    random targets + samples. Returns rows ready for charting / comparison
    with pac_bound_rectangles()."""
    rng = np.random.default_rng(seed)
    rows = []
    for eps in eps_values:
        for delta in delta_values:
            for m in m_grid:
                failures = 0
                for _ in range(trials):
                    # random target rectangle (lo < hi in both axes)
                    t = np.sort(rng.uniform(0, 1, size=(2, 2)), axis=0)
                    target = np.array([t[0, 0], t[0, 1], t[1, 0], t[1, 1]])
                    pts = rng.uniform(0, 1, size=(m, 2))
                    labels = predict_rectangle(target, pts)
                    box = fit_rectangle(pts, labels)
                    if rectangle_error(box, target) > eps:
                        failures += 1
                rows.append({
                    "eps": eps, "delta": delta, "m": m,
                    "failure_rate": failures / trials,
                    "bound_m": pac_bound_rectangles(eps, delta),
                    "theory_ok": m >= pac_bound_rectangles(eps, delta),
                })
    return rows
