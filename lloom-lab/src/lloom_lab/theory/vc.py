"""E3: growth function, shattering, and effective VC dimension.

The growth function Π_H(m) counts distinct dichotomies (labelings) a
hypothesis class realizes on m points. Sauer's lemma bounds it:
    Π_H(m) <= sum_{i=0}^{d} C(m, i)   for m points, VC dimension d
(with the trivial 2^m while m <= d). We estimate Π_H empirically by sampling
point sets and checking realizability of every dichotomy — for the rectangle
class realization has an exact test (a subset S is realizable iff the tightest
box around S contains no point outside S), because any rectangle containing S
must contain S's minimal bounding box.

"Effective VC dimension" = the largest m whose shatter rate stays >= 0.5
over sampled configurations — the empirical analogue of the paper's
effective-capacity analysis of promptable concept classes.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np

# A realizer decides, for a point set X (n x dims) and a desired labeling
# y (+1/-1 or 0/1), whether SOME hypothesis in the class produces exactly y.
Realizer = Callable[[np.ndarray, np.ndarray], bool]


def sauer_bound(m: int, d: int) -> int:
    """Sauer's lemma envelope: sum_{i=0}^{d} C(m, i) (capped at 2^m)."""
    if m <= d:
        return 2 ** m
    return sum(math.comb(m, i) for i in range(min(d, m) + 1))


def rectangle_realizer(X: np.ndarray, y: np.ndarray) -> bool:
    """Exact realizability test for axis-parallel rectangles in [0,1]^k.

    S is realizable iff the tightest box around its positives excludes every
    negative (the minimal box is contained in every rectangle covering S).
    """
    X = np.asarray(X, dtype=float)
    pos = X[y.astype(bool)]
    if len(pos) == 0:
        return not y.any()  # empty rectangle labels everything negative
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    inside = ((X >= lo) & (X <= hi)).all(axis=1)
    return bool((inside == y.astype(bool)).all())


def threshold_realizer_1d(X: np.ndarray, y: np.ndarray) -> bool:
    """Realizability by a threshold on one coordinate (class with VC dim 1).

    {x >= t} classifiers produce exactly the sorted step labelings 0…01…1
    (including all-0 and all-1), so `y` is realizable iff the labels sorted
    by coordinate are non-decreasing.
    """
    x = np.asarray(X, dtype=float).ravel()
    order = np.argsort(x)
    ys = y[order]
    return bool(np.all(np.diff(ys.astype(int)) >= 0)) if len(ys) else True


def interval_realizer_1d(X: np.ndarray, y: np.ndarray) -> bool:
    """Realizability by an interval [a,b] on one coordinate (VC dim 2)."""
    x = np.asarray(X, dtype=float).ravel()
    pos = x[y.astype(bool)]
    if len(pos) == 0:
        return not y.any()
    inside = (x >= pos.min()) & (x <= pos.max())
    return bool((inside == y.astype(bool)).all())


def growth_function(X_sets: list[np.ndarray], realizer: Realizer) -> list[int]:
    """Π_H(m) for each provided point set: count realizable dichotomies."""
    out = []
    for X in X_sets:
        n = len(X)
        count = 0
        for bits in itertools.product((0, 1), repeat=n):
            if realizer(X, np.array(bits)):
                count += 1
        out.append(count)
    return out


def shatter_rate(
    sample_points: Callable[[int], np.ndarray],
    realizer: Realizer,
    m: int,
    trials: int = 100,
) -> float:
    """Fraction of m-point sets that are fully shattered (all 2^m dichotomies)."""
    shattered = 0
    for _ in range(trials):
        X = sample_points(m)
        realizable = all(
            realizer(X, np.array(bits))
            for bits in itertools.product((0, 1), repeat=m)
        )
        shattered += realizable
    return shattered / trials


def effective_vc_dim(
    sample_points: Callable[[int], np.ndarray],
    realizer: Realizer,
    d_max: int = 6,
    trials: int = 100,
    rate_threshold: float = 0.5,
) -> int:
    """Largest m with shatter rate >= threshold (0 if none at m=1)."""
    dim = 0
    for m in range(1, d_max + 1):
        if shatter_rate(sample_points, realizer, m, trials) < rate_threshold:
            break
        dim = m
    return dim
