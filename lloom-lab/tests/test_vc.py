"""E3 theory: growth function, shattering, Sauer's lemma (offline)."""

import numpy as np

from lloom_lab.theory.vc import (
    effective_vc_dim,
    growth_function,
    interval_realizer_1d,
    rectangle_realizer,
    sauer_bound,
    shatter_rate,
    threshold_realizer_1d,
)


def _grid(*coords):
    return np.array(coords, dtype=float)


def test_threshold_vc_dim_is_1():
    def pts(m):
        return np.linspace(0.1, 0.9, m).reshape(-1, 1)
    # 1 point: both labelings; 2 points: (0,1) unreachable -> not shattered
    assert shatter_rate(pts, threshold_realizer_1d, 1) == 1.0
    assert shatter_rate(pts, threshold_realizer_1d, 2) == 0.0
    assert effective_vc_dim(pts, threshold_realizer_1d, d_max=3) == 1


def test_interval_vc_dim_is_2():
    def pts(m):
        return np.linspace(0.1, 0.9, m).reshape(-1, 1)
    assert shatter_rate(pts, interval_realizer_1d, 2) == 1.0
    # 3 collinear points: labeling 1-0-1 (outer positives) is unrealizable
    X = _grid([0.2], [0.5], [0.8])
    assert not interval_realizer_1d(X, np.array([1, 0, 1]))
    assert effective_vc_dim(pts, interval_realizer_1d, d_max=3) == 2


def test_rectangle_vc_dim_is_4():
    # The classic shatterable configuration is the *diamond* (one point
    # extreme in each of the four directions). Random 4-point sets are NOT
    # always shattered — e.g. the subset {leftmost-x, rightmost-x} is only
    # realizable if the other points' y-coords fall outside the pair's range.
    diamond = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
                        [5.0, 5.0]])  # 5th point breaks shattering (VC dim 4)
    def pts(m):
        return diamond[:m]
    assert effective_vc_dim(pts, rectangle_realizer, d_max=5, trials=1) == 4
    # and no 5-point set can be shattered
    assert shatter_rate(pts, rectangle_realizer, 5, trials=1) == 0.0


def test_growth_function_counts():
    X = _grid([0.2], [0.5], [0.8])
    g = growth_function([X], threshold_realizer_1d)
    assert g[0] == 4  # 000, 001, 011, 111
    g2 = growth_function([X], interval_realizer_1d)
    # every labeling except 1-0-1 (outer positives, middle negative)
    assert g2[0] == 7
    assert not interval_realizer_1d(X, np.array([1, 0, 1]))


def test_sauer_envelope():
    assert sauer_bound(2, 5) == 4          # m <= d: trivial 2^m
    assert sauer_bound(4, 4) == 16
    # m=5, d=4: 1 + 5 + 10 + 10 + 5 = 31 < 32
    assert sauer_bound(5, 4) == 31
    # growth of rectangles can never exceed the Sauer envelope
    rng = np.random.default_rng(3)
    for m in range(1, 6):
        X = rng.uniform(0.2, 0.8, size=(m, 2))
        n_dichotomies = growth_function([X], rectangle_realizer)[0]
        assert n_dichotomies <= sauer_bound(m, 4)


def test_rectangle_realizer_exact():
    X = _grid([0.3, 0.3], [0.7, 0.3], [0.3, 0.7], [0.7, 0.7])
    # S = two left points: tightest box excludes the right pair -> realizable
    assert rectangle_realizer(X, np.array([1, 1, 0, 0]))
    # S = the two left points plus top-right: box [0.3-0.7]x[0.3-0.7]
    # contains bottom-right (0.7, 0.3) which is negative -> not realizable
    assert not rectangle_realizer(X, np.array([1, 0, 1, 1]))
    assert rectangle_realizer(X, np.array([0, 0, 0, 0]))
