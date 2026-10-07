"""E2 theory: rectangle learner + PAC bound (deterministic, offline)."""

import numpy as np
import pytest

from lloom_lab.theory.rectangles import (
    empirical_sample_complexity,
    fit_rectangle,
    pac_bound_rectangles,
    predict_rectangle,
    rectangle_error,
)


def test_pac_bound_shape_and_monotonicity():
    assert pac_bound_rectangles(0.1, 0.1) == int(np.ceil(40 * np.log(130)))
    assert pac_bound_rectangles(0.05, 0.1) > pac_bound_rectangles(0.1, 0.1)
    assert pac_bound_rectangles(0.1, 0.01) > pac_bound_rectangles(0.1, 0.1)


def test_pac_bound_validates_inputs():
    with pytest.raises(ValueError):
        pac_bound_rectangles(0.0, 0.1)
    with pytest.raises(ValueError):
        pac_bound_rectangles(0.1, 1.5)


def test_consistent_learner_recovers_target_on_dense_sample():
    rng = np.random.default_rng(0)
    target = np.array([0.2, 0.6, 0.3, 0.8])
    pts = rng.uniform(0, 1, size=(4000, 2))
    labels = predict_rectangle(target, pts)
    box = fit_rectangle(pts, labels)
    assert rectangle_error(box, target) < 0.05


def test_error_is_symmetric_difference_area():
    # identical boxes -> 0 error; disjoint boxes -> sum of areas
    a = np.array([0.0, 0.5, 0.0, 0.5])
    assert rectangle_error(a, a.copy()) == 0.0
    b = np.array([0.5, 1.0, 0.0, 0.5])
    assert abs(rectangle_error(a, b) - 0.5) < 1e-12


def test_no_positives_gives_empty_hypothesis():
    pts = np.array([[0.1, 0.1], [0.9, 0.9]])
    labels = np.array([0, 0])
    box = fit_rectangle(pts, labels)
    assert not predict_rectangle(box, pts).any()


def test_empirical_failure_respects_the_bound():
    # at the bound's m the failure rate must be <= delta (statistical check,
    # generous tolerance for finite trials, fixed seed for determinism)
    rows = empirical_sample_complexity(
        eps_values=[0.2], delta_values=[0.3], m_grid=[20, 60, 200],
        trials=300, seed=7)
    by_m = {r["m"]: r for r in rows}
    m_bound = pac_bound_rectangles(0.2, 0.3)
    assert by_m[min(m for m in [20, 60, 200] if m >= m_bound)]["failure_rate"] <= 0.3
    # failure rate should shrink as m grows well past the bound
    assert by_m[200]["failure_rate"] <= by_m[20]["failure_rate"] + 0.05
