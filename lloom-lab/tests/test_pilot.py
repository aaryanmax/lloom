"""Pilot aggregation + verdict logic (pure functions, no LLM, no I/O)."""

import statistics

from lloom_lab.experiments import pilot


def _scores(spec_by_n, gen_by_n=None):
    """scores list with per-n values: {n: [v_trial0, v_trial1]}."""
    gen_by_n = gen_by_n or {}
    out = []
    for n, vals in spec_by_n.items():
        for trial, v in enumerate(vals):
            g = gen_by_n.get(n, [0.9] * len(vals))[trial]
            out.append({"n": n, "trial": trial, "generic": g, "specific": v,
                        "coverage": 0.5, "n_generated": 5, "errors": {}})
    return out


def test_summarize_stats():
    scores = _scores({10: [0.4, 0.6], 100: [0.5, 0.5]},
                     gen_by_n={10: [0.8, 1.0]})
    s = pilot.summarize(scores, [10, 100], 2)
    e10 = s["per_n"][0]
    assert e10["n"] == 10 and e10["n_completed"] == 2
    assert e10["specific"]["mean"] == 0.5
    assert e10["specific"]["std"] == statistics.stdev([0.4, 0.6])
    assert abs(e10["specific"]["cv"]
               - statistics.stdev([0.4, 0.6]) / 0.5) < 1e-9
    # single-value groups get std 0
    assert s["per_n"][1]["specific"]["std"] == 0.0


def test_summarize_handles_missing_n():
    s = pilot.summarize(_scores({10: [0.4]}), [10, 25, 100], 2)
    assert s["per_n"][1]["specific"]["mean"] is None
    assert s["per_n"][1]["n_completed"] == 0


def test_decide_go():
    scores = _scores({10: [0.2], 25: [0.3], 50: [0.4], 100: [0.5, 0.55]})
    s = pilot.summarize(scores, [10, 25, 50, 100], 2)
    d = pilot.decide(s)
    assert d["verdict"] == "GO"


def test_decide_noisy_is_nogo():
    # high variance at n=100: values 0.2 vs 0.8 -> CV 0.6 > 0.25
    scores = _scores({100: [0.2, 0.8]})
    s = pilot.summarize(scores, [100], 2)
    d = pilot.decide(s)
    assert d["verdict"] == "NO-GO"
    assert any("noisy" in r for r in d["reasons"])


def test_decide_floor_is_nogo():
    scores = _scores({100: [0.1, 0.12]})
    s = pilot.summarize(scores, [100], 2)
    d = pilot.decide(s)
    assert d["verdict"] == "NO-GO"
    assert any("floor" in r for r in d["reasons"])


def test_decide_incomplete_is_nogo():
    scores = _scores({100: [0.5]})  # 1 of 2 trials
    s = pilot.summarize(scores, [100], 2)
    d = pilot.decide(s)
    assert d["verdict"] == "NO-GO"
    assert any("incomplete" in r for r in d["reasons"])


def test_run_id_format():
    assert pilot.run_id(10, 0) == "n010_t0"
    assert pilot.run_id(100, 1) == "n100_t1"
