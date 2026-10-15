"""Calibration study of the FDTest decision rule (scripts/fdtest_interval_calibration.py): aggregation by hand, seeding,
and a smoke run with tiny sizes. The study itself (many replicas) is run by hand, not here."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import fdtest_interval_calibration as cal

ROOT = Path(__file__).resolve().parents[1]
TINY = ["--reps", "2", "--Ms", "60", "--K", "10", "--t-relax", "30", "--t2", "60", "--t4", "60", "--n-lags", "10",
        "--n-boot", "12", "--n-inner", "4", "--n-splits", "3"]


def _rec(rep, d2, floor, methods, scenario="S2", M=60, true=0.3):
    return {"scenario": scenario, "M": M, "rep": rep, "d2": d2, "floor": floor, "true": true, "methods": methods}


def test_aggregate_hand_values_rate_coverage_width_ratio_and_nan_intervals():
    # Break caught: coverage counted over NaN intervals, a wrong width/median, or a rate over the wrong denominator.
    nan = float("nan")
    recs = [_rec(0, 0.4, 0.1, {"reverse": {"excl": True, "lo": 0.2, "hi": 0.7}, "plain3x": {"excl": True}}),
            _rec(1, 0.2, 0.1, {"reverse": {"excl": True, "lo": 0.35, "hi": 0.9}, "plain3x": {"excl": False}}),
            _rec(2, 0.3, 0.1, {"reverse": {"excl": False, "lo": nan, "hi": nan}, "plain3x": {"excl": False}}),
            _rec(3, 0.6, 0.2, {"reverse": {"excl": False, "lo": 0.0, "hi": 0.5}, "plain3x": {"excl": False}})]
    cells = {(c["scenario"], c["M"], c["method"]): c for c in cal.aggregate(recs)}
    c = cells[("S2", 60, "reverse")]
    assert c["n"] == 4 and c["rate"] == 0.5 and c["rate_se"] == pytest.approx(0.25)        # sqrt(.5 * .5 / 4)
    assert c["n_ci"] == 3 and c["coverage"] == pytest.approx(2 / 3)                        # [.2, .7] and [0, .5] contain .3
    assert c["median_width"] == pytest.approx(0.5)                                         # widths .5, .55, .5
    assert c["median_ratio"] == pytest.approx(3.0)                                         # d2/floor = 4, 2, 3, 3
    p = cells[("S2", 60, "plain3x")]
    assert p["rate"] == 0.25 and np.isnan(p["coverage"]) and np.isnan(p["median_width"]) and p["median_ratio"] == pytest.approx(3.0)
    # S0: the "true value" is 0 by convention (fraction of intervals containing 0)
    s0 = [_rec(0, 0.4, 0.1, {"reverse": {"excl": False, "lo": -0.1, "hi": 0.5}}, scenario="S0", true=None),
          _rec(1, 0.4, 0.1, {"reverse": {"excl": True, "lo": 0.05, "hi": 0.5}}, scenario="S0", true=None)]
    assert cal.aggregate(s0)[0]["coverage"] == pytest.approx(0.5)


def test_seeds_are_deterministic_and_distinct_per_scenario_M_and_replica():
    # Break caught: seeds shared between cells (correlated replicas) or dependent on the process that runs them.
    seeds = {(s, M, r): cal.replica_seeds(s, M, r) for s in range(3) for M in (60, 120) for r in range(4)}
    assert cal.replica_seeds(1, 60, 2) == seeds[(1, 60, 2)]
    flat = [x for v in seeds.values() for x in v]
    assert len(set(flat)) == len(flat)


def test_nested_bootstrap_se_is_reproducible_positive_and_uses_the_resample_weights():
    # Break caught: an inner bootstrap that ignores the outer resample (se identical for all outer draws) or is not seeded.
    cfg = dict(cal.DEFAULTS, K=10, t_relax=30, t2=60, t4=60, n_lags=10, n_boot=6, n_inner=5)
    stats, masks = cal.simulate_stats(cal.SCENARIOS["S1"], 40, cfg, 5)
    w = cal.ft.bootstrap_weights(40, 6, np.random.default_rng(0))
    a = cal.nested_bootstrap_se(stats, masks, w, 5, 10, np.random.default_rng(1))
    b = cal.nested_bootstrap_se(stats, masks, w, 5, 10, np.random.default_rng(1))
    assert a.shape == (6,) and np.array_equal(a, b, equal_nan=True) and (a[np.isfinite(a)] > 0).all()
    assert np.isfinite(a).all() and np.ptp(a) > 0
    one_hot = np.zeros((1, 40)); one_hot[0, 0] = 1.0                       # every inner resample is the same single cycle
    assert not cal.nested_bootstrap_se(stats, masks, one_hot, 5, 10, np.random.default_rng(1))[0] > 0     # se 0 (or undefined)


def test_replica_uses_the_fixed_window_over_all_lags_not_the_data_dependent_cutoff():
    # Break caught: a replica that lets select_cutoff choose the window (the true delta2 of S2 refers to lags 0..n_lags).
    cfg = dict(cal.DEFAULTS, K=20, t_relax=30, t2=100, t4=100, n_lags=30, n_boot=5, n_inner=3, n_splits=2)
    rec = cal.run_replica(("S1", 60, 0, cfg))
    stats, masks = cal.simulate_stats(cal.SCENARIOS["S1"], 60, cfg, cal.replica_seeds(1, 60, 0)[0])
    cur = cal.ft.binary_curves(stats)
    fixed = cal.ft.class_delta2(cur["z_mean"], cur["c_zz"], masks, cutoff=30)[0][0]
    auto, tau = cal.ft.delta2_from_sums(*cal.ft.class_sums(cur["z_mean"], cur["c_zz"], masks), return_tau=True)
    assert tau[0] < 30 and auto[0] != fixed                                   # this seed: the automatic cutoff would differ
    assert rec["d2"] == fixed and set(rec["methods"]) == set(cal.METHODS) and rec["true"] == 0.0


def _run(out, *extra):
    return subprocess.run([sys.executable, "-m", "scripts.fdtest_interval_calibration", *TINY, "--out", str(out), *extra],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_smoke_run_writes_table_and_json_and_is_independent_of_the_number_of_processes(tmp_path):
    a, b = tmp_path / "a" / "cal.md", tmp_path / "b" / "cal.md"
    r = _run(a, "--procs", "1")
    assert r.returncode == 0, r.stderr
    r2 = _run(b, "--procs", "2")
    assert r2.returncode == 0, r2.stderr
    text = a.read_text()
    assert "reps = 2" in text and "runtime" in text and "| S0 |" in text and "| S1 |" in text and "| S2 |" in text
    for m in ("percentile", "reverse", "sqrt_reverse", "studentized", "plain3x"):
        assert f"| {m} |" in text
    raw = a.with_suffix(".json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    res = json.loads(raw)
    assert len(res["cells"]) == 3 * 1 * 5 and len(res["replicas"]) == 3 * 1 * 2
    assert res["true_delta2"]["S1"] == 0.0 and res["true_delta2"]["S2"] > 0.05 and res["true_delta2"]["S0"] is None
    for c in res["cells"]:
        assert 0.0 <= c["rate"] <= 1.0 and c["n"] == 2
    res2 = json.loads(b.with_suffix(".json").read_text())
    assert res2["cells"] == res["cells"] and res2["replicas"] == res["replicas"]                 # same seeds, same numbers
