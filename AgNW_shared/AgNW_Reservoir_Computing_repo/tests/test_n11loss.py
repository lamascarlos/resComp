"""N11 losses and decision rules (agnw_rc/n11loss.py)."""
import numpy as np
import pytest

from agnw_rc import n11loss as L

DATA = {"onset_V": 2.5,
        "shape": [{"V": 2.8, "ratio": 0.2, "log_delta": 0.01}, {"V": 4.0, "ratio": 0.6, "log_delta": 0.01},
                  {"V": 6.0, "ratio": 1.0, "log_delta": 0.01}],
        "theta": [{"V": 3.0, "theta_per_s": 0.1, "log_sd": 0.1, "unresolved": False},
                  {"V": 3.1, "theta_per_s": 0.4, "log_sd": 0.1, "unresolved": False},
                  {"V": 5.0, "theta_per_s": 0.2, "log_sd": 0.02, "unresolved": False},
                  {"V": 6.0, "theta_per_s": 9.9, "log_sd": 0.1, "unresolved": True}],
        "relaxation": {"one": {"tau_s": 56.0}, "two": {"tau1_s": 27.0, "tau2_s": 7e4}, "best": "two"}}


def _sim(scale=1.0):
    return {"onset_V": 2.5 * scale, "relax_tau_s": 56.0 * scale,
            "shape": [{"V": 2.8, "ratio": 0.2 * scale}, {"V": 4.0, "ratio": 0.6 * scale}, {"V": 6.0, "ratio": 1.0}],
            "theta": [{"V": 3.0, "theta_per_s": 0.25 * scale, "unresolved": False},
                      {"V": 5.0, "theta_per_s": 0.2 * scale, "unresolved": False}]}


def test_theta_targets_smooth_neighbours_and_widen_delta():
    t = {r["V"]: r for r in L.theta_targets(DATA["theta"])}
    assert set(t) == {3.0, 3.1, 5.0}                                   # unresolved dropped
    assert t[3.0]["theta"] == pytest.approx(0.25) and t[3.0]["delta"] == pytest.approx(np.std(np.log([0.1, 0.4])))
    assert t[5.0]["theta"] == 0.2 and t[5.0]["delta"] == L.DELTA_FLOOR


def test_relax_target_delta_covers_one_vs_two_exponentials():
    tau, d = L.relax_target(DATA["relaxation"])
    assert tau == 56.0 and d == pytest.approx(np.log(56 / 27) / 2)


def test_stage1_loss_is_zero_at_the_data_and_grows_away():
    assert L.stage1_terms(_sim(), DATA)["loss"] == pytest.approx(0.0, abs=1e-12)
    assert L.stage1_terms(_sim(1.3), DATA)["loss"] > 1.0
    assert L.stage1_terms({**_sim(), "onset_V": None}, DATA)["loss"] == np.inf


def test_calibration_verdict_rules():
    v = L.calibration_verdict(_sim(), DATA)
    assert v["pass"] and v["onset_ok"] and v["shape_rms"] == pytest.approx(0) and v["theta_share"] == 1.0
    bad = L.calibration_verdict(_sim(1.15), DATA)                      # onset 15 % off
    assert not bad["onset_ok"] and not bad["pass"]


def test_poisson_deviance_and_aic():
    assert L.poisson_deviance(10, 10.0) == pytest.approx(0.0)
    assert L.poisson_deviance(0, 2.0) == pytest.approx(4.0)
    assert L.poisson_deviance(5, 0.0) == L.poisson_deviance(5, 0.5)   # mean floored at 0.5
    assert L.aic(10.0, "constant") == 14.0 and L.aic(10.0, "power") == 16.0


def _rows():
    return {V: {"jumps": {"events": 40, "duration_s": 500.0, "rate_ci_per_s": [0.05, 0.11], "sd_rel": 3e-4},
                "sd_log_delta": 0.05, "sizes": list(np.linspace(-0.01, 0.01, 40))} for V in (3.0, 4.0)}


def test_stage2_terms_zero_when_simulation_matches():
    rows = _rows()
    sim = {V: {"events": 40.0, "duration_s": 500.0, "sd_rel": 3e-4, "rate_per_s": 0.08,
               "sizes": np.linspace(-0.01, 0.01, 40)} for V in rows}
    data_sizes = np.concatenate([r["sizes"] for r in rows.values()])
    t = L.stage2_terms(sim, rows, data_sizes)
    assert t["loss"] == pytest.approx(0.0, abs=1e-9)
    sim[3.0]["events"] = 80.0
    assert L.stage2_terms(sim, rows, data_sizes)["loss"] > 5


def test_choose_law_needs_both_aic_and_the_interval_condition():
    rows = _rows()
    inside = {3.0: 0.08, 4.0: 0.08}
    outside = {3.0: 0.08, 4.0: 0.30}
    # constant misses 4 V, power covers it and is much better: power chosen
    law, _ = L.choose_law({"constant": {"aic": 50, "rate_by_V": outside}, "power": {"aic": 30, "rate_by_V": inside}}, rows)
    assert law == "power"
    # constant misses nothing: constant stays even with a worse AIC (Review Focus 5)
    law, _ = L.choose_law({"constant": {"aic": 50, "rate_by_V": inside}, "power": {"aic": 30, "rate_by_V": inside}}, rows)
    assert law == "constant"
    # AIC gain not above 4: constant stays
    law, _ = L.choose_law({"constant": {"aic": 33, "rate_by_V": outside}, "thermal": {"aic": 30, "rate_by_V": inside}}, rows)
    assert law == "constant"


def test_r0_interval_is_the_delta_aic_4_profile():
    rows = [{"r0": 1e-3, "aic": 20.0}, {"r0": 3e-3, "aic": 12.0}, {"r0": 1e-2, "aic": 15.0}, {"r0": 3e-2, "aic": 17.0}]
    assert L.r0_interval(rows) == (3e-3, 1e-2)
