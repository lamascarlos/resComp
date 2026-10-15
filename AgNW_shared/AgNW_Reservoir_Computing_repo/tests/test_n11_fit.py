"""N11 stages 1-2 driver (scripts/n11_fit.py) on the tiny MINI strip."""
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("n11_fit", ROOT / "scripts" / "n11_fit.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
DATA = json.loads((ROOT / "results" / "n11_data.json").read_text())


@pytest.fixture
def mini(tmp_path):
    return dataclasses.replace(F.MINI, raw=tmp_path / "raw", out=tmp_path, fig=tmp_path)


def test_names_are_unique_and_carry_every_simulation_field(mini):
    p = F.point(1, 40.0, 0.3, 1.0, 100.0, mini)
    a = p | {"kind": "step", "V": 2.5}
    variants = [a, a | {"V": 2.6}, a | {"dt": 1e-3}, a | {"seed": 2}, a | {"vs": 41.0},
                a | {"kind": "noise", "law": "power", "r0": 1e-3, "alpha": 1.0, "P0": 1e-6, "frac": 1.0},
                a | {"kind": "noise", "law": "power", "r0": 1e-3, "alpha": 1.0, "P0": 2e-6, "frac": 1.0}]
    names = [F.name(r) for r in variants]
    assert len(set(names)) == len(names)


def test_vscale_rescaling_keeps_the_junction_threshold():
    assert F.scale_vs(40.0, 0.3) == 40.0
    k = F.VolatileParams().k_p0
    v = F.scale_vs(40.0, 30.0)
    assert v / 40.0 == pytest.approx(np.log(1 / (k * 30)) / np.log(1 / (k * 0.3)), rel=1e-3)


def test_vscale_step_moves_the_right_way():
    assert F.next_vscale(40.0, None, 2.5) == 60.0               # never reached: more voltage sensitivity
    assert F.next_vscale(40.0, F.V_ONSET[0], 2.5) == 28.0       # already on at the lowest voltage
    assert F.next_vscale(40.0, 3.0, 2.5) == pytest.approx(48.0)  # onset ~ 1 / v_scale


def test_stage1_end_to_end_on_the_mini_strip(mini):
    grid = {"tau": (0.3,), "sp": (1.0,), "km": (100.0,)}
    volts = (2.5, 5.5, 6.6)
    v = F.stage1_vscale(1, mini, DATA, vs0=40.0, max_iter=1, volts=volts)
    assert v["history"] and "v_scale_ref" in v
    F.stage1_grid(1, mini, DATA, grid=grid, volts=volts)
    rep = F.stage1_report(mini, DATA, grid=grid, volts=volts)
    assert len(rep["points"]) == 1 and "loss" in rep["points"][0] and rep["best"] is not None
    assert (mini.out / "n11_fit_stage1.json").exists()


def test_report_skips_incomplete_points(mini):
    grid = {"tau": (0.3, 3.0), "sp": (1.0,), "km": (100.0,)}
    (mini.out / "n11_stage1_vscale.json").write_text(json.dumps({"v_scale_ref": 40.0, "history": []}))
    rep = F.stage1_report(mini, DATA, grid=grid, volts=(2.5,))
    assert rep["points"] == [] and rep["incomplete"] == 2


def _s1(mini):
    return {"best": {"params": {"vs": 40.0, "tau": 0.3, "sp": 1.0, "km": 100.0},
                     "P_formed_median_at_1p5_onset": 1e-7}}


def test_coarse_configs_cover_the_three_laws():
    cfg = F.coarse_configs()
    by = {law: [c for c in cfg if c["law"] == law] for law in ("constant", "power", "thermal")}
    assert len(by["constant"]) == 4 * 2 and len(by["power"]) == 4 * 3 * 2 and len(by["thermal"]) == 4 * 3 * 2


def test_stage2_plan_picks_six_spread_voltages(mini):
    plan = F.stage2_plan(mini, DATA, _s1(mini))
    v = plan["voltages"]
    assert len(v) == 6 and v == sorted(v) and set(v) <= set(DATA["stage2_voltages"])
    assert v[0] == DATA["stage2_voltages"][0] and v[-1] == DATA["stage2_voltages"][-1] and plan["P0_W"] == 1e-7


def test_stage2_end_to_end_on_the_mini_strip(mini):
    plan = F.stage2_plan(mini, DATA, _s1(mini))
    plan["voltages"] = plan["voltages"][:2]
    plan["coarse"] = [{"law": "constant", "r0": 1e-2, "frac": 1.0}, {"law": "power", "r0": 1e-2, "alpha": 1.0, "frac": 1.0}]
    (mini.out / "n11_stage2_plan.json").write_text(json.dumps(plan))
    base, noise = F.stage2_runs(plan, mini)
    F.run_many(base, 1, mini)
    F.run_many(noise, 1, mini)
    rep = F.stage2_report(mini, DATA)
    assert rep["chosen_law"] in ("constant", "power", "thermal") and len(rep["configs"]) == 2
    assert all(np.isfinite(c["aic"]) or c["status"] == "disconnected" for c in rep["configs"])
    assert (mini.out / "n11_fit_stage2.json").exists()


def test_disconnected_run_gets_infinite_loss(mini, monkeypatch):
    plan = F.stage2_plan(mini, DATA, _s1(mini))
    plan["voltages"] = plan["voltages"][:1]
    cfg = {"law": "constant", "r0": 1e-2, "frac": 1.0}
    p = F.point(1, 40.0, 0.3, 1.0, 100.0, mini) | {"kind": "noise", "V": plan["voltages"][0], "P0": 1e-7} | cfg
    real = F.load
    monkeypatch.setattr(F, "load", lambda r, s: {"t": np.arange(10) * 0.6, "G": np.r_[np.ones(5), np.zeros(5)],
                                                 "mean_rupture_rate": 0.01} if r.get("kind") == "noise" else real(r, s))
    st = F.config_stats(plan, cfg, mini)
    assert st["status"] == "disconnected"


def test_checks_use_the_current_best_point(mini, monkeypatch):
    """Smoke run 2026-10-13: checks run after `refine` must use the best point including the refined ones, so the
    report is refreshed first (a stale best left every check None)."""
    best = {"vs": 6.7, "tau": 0.3, "sp": 0.0, "km": 3.162}
    monkeypatch.setattr(F, "stage1_report", lambda setup, data, **kw: {"best": {"params": best}})
    seen = []
    monkeypatch.setattr(F, "run_many", lambda rs, procs, setup: seen.extend(rs) or 0)
    F.stage1_checks(1, mini, DATA)
    assert seen and all(r["km"] == 3.162 and r["vs"] == 6.7 for r in seen)


def test_report_keeps_every_refined_point(mini):
    """Smoke run 2026-10-13: the report listed only the refinement around the current best, so points of an earlier
    refinement vanished when the best moved. Every refine grid that was run is recorded and reported."""
    (mini.out / "n11_stage1_vscale.json").write_text(json.dumps({"v_scale_ref": 40.0, "history": []}))
    (mini.out / "n11_stage1_refines.json").write_text(json.dumps([{"tau": [3.0], "sp": [1.0], "km": [100.0]}]))
    rep = F.stage1_report(mini, DATA, grid={"tau": (0.3,), "sp": (1.0,), "km": (100.0,)}, volts=(2.5,))
    assert rep["incomplete"] == 2


# ---- final-review fixes (2026-10-13): multi-node safety and robustness of the reports
def test_noise_run_without_its_base_is_skipped_not_crashed(mini):
    r = F.point(1, 40.0, 0.3, 1.0, 100.0, mini) | {"kind": "noise", "V": 2.5, "P0": 1e-7, "law": "constant",
                                                    "r0": 1e-2, "frac": 1.0}
    msg = F.run_one(r, mini)
    assert "base" in msg and not F.raw_file(r, mini).exists()


def test_marker_claim_is_exclusive_and_stale_markers_are_taken_over(mini):
    mini.raw.mkdir(parents=True)
    m = mini.raw / "x.running"
    assert F.claim(m) and not F.claim(m)
    m.write_text("otherhost 1 0")                                      # written at epoch 0: stale
    assert F.claim(m)


def test_refine_reuses_the_recorded_grid(mini, monkeypatch):
    rec = {"tau": [0.1, 0.3, 0.9], "sp": [0.5, 1.0, 1.5], "km": [30.0, 100.0, 300.0]}
    (mini.out / "n11_stage1_refines.json").write_text(json.dumps([rec]))
    monkeypatch.setattr(F, "stage1_report", lambda setup, data, **kw: {"best": {"params": {"tau": 9.0, "sp": 2.0,
                                                                                            "km": 1000.0}}})
    seen = {}
    monkeypatch.setattr(F, "stage1_grid", lambda procs, setup, data, grid, volts: seen.update(grid=grid))
    F.stage1_refine(1, mini, DATA)
    assert {k: list(v) for k, v in seen["grid"].items()} == rec


def test_point_stats_survive_a_failing_relaxation_fit(mini, monkeypatch):
    t = np.arange(0, 3.0, 0.6)
    monkeypatch.setattr(F, "load", lambda r, s: {"t": t, "G": 1e-6 * (1 + t), "P_formed_median": 1e-7})
    monkeypatch.setattr(F.S, "fit_relaxation", lambda *a: (_ for _ in ()).throw(RuntimeError("no fit")))
    st = F.point_stats(F.point(1, 40.0, 0.3, 1.0, 100.0, mini), (2.5, 5.5), mini, DATA)
    assert st is not None and st["relax_tau_s"] is None


def test_nonfinite_conductance_marks_the_configuration_failed(mini, monkeypatch):
    plan = F.stage2_plan(mini, DATA, _s1(mini))
    plan["voltages"] = plan["voltages"][:1]
    monkeypatch.setattr(F, "load", lambda r, s: {"t": np.arange(10) * 0.6, "G": np.r_[np.ones(5), np.full(5, np.nan)],
                                                 "mean_rupture_rate": 0.01})
    assert F.config_stats(plan, {"law": "constant", "r0": 1e-2, "frac": 1.0}, mini)["status"] == "disconnected"


def test_stage2_plan_rejects_an_undefined_P0(mini):
    s1 = _s1(mini)
    s1["best"]["P_formed_median_at_1p5_onset"] = float("nan")
    with pytest.raises(SystemExit):
        F.stage2_plan(mini, DATA, s1)


def test_vscale_stops_with_a_message_when_runs_are_missing(mini, monkeypatch):
    monkeypatch.setattr(F, "run_many", lambda rs, procs, setup: 0)
    with pytest.raises(SystemExit, match="missing"):
        F.stage1_vscale(1, mini, DATA, max_iter=1, volts=(2.5,))
