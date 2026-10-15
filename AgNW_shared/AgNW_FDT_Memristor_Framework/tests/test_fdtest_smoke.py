"""Smoke test of the whole chain (pilot -> run -> analyze) on a tiny network: it checks the plumbing (files, keys,
no crash, exit status), not the physics. N = 150 wires do not percolate, so g_eff is numerically ~0 and the
acceptance gate is allowed to fail; the analysis must still finish and write its results."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """pilot + two seeds of runs in a temporary directory; returns (directory, analyzer process)."""
    tmp_path = tmp_path_factory.mktemp("fdtest_smoke")
    out = str(tmp_path)
    for V in ("1.0", "1.5"):
        r = _run("scripts.fdtest_pilot", "run", "150", V, "--n-steps", "800", "--k", "30", "--out", out)
        assert r.returncode == 0, r.stderr
    r = _run("scripts.fdtest_pilot", "summarize", "150", "--block", "200", "--tail", "800", "--out", out)
    assert r.returncode in (0, 2), r.stderr                       # 2 only if no voltage is stationary (1 = crash)
    for seed in ("1", "2"):
        r = _run("scripts.fdtest_run", "150", "--v0", "1.5", "--eps", "0.1", "--t-relax", "100", "--t-samp", "100",
                 "--cycles", "4", "--seed", seed, "--k", "30", "--record-every", "10", "--n-lags", "8", "--out", out)
        assert r.returncode == 0, r.stderr
    r = _run("scripts.fdtest_analyze", "150", "--out", out, "--n-boot", "8")
    return tmp_path, r


def test_pilot_run_and_analyze_end_to_end(pipeline):
    tmp_path, r = pipeline
    assert r.returncode in (0, 2), r.stderr                       # 2 = gate failed, allowed on a non-conducting network
    text = (tmp_path / "fdtest_results_N150.json").read_text()
    assert "NaN" not in text and "Infinity" not in text           # strict JSON: non-finite values are written as null
    result = json.loads(text)
    assert {"schedule", "geff", "classes", "gate", "gate_passed", "associations"} <= set(result)
    assert result["M"] == 8 and result["schedule"]["n_cycles"] == 4
    assert (r.returncode == 0) == result["gate_passed"]
    for name in ("geff", "classes", "junctions"):
        assert (tmp_path / f"fdtest_{name}_N150.png").exists()
    # this network has no valid junction and no class can support an SNR claim: the gates must fail closed
    assert result["n_valid_junctions"] == 0 and all(v["n_valid"] == 0 for v in result["classes"].values())
    assert result["gate"]["p0_halves_junctions"]["passed"] is False
    assert result["gate"]["power_classes"]["passed"] is False
    assert set(result["gate"]["power_classes"]["failing_classes"]) == set(result["classes"])
    assert result["gate_passed"] is False


def test_results_carry_the_decision_fields_and_advisory_gates_do_not_enter_gate_passed(pipeline):
    tmp_path, r = pipeline
    result = json.loads((tmp_path / "fdtest_results_N150.json").read_text())
    for key in ("delta2", "ci95", "ci95_reverse", "excess", "excludes_floor", "noise_floor", "tau_c", "cutoff_at_end",
                "achieved_response_sigma", "overlap_fraction", "excluded_fraction"):
        assert key in result["geff"]
    for v in result["classes"].values():
        for key in ("ci95", "ci95_reverse", "excess", "excludes_floor", "tau_c", "cutoff_at_end", "achieved_shift_sigma"):
            assert key in v
        assert v["excludes_floor"] is False                        # fail closed on undefined estimates
    assert result["geff"]["excludes_floor"] is False
    assert result["associations_note"].startswith("descriptive only")
    assert set(result["scaling"]) == {"n_cycles", "geff", "classes"}
    gates = result["gate"]
    advisory = [k for k, g in gates.items() if g.get("advisory")]
    assert sorted(advisory) == ["scaling_classes", "scaling_geff"]
    # gate_passed ignores the advisory gates: with every required gate faked to pass it would be True regardless of them
    required = [g["passed"] for k, g in gates.items() if k not in advisory]
    assert result["gate_passed"] == all(required)
    assert {"binned", "bin_scheme", "lag_noise_corr"} <= set(result)
    scheme = result["bin_scheme"]
    assert scheme["edges"][0] == 0 and scheme["edges"][-1] == result["schedule"]["n_lags"] and scheme["n_bins"] == len(scheme["edges"]) - 1
    assert set(result["binned"]) == {"geff", "classes"} and set(result["binned"]["classes"]) == set(result["classes"])
    for block in (result["binned"]["geff"], *result["binned"]["classes"].values()):
        for key in ("delta2", "ci95", "ci95_reverse", "noise_floor", "excess", "excludes_floor", "tau_c", "cutoff_at_end"):
            assert key in block
        assert block["excludes_floor"] is False                    # fail closed on undefined estimates
    assert "corr_length" in result["lag_noise_corr"]["geff"] and set(result["lag_noise_corr"]["classes"]) == set(result["classes"])
    assert "lag-binned violation" in r.stdout
    fw = result["fixed_window"]                                    # fixed analysis window: default = the whole window (n_lags lags)
    assert fw["window"] == result["schedule"]["n_lags"] - 1 and result["interval_method"] == "reverse"
    assert set(fw) == {"window", "geff", "classes"} and set(fw["classes"]) == set(result["classes"])
    for block in (fw["geff"], *fw["classes"].values()):
        for key in ("delta2", "ci95", "ci95_reverse", "noise_floor", "excess", "excludes_floor", "tau_c", "cutoff_at_end"):
            assert key in block
        assert block["tau_c"] in (None, fw["window"]) and block["excludes_floor"] is False
    assert "fixed window" in r.stdout
    assert "advisory checks" in r.stdout and "excludes_floor" in r.stdout
    assert "acceptance gate: FAILED (" in r.stdout and "scaling_" not in r.stdout.split("acceptance gate:")[1]


def _copy_runs(src, dst):
    for f in src.glob("run_N150_seed*.npz"):
        shutil.copy(f, dst / f.name)


def test_n_lags_override_changes_the_number_of_lags_used(pipeline, tmp_path):
    src, _ = pipeline
    _copy_runs(src, tmp_path)
    r = _run("scripts.fdtest_analyze", "150", "--out", str(tmp_path), "--n-boot", "8", "--n-lags", "6")
    assert r.returncode in (0, 2), r.stderr
    assert "6 lags" in r.stdout.splitlines()[0]
    result = json.loads((tmp_path / "fdtest_results_N150.json").read_text())
    assert result["schedule"]["n_lags"] == 6 and result["schedule"]["n_lags_stored"] == 8


def test_n_lags_above_the_allowed_maximum_is_rejected_before_any_work(pipeline, tmp_path):
    src, _ = pipeline                                       # T_samp = T_relax = 100, record_every = 10: at most 10 lags
    _copy_runs(src, tmp_path)
    for bad in ("11", "1"):
        r = _run("scripts.fdtest_analyze", "150", "--out", str(tmp_path), "--n-boot", "8", "--n-lags", bad)
        assert r.returncode == 1 and "10" in r.stderr and "--n-lags" in r.stderr, r.stderr
        assert "Traceback" not in r.stderr and r.stdout == ""
    assert not (tmp_path / "fdtest_results_N150.json").exists()


def test_no_binned_removes_the_binned_block_and_leaves_the_unbinned_results_unchanged(pipeline, tmp_path):
    src, _ = pipeline
    _copy_runs(src, tmp_path)
    r = _run("scripts.fdtest_analyze", "150", "--out", str(tmp_path), "--n-boot", "8", "--no-binned")
    assert r.returncode in (0, 2), r.stderr
    text = (tmp_path / "fdtest_results_N150.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    plain = json.loads(text)
    assert not ({"binned", "bin_scheme", "lag_noise_corr"} & set(plain)) and "lag-binned" not in r.stdout
    full = json.loads((src / "fdtest_results_N150.json").read_text())
    for key in ("geff", "classes", "gate", "gate_passed", "scaling", "associations"):
        assert plain[key] == full[key], key                          # the binned block does not disturb the unbinned analysis
    for flags in (["--bin-ratio", "1.0"], ["--bin-ratio", "inf"], ["--bin-ratio", "nan"], ["--bin-first", "0"]):
        r = _run("scripts.fdtest_analyze", "150", "--out", str(tmp_path), "--n-boot", "8", *flags)
        assert r.returncode == 1 and flags[0] in r.stderr and "Traceback" not in r.stderr, (flags, r.stderr)


@pytest.mark.parametrize("corrupt, message", [("ids", "sampled_junction_ids"), ("snapshots", "G_snapshots")])
def test_analyzer_rejects_run_files_inconsistent_with_the_network(pipeline, tmp_path, corrupt, message):
    src, _ = pipeline
    _copy_runs(src, tmp_path)
    target = tmp_path / "run_N150_seed2.npz"
    z = dict(np.load(target))
    if corrupt == "ids":
        z["sampled_junction_ids"] = z["sampled_junction_ids"][::-1].copy()
    else:
        z["G_snapshots"] = z["G_snapshots"][:, :-1].copy()
    np.savez_compressed(target, **z)
    r = _run("scripts.fdtest_analyze", "150", "--out", str(tmp_path), "--n-boot", "8")
    assert r.returncode == 1 and message in r.stderr and "run_N150_seed2.npz" in r.stderr
    assert not (tmp_path / "fdtest_results_N150.json").exists()


def test_fixed_window_option_no_fixed_and_interval_method(pipeline, tmp_path):
    src, _ = pipeline
    _copy_runs(src, tmp_path)
    full = json.loads((src / "fdtest_results_N150.json").read_text())
    r = _run("scripts.fdtest_analyze", "150", "--out", str(tmp_path), "--n-boot", "8", "--fixed-window", "3", "--interval-method", "sqrt_reverse")
    assert r.returncode in (0, 2), r.stderr
    got = json.loads((tmp_path / "fdtest_results_N150.json").read_text())
    assert got["fixed_window"]["window"] == 3 and got["interval_method"] == "sqrt_reverse"
    assert all(v["tau_c"] in (None, 3) for v in got["fixed_window"]["classes"].values())
    assert "fixed window" in r.stdout
    r = _run("scripts.fdtest_analyze", "150", "--out", str(tmp_path), "--n-boot", "8", "--no-fixed")
    assert r.returncode in (0, 2), r.stderr
    plain = json.loads((tmp_path / "fdtest_results_N150.json").read_text())
    assert "fixed_window" not in plain and "fixed window" not in r.stdout
    for key in ("geff", "classes", "gate", "gate_passed", "scaling", "associations", "binned", "lag_noise_corr"):
        assert plain[key] == full[key], key                          # the fixed block does not disturb the other results
    for flags in (["--fixed-window", "-1"], ["--fixed-window", "8"], ["--fixed-window", "3", "--no-fixed"]):
        r = _run("scripts.fdtest_analyze", "150", "--out", str(tmp_path), "--n-boot", "8", *flags)
        assert r.returncode == 1 and flags[0] in r.stderr and "Traceback" not in r.stderr, (flags, r.stderr)
