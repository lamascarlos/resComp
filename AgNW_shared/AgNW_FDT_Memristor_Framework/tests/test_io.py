import json
from pathlib import Path
import numpy as np
import pytest
from agnw_sim.simulate import run_simulation
from agnw_sim.io import save_result, load_result, reconstruct_network, restore_rng
from agnw_sim.visualize import plot_network


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    result = run_simulation(
        n_wires=100, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.3,
        n_steps=10, dt=1e-3, rng=rng, band_frac=0.06,
    )
    path = str(tmp_path / "run.h5")
    save_result(path, result)
    data = load_result(path)

    np.testing.assert_array_equal(data["t"], result.t)
    np.testing.assert_array_equal(data["V_in"], result.V_in)
    np.testing.assert_array_equal(data["I_total"], result.I_total)
    np.testing.assert_array_equal(data["G_sample"], result.G_sample)
    np.testing.assert_array_equal(data["junctions"], result.network.junctions)
    np.testing.assert_array_equal(data["wire_endpoints"], result.network.wire_endpoints)
    np.testing.assert_array_equal(data["electrode_in"], result.network.electrode_in)
    np.testing.assert_array_equal(data["electrode_gnd"], result.network.electrode_gnd)
    np.testing.assert_array_equal(data["isolated_wires"], result.network.isolated_wires)
    assert data["metadata"]["n_wires"] == 100
    assert data["metadata"]["G_on"] == result.params.G_on
    assert data["metadata"]["n_junctions"] == result.network.junctions.shape[0]
    assert data["metadata"]["n_steps"] == 10
    assert abs(float(data["metadata"]["band_frac"]) - 0.06) < 1e-12
    assert json.loads(data["metadata"]["initial_rng_state_json"]) == result.initial_rng_state
    assert isinstance(data["metadata"]["git_commit_hash"], str)


def test_reconstruct_network_usable_by_visualize(tmp_path):
    rng = np.random.default_rng(5)
    result = run_simulation(
        n_wires=100, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.3,
        n_steps=5, dt=1e-3, rng=rng,
    )
    path = str(tmp_path / "run2.h5")
    save_result(path, result)
    data = load_result(path)
    network = reconstruct_network(data)
    ax = plot_network(network)
    assert len(ax.lines) == network.n_wires
    assert network.n_wires == result.network.n_wires
    assert network.area == result.network.area
    assert network.wire_length == result.network.wire_length


def test_restore_rng_reproduces_identical_run(tmp_path):
    seed_rng = np.random.default_rng(9)
    result_a = run_simulation(
        n_wires=100, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.3,
        n_steps=10, dt=1e-3, rng=seed_rng,
    )
    path = str(tmp_path / "run3.h5")
    save_result(path, result_a)
    data = load_result(path)

    restored_rng = restore_rng(data["metadata"]["initial_rng_state_json"])
    result_b = run_simulation(
        n_wires=100, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.3,
        n_steps=10, dt=1e-3, rng=restored_rng,
    )
    np.testing.assert_array_equal(result_a.I_total, result_b.I_total)
    np.testing.assert_array_equal(result_a.G_sample, result_b.G_sample)
    np.testing.assert_array_equal(result_a.network.junctions, result_b.network.junctions)


def test_git_commit_hash_is_nonempty_in_this_repo(tmp_path):
    rng = np.random.default_rng(6)
    result = run_simulation(
        n_wires=50, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.2,
        n_steps=3, dt=1e-3, rng=rng,
    )
    path = str(tmp_path / "run4.h5")
    save_result(path, result)
    data = load_result(path)
    # This test runs inside a real git checkout, so the hash should be a
    # 40-character hex SHA, not the "" fallback for a missing repo.
    assert len(data["metadata"]["git_commit_hash"]) == 40


def test_git_commit_hash_correct_even_when_cwd_is_outside_repo(tmp_path, monkeypatch):
    # _git_commit_hash() must not depend on the caller's cwd: if invoked
    # while the process cwd is somewhere outside this repo (e.g. tmp_path,
    # which pytest guarantees is outside the repo), it must still report
    # THIS repo's real commit hash, not "" and not some other repo's HEAD.
    rng = np.random.default_rng(6)
    result = run_simulation(
        n_wires=50, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.2,
        n_steps=3, dt=1e-3, rng=rng,
    )
    path = str(tmp_path / "run5.h5")

    monkeypatch.chdir(tmp_path)
    save_result(path, result)

    data = load_result(path)
    hash_from_outside_cwd = data["metadata"]["git_commit_hash"]

    # Ground truth: the real repo's HEAD, computed with an explicit cwd
    # pinned to this test file's own repo checkout.
    import subprocess
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=5, check=True,
        cwd=str(Path(__file__).resolve().parent),
    ).stdout.strip()

    assert len(hash_from_outside_cwd) == 40
    assert hash_from_outside_cwd == expected


def test_wire_diameter_roundtrip(tmp_path):
    for D in (None, 90e-9):
        result = run_simulation(
            n_wires=60, area=1e-6, length=70e-6, waveform_fn=lambda t: 0.3,
            n_steps=3, dt=1e-3, rng=np.random.default_rng(6), wire_diameter=D,
        )
        path = str(tmp_path / f"run_{D}.h5")
        save_result(path, result)
        stored = float(load_result(path)["metadata"]["wire_diameter"])
        if D is None:
            assert np.isnan(stored)      # NaN encodes ideal wires
        else:
            assert stored == D
