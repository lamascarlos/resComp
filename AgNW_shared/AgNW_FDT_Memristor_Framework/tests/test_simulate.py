import numpy as np
import pytest
from agnw_sim.simulate import run_simulation, SimulationResult
from agnw_sim.dynamics import SimParams


def test_run_simulation_produces_expected_shapes():
    rng = np.random.default_rng(1)
    result = run_simulation(
        n_wires=150, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.5,
        n_steps=50, dt=1e-3, rng=rng,
    )
    assert isinstance(result, SimulationResult)
    assert result.t.shape == (50,)
    assert result.V_in.shape == (50,)
    assert result.I_total.shape == (50,)
    assert result.G_sample.shape[0] == 50
    assert np.all(result.V_in == 0.5)


def test_run_simulation_zero_junctions_does_not_crash():
    rng = np.random.default_rng(2)
    result = run_simulation(
        n_wires=2, area=1e-6, length=1e-9,
        waveform_fn=lambda t: 1.0,
        n_steps=5, dt=1e-3, rng=rng,
    )
    assert result.I_total.shape == (5,)
    assert np.isfinite(result.I_total).all()


def test_run_simulation_conductance_can_switch_on():
    # NOTE: n_wires=1500, not the 300 used elsewhere in this file. For
    # area=1e-6, length=70e-6, the 2D stick-percolation threshold is
    # ~1150 wires (Balberg/Pike-Seager critical avg-contacts criterion).
    # At n_wires=300 (~26% of critical density) the cold-start network
    # (G uniformly G_off) never has a real V_IN-to-GND conduction path,
    # so every junction's V_mem is exactly 0 (dead-end/no-current, not
    # merely small) and SET can never fire regardless of steps or rng
    # seed -- confirmed empirically across seeds 0-199, 0/200 percolate
    # at n_wires=300 for these area/length values. n_wires=1500 is safely
    # above threshold (confirmed percolating for the seed below) so this
    # test actually exercises the SET switching path it's meant to check.
    rng = np.random.default_rng(3)
    params = SimParams(P0=1.0, alpha=1000.0, V_th=0.001, dt=1e-3)
    result = run_simulation(
        n_wires=1500, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 1.0,
        n_steps=20, dt=1e-3, rng=rng, params=params,
    )
    assert np.any(result.G_sample > params.G_off)


def test_run_simulation_consistent_dt_and_params_dt_works():
    rng = np.random.default_rng(5)
    params = SimParams(dt=1e-3)
    result = run_simulation(
        n_wires=150, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.5,
        n_steps=10, dt=1e-3, rng=rng, params=params,
    )
    assert result.t.shape == (10,)
    assert np.allclose(result.t, np.arange(10) * 1e-3)


def test_run_simulation_conflicting_dt_and_params_dt_raises():
    rng = np.random.default_rng(6)
    params = SimParams(dt=1e-6)
    with pytest.raises(ValueError):
        run_simulation(
            n_wires=150, area=1e-6, length=70e-6,
            waveform_fn=lambda t: 0.5,
            n_steps=10, dt=1.0, rng=rng, params=params,
        )


def test_run_simulation_sampled_junction_ids_within_range():
    rng = np.random.default_rng(4)
    result = run_simulation(
        n_wires=300, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.3,
        n_steps=10, dt=1e-3, rng=rng, n_sampled_junctions=50,
    )
    n_junctions = result.network.junctions.shape[0]
    assert result.sampled_junction_ids.shape[0] == min(50, n_junctions)
    if n_junctions:
        assert result.sampled_junction_ids.max() < n_junctions


def test_run_simulation_captures_initial_rng_state():
    rng = np.random.default_rng(42)
    expected_state = rng.bit_generator.state
    result = run_simulation(
        n_wires=100, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.3,
        n_steps=5, dt=1e-3, rng=rng,
    )
    assert isinstance(result.initial_rng_state, dict)
    assert result.initial_rng_state == expected_state


def test_run_simulation_initial_rng_state_differs_across_seeds():
    result_a = run_simulation(
        n_wires=50, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.1,
        n_steps=3, dt=1e-3, rng=np.random.default_rng(1),
    )
    result_b = run_simulation(
        n_wires=50, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.1,
        n_steps=3, dt=1e-3, rng=np.random.default_rng(2),
    )
    assert result_a.initial_rng_state != result_b.initial_rng_state


def test_run_simulation_records_band_frac():
    result = run_simulation(
        n_wires=100, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.3,
        n_steps=5, dt=1e-3, rng=np.random.default_rng(7), band_frac=0.08,
    )
    assert result.band_frac == 0.08


def test_run_simulation_band_frac_defaults_to_005():
    result = run_simulation(
        n_wires=100, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.3,
        n_steps=5, dt=1e-3, rng=np.random.default_rng(8),
    )
    assert result.band_frac == 0.05
