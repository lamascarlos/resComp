import numpy as np
from agnw_sim.simulate import run_simulation
from agnw_sim.dynamics import SimParams
from agnw_sim.waveforms import triangle, ramp
from agnw_sim.analysis import hysteresis_loop_area


def test_end_to_end_small_network_produces_nontrivial_current():
    rng = np.random.default_rng(11)
    params = SimParams(P0=0.3, alpha=100.0, V_th=0.01, P_decay=0.01, dt=1e-3)
    result = run_simulation(
        n_wires=1500, area=1e-6, length=70e-6,
        waveform_fn=lambda t: float(ramp(np.array([t]), rate=2.0)[0]),
        n_steps=300, dt=1e-3, params=params, rng=rng,
    )
    assert np.any(result.I_total != 0.0)
    assert np.std(result.I_total) > 0


def test_hysteresis_loop_area_nonzero_above_percolation():
    dt = 1e-3
    period = 0.2
    # n_steps=201 (not 200) so that t runs 0..0.2 *inclusive*, i.e. the
    # sampled triangle wave returns to its starting value (V_in[0] ==
    # V_in[-1]): a genuinely closed cycle. At n_steps=200 the last sample
    # is t=0.199, one dt short of closing, so V_in[-1] != V_in[0] and
    # np.trapezoid picks up a spurious nonzero "area" from the open path
    # alone -- for ANY I(V) curve, hysteretic or not.
    n_steps = 201

    # amplitude=1.0V (up from the original 0.5V) and P0=0.4 (up from 0.3)
    # so that junctions actually SET during the cycle -- with the original
    # weak drive essentially nothing switched (the reviewer measured 1/3419
    # junctions ever switching), so the old test's nonzero loop_area was
    # floating-point/discretization noise from the open path, not real
    # hysteresis. V_th and alpha are unchanged from the original.
    amplitude = 1.0
    P0 = 0.4
    alpha = 100.0
    V_th = 0.01
    P_decay = 0.05

    waveform_fn = lambda t: float(triangle(np.array([t]), amplitude=amplitude, period=period)[0])

    rng = np.random.default_rng(12)
    params = SimParams(P0=P0, alpha=alpha, V_th=V_th, P_decay=P_decay, dt=dt)
    result = run_simulation(
        n_wires=1500, area=1e-6, length=70e-6,
        waveform_fn=waveform_fn,
        n_steps=n_steps, dt=dt, params=params, rng=rng,
    )
    loop_area = hysteresis_loop_area(result.I_total, result.V_in)

    # Sanity check that real switching actually happened this cycle (not
    # just the cold-start G_off value) -- this is what makes the loop_area
    # assertion below meaningful rather than accidental.
    assert np.any(result.G_sample > params.G_off)

    # Null control: same network/geometry/seed, but with P0=P_decay=0 so
    # p_set and p_reset are always exactly 0 -- G is frozen at its
    # cold-start value for the whole run (a pure, unchanging ohmic
    # network). Any nonzero loop_area here is pure floating-point /
    # discretization noise from the (now closed) triangle path, not
    # hysteresis, since a fixed-G network's I(V) is a single-valued line
    # through the origin traversed forward and back.
    rng_control = np.random.default_rng(12)
    control_params = SimParams(P0=0.0, P_decay=0.0, dt=dt)
    control_result = run_simulation(
        n_wires=1500, area=1e-6, length=70e-6,
        waveform_fn=waveform_fn,
        n_steps=n_steps, dt=dt, params=control_params, rng=rng_control,
    )
    control_loop_area = hysteresis_loop_area(control_result.I_total, control_result.V_in)

    # Empirically: real loop_area ~= 2.2e-12, control loop_area ~= 1.4e-25
    # (ratio ~1.6e13) for this exact setup. The noise floor for a truly
    # closed ohmic cycle is essentially machine-epsilon-scale, so a
    # 100x margin is an extremely conservative threshold that cleanly
    # separates real hysteresis (many orders of magnitude larger) from
    # discretization/floating-point noise, without being tuned to the
    # exact observed ratio.
    assert abs(loop_area) > 100 * abs(control_loop_area)


def test_network_above_threshold_percolates():
    rng = np.random.default_rng(13)
    result = run_simulation(
        n_wires=1500, area=1e-6, length=70e-6,
        waveform_fn=lambda t: 0.1,
        n_steps=1, dt=1e-3, rng=rng,
    )
    # electrode_in and electrode_gnd must both be non-empty for a 1500-wire
    # network in a 1mm^2 area with 70um wires (well above N_c~1200)
    assert result.network.electrode_in.size > 0
    assert result.network.electrode_gnd.size > 0
