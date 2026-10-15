import numpy as np
import pytest

from agnw_sim import fdtest_protocol as fp
from agnw_sim.geometry import Network
from agnw_sim.dualnode import DualNodeCircuit


def _schedule(**kw):
    base = dict(V0=2.0, eps=0.1, T_relax=100, T_samp=100, n_cycles=3, record_every=10, n_lags=8)
    base.update(kw)
    return fp.Schedule(**base)


# ------------------------------------------------------------------------------------ schedule
def test_schedule_voltage_follows_the_four_stages():
    # Break caught: an off-by-one in the removal instant (P_eps would be sampled after the perturbation ended).
    s = _schedule()
    assert s.T_cycle == 400 and s.n_steps == 1200
    assert [s.voltage(i) for i in (0, 199)] == [2.1, 2.1]          # stages I and II (steps 0..199)
    assert [s.voltage(i) for i in (200, 399)] == [2.0, 2.0]        # stages III and IV
    assert s.voltage(400) == 2.1                                    # next cycle starts perturbed


def test_schedule_waveform_matches_voltage_for_time_in_seconds():
    s = _schedule()
    f = s.waveform_fn(1e-3)
    assert f(199 * 1e-3) == 2.1 and f(200 * 1e-3) == 2.0


def test_schedule_validation():
    bad_cases = [dict(T_relax=105),                       # not a multiple of record_every
                 dict(T_samp=55),
                 dict(n_lags=20),                         # 20 * 10 > T_relax: lags would run into stage IV
                 dict(T_relax=300, T_samp=50, n_lags=8),  # only 5 recorded samples in stage IV for 8 lags
                 dict(record_every=0), dict(n_cycles=0), dict(n_lags=1)]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            _schedule(**bad)


def test_snapshots_are_taken_at_the_end_of_stages_two_and_four():
    assert _schedule().snapshot_steps() == [199, 399, 599, 799, 999, 1199]


def test_extract_stage_arrays_cuts_the_recorded_series_at_the_stage_boundaries():
    # record index i corresponds to step 10 * i; cycle 0: stage II = steps 100..199, removal at step 200,
    # stage IV = steps 300..399
    s = _schedule()
    x2, x3, x4 = fp.extract_stage_arrays(np.arange(s.n_steps // 10), s)
    np.testing.assert_array_equal(x2[0] * 10, np.arange(100, 200, 10))                    # stage II of cycle 0
    np.testing.assert_array_equal(x3[0] * 10, np.arange(200, 280, 10))                    # 8 lags from the removal instant
    np.testing.assert_array_equal(x4[0] * 10, np.arange(300, 400, 10))                    # stage IV
    assert x2[1][0] * 10 == 500 and x3[1][0] * 10 == 600 and x4[1][0] * 10 == 700        # cycle 1 (base 400)
    with pytest.raises(ValueError):
        fp.extract_stage_arrays(np.arange(10), s)


def test_extract_switch_on_arrays_takes_the_first_records_of_stage_one():
    # record index i corresponds to step 10 * i; the cycle c starts at step 400 c with stage I (perturbation on)
    s = _schedule()
    xI = fp.extract_switch_on_arrays(np.arange(s.n_steps // 10), s)
    assert xI.shape == (3, 8)
    np.testing.assert_array_equal(xI[0] * 10, np.arange(0, 80, 10))                        # cycle 0: steps 0..70
    np.testing.assert_array_equal(xI[1] * 10, np.arange(400, 480, 10))                     # cycle 1 starts at step 400
    assert xI[2][0] * 10 == 800 and xI[2][-1] * 10 == 870
    # lag 0 of cycle c is the record right after the last stage IV record of cycle c - 1 (the unperturbed steady state)
    x4 = fp.extract_stage_arrays(np.arange(s.n_steps // 10), s)[2]
    assert xI[1][0] - x4[0][-1] == 1
    with pytest.raises(ValueError):
        fp.extract_switch_on_arrays(np.arange(10), s)


# ------------------------------------------------------------------------------------ operating point
def test_pilot_summary_drift_and_saturation_hand_values():
    V = np.full(8, 2.0)
    I_total = 2.0 * np.array([1, 1, 1, 1, 3, 3, 3, 3], dtype=float)     # g_eff = 1, 1, 1, 1, 3, 3, 3, 3
    G = np.zeros((8, 3))
    G[:, 0] = 1e-3                                                        # always ON
    G[::2, 1] = 1e-3                                                      # ON half of the time
    out = fp.pilot_summary(I_total, V, G, block=4, tail=8)                # two blocks: means 1 and 3
    assert out["drift"] == pytest.approx(1.0)                             # |3 - 1| / mean(=2)
    assert out["mean_geff"] == pytest.approx(2.0)
    assert out["frac_intermediate"] == pytest.approx(1 / 3)               # only the alternating junction
    assert out["p_on"].tolist() == [1.0, 0.5, 0.0]


def test_pilot_summary_needs_two_blocks():
    with pytest.raises(ValueError):
        fp.pilot_summary(np.ones(10), np.ones(10), np.zeros((10, 2)), block=10, tail=10)


def test_choose_v0_picks_the_least_saturated_stationary_voltage():
    summaries = {1.6: dict(drift=0.05, frac_intermediate=0.5),         # least saturated but not stationary
                 1.8: dict(drift=0.01, frac_intermediate=0.30),
                 2.0: dict(drift=0.015, frac_intermediate=0.20),
                 2.5: dict(drift=0.004, frac_intermediate=0.05)}
    assert fp.choose_v0(summaries) == 1.8
    assert fp.choose_v0({1.6: dict(drift=0.5, frac_intermediate=0.5)}) is None
    assert fp.choose_v0({1.6: dict(drift=np.nan, frac_intermediate=0.5)}) is None


def test_choose_epsilon_uses_the_local_slope_of_the_pilot_curve():
    # slope between 1.6 V and 2.0 V = (0.7e-3 - 0.2e-3) / 0.4 = 1.25e-3 S/V; eps = 0.5 * 1e-5 / 1.25e-3 = 0.004 V
    eps = fp.choose_epsilon([1.6, 1.8, 2.0, 2.5], [0.2e-3, 0.5e-3, 0.7e-3, 0.9e-3], 1e-5, 1.8)
    assert eps == pytest.approx(0.004)
    with pytest.raises(ValueError):
        fp.choose_epsilon([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 0.1, 2.0)


# ------------------------------------------------------------------------------------ junction properties
def _chain(junctions, n_wires):
    L = 70e-6
    return Network(n_wires=n_wires, wire_endpoints=np.zeros((n_wires, 2, 2)),
                   junctions=np.array(junctions), junction_pos=np.zeros((len(junctions), 2)),
                   electrode_in=np.array([0]), electrode_gnd=np.array([n_wires - 1]),
                   isolated_wires=np.array([], dtype=int), area=1e-8, wire_length=L)


def test_junction_strength_counts_the_other_junctions_on_both_wires():
    # wires 0-1-2 in a chain: deg = [1, 2, 1] -> both junctions have strength 1 + 2 - 2 = 1
    assert fp.junction_strength(_chain([[0, 1], [1, 2]], 3)).tolist() == [1, 1]
    # a triangle: deg = [2, 2, 2] -> strength 2 + 2 - 2 = 2 each
    assert fp.junction_strength(_chain([[0, 1], [1, 2], [0, 2]], 3)).tolist() == [2, 2, 2]


def test_current_share_of_a_series_circuit_is_one_and_backbone_needs_the_junction_to_be_on():
    L = 70e-6
    w0 = np.array([[0.0, 0.0], [L, 0.0]])
    mid = np.array([30e-6, 0.0])
    d = np.array([0.6, 0.8])
    w1 = np.array([mid - 0.5 * L * d, mid + 0.5 * L * d])
    net = Network(n_wires=2, wire_endpoints=np.stack([w0, w1]), junctions=np.array([[0, 1]]),
                  junction_pos=mid[None, :], electrode_in=np.array([0]), electrode_gnd=np.array([1]),
                  isolated_wires=np.array([], dtype=int), area=1e-8, wire_length=L)
    circuit = DualNodeCircuit(net, 90e-9)
    share_on, backbone_on = fp.current_share(circuit, np.array([[1]]), 1.0)
    share_off, backbone_off = fp.current_share(circuit, np.array([[0]]), 1.0)
    assert share_on[0] == pytest.approx(1.0) and backbone_on[0] == 1.0     # all the current, and ON
    assert share_off[0] == pytest.approx(1.0) and backbone_off[0] == 0.0   # same current fraction but OFF


def test_junction_classes_terciles_and_backbone():
    strength = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    backbone = np.array([1, 1, 1, 0, 0, 0, 0.6, 0.4, 0.5])
    masks, names = fp.junction_classes(strength, backbone)
    assert masks.shape == (5, 9) and len(names) == 5
    assert masks[:3].sum(axis=0).tolist() == [1] * 9                       # the three strength classes partition
    assert masks[3].tolist() == [1, 1, 1, 0, 0, 0, 1, 0, 1]                # backbone >= 0.5
    assert (masks[3] ^ masks[4]).all()


# ------------------------------------------------------------------------------------ acceptance gate
def test_gate_drift_flags_a_trend_in_stage_four():
    flat = np.ones((4, 30))
    trend = np.ones((4, 30)) * np.linspace(1.0, 1.1, 30)
    assert fp.gate_drift(flat)["passed"]
    out = fp.gate_drift(trend)
    # three blocks of 10 samples of linspace(1, 1.1, 30): block means 1 + 0.1 * (4.5, 14.5, 24.5) / 29,
    # so drift = (max - min) / mean = (0.1 * 20 / 29) / 1.05 = 0.06568
    assert not out["passed"] and out["value"] == pytest.approx(0.1 * 20 / 29 / 1.05, rel=1e-9)


def test_gate_halves_uses_per_cycle_differences():
    rng = np.random.default_rng(0)
    same = rng.normal(size=(50, 40, 20))                                    # 20 series with the same mean in both halves
    shifted = same.copy(); shifted[:, :20] += 2.0                           # first half shifted for every series
    assert fp.gate_halves(same)["passed"]
    assert not fp.gate_halves(shifted)["passed"]


def test_gate_halves_is_paired_across_cycles():
    rng = np.random.default_rng(1)
    offset = rng.normal(scale=5.0, size=(50, 1, 1))                         # large per-cycle offset shared by both halves
    x = offset + rng.normal(scale=0.1, size=(50, 40, 20))
    x[:, :20] += 0.3                                                        # small but perfectly consistent half shift
    assert not fp.gate_halves(x)["passed"]                                  # paired: the offset cancels, t is huge
    a, b = x[:, :20, 0].mean(axis=1), x[:, 20:, 0].mean(axis=1)
    welch_t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / 50 + b.var(ddof=1) / 50)
    assert abs(welch_t) < 1.0                                               # an unpaired test would not see it


def test_gate_halves_min_fraction():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 40, 20))
    one = x.copy(); one[:, :20, :1] += 2.0                                  # 1 of 20 series shifted
    three = x.copy(); three[:, :20, :3] += 2.0                              # 3 of 20 series shifted
    assert fp.gate_halves(one, min_fraction=0.9)["passed"]
    assert fp.gate_halves(one)["value"] == pytest.approx(0.95)
    assert not fp.gate_halves(three)["passed"]                              # 85 % consistent < default 95 %
    assert fp.gate_halves(three)["value"] == pytest.approx(0.85)
    assert fp.gate_halves(three, min_fraction=0.8)["passed"]


def test_gate_odd_even_needs_mutual_containment():
    assert fp.gate_odd_even(0.10, (0.05, 0.16), 0.12, (0.06, 0.20))["passed"]
    # even outside odd's interval, odd inside even's
    assert not fp.gate_odd_even(0.10, (0.05, 0.16), 0.30, (0.06, 0.20))["passed"]
    # odd outside even's interval, even inside odd's (the opposite violation)
    assert not fp.gate_odd_even(0.10, (0.05, 0.16), 0.12, (0.11, 0.20))["passed"]


def test_gate_scaling_is_an_endpoint_check():
    assert fp.gate_scaling([0.5, 0.2, 0.05])["passed"]
    assert fp.gate_scaling([0.5, 0.4])["passed"]                            # shrinks, however little
    assert not fp.gate_scaling([0.5, 0.9])["passed"]                        # grows
    assert not fp.gate_scaling([0.05, 0.2, 0.5], floor=0.01)["passed"]
    assert fp.gate_scaling([0.05, 0.06], floor=0.05)["passed"]              # last <= 2 * floor although last > first


def test_gate_power_clean_and_low_snr():
    assert fp.gate_power(np.array([1.0, 1.1, 0.9, 1.05, 0.95]))["passed"]
    assert not fp.gate_power(np.array([0.1, -0.1, 0.2, -0.2, 0.0]))["passed"]
    out = fp.gate_power(np.full(5, 0.7))                                    # nonzero mean, zero spread: infinite SNR
    assert out["passed"] and out["value"] == np.inf and out["min_snr"] == 3.0


@pytest.mark.parametrize("z", [np.array([np.nan, 1.0, 2.0]), np.array([1.0, np.inf, 2.0]), np.zeros(5), np.array([1.0]),
                               np.array([])])
def test_gate_power_fails_closed_on_degenerate_input(z):
    out = fp.gate_power(z)
    assert out["passed"] is False and out["min_snr"] == 3.0 and "value" in out


# ------------------------------------------------------------------------------------ achieved perturbation
def test_achieved_response_hand_values():
    # Break caught: a wrong sign / normalisation of the a-posteriori response, or an overlap fraction that is not the
    # share of stage-II samples inside the central 95 % of the stage-IV distribution.
    x4 = np.arange(101, dtype=float)[None, :]                     # mean 50, std sqrt(850), central 95 % = [2.5, 97.5]
    x2 = np.array([[0.0, 2.0, 3.0, 60.0, 97.0, 98.0]])            # mean 260 / 6
    r = fp.achieved_response(x2, x4)
    assert r["achieved_response_sigma"] == pytest.approx((260 / 6 - 50) / np.sqrt(850), rel=1e-12)
    assert r["overlap_fraction"] == pytest.approx(3 / 6)
    assert np.isnan(fp.achieved_response(x2, np.ones((1, 10)))["achieved_response_sigma"])   # zero spread: undefined
