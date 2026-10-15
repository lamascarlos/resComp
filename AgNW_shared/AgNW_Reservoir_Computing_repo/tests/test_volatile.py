"""N10 volatile two-time-scale junction (agnw_rc/volatile.py, docs/N10_DESIGN.md)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from agnw_rc import config as C
from agnw_rc.multidrive import simulate_multi
from agnw_rc.volatile import VolatileParams, conductance, junction_kd0, rates, step_state

P = VolatileParams()


def _run(state, V, p, n, kd0=None):
    kd0 = np.full(state.shape[1], 1 / p.tau_s) if kd0 is None else kd0
    for _ in range(n):
        state = step_state(state, V, p, kd0)
    return state


def test_milano_special_case_matches_the_ode():
    """k_c = 0, no power rupture, linear conductance: x follows dg/dt = k_p (1 - g) - k_d g of Milano et al. (2022)
    exactly for piecewise-constant voltages, whatever the step."""
    p = P.with_(k_P0=0.0, conductance="linear", dt=5e-3)
    kd0 = 1 / p.tau_s
    V_seq = [0.0, 0.45, 0.55, 0.3, 0.0, 0.6]
    x = np.array([[0.1], [0.0]])
    g = 0.1
    for V in V_seq:
        x = _run(x, np.array([V]), p, 40)
        kp = p.k_p0 * np.exp(p.eta_p * V)
        kd = kd0 * np.exp(-p.eta_d * V)
        sol = solve_ivp(lambda t, y: kp * (1 - y) - kd * y, (0, 40 * p.dt), [g], rtol=1e-11, atol=1e-13)
        g = sol.y[0, -1]
        assert x[0, 0] == pytest.approx(g, rel=1e-7, abs=1e-12)
    G = conductance(x, p)
    assert G[0] == pytest.approx(p.G_off + (p.G_on - p.G_off) * g, rel=1e-7)


def test_zero_bias_relaxation_is_exponential_with_tau_s():
    p = P.with_(tau_s=0.5, dt=1e-2)
    x = _run(np.array([[1.0], [0.0]]), np.array([0.0]), p, 50)          # 0.5 s
    k = p.k_p0 + 1 / p.tau_s
    x_inf = p.k_p0 / k
    assert x[0, 0] == pytest.approx(x_inf + (1 - x_inf) * np.exp(-k * 0.5), rel=1e-9)
    assert x[0, 0] == pytest.approx(np.exp(-1.0), rel=1e-5)                       # ~ exp(-t / tau_s)


def test_steady_state_and_stability_for_any_step():
    V = np.array([0.0, 0.3, 0.49, 0.7, 5.0])
    G = conductance(np.zeros((2, 5)), P)
    kp, kd, kP = rates(V, G, P.with_(k_P0=0.0), 1 / P.tau_s)
    for dt in (1e-3, 1.0, 100.0):
        x = _run(np.zeros((2, 5)), V, P.with_(dt=dt, k_P0=0.0), 3 if dt > 1 else 20_000 if dt < 1e-2 else 50)
        assert np.all((x >= 0) & (x <= 1))
        np.testing.assert_allclose(x[0], kp / (kp + kd), rtol=1e-6)


def test_power_rupture_limits_the_filament():
    """Above P_c growth stops and the filament ruptures, so a junction held at 2 V cannot stay fully formed
    (G_on V^2 = 4 mW > P_c); without rupture it closes completely."""
    p = P.with_(dt=1e-3)
    V = np.array([2.0])
    no_rupture = _run(np.zeros((2, 1)), V, p.with_(k_P0=0.0), 100)
    assert no_rupture[0, 0] > 0.99
    x, xs = np.zeros((2, 1)), []
    for _ in range(2000):
        x = step_state(x, V, p, np.array([1 / p.tau_s]))
        xs.append(x[0, 0])
    xs = np.array(xs[1000:])
    power = conductance(np.stack([xs, np.zeros_like(xs)]), p) * 4.0
    assert xs.min() < 0.9                                  # repeatedly ruptured
    assert np.mean(power > p.P_c) < 0.6                    # overpowered at most about half of the time


def test_consolidation_needs_a_formed_filament_and_voltage_and_decays_slowly():
    p = P.with_(k_c=0.1, dt=1e-2, k_P0=0.0)
    full = np.array([[1.0], [0.0]])
    strong = _run(full, np.array([0.6]), p, 1000)                        # 10 s at 0.6 V: x stays ~1
    assert strong[1, 0] == pytest.approx(1 - np.exp(-1.0), rel=0.05)   # k_c t = 1
    weak = _run(np.array([[0.0], [0.0]]), np.array([0.1]), p, 1000)    # no filament: no consolidation
    assert weak[1, 0] < 1e-6
    decayed = _run(np.array([[0.0], [0.5]]), np.array([0.0]), p.with_(tau_l=100.0), 1000)
    assert decayed[1, 0] == pytest.approx(0.5 * np.exp(-0.1), rel=1e-3)


def test_conductance_endpoints_and_log_interpolation():
    s = np.array([[0.0, 0.5, 1.0, 0.7], [0.0, 0.0, 0.0, 0.6]])
    G = conductance(s, P)
    np.testing.assert_allclose(G, [P.G_off, np.sqrt(P.G_off * P.G_on), P.G_on, P.G_on])


def test_junction_spread_is_lognormal_and_fixed_by_network():
    p = P.with_(tau_s=0.3, tau_spread=1.0)
    a, b = junction_kd0(p, 20_000, 1), junction_kd0(p, 20_000, 1)
    np.testing.assert_array_equal(a, b)
    assert np.median(1 / a) == pytest.approx(0.3, rel=0.05)
    assert np.std(np.log(a)) == pytest.approx(1.0, rel=0.05)
    np.testing.assert_array_equal(junction_kd0(p.with_(tau_spread=0), 5, 1), np.full(5, 1 / 0.3))


def test_spec_names_params_and_washout():
    s = C.RunSpec("iid", 300, 4.0, 1, layout="two_bands", encoding="direct", model="volatile", tau_s=0.3,
                  tau_spread=1.0, consolidation=True, stride=2)
    assert s.name.endswith("_vol_ts0.3_sp1_con_st2")
    vp = C.volatile_params(s)
    assert vp.tau_s == 0.3 and vp.tau_spread == 1.0 and vp.k_c == C.VOLATILE_K_C and vp.dt == 2e-3
    assert C.volatile_params(C.RunSpec("iid", 300, 4.0, 1, model="volatile")).k_c == 0.0
    assert C.model_washout_steps("volatile", tau_s=10.0) == 30_000
    assert C.model_washout_steps("volatile", tau_s=0.01) == 3_000


def test_network_run_shapes_and_state_bounds():
    spec = C.RunSpec("iid", 20, 4.0, 1, n_samples=6, layout="two_bands", encoding="direct", model="volatile",
                     tau_s=0.1, tau_spread=0.5, consolidation=True, stride=2, n_virtual=5)
    raw = simulate_multi(spec, washout_steps=40, save_states=True)
    n_j = raw["n_junctions"]
    assert raw["junction_x"].shape == (6, n_j) and raw["junction_y"].shape == (6, n_j)
    assert raw["current"].shape == (6, 5 * 2)
    x = raw["junction_x"].astype(float)
    assert np.all((x >= 0) & (x <= 1)) and x.max() > 0
    from agnw_rc.evaluate import observables
    assert observables(raw)["junctions"].shape == (6, n_j)


def test_pads_float_layout_and_electrode_readout():
    """N10-E: 4 driven, 4 grounded and 8 floating pads; the "electrodes" readout holds the 8 driven/grounded
    currents and the 8 floating-pad voltages at every virtual node, and floating pads sit between the driven and
    grounded potentials."""
    from agnw_rc.electrodes import make_layout
    from agnw_rc.evaluate import observables, state_observables
    net = C.make_network(1)
    roles = [e.role for e in make_layout("pads_float", net)]
    assert roles.count("in") == 4 and roles.count("gnd") == 4 and roles.count("float") == 8
    spec = C.RunSpec("iid", 20, 1.0, 1, n_samples=6, layout="pads_float", encoding="direct", model="volatile",
                     tau_s=0.3, stride=2, n_virtual=5)
    raw = simulate_multi(spec, washout_steps=40)
    F = observables(raw)
    assert F["electrodes"].shape == (6, 5 * 16) and raw["v_float"].shape == (6, 5 * 8)
    v_in = raw["v_electrodes"][:, :1]
    assert np.all((raw["v_float"] >= -1e-9) & (raw["v_float"] <= v_in + 1e-9))
    assert "electrodes" in state_observables(raw)


def test_rupture_events_drop_the_filament_at_the_given_rate():
    """Random complete ruptures: with x held high by the drive, the fraction of junctions hit per step equals
    rupture_rate * dt, and a hit sets x to 0 (rupture_frac = 1)."""
    p = P.with_(rupture_rate=5.0, dt=1e-3, k_P0=0.0)
    rng = np.random.default_rng(0)
    n = 200_000
    x = np.stack([np.full(n, 0.9), np.zeros(n)])
    new = step_state(x, np.zeros(n), p, np.full(n, 1 / p.tau_s), rng)
    dropped = new[0] == 0
    assert dropped.mean() == pytest.approx(5.0 * 1e-3, rel=0.1)
    assert np.all(new[0][~dropped] > 0.8)
    with pytest.raises(ValueError):
        step_state(x, np.zeros(n), p, np.full(n, 1 / p.tau_s))


def test_large_pads_contact_more_wires():
    from agnw_rc.electrodes import make_layout
    net = C.make_network(1)
    small = sum(len(e.wires) for e in make_layout("pads", net))
    large = sum(len(e.wires) for e in make_layout("pads_large", net))
    assert large > 1.5 * small


def test_n10k_fields_default_to_old_names_and_override_params():
    """N10-K fields leave every existing run name unchanged and override k_c / tau_l only when set."""
    old = C.RunSpec("iid", 100, 0.38, 1, v_width=0.16, n_samples=800, layout="pads", encoding="weights",
                    model="volatile", tau_s=0.3, tau_spread=1.0, rupture=0.01, washout=6700, n_virtual=50, stride=2)
    assert old.name == "iid_h100_v0.38_w0.16_pr1e-04_n1_r0_s800_pads_weights_vol_ts0.3_sp1_rup0.01_wo6700_nv50_st2"
    k = C.RunSpec("iid", 100, 0.38, 1, layout="pads", model="volatile", tau_s=0.3, consolidation=True, k_c=1.0,
                  tau_l=1e4, p_c2=1e-5, forming=10.0, forming_v=0.7, stride=2)
    assert "_con_kc1_tl10000_pc1e-05_form10v0.7" in k.name
    vp = C.volatile_params(k)
    assert vp.k_c == 1.0 and vp.tau_l == 1e4 and vp.P_c2 == 1e-5
    assert C.volatile_params(C.RunSpec("iid", 100, 0.38, 1, model="volatile", k_c=1.0)).k_c == 0.0   # needs the flag


def test_forming_builds_consolidation_only_when_on():
    """A strong forming drive consolidates junctions only with consolidation on; without forming, no formed_* keys."""
    base = dict(n_samples=4, layout="pads", encoding="weights", model="volatile", tau_s=0.3, stride=2, n_virtual=5,
                forming=1.0, forming_v=1.5, k_c=5.0)
    on = simulate_multi(C.RunSpec("iid", 20, 0.4, 1, consolidation=True, **base), washout_steps=40)
    off = simulate_multi(C.RunSpec("iid", 20, 0.4, 1, consolidation=False, **base), washout_steps=40)
    assert on["formed_y"].max() > 0.1 and np.all(off["formed_y"] == 0)
    assert np.all((on["formed_y"] >= 0) & (on["formed_y"] <= 1))
    plain = simulate_multi(C.RunSpec("iid", 20, 0.4, 1, n_samples=4, layout="pads", encoding="weights",
                                     model="volatile", tau_s=0.3, stride=2, n_virtual=5), washout_steps=40)
    assert "formed_y" not in plain
    with pytest.raises(ValueError):
        simulate_multi(C.RunSpec("iid", 20, 0.4, 1, n_samples=4, layout="pads", forming=1.0, forming_v=1.0),
                       washout_steps=40)


def test_names_keep_two_decimals_and_resolve_small_windows():
    assert C.RunSpec("iid", 100, 0.38, 1, v_width=0.16).name.startswith("iid_h100_v0.38_w0.16_")
    a, b = C.RunSpec("iid", 100, 0.045, 1, v_width=0.019), C.RunSpec("iid", 100, 0.054, 1, v_width=0.019)
    assert a.name != b.name and "_v0.045_w0.019_" in a.name


def test_n10l_side_scales_the_network_and_pads():
    """N10-L: side = 1 reproduces the working network and "pads_scaled" = "pads"; side 2 keeps the wire density, the
    scaled pads keep their relative size, and run names change only when side is set."""
    from agnw_rc.electrodes import make_layout
    a, b = C.make_network(1), C.make_network(1, side=1.0)
    assert np.array_equal(a.wire_endpoints, b.wire_endpoints)
    pa, ps = make_layout("pads", a), make_layout("pads_scaled", a)
    assert all(np.array_equal(x.wires, y.wires) for x, y in zip(pa, ps))
    big = C.make_network(1, side=2.0)
    assert big.wire_endpoints.shape[0] == 4 * C.N_WIRES and abs(big.area - 4 * C.AREA) < 1e-15
    n_small = np.mean([len(e.wires) for e in ps])
    n_big = np.mean([len(e.wires) for e in make_layout("pads_scaled", big)])
    assert 2.5 < n_big / n_small < 6.0                                    # 4x pad area, same density
    assert "_L2_" not in C.RunSpec("iid", 100, 0.4, 1, layout="pads").name
    assert "_pads_scaled_weights_L2" in C.RunSpec("iid", 100, 0.4, 1, layout="pads_scaled", side=2.0).name


def test_junction_states_are_optional_and_do_not_change_the_run():
    """Since 2026-10-08 junction states are not stored by default: same currents and on_fraction as a run that stores
    them, no state keys, and the scoring simply has no "junctions" readout."""
    from agnw_rc.evaluate import observables, state_observables
    spec = C.RunSpec("iid", 20, 0.4, 1, n_samples=6, layout="pads", encoding="weights", model="volatile", tau_s=0.3,
                     stride=2, n_virtual=5)
    with_s = simulate_multi(spec, washout_steps=40, save_states=True)
    without = simulate_multi(spec, washout_steps=40)
    assert C.SAVE_JUNCTION_STATES is False and "junction_x" not in without and "junction_y" not in without
    np.testing.assert_array_equal(with_s["current"], without["current"])
    np.testing.assert_allclose(with_s["on_fraction"], without["on_fraction"])
    assert "junctions" in observables(with_s) and "junctions" not in observables(without)
    assert "junctions" not in state_observables(without)          # consistency loops over these keys


# ---- N11: state-dependent rupture laws (docs/PLAN_N11_CALIBRATION.md section 4)
from agnw_rc.volatile import RUPTURE_LAWS, rupture_rates  # noqa: E402


def test_rupture_law_defaults_and_power_alpha0_reproduce_the_constant_law():
    """N11: the default law is the constant N10 rate, and the power law with alpha = 0 (or the thermal law with
    P* = inf) draws the same random numbers and hits the same junctions."""
    assert P.rupture_law == "constant" and RUPTURE_LAWS == ("constant", "power", "thermal")
    n = 50_000
    x = np.stack([np.full(n, 0.9), np.zeros(n)])
    V = np.linspace(0.0, 0.05, n)
    kd0 = np.full(n, 1 / P.tau_s)
    base = P.with_(rupture_rate=5.0, k_P0=0.0)
    ref = step_state(x, V, base, kd0, np.random.default_rng(3))
    for p in (base.with_(rupture_law="power", rupture_alpha=0.0), base.with_(rupture_law="thermal")):
        np.testing.assert_array_equal(step_state(x, V, p, kd0, np.random.default_rng(3)), ref)


def test_rupture_rates_follow_the_specified_laws():
    Pw = np.array([0.0, 1e-7, 4e-7, 1e-6])
    c = P.with_(rupture_rate=0.5)
    np.testing.assert_allclose(rupture_rates(Pw, c), 0.5)
    pw = c.with_(rupture_law="power", rupture_alpha=2.0, rupture_P0=1e-7)
    np.testing.assert_allclose(rupture_rates(Pw, pw), 0.5 * (Pw / 1e-7) ** 2)
    th = c.with_(rupture_law="thermal", rupture_Pstar=2e-7)
    np.testing.assert_allclose(rupture_rates(Pw, th), 0.5 * np.exp(Pw / 2e-7))
    huge = rupture_rates(np.array([1.0]), c.with_(rupture_law="thermal", rupture_Pstar=1e-12))
    assert np.all(np.isfinite(huge))                                   # exponent capped


def test_power_law_ruptures_hit_hot_junctions_more_often():
    """Junctions at 4 x the reference power break 4 x as often (alpha = 1), counted over 40 000 trials each."""
    n = 40_000
    p = P.with_(rupture_law="power", rupture_alpha=1.0, rupture_P0=1e-7, rupture_rate=2.0, dt=1e-3, k_P0=0.0)
    x = np.stack([np.ones(2 * n), np.zeros(2 * n)])                    # G = G_on = 1e-3 S
    V = np.r_[np.full(n, 0.01), np.full(n, 0.02)]                      # P = 1e-7 W and 4e-7 W
    new = step_state(x, V, p, np.full(2 * n, 1 / p.tau_s), np.random.default_rng(5))
    hits = (new[0] == 0).reshape(2, n).sum(axis=1)                     # expected 80 and 320
    assert abs(hits[0] - 80) < 4 * np.sqrt(80) and abs(hits[1] - 320) < 4 * np.sqrt(320)


def test_partial_rupture_removes_the_given_fraction():
    p = P.with_(rupture_rate=1e6, rupture_frac=0.5, k_P0=0.0)          # rate x dt >> 1: every junction is hit
    x = np.stack([np.full(10, 0.8), np.zeros(10)])
    kd0 = np.full(10, 1 / p.tau_s)
    intact = step_state(x, np.zeros(10), p.with_(rupture_rate=0.0), kd0)
    np.testing.assert_allclose(step_state(x, np.zeros(10), p, kd0, np.random.default_rng(0))[0], 0.5 * intact[0])


def test_unknown_rupture_law_and_negative_alpha_are_rejected():
    with pytest.raises(ValueError):
        P.with_(rupture_law="linear")
    with pytest.raises(ValueError):
        P.with_(rupture_law="power", rupture_alpha=-1.0)
