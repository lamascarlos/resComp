import numpy as np
from agnw_sim.dynamics import SimParams, p_set, p_reset, update_dynamics


def test_p_set_below_threshold_is_zero():
    params = SimParams(P0=0.3, alpha=50.0, V_th=0.01, dt=1e-3)
    assert p_set(np.array([0.005]), params)[0] == 0.0


def test_p_set_matches_formula_above_threshold():
    params = SimParams(P0=0.3, alpha=50.0, V_th=0.01, dt=1e-3)
    V_mem = np.array([0.02])
    expected = params.dt * params.P0 * (1 - np.exp(-params.alpha * (V_mem - params.V_th)))
    np.testing.assert_allclose(p_set(V_mem, params), expected)


def test_p_reset_matches_formula():
    params = SimParams(P_decay=0.02, dt=1e-3)
    I_mem = np.array([0.5])
    expected = params.dt * params.P_decay * I_mem ** 2
    np.testing.assert_allclose(p_reset(I_mem, params), expected)


def test_p_reset_clipped_at_one():
    params = SimParams(P_decay=1.0, dt=1.0)
    assert p_reset(np.array([1000.0]), params)[0] == 1.0


def test_p_set_monotonic_in_voltage():
    params = SimParams()
    V = np.linspace(0, 0.1, 50)
    p = p_set(V, params)
    assert np.all(np.diff(p) >= -1e-15)


def test_p_reset_monotonic_in_current_magnitude():
    params = SimParams()
    I = np.linspace(0, 1.0, 50)
    p = p_reset(I, params)
    assert np.all(np.diff(p) >= -1e-15)


def test_probabilities_always_bounded():
    params = SimParams()
    rng = np.random.default_rng(0)
    V = np.abs(rng.uniform(-1, 1, 1000))
    I = rng.uniform(-100, 100, 1000)
    ps = p_set(V, params)
    pr = p_reset(I, params)
    assert np.all((ps >= 0) & (ps <= 1))
    assert np.all((pr >= 0) & (pr <= 1))


def test_set_switching_frequency_matches_analytic_probability():
    rng = np.random.default_rng(123)
    params = SimParams(P0=0.3, alpha=50.0, V_th=0.01, dt=1e-3)
    n_trials = 200_000
    V_mem = np.full(n_trials, 0.02)
    G = np.full(n_trials, params.G_off)
    I_mem = np.zeros(n_trials)

    new_G = update_dynamics(G, V_mem, I_mem, params, rng)
    empirical_p = np.mean(new_G == params.G_on)
    analytic_p = p_set(V_mem[:1], params)[0]

    se = np.sqrt(analytic_p * (1 - analytic_p) / n_trials)
    assert abs(empirical_p - analytic_p) < 5 * se


def test_reset_switching_frequency_matches_analytic_probability():
    rng = np.random.default_rng(456)
    params = SimParams(P_decay=0.05, dt=1e-3)
    n_trials = 200_000
    I_mem = np.full(n_trials, 2.0)
    G = np.full(n_trials, params.G_on)
    V_mem = np.zeros(n_trials)

    new_G = update_dynamics(G, V_mem, I_mem, params, rng)
    empirical_p = np.mean(new_G == params.G_off)
    analytic_p = p_reset(I_mem[:1], params)[0]

    se = np.sqrt(analytic_p * (1 - analytic_p) / n_trials)
    assert abs(empirical_p - analytic_p) < 5 * se


def test_update_dynamics_off_junction_never_resets():
    # V_mem=0 (below V_th) -> P_SET=0, so no SET fires either; I_mem huge
    # would force RESET with near-certainty if this junction were ON. An
    # OFF junction must stay exactly G_off regardless.
    rng = np.random.default_rng(1)
    params = SimParams()
    G = np.array([params.G_off])
    new_G = update_dynamics(G, V_mem=np.array([0.0]), I_mem=np.array([1e6]), params=params, rng=rng)
    assert new_G[0] == params.G_off


def test_update_dynamics_on_junction_never_sets():
    # I_mem=0 -> P_RESET=0, so no RESET fires either; V_mem huge would
    # force SET with near-certainty if this junction were OFF. An ON
    # junction must stay exactly G_on regardless.
    rng = np.random.default_rng(2)
    params = SimParams()
    G = np.array([params.G_on])
    new_G = update_dynamics(G, V_mem=np.array([1000.0]), I_mem=np.array([0.0]), params=params, rng=rng)
    assert new_G[0] == params.G_on


def test_from_step_probs_reproduces_per_step_probabilities():
    # Lamas 2026 Table 1 lists dt*P0 and dt*P_decay (per-step products).
    params = SimParams.from_step_probs(p0_step=0.4, p_decay_step=0.01, dt=1e-3)
    # Saturated SET probability per step is exactly dt*P0 = 0.4.
    np.testing.assert_allclose(p_set(np.array([1e6]), params)[0], 0.4)
    # RESET per step is dt*P_decay*I^2 = 0.01 * I^2 (I in units of I_unit).
    np.testing.assert_allclose(p_reset(np.array([2.0]), params)[0], 0.01 * 4.0)


def test_p_reset_respects_current_unit():
    # With I_unit = 1e-3 A (mA), a 2 mA current must give the same P_RESET as
    # I = 2 in per-step-coefficient units.
    params = SimParams.from_step_probs(0.4, 0.01, dt=1e-3, I_unit=1e-3)
    np.testing.assert_allclose(p_reset(np.array([2e-3]), params)[0], 0.01 * 4.0)


def test_default_current_unit_is_amperes():
    assert SimParams().I_unit == 1.0


def test_p_relax_defaults_to_zero():
    assert SimParams().p_relax == 0.0


def test_spontaneous_relaxation_turns_on_junctions_off_without_bias():
    # No voltage, no current: only the bias-independent decay can act.
    params = SimParams(P0=0.0, P_decay=0.0, p_relax=0.1, dt=1e-3)
    n = 200_000
    G = np.full(n, params.G_on)
    new_G = update_dynamics(G, np.zeros(n), np.zeros(n), params, np.random.default_rng(0))
    frac_off = np.mean(new_G <= params.G_off)
    assert abs(frac_off - 0.1) < 4 * np.sqrt(0.1 * 0.9 / n)


def test_relaxation_combines_with_reset_as_independent_channels():
    # P_RESET = 0.2 (from current) and p_relax = 0.1 -> P(off) = 1 - 0.8*0.9 = 0.28
    params = SimParams.from_step_probs(0.0, 0.2, dt=1e-3, p_relax=0.1)   # I_unit = 1 A
    n = 200_000
    G = np.full(n, params.G_on)
    new_G = update_dynamics(G, np.zeros(n), np.ones(n), params, np.random.default_rng(1))
    frac_off = np.mean(new_G <= params.G_off)
    assert abs(frac_off - 0.28) < 4 * np.sqrt(0.28 * 0.72 / n)


def test_relaxation_does_not_affect_off_junctions():
    params = SimParams(P0=0.0, P_decay=0.0, p_relax=0.5, dt=1e-3)
    G = np.full(1000, params.G_off)
    new_G = update_dynamics(G, np.zeros(1000), np.zeros(1000), params, np.random.default_rng(2))
    np.testing.assert_array_equal(new_G, G)


def test_zero_relaxation_keeps_random_stream_and_results_unchanged():
    # p_relax = 0 must reproduce the previous update bit for bit.
    kw = dict(P0=0.4, alpha=1.0, V_th=0.1, P_decay=1.0, dt=1e-3)
    G = np.where(np.random.default_rng(3).random(5000) < 0.5, 1e-3, 1e-9)
    V = np.full(5000, 0.3); I = np.full(5000, 5e-3)
    a = update_dynamics(G, V, I, SimParams(**kw), np.random.default_rng(4))
    b = update_dynamics(G, V, I, SimParams(**kw, p_relax=0.0), np.random.default_rng(4))
    np.testing.assert_array_equal(a, b)


# ---- fixed lifetime after SET (deterministic spontaneous relaxation) ----
from agnw_sim.dynamics import update_dynamics_lifetime


def _quiet(**kw):
    """No SET, no current-driven RESET: only the lifetime can act."""
    return SimParams(P0=0.0, P_decay=0.0, dt=1e-3, **kw)


def test_lifetime_defaults_to_disabled():
    assert SimParams().lifetime_steps == 0


def test_on_junction_stays_on_exactly_lifetime_steps():
    L = 5
    params = _quiet(lifetime_steps=L)
    G = np.full(10, params.G_on)
    age = np.zeros(10, dtype=int)
    rng = np.random.default_rng(0)
    for call in range(1, L + 1):
        G, age = update_dynamics_lifetime(G, age, np.zeros(10), np.zeros(10), params, rng)
        if call < L:
            assert np.all(G == params.G_on), f"expired too early at call {call}"
    assert np.all(G == params.G_off)          # off after the L-th update
    assert np.all(age == 0)                    # OFF junctions carry no age


def test_set_junction_starts_with_age_zero_and_then_counts():
    # certain SET (p=1), then let it age
    params = SimParams.from_step_probs(1.0, 0.0, dt=1e-3, alpha=1000.0, V_th=0.0, lifetime_steps=3)
    G = np.full(4, params.G_off)
    age = np.zeros(4, dtype=int)
    rng = np.random.default_rng(1)
    G, age = update_dynamics_lifetime(G, age, np.full(4, 1.0), np.zeros(4), params, rng)
    assert np.all(G == params.G_on) and np.all(age == 0)     # just SET: age 0
    quiet = _quiet(lifetime_steps=3)
    for expected_age in (1, 2):
        G, age = update_dynamics_lifetime(G, age, np.zeros(4), np.zeros(4), quiet, rng)
        assert np.all(G == quiet.G_on) and np.all(age == expected_age)
    G, age = update_dynamics_lifetime(G, age, np.zeros(4), np.zeros(4), quiet, rng)
    assert np.all(G == quiet.G_off)


def test_reset_before_expiry_clears_age():
    params = SimParams.from_step_probs(0.0, 1.0, dt=1e-3, lifetime_steps=100)   # certain RESET at I=1 A
    G = np.full(5, params.G_on)
    age = np.full(5, 7, dtype=int)
    G, age = update_dynamics_lifetime(G, age, np.zeros(5), np.ones(5), params, np.random.default_rng(2))
    assert np.all(G == params.G_off) and np.all(age == 0)


def test_lifetime_zero_matches_plain_update_and_stream():
    kw = dict(P0=0.4, alpha=1.0, V_th=0.1, P_decay=1.0, dt=1e-3)
    G = np.where(np.random.default_rng(3).random(3000) < 0.5, 1e-3, 1e-9)
    V = np.full(3000, 0.3); I = np.full(3000, 5e-3)
    plain = update_dynamics(G, V, I, SimParams(**kw), np.random.default_rng(9))
    G2, age2 = update_dynamics_lifetime(G, np.zeros(3000, dtype=int), V, I, SimParams(**kw), np.random.default_rng(9))
    np.testing.assert_array_equal(plain, G2)
