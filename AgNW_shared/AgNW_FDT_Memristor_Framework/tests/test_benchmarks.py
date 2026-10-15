import numpy as np
import pytest

from agnw_sim.benchmarks import two_state_fdtest, renewal_fdtest

S0, R0 = 0.0725, 0.0625          # per-step probabilities of the Lamas preset at V_mem = 0.3 V, I_mem = 2.5 mA


def _dense_renewal(s0, r0, s1, r1, L, n_lags):
    """Independent reference: explicit (L+1)x(L+1) transition matrices, no shift recursion."""
    n = L + 1

    def P(s, r):
        M = np.zeros((n, n))
        M[0, 0], M[0, 1] = 1 - s, s
        for j in range(L - 1):
            M[j + 1, 0], M[j + 1, j + 2] = r, 1 - r
        M[L, 0] = 1.0
        return M

    def stationary(M):
        w, v = np.linalg.eig(M.T)
        p = np.real(v[:, np.argmin(abs(w - 1))])
        return p / p.sum()

    P0, Pe = P(s0, r0), P(s1, r1)
    pi0, pie = stationary(P0), stationary(Pe)
    g = np.r_[0.0, np.ones(L)]
    p0, pe = pi0 @ g, pie @ g
    zg = np.where(g == 1, pe / p0 - 1, (1 - pe) / (1 - p0) - 1)
    c = np.array([np.sum(pi0 * zg * (np.linalg.matrix_power(P0, k) @ zg)) for k in range(n_lags + 1)])
    zm = np.array([(pie @ np.linalg.matrix_power(P0, k)) @ zg for k in range(n_lags + 1)])
    return c, zm


def test_markov_junction_obeys_the_relation_for_any_perturbation():
    # Break caught: a wrong Z, transition matrix or ordering in the two-state curves.
    for s1, r1 in [(0.09, R0), (S0, 0.075), (0.09, 0.075), (0.02, 0.2)]:
        out = two_state_fdtest(S0, R0, s1, r1, 40)
        assert out["delta2"] < 1e-20
        np.testing.assert_allclose(out["c_zz"], out["z_mean"], atol=1e-14)


def test_two_state_z_is_hand_derivable():
    # p0 = s0/(s0+r0) = 0.5 and pe = 0.6 -> Z(ON) = 0.6/0.5 - 1 = 0.2, Z(OFF) = 0.4/0.5 - 1 = -0.2
    out = two_state_fdtest(0.1, 0.1, 0.15, 0.1, 5)
    np.testing.assert_allclose(out["z"], [-0.2, 0.2], atol=1e-15)
    # lag 0: C_ZZ(0) = E_0[Z^2] = 0.04 and <Z(0)> = E_eps[Z] = 0.4 * (-0.2) + 0.6 * 0.2 = 0.04
    assert out["c_zz"][0] == pytest.approx(0.04, abs=1e-15)
    assert out["z_mean"][0] == pytest.approx(0.04, abs=1e-15)


@pytest.mark.parametrize("L", [4, 8, 12])
def test_renewal_curves_match_an_independent_dense_computation(L):
    # Break caught: an error in the extended-chain shift recursion or in the C_ZZ decomposition.
    s1, r1 = 0.09, 0.075
    out = renewal_fdtest(S0, R0, s1, r1, L, 40)
    c, zm = _dense_renewal(S0, R0, s1, r1, L, 40)
    np.testing.assert_allclose(out["c_zz"], c, atol=1e-13)
    np.testing.assert_allclose(out["z_mean"], zm, atol=1e-13)


def test_fixed_lifetime_hides_nothing_from_a_set_only_perturbation():
    # Break caught: treating the hidden age as visible to a SET-side perturbation. SET only alters the
    # memoryless OFF exit, so delta2 is exactly zero even though g is not Markov (audit section 6.2).
    for L in (4, 8, 16, 64):
        assert renewal_fdtest(S0, R0, 0.09, R0, L, 60)["delta2"] < 1e-20


def test_fixed_lifetime_violates_the_relation_for_a_reset_perturbation():
    # Break caught: an estimator/benchmark pair that cannot see the hidden age. Values from the
    # independent dense computation above (L = 8: 0.25 RESET-only, ~0.011 for both channels).
    reset_only = renewal_fdtest(S0, R0, S0, 0.075, 8, 60)["delta2"]
    both = renewal_fdtest(S0, R0, 0.09, 0.075, 8, 60)["delta2"]
    assert reset_only == pytest.approx(0.25, abs=0.01)
    assert both == pytest.approx(0.011, abs=0.002)


def test_a_lifetime_that_never_binds_recovers_the_markov_junction():
    # Break caught: wrong L -> infinity limit (lifetime 400 steps >> 1/r0 = 16 steps).
    ren = renewal_fdtest(S0, R0, 0.09, 0.075, 400, 40)
    mk = two_state_fdtest(S0, R0, 0.09, 0.075, 40)
    assert ren["delta2"] < 1e-12
    np.testing.assert_allclose(ren["c_zz"], mk["c_zz"], atol=1e-10)
