"""N4 continuous-filament junction model: rate law, limits of the conductance, Simmons formula, noise."""
import numpy as np
import pytest

from agnw_rc.filament import (E, G0, H, M_E, FilamentParams, conductance, gap, lambda_rate, step_lambda,
                              tunnel_conductance)

P = FilamentParams()


def test_rate_law_regions():
    lam = np.array([0.005, 0.005, 0.005, -0.005, 0.005])
    V = np.array([0.03, -0.03, 0.007, 0.0, 0.002])
    r = lambda_rate(V, lam, P)
    assert r[0] == pytest.approx(0.02)                    # growth, positive polarity
    assert r[1] == pytest.approx(-0.02)                   # growth towards negative Lambda (bipolar)
    assert r[2] == 0.0                                    # hold between V_reset and V_set
    assert r[3] == pytest.approx(10 * 0.005)              # dissolution of a negative state: towards 0
    assert r[4] == pytest.approx(-10 * 0.003)             # dissolution of a positive state


def test_step_clips_and_does_not_cross_zero_when_dissolving():
    p = P.with_(dt=1.0)
    assert step_lambda(np.array([0.001]), np.array([0.0]), p)[0] == 0.0       # would overshoot below 0
    assert step_lambda(np.array([0.0149]), np.array([1.0]), p)[0] == P.lambda_max
    assert step_lambda(np.array([0.0]), np.array([0.0]), p)[0] == 0.0          # sgn(0) = 0: stays closed


def test_conductance_limits():
    assert conductance(0.0, P) == pytest.approx(P.G_off, rel=1e-6)            # 5 nm gap: tunnelling negligible
    closed = 1 / (1 / P.G_on) + P.G_off
    assert conductance(P.lambda_crit, P) == pytest.approx(closed)
    assert conductance(P.lambda_max, P) == pytest.approx(closed)
    lam = np.linspace(0, P.lambda_crit, 200)
    assert np.all(np.diff(conductance(lam, P)) >= 0)                           # monotone in |Lambda|
    assert conductance(-0.009, P) == conductance(0.009, P)                    # symmetric


def test_simmons_formula_independent_evaluation():
    s = 0.5e-9
    m = 0.99 * M_E
    phi = 0.82 * E
    # J = (e^2 / (h^2 s)) (3/2) sqrt(2 m phi) V exp(-4 pi s sqrt(2 m phi) / h)  (Simmons low-bias limit)
    J_over_V = E ** 2 / (H ** 2 * s) * 1.5 * np.sqrt(2 * m * phi) * np.exp(-4 * np.pi * s * np.sqrt(2 * m * phi) / H)
    assert tunnel_conductance(s, P) == pytest.approx(0.17e-18 * J_over_V, rel=1e-12)
    assert gap(0.0, P) == pytest.approx(5e-9)
    assert gap(P.lambda_crit, P) == 0.0
    assert G0 == pytest.approx(7.748091729e-5, rel=1e-9)


def test_noise_increment_statistics():
    p = P.with_(sigma=0.01, dt=1e-3)
    rng = np.random.default_rng(0)
    lam0 = np.full(200_000, 0.007)
    V = np.full(200_000, 0.007)                                               # hold region: no drift
    d = step_lambda(lam0, V, p, rng) - lam0
    assert d.mean() == pytest.approx(0.0, abs=2e-6)
    assert d.std() == pytest.approx(0.01 * np.sqrt(1e-3), rel=0.01)
    with pytest.raises(ValueError):
        step_lambda(lam0, V, p)                                               # sigma > 0 needs an rng


def test_unipolar_growth_survives_polarity_reversal():
    p = P.with_(unipolar=True)
    lam = np.array([0.005, 0.005, 0.0])
    r = lambda_rate(np.array([-0.03, 0.03, 0.002]), lam, p)
    assert r[0] == pytest.approx(0.02) and r[1] == pytest.approx(0.02)     # both polarities grow Lambda
    assert r[2] == 0.0                                                     # closed junction does not dissolve below 0
    bip = lambda_rate(np.array([-0.03]), np.array([0.005]), P)
    assert bip[0] == pytest.approx(-0.02)                                  # bipolar: negative V drives Lambda down
