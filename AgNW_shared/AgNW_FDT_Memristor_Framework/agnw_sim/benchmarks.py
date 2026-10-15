"""Exact FDTest curves for exactly solvable single junctions at fixed (V_mem, I_mem).

A junction is a two-state chain (OFF, ON): OFF -> ON with probability s per step, ON -> OFF with
probability b per step. The FDTest perturbation is a static change of both switching probabilities,
(s0, r0) -> (s1, r1): s through V_mem, r through I_mem. The unperturbed dynamics use (s0, r0).
With Z(g) = P_eps(g) / P_0(g) - 1 the benchmark curves are

    C_ZZ(n)  = sum_g P_0(g) Z(g) E[Z(g_n) | g_0 = g]     (stationary, unperturbed)
    <Z(n)>   = E_eps[Z(g_n)]                              (start from P_eps, unperturbed dynamics)

and delta2 = sum_n (C_ZZ - <Z>)^2 / sum_n <Z>^2 over n = 0..n_lags. Known results (all pure algebra):
the Markov junction has delta2 = 0 for any perturbation; the fixed-lifetime junction (ON for at most L
steps after SET, hidden age) has delta2 = 0 for a SET-only perturbation (SET only alters the memoryless
OFF exit) and delta2 > 0 whenever r changes.
"""
import numpy as np


def _delta2(c_zz, z_mean):
    return float(np.sum((c_zz - z_mean) ** 2) / np.sum(z_mean ** 2))


def two_state_fdtest(s0, r0, s1, r1, n_lags):
    """Markov two-state junction. Returns dict(z=[Z(OFF), Z(ON)], c_zz, z_mean, delta2) with the
    curves indexed n = 0..n_lags."""
    P = np.array([[1 - s0, s0], [r0, 1 - r0]])
    pi0 = np.array([r0, s0]) / (s0 + r0)
    pie = np.array([r1, s1]) / (s1 + r1)
    z = pie / pi0 - 1.0
    c_zz = np.empty(n_lags + 1)
    z_mean = np.empty(n_lags + 1)
    Pn = np.eye(2)
    for n in range(n_lags + 1):
        c_zz[n] = np.sum(pi0 * z * (Pn @ z))
        z_mean[n] = (pie @ Pn) @ z
        Pn = Pn @ P
    return {"z": z, "c_zz": c_zz, "z_mean": z_mean, "delta2": _delta2(c_zz, z_mean)}


def _stationary(s, r, L):
    """Stationary distribution of the extended chain: (P(OFF), P(ON, age j) for j = 0..L-1)."""
    w = s * (1.0 - r) ** np.arange(L)
    off = 1.0 / (1.0 + w.sum())
    return off, off * w


def _step(off, on_by_age, s, r):
    new_on = np.empty_like(on_by_age)
    new_on[0] = s * off
    new_on[1:] = (1.0 - r) * on_by_age[:-1]
    new_off = off * (1.0 - s) + r * on_by_age[:-1].sum() + on_by_age[-1]
    return new_off, new_on


def renewal_fdtest(s0, r0, s1, r1, L, n_lags):
    """Fixed-lifetime junction observed through g only (the age is hidden): an ON junction turns OFF
    with probability r per step or, at the latest, after L steps. Same return value as two_state_fdtest."""
    off0, on0 = _stationary(s0, r0, L)
    offe, one = _stationary(s1, r1, L)
    p0, pe = on0.sum(), one.sum()
    z_off, z_on = (1.0 - pe) / (1.0 - p0) - 1.0, pe / p0 - 1.0
    dz = z_on - z_off
    c_zz = np.empty(n_lags + 1)
    z_mean = np.empty(n_lags + 1)
    j_off, j_on = 0.0, on0.copy()        # ON-restricted stationary vector: its ON mass is P(g_0 = ON, g_n = ON)
    e_off, e_on = offe, one.copy()       # perturbed stationary vector evolving under the unperturbed chain
    for n in range(n_lags + 1):
        joint = j_on.sum()
        # C_ZZ(n) = dz * [Z(ON) P(ON_0, ON_n) + Z(OFF) P(OFF_0, ON_n)]  (E_0[Z] = 0 removes the constant part)
        c_zz[n] = dz * (z_on * joint + z_off * (p0 - joint))
        z_mean[n] = z_off + dz * e_on.sum()
        j_off, j_on = _step(j_off, j_on, s0, r0)
        e_off, e_on = _step(e_off, e_on, s0, r0)
    return {"z": np.array([z_off, z_on]), "c_zz": c_zz, "z_mean": z_mean, "delta2": _delta2(c_zz, z_mean)}
