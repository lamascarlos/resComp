"""Exact + Monte-Carlo audit of the two spontaneous-relaxation mechanisms for a single junction at fixed
(V_mem, I_mem): exponential (SimParams.p_relax) and fixed lifetime (SimParams.lifetime_steps).
Run from the repo root with the project venv:  .venv/bin/python docs/theory_audit/verify_relaxation.py [DPI]

Fixed lifetime = alternating renewal process, exact via the extended Markov chain
  states: OFF, (ON, age j=0..L-1);  OFF->(ON,0) w.p. s;  (ON,j)->OFF w.p. r (j<L-1), (ON,L-1)->OFF w.p. 1;
  (ON,j)->(ON,j+1) w.p. 1-r.
Response/correlation: X(n) = s * dp_n / (ds * [C(0)-C(n)]); X == 1 for the two-state Markov chain (exact).
"""
import sys
import numpy as np
from agnw_sim.dynamics import SimParams, p_set, p_reset, update_dynamics, update_dynamics_lifetime


# ------------------------------------------------------------------ exact renewal curves
def stationary(s, r, L):
    """(pi_off, pi_a[0..L-1])."""
    j = np.arange(L)
    w = s * (1 - r) ** j
    pi_off = 1.0 / (1.0 + w.sum())            # w.sum() = s E[D]
    return pi_off, pi_off * w


def _step(off, a, s, r):
    new_off = off * (1 - s) + r * a[:-1].sum() + a[-1]
    new_a = np.empty_like(a)
    new_a[0] = s * off
    new_a[1:] = (1 - r) * a[:-1]
    return new_off, new_a


def renewal_curves(s, r, L, nmax, ds=1.0, dr=0.0):
    """Returns (C0_minus_Cn[0..nmax], dp[0..nmax], p_star): exact autocorrelation deficit of g and the step
    response of P(ON) to a perturbation ds of the SET probability (and dr of the RESET probability)."""
    pi_off, pi_a = stationary(s, r, L)
    p_star = pi_a.sum()
    # C(n) = P(g0=1, gn=1) - p*^2, evolving the ON-restricted stationary vector
    off, a = 0.0, pi_a.copy()
    C = np.empty(nmax + 1)
    # response: d pi_{n+1} = d pi_n P + source
    doff, da = 0.0, np.zeros(L)
    dp = np.empty(nmax + 1)
    for n in range(nmax + 1):
        C[n] = a.sum() - p_star ** 2
        dp[n] = da.sum()
        off, a = _step(off, a, s, r)
        doff, da = _step(doff, da, s, r)
        doff += -ds * pi_off + dr * pi_a[:-1].sum()
        da[0] += ds * pi_off
        da[1:] -= dr * pi_a[:-1]
    return C[0] - C, dp, p_star


def dense_curves(s, r, L, nmax, ds=1.0, dr=0.0):
    """Same quantities from explicit matrices (small L only), to validate the recursion above."""
    n_st = L + 1
    P = np.zeros((n_st, n_st)); dP = np.zeros((n_st, n_st))
    P[0, 0], P[0, 1] = 1 - s, s
    dP[0, 0], dP[0, 1] = -ds, ds
    for j in range(L - 1):
        P[j + 1, 0], P[j + 1, j + 2] = r, 1 - r
        dP[j + 1, 0], dP[j + 1, j + 2] = dr, -dr
    P[L, 0] = 1
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(abs(w - 1))]); pi /= pi.sum()
    g = np.r_[0.0, np.ones(L)]
    p_star = pi @ g
    gt = g - p_star
    C = np.array([pi @ (gt * (np.linalg.matrix_power(P, n) @ gt)) for n in range(nmax + 1)])
    dpi = np.zeros(n_st); dp = np.empty(nmax + 1)
    for n in range(nmax + 1):
        dp[n] = dpi @ g
        dpi = dpi @ P + pi @ dP
    return C[0] - C, dp, p_star


def Y_ratio(r, Cdef, dp_dr, dr=1.0):
    """RESET-side analogue: Y(n) = -r dp_n / (dr [C(0)-C(n)]); == 1 for a two-state Markov junction with b = r."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -r * dp_dr / (dr * Cdef)


def X_ratio(s, Cdef, dp, ds=1.0):
    n = np.arange(len(dp))
    with np.errstate(divide="ignore", invalid="ignore"):
        return s * dp / (ds * Cdef)


if __name__ == "__main__":
    DPI = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print("=== (A) recursion vs explicit matrices, L=8")
    s_, r_ = 0.0725, 0.0625
    for (ds, dr) in ((1e-3, 0.0), (0.0, 1e-3), (1e-3, 2e-3)):
        c1, d1, p1 = renewal_curves(s_, r_, 8, 60, ds, dr)
        c2, d2, p2 = dense_curves(s_, r_, 8, 60, ds, dr)
        print(f"  ds={ds:g} dr={dr:g}: max|C diff| {np.abs(c1 - c2).max():.1e}, max|dp diff| {np.abs(d1 - d2).max():.1e}, "
              f"|p* diff| {abs(p1 - p2):.1e}")

    # ------------------------------------------------------------------ Monte Carlo with the package
    def mc(params, L, M=2_000_000, n_max=60, dV=0.01, V_mem=0.30, I_mem=2.5e-3, seed=5, dI=0.0):
        s = float(p_set(np.array([V_mem]), params)[0]); r = float(p_reset(np.array([I_mem]), params)[0])
        ds = float(p_set(np.array([V_mem + dV]), params)[0]) - s
        dr = float(p_reset(np.array([I_mem + dI]), params)[0]) - r
        rng0 = np.random.default_rng(seed)
        if L:
            pi_off, pi_a = stationary(s, r, L)
            u = rng0.random(M); cdf = np.r_[pi_off, pi_off + np.cumsum(pi_a)]
            state = np.searchsorted(cdf, u)          # 0 = OFF, k = ON with age k-1
            G0 = np.where(state > 0, params.G_on, params.G_off)
            age0 = np.where(state > 0, state - 1, 0)
            step = lambda G, age, V, rr, I=None: update_dynamics_lifetime(G, age, np.full(M, V), np.full(M, I_mem if I is None else I), params, rr)
        else:
            b = 1 - (1 - r) * (1 - params.p_relax)
            G0 = np.where(rng0.random(M) < s / (s + b), params.G_on, params.G_off); age0 = None
            step = lambda G, age, V, rr, I=None: (update_dynamics(G, np.full(M, V), np.full(M, I_mem if I is None else I), params, rr), None)
        ctl, prt, ac, ap = G0.copy(), G0.copy(), age0, None if age0 is None else age0.copy()
        rc, rp = np.random.default_rng(99), np.random.default_rng(99)
        g_init = (G0 >= params.G_on).astype(float)
        Cdef, dp = [], []
        c0 = None
        for n in range(n_max + 1):
            gc = (ctl >= params.G_on).astype(float); gp = (prt >= params.G_on).astype(float)
            cn = np.mean((g_init - g_init.mean()) * (gc - gc.mean()))
            c0 = cn if c0 is None else c0
            Cdef.append(c0 - cn); dp.append(gp.mean() - gc.mean())
            ctl, ac = step(ctl, ac, V_mem, rc); prt, ap = step(prt, ap, V_mem + dV, rp, I_mem + dI)
        return np.array(Cdef), np.array(dp), s, r, (ds, dr)

    print("\n=== (B) fixed lifetime L=8, Monte Carlo (package update_dynamics_lifetime, M=2e6) vs exact")
    prm = SimParams.from_step_probs(0.4, 0.01, dt=1e-3, alpha=1.0, V_th=0.1, I_unit=1e-3, lifetime_steps=8)
    Cd, dp, s, r, (ds, _) = mc(prm, 8)
    Ce, de, ps = renewal_curves(s, r, 8, 60, ds=ds)
    print(f"  s={s:.4f} r={r:.4f} p*={ps:.4f} ds={ds:.5f}")
    print("   n   C0-Cn MC   exact    |   dp_n MC    exact     |  X_MC   X_exact")
    for n in (1, 3, 5, 8, 12, 20, 40):
        print(f"  {n:3d}  {Cd[n]:.5f}  {Ce[n]:.5f}  |  {dp[n]:.6f}  {de[n]:.6f}  |  {s*dp[n]/(ds*Cd[n]):.3f}  {s*de[n]/(ds*Ce[n]):.3f}")
    print(f"  max|MC-exact|: C {np.abs(Cd - Ce).max():.1e} (stat ~2e-4), dp {np.abs(dp - de).max():.1e} (stat ~1e-4)")

    print("\n=== (C) exponential relaxation eps=0.02 (package p_relax), Monte Carlo vs two-state formula")
    prm = SimParams.from_step_probs(0.4, 0.01, dt=1e-3, alpha=1.0, V_th=0.1, I_unit=1e-3, p_relax=0.02)
    Cd, dp, s, r, (ds, _) = mc(prm, 0)
    b = 1 - (1 - r) * (1 - 0.02); lam = s + b; pst = s / lam
    Cth = pst * (1 - pst) * (1 - (1 - lam) ** np.arange(61))          # deficit C(0)-C(n)
    ds_over_s = ds / s
    print(f"  s={s:.4f} r={r:.4f} b={b:.4f} p*={pst:.4f};  expected constant ratio X = 1")
    for n in (3, 10, 20, 40):
        print(f"  n={n:3d}: C-deficit MC {Cd[n]:.5f} theory {Cth[n]:.5f} | X_MC = {s*dp[n]/(ds*Cd[n]):.3f}")

    # ------------------------------------------------------------------ realistic parameters, exact
    print("\n=== (D) exact X(n) = s dp_n/(ds [C(0)-C(n)]) for the fixed-lifetime junction; two-state Markov: X == 1")
    cases = [("s=0.01, r=2.5e-5 (I~0.05 mA), L=100", 0.01, 2.5e-5, 100),
             ("s=0.01, r=2.5e-5, L=1000", 0.01, 2.5e-5, 1000),
             ("s=0.01, r=2.5e-5, L=5000", 0.01, 2.5e-5, 5000),
             ("s=0.01, r=2.5e-5, L=20000", 0.01, 2.5e-5, 20000),
             ("s=0.01, r=6e-2 (RESET dominates), L=20000", 0.01, 0.0625, 20000)]
    curves = {}
    for name, s, r, L in cases:
        nmax = min(6 * L, 120000)
        Cdef, dpv, ps = renewal_curves(s, r, L, nmax)
        X = X_ratio(s, Cdef, dpv)
        curves[name] = (np.arange(nmax + 1), X, L)
        picks = sorted({n for n in (1, 10, 100, 1000, L // 4, L // 2, L, 2 * L, 4 * L) if 0 < n <= nmax})
        print(f"  {name}: p*={ps:.3f}")
        print("     " + "  ".join(f"n={n}: {X[n]:.3f}" for n in picks))
    print("\n=== (E) RESET-side response: Y(n) = -r dp_n/(dr [C(0)-C(n)]) for a perturbation of the current-driven RESET; Markov: Y == 1")
    ycurves = {}
    for name, s, r, L in cases:
        nmax = min(6 * L, 120000)
        Cdef, dpr, ps = renewal_curves(s, r, L, nmax, ds=0.0, dr=1e-3 * r)
        Y = Y_ratio(r, Cdef, dpr, 1e-3 * r)
        ycurves[name] = (np.arange(nmax + 1), Y, L)
        picks = sorted({n for n in (1, 10, 100, 1000, L // 4, L // 2, L, 2 * L, 4 * L) if 0 < n <= nmax})
        print(f"  {name}:   " + "  ".join(f"n={n}: {Y[n]:.3f}" for n in picks))

    print("\n=== (F) Monte Carlo check of (E): fixed lifetime L=8, perturb the current I_mem (package), M=2e6")
    prm = SimParams.from_step_probs(0.4, 0.01, dt=1e-3, alpha=1.0, V_th=0.1, I_unit=1e-3, lifetime_steps=8)
    Cd, dpm, s, r, (_, dr) = mc(prm, 8, dV=0.0, dI=0.2e-3, seed=7)
    Ce, dpe, ps = renewal_curves(s, r, 8, 60, ds=0.0, dr=dr)
    print(f"  r={r:.4f} dr={dr:.5f}")
    print("   n   dp_n MC     exact     |  Y_MC   Y_exact")
    for n in (1, 3, 5, 8, 12, 20, 40):
        print(f"  {n:3d}  {dpm[n]:+.6f}  {dpe[n]:+.6f}  |  {-r*dpm[n]/(dr*Cd[n]):.3f}  {-r*dpe[n]/(dr*Ce[n]):.3f}")
    print(f"  max|MC-exact| dp: {np.abs(dpm - dpe).max():.1e} (stat ~1e-4)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(20, 4.8))
    for name, (n, X, L) in curves.items():
        ax[0].semilogx(n[1:], X[1:], label=name)
    ax[0].axhline(1, color="k", lw=0.8, ls="--"); ax[0].set_ylim(0.0, 2.0)
    ax[0].set_xlabel("n (steps)"); ax[0].set_ylabel("X(n) = s dp_n / (ds [C(0)-C(n)])")
    ax[0].set_title("SET-side perturbation: exactly 1 for fixed lifetime too"); ax[0].legend(fontsize=6); ax[0].grid(alpha=0.3)
    for name, (n, Y, L) in ycurves.items():
        ax[1].semilogx(n[1:], Y[1:], label=name)
    ax[1].axhline(1, color="k", lw=0.8, ls="--"); ax[1].set_ylim(-2, 4)
    ax[1].set_xlabel("n (steps)"); ax[1].set_ylabel("Y(n) = -r dp_n / (dr [C(0)-C(n)])")
    ax[1].set_title("RESET-side perturbation: deviates from 1 (renewal ON sojourn)"); ax[1].legend(fontsize=6); ax[1].grid(alpha=0.3)
    nL, YL, L = ycurves["s=0.01, r=2.5e-5, L=20000"]
    ax[2].plot(nL[1:] / L, YL[1:], color="purple"); ax[2].axhline(1, color="k", lw=0.8, ls="--")
    ax[2].set_xlabel("n / L"); ax[2].set_ylabel("Y(n)"); ax[2].set_title("L=20000 (s=0.01, r=2.5e-5), linear time axis"); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("docs/theory_audit/relaxation_ratio.png", dpi=DPI)
    print("\nwrote docs/theory_audit/relaxation_ratio.png")
