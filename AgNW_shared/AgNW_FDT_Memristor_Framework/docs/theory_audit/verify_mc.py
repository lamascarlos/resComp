"""Monte-Carlo checks of the exact two-state-junction results against the
package's own Bernoulli dynamics (agnw_sim.dynamics.update_dynamics), at fixed
V_mem and I_mem (no network feedback). Run from the repo root with the project
venv:  .venv/bin/python docs/theory_audit/verify_mc.py

Tests
 (1) one-step mean/variance of the ensemble fraction g=n_ON/N vs exact
     Kramers-Moyal and vs the doc's Eq. 6 / Eq. 7.
 (2) stationary autocorrelation C(n) vs p*(1-p*)(1-s-r)^n.
 (3) step response to dV vs dp_n = ds(1-p*)(1-(1-s-r)^n)/(s+r), using common
     random numbers (identical RNG streams for control/perturbed ensembles).
 (4) FDT ratio chi_step(n)/[C(0)-C(n)] vs the exact constant ds/s (not beta q).
"""
import numpy as np
from agnw_sim.dynamics import SimParams, p_set, p_reset, update_dynamics

params = SimParams.from_step_probs(0.4, 0.01, dt=1e-3, alpha=1.0, V_th=0.1, I_unit=1e-3)
V_MEM, I_MEM = 0.30, 2.5e-3                    # 0.3 V across, 2.5 mA through the junction
s = float(p_set(np.array([V_MEM]), params)[0])
r = float(p_reset(np.array([I_MEM]), params)[0])
lam = s + r
p_star = s / lam
print(f"s = {s:.5f}, r = {r:.5f}, lambda = {lam:.5f}, p* = {p_star:.4f}\n")


def step(G, V, I, rng):
    n = len(G)
    return update_dynamics(G, np.full(n, V), np.full(n, I), params, rng)


# ------------------------------------------------ (1) one-step moments of g = n_ON / N
N, reps, g0 = 50, 200_000, 0.4
n_on = int(round(g0 * N))
rng = np.random.default_rng(11)
G = np.full((reps, N), params.G_off)
G[:, :n_on] = params.G_on
G1 = step(G.ravel(), V_MEM, I_MEM, rng).reshape(reps, N)
dg = ((G1 >= params.G_on).sum(axis=1) - n_on) / N
mean_th = s * (1 - g0) - r * g0
var_km = ((1 - g0) * s * (1 - s) + g0 * r * (1 - r)) / N
var_doc = s * (1 - g0) ** 2 + r * g0 ** 2          # 2 B_doc, Eq. 7 (no 1/N)
se_mean = dg.std() / np.sqrt(reps)
print("(1) one step, N=50, g=0.4")
print(f"    E[dg]  MC {dg.mean():.5f} +- {se_mean:.5f}   Eq.6/exact {mean_th:.5f}")
print(f"    Var[dg] MC {dg.var():.3e}   exact KM {var_km:.3e}   doc Eq.7 (2B) {var_doc:.3e}\n")

# --------------------------- (2)-(4): independent junctions at stationarity, CRN response
M, n_max, dV = 2_000_000, 40, 0.01
rng0 = np.random.default_rng(5)
G0 = np.where(rng0.random(M) < p_star, params.G_on, params.G_off)
ctl, prt = G0.copy(), G0.copy()
rc, rp = np.random.default_rng(99), np.random.default_rng(99)     # common random numbers
g_init = (G0 >= params.G_on).astype(float)
ds = float(p_set(np.array([V_MEM + dV]), params)[0]) - s
C_mc, dp_mc = [], []
for nstep in range(n_max + 1):
    gc = (ctl >= params.G_on).astype(float)
    gp = (prt >= params.G_on).astype(float)
    C_mc.append(np.mean((g_init - g_init.mean()) * (gc - gc.mean())))
    dp_mc.append(gp.mean() - gc.mean())
    ctl = step(ctl, V_MEM, I_MEM, rc)
    prt = step(prt, V_MEM + dV, I_MEM, rp)
C_mc, dp_mc = np.array(C_mc), np.array(dp_mc)
nn = np.arange(n_max + 1)
C_th = p_star * (1 - p_star) * (1 - lam) ** nn
dp_th = ds * (1 - p_star) * (1 - (1 - lam) ** nn) / lam
print("(2) autocorrelation C(n):  n  MC  theory")
for nsel in (0, 1, 5, 10, 20, 40):
    print(f"    {nsel:3d}  {C_mc[nsel]:.5f}  {C_th[nsel]:.5f}")
print(f"    max |MC - theory| = {np.abs(C_mc - C_th).max():.2e}   (statistical ~ {p_star*(1-p_star)/np.sqrt(M):.1e})\n")
# exact solution for the FINITE step ds (perturbed chain relaxes to p*' at rate 1-(lam+ds))
lam_p = lam + ds
p_star_p = (s + ds) / lam_p
dp_exact = p_star_p + (p_star - p_star_p) * (1 - lam_p) ** nn - p_star
print("(3) step response dp_n:  n  MC  linear-response  exact-finite-step")
for nsel in (1, 5, 10, 20, 40):
    print(f"    {nsel:3d}  {dp_mc[nsel]:.6f}  {dp_th[nsel]:.6f}  {dp_exact[nsel]:.6f}")
print(f"    max |MC - linear| = {np.abs(dp_mc - dp_th).max():.2e};  max |MC - exact finite| = "
      f"{np.abs(dp_mc - dp_exact).max():.2e}  (finite step ds/lam = {ds/lam:.1%} explains the linear-theory gap)\n")
den = C_mc[0] - C_mc
ratio = dp_mc[3:] / (dV * den[3:])
s_prime_over_s = ds / dV / s
print("(4) chi_step(n)/[C(0)-C(n)] (per volt), should be n-independent = ds/(dV s):")
print(f"    exact {s_prime_over_s:.4f} 1/V;  analytic d ln s/dV = alpha/(exp(alpha(V-Vth))-1) = "
      f"{1.0/(np.exp(1.0*(V_MEM-0.1))-1):.4f} 1/V")
print(f"    MC   n=3: {ratio[0]:.4f}  n=10: {ratio[7]:.4f}  n=20: {ratio[17]:.4f}  n=40: {ratio[-1]:.4f}")
