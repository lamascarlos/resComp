"""Symbolic audit of AgNW_FDT_Memristor_Framework.md (Eqs. 2, 8, 10, 12-14, 16,
Sec. 10.1) and derivation of the exact two-state-junction FDT results.
Run with a Python that has sympy (system python3):  python3 docs/theory_audit/verify_symbolic.py
Uses verify_helpers.py (copy of the math-derivation skill's verify.py)."""
import os
import sys
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_helpers import VerifyResult, check_equal, check_limit, check_ode_solution, report  # noqa: E402
from sympy.physics.units import (Quantity, ampere, convert_to, joule, kilogram, meter,  # noqa: E402
                                 second, volt)

BASE = [kilogram, meter, second, ampere]


def refute(lhs, rhs):
    """Confirm a CLAIMED identity is false (check_equal must return ok=False)."""
    r = check_equal(lhs, rhs)
    return VerifyResult(r.ok is False, r.method,
                        "claimed identity is refuted (difference != 0)" if r.ok is False
                        else f"NOT refuted: {r.detail}")


def dimensionless(expr):
    conv = convert_to(expr, BASE)
    ok = not conv.atoms(Quantity)
    return VerifyResult(ok, "symbolic", f"reduces to {conv}")


def not_dimensionless(expr):
    r = dimensionless(expr)
    return VerifyResult(not r.ok, r.method, "not dimensionless: " + r.detail if not r.ok else r.detail)


checks = []
t = sp.symbols("t", positive=True)

# ---------------------------------------------------------------- Eq. 8 (sign)
mu, k, T = sp.symbols("mu k T", positive=True)
beta = 1 / T
sigma2 = sp.solve(sp.Eq(2 * (-mu * k) * sp.Symbol("s2") + 2 * mu * T, 0), sp.Symbol("s2"))[0]  # Lyapunov
C_ou = sigma2 * sp.exp(-mu * k * t)
R_ou = mu * sp.exp(-mu * k * t)     # impulse response of x' = -mu k x + mu F + noise
checks += [
    ("OU: stationary variance = T/k (Lyapunov)", check_equal(sigma2, T / k)),
    ("OU: R' = -mu k R (impulse response ODE)", check_ode_solution(sp.diff(R_ou, t) + mu * k * R_ou)),
    ("Eq.8 correct: R(t) = -beta dC/dt", check_equal(R_ou, -beta * sp.diff(C_ou, t))),
    ("Eq.8 as written: chi = +beta dC/dt", refute(R_ou, beta * sp.diff(C_ou, t))),
    ("Eq.8 step form: chi_step = beta[C(0)-C(t)] = int_0^t R",
     check_equal(beta * (C_ou.subs(t, 0) - C_ou), sp.integrate(R_ou.subs(t, sp.Symbol("u", positive=True)),
                                                                 (sp.Symbol("u", positive=True), 0, t)))),
]

# -------------------------------------- exact two-state junction (continuous time)
a, b = sp.symbols("a b", positive=True)         # rates OFF->ON, ON->OFF
a1, b1 = sp.symbols("a1 b1", real=True)         # d a/dV, d b/dV
lam = a + b
p = a / lam
R2 = (a1 * (1 - p) - b1 * p) * sp.exp(-lam * t)
C2 = p * (1 - p) * sp.exp(-lam * t)
checks += [
    ("2-state: R' = -lam R", check_ode_solution(sp.diff(R2, t) + lam * R2)),
    ("2-state: C' = -lam C", check_ode_solution(sp.diff(C2, t) + lam * C2)),
    ("2-state: int_0^inf R dt = d p*/dV (independent static check)",
     check_equal(sp.integrate(R2, (t, 0, sp.oo)), (a1 * b - b1 * a) / lam**2)),
]
# Equilibrium (local detailed balance a/b = exp(beta q V)): FDT holds exactly.
V, q, kk, be = sp.symbols("V q kk be", positive=True)
a_eq, b_eq = kk * sp.exp(be * q * V / 2), kk * sp.exp(-be * q * V / 2)
p_eq = a_eq / (a_eq + b_eq)
R_eq = (sp.diff(a_eq, V) * (1 - p_eq) - sp.diff(b_eq, V) * p_eq) * sp.exp(-(a_eq + b_eq) * t)
C_eq = p_eq * (1 - p_eq) * sp.exp(-(a_eq + b_eq) * t)
checks.append(("2-state, detailed-balance rates: R = -(beta q) dC/dt (FDT holds)",
               check_equal(R_eq, -be * q * sp.diff(C_eq, t))))
checks.append(("2-state, general: R/(-dC/dt) = a1/a - b1/b = d ln(a/b)/dV  (exact, any t)",
               check_equal(sp.simplify(R2 / (-sp.diff(C2, t))), a1 / a - b1 / b)))
# AgNW junction: only SET depends on V (RESET depends on I): b1 = 0.
checks.append(("2-state, b1=0: R/(-dC/dt) = a1/a, independent of t",
               check_equal(sp.simplify(R2.subs(b1, 0) / (-sp.diff(C2, t))), a1 / a)))
al, Vth, Vv, P0dt = sp.symbols("alpha V_th V P0dt", positive=True)
s_of_V = P0dt * (1 - sp.exp(-al * (Vv - Vth)))
checks += [
    ("SET rate: d ln s/dV = alpha/(exp(alpha(V-Vth))-1)",
     check_equal(sp.diff(s_of_V, Vv) / s_of_V, al / (sp.exp(al * (Vv - Vth)) - 1))),
    ("SET rate near threshold: (V-Vth) s'/s -> 1",
     check_limit((Vv - Vth) * al / (sp.exp(al * (Vv - Vth)) - 1), Vv, Vth, 1)),
]
# Discrete time (what the simulator does): p_{n+1} = (1-r) p_n + s (1-p_n).
n = sp.symbols("n", integer=True, nonnegative=True)
s_, r_, ds = sp.symbols("s r ds", positive=True)
lam_d = s_ + r_
p_d = s_ / lam_d
dp = lambda m: ds * (1 - p_d) * (1 - (1 - lam_d) ** m) / lam_d
Cd = lambda m: p_d * (1 - p_d) * (1 - lam_d) ** m
checks += [
    ("discrete: stationary p* = s/(s+r)", check_equal(p_d, (1 - r_) * p_d + s_ * (1 - p_d))),
    ("discrete: step response recursion dp_{n+1} = (1-lam) dp_n + ds(1-p*)",
     check_equal(dp(n + 1), (1 - lam_d) * dp(n) + ds * (1 - p_d))),
    ("discrete: dp_0 = 0", check_equal(dp(0), 0)),
    ("discrete: C(n) = p*(1-p*)(1-lam)^n obeys C(n+1) = (1-lam) C(n)",
     check_equal(Cd(n + 1), (1 - lam_d) * Cd(n))),
    ("discrete: step response / [C(0)-C(n)] = ds/s, independent of n",
     check_equal(sp.simplify(dp(3) / (Cd(0) - Cd(3))), ds / s_)),
]

# ------------------------------------------- Eqs. 5-7, 10 (Fokker-Planck audit)
g = sp.symbols("g", real=True)
A, B, rho = (sp.Function(nm)(g) for nm in ("A", "B", "rho"))
integrand = g * (-sp.diff(A * rho, g) + sp.diff(B * rho, g, 2))
antideriv = -g * A * rho + g * sp.diff(B * rho, g) - B * rho
resid10 = sp.simplify(sp.expand((sp.diff(antideriv, g) - (integrand - A * rho)).doit()))
checks.append(("Eq.10: d/dg[-gA rho + g (B rho)' - B rho] = g*RHS_FP - A rho, i.e. "
               "int g d_t rho = [-gJ - B rho]_0^1 + <A>, J = A rho - (B rho)'",
               VerifyResult(resid10 == 0, "symbolic (abstract functions)", f"residual = {resid10}")))
# Ensemble fraction g = n/N of independent two-state junctions: exact per-step variance.
N = sp.symbols("N", positive=True)
var_step = (N * (1 - g) * s_ * (1 - s_) + N * g * r_ * (1 - r_)) / N**2
KM_leading = (s_ * (1 - g) + r_ * g) / N
B_doc = sp.Rational(1, 2) * (s_ * (1 - g) ** 2 + r_ * g ** 2)     # Eq. 7
B_km = sp.Rational(1, 2) * KM_leading
from sympy.stats import Bernoulli, variance  # noqa: E402
checks += [
    ("Bernoulli(s) variance = s(1-s) (sympy.stats)",
     check_equal(variance(Bernoulli("x", s_)), s_ * (1 - s_))),
    ("KM: N*Var[dg] = [s(1-g)+r g] - [s^2(1-g)+r^2 g]  (independent junctions; leading order s,r<<1). "
     "MC check of the full variance: verify_mc.py",
     check_equal(var_step * N, (s_ * (1 - g) + r_ * g) - (s_**2 * (1 - g) + r_**2 * g))),
    ("Eq.7 B(g) has (1-g)^2, g^2 and no 1/N: differs from Kramers-Moyal B",
     refute(B_doc, B_km)),
    ("Eq.7 vs KM(N=1) differ in g-dependence but coincide at the boundaries g=0 and g=1",
     check_equal(sp.simplify((B_doc - sp.Rational(1, 2) * (s_ * (1 - g) + r_ * g)).subs(g, 0)) +
                 sp.simplify((B_doc - sp.Rational(1, 2) * (s_ * (1 - g) + r_ * g)).subs(g, 1)), 0)),
    ("...and differ in the interior (g=1/2)",
     refute(B_doc.subs(g, sp.Rational(1, 2)), sp.Rational(1, 2) * (s_ * (1 - g) + r_ * g).subs(g, sp.Rational(1, 2)))),
]

# -------------------------------------------------------- dimensions (Eqs. 2,12,13,14,16)
Ppd_doc = second / ampere**2            # as printed in the doc (s A^-2)
Ppd_fix = 1 / (second * ampere**2)      # dimensionally consistent with Eq. 2
P0u, alu = 1 / second, 1 / volt
dt, I, V_ = second, ampere, volt
checks += [
    ("Eq.2 with doc units [P_decay]=s A^-2: dt P_decay I^2 dimensionless?",
     not_dimensionless(dt * Ppd_doc * I**2)),
    ("Eq.2 with [P_decay]=s^-1 A^-2: dt P_decay I^2 dimensionless", dimensionless(dt * Ppd_fix * I**2)),
    ("Eq.14: P_decay/(P0 alpha) dimensionless? (X-1 must be)", not_dimensionless(Ppd_fix / (P0u * alu))),
    ("Eq.13: (I^2/V^2) P_decay/(P0 alpha) dimensionless?",
     not_dimensionless((I**2 / V_**2) * Ppd_fix / (P0u * alu))),
    ("Eq.12: [int <gdot gdot> ds / (kB T)] has units of chi = 1/V?",
     not_dimensionless((1 / (second * joule)) * volt)),
    ("Eq.16: [oint I dV] = A V has units of energy?", not_dimensionless(ampere * volt / joule)),
]

# ----------------------------------------------------------- Eq. 16: loop area != dissipation
V0, Rr, w = sp.symbols("V0 R omega", positive=True)
Vt = V0 * sp.sin(w * t)
I_lin = Vt / Rr
per = 2 * sp.pi / w
loop = sp.integrate(I_lin * sp.diff(Vt, t), (t, 0, per))
diss = sp.integrate(I_lin * Vt, (t, 0, per))
checks += [
    ("Eq.16: linear resistor: oint I dV = 0 over a cycle", check_equal(loop, 0)),
    ("...but dissipated energy int I V dt = pi V0^2/(R w) > 0", check_equal(diss, sp.pi * V0**2 / (Rr * w))),
]

# ----------------------------------------------------------- Sec. 10.1: N_c
nc_L2 = sp.Float("5.63726")        # literature value for 2D sticks (quoted from memory; verify citation)
Nc = nc_L2 * sp.Float("1e-6") / sp.Float("70e-6") ** 2
checks.append(("Sec.10.1: N_c = 5.637 A/L^2 for A=1 mm^2, L=70 um is ~1150 (doc: ~1200)",
               VerifyResult(abs(float(Nc) - 1150) < 5, "symbolic", f"N_c = {float(Nc):.1f}")))

# ================================================================ relaxation mechanisms (added 2026-09-20)
# (a) exponential spontaneous relaxation: extra ON->OFF probability eps, independent of V and I.
#     Per step an ON junction turns OFF with b = 1-(1-r)(1-eps) (independent channels).
eps, ds_, dr_ = sp.symbols("epsilon ds dr", positive=True)
b_ = 1 - (1 - r_) * (1 - eps)
lam_e = s_ + b_
p_e = s_ / lam_e
db_ = sp.diff(b_, r_) * dr_                       # a perturbation of the current-driven RESET
dp_e = lambda m: ((1 - p_e) * ds_ - p_e * db_) * (1 - (1 - lam_e) ** m) / lam_e
C0_Cn = lambda m: p_e * (1 - p_e) * (1 - (1 - lam_e) ** m)
checks += [
    ("relax(exp): stationary p* = s/(s+b)", check_equal(p_e, (1 - b_) * p_e + s_ * (1 - p_e))),
    ("relax(exp): step response solves dp_{n+1} = (1-lam) dp_n + (1-p*) ds - p* db",
     check_equal(dp_e(n + 1), (1 - lam_e) * dp_e(n) + (1 - p_e) * ds_ - p_e * db_)),
    ("relax(exp): chi_step/[C(0)-C(n)] = ds/s - db/b  (any n, any eps)",
     check_equal(sp.simplify(dp_e(3) / C0_Cn(3)), ds_ / s_ - db_ / b_)),
    ("relax(exp): SET-only perturbation (dr=0): ratio = ds/s, independent of eps and r",
     check_equal(sp.simplify((dp_e(3) / C0_Cn(3)).subs(dr_, 0)), ds_ / s_)),
]
x_ = sp.symbols("x", real=True)                   # x = dI/I from the network
checks.append(("relax(exp): RESET part of the ratio is diluted by eps: db/b -> 2 r/(r+eps) dI/I  (b~r+eps, r~I^2)",
               check_equal(dr_.subs(dr_, 2 * r_ * x_) / (r_ + eps), 2 * r_ / (r_ + eps) * x_)))

# (b) fixed lifetime L after SET: alternating renewal process, ON sojourn D = min(K, L), K~Geom(r);
#     OFF sojourn ~ Geom(s). Extended Markov chain: OFF, (ON, age j = 0..L-1).
Ls = sp.symbols("L", integer=True, positive=True)
k = sp.symbols("k", integer=True, positive=True)
E_D = sp.summation((1 - r_) ** (k - 1), (k, 1, Ls))
checks.append(("renewal: E[D] = sum_{k=1}^L (1-r)^(k-1) = (1-(1-r)^L)/r",
               check_equal(sp.simplify(E_D), (1 - (1 - r_) ** Ls) / r_)))
# explicit L=4 chain, states [OFF, a0, a1, a2, a3]
Lc = 4
pi = sp.symbols("pi0:%d" % (Lc + 1), positive=True)
P = sp.zeros(Lc + 1, Lc + 1)
P[0, 0], P[0, 1] = 1 - s_, s_
for j in range(Lc - 1):
    P[j + 1, 0], P[j + 1, j + 2] = r_, 1 - r_
P[Lc, 0] = 1
sol = sp.solve(list(sp.Matrix([pi]) * P - sp.Matrix([pi])) + [sum(pi) - 1], pi, dict=True)[0]
EDc = sum((1 - r_) ** j for j in range(Lc))
pi0_claim = 1 / (1 + s_ * EDc)
checks += [
    ("renewal (L=4): stationary pi_OFF = 1/(1 + s E[D])", check_equal(sol[pi[0]], pi0_claim)),
    ("renewal (L=4): stationary pi_ON,j = pi_OFF s (1-r)^j  (j=2)",
     check_equal(sol[pi[3]], pi0_claim * s_ * (1 - r_) ** 2)),
    ("renewal: p* = E[D]/(E[D]+1/s)  (L=4)", check_equal(1 - sol[pi[0]], EDc / (EDc + 1 / s_))),
]
# L -> infinity: t = (1-r)^L -> 0 because 0 < 1-r < 1, so p*(t) -> s/(s+r)
t_ = sp.symbols("t", positive=True)
E_t = (1 - t_) / r_
checks.append(("renewal: L->inf recovers the two-state p* = s/(s+r)  (p*(t) as t=(1-r)^L -> 0)",
               check_limit(E_t / (E_t + 1 / s_), t_, 0, s_ / (s_ + r_))))
# detailed balance is violated by construction (a directed cycle: OFF -> a0 -> ... -> a_{L-1} -> OFF)
flux_fwd = sol[pi[2]] * P[2, 3]          # a1 -> a2
flux_bwd = sol[pi[3]] * P[3, 2]          # a2 -> a1 (impossible: age only increases)
checks += [
    ("renewal: a1->a2 has probability flux, a2->a1 has none => detailed balance violated",
     refute(sp.simplify(flux_fwd - flux_bwd), 0)),
    ("renewal: reverse transition probability P(a2 -> a1) = 0", check_equal(P[3, 2], 0)),
]

# RESET-side response of the renewal junction (explicit L=4 matrices; independent of the closed form below)
dr_s = sp.symbols("dr_s", positive=True)
dP = sp.zeros(Lc + 1, Lc + 1)
for j in range(Lc - 1):
    dP[j + 1, 0], dP[j + 1, j + 2] = dr_s, -dr_s
g_vec = sp.Matrix([0] + [1] * Lc)
pi_vec = sp.Matrix([[sol[pi[i]] for i in range(Lc + 1)]])
p_on = sp.simplify((pi_vec * g_vec)[0])
gt = g_vec - p_on * sp.ones(Lc + 1, 1)
C_of = lambda m: (pi_vec * sp.Matrix([gt[i] * (P ** m * gt)[i] for i in range(Lc + 1)]))[0]
C0_C1 = sp.simplify(C_of(0) - C_of(1))
dp1 = sp.simplify((pi_vec * dP * g_vec)[0])              # d pi_1 = pi* dP (first update after the perturbation starts)
Y1_matrix = sp.simplify(-r_ * dp1 / (dr_s * C0_C1))
pi_last = sol[pi[Lc]]
Y1_claim = r_ * (p_on - pi_last) / (r_ * (p_on - pi_last) + pi_last)
checks.append(("renewal (L=4): RESET-side ratio at n=1, Y(1) = -r dp_1/(dr[C(0)-C(1)]) = fraction of ON-exit flux due to RESET "
               "= r(p*-pi_{L-1}) / [r(p*-pi_{L-1}) + pi_{L-1}]   (from explicit matrices)",
               check_equal(Y1_matrix, sp.simplify(Y1_claim))))
checks.append(("renewal: Y(1) = 1 when the lifetime never binds (pi_{L-1} = 0)  [algebraic substitution]",
               VerifyResult(sp.simplify(Y1_claim.subs(pi_last, 0)) == 1, "symbolic", "Y(1) at pi_{L-1}=0 equals 1")))

# SET-side response of the renewal junction: X(n) = s dp_n/(ds [C(0)-C(n)]) == 1 (explicit L=3 matrices, n = 1..4)
L3 = 3
pis = sp.symbols("q0:%d" % (L3 + 1), positive=True)
P3 = sp.zeros(L3 + 1, L3 + 1); dP3 = sp.zeros(L3 + 1, L3 + 1)
P3[0, 0], P3[0, 1] = 1 - s_, s_
dP3[0, 0], dP3[0, 1] = -ds_, ds_
for j in range(L3 - 1):
    P3[j + 1, 0], P3[j + 1, j + 2] = r_, 1 - r_
P3[L3, 0] = 1
sol3 = sp.solve(list(sp.Matrix([pis]) * P3 - sp.Matrix([pis])) + [sum(pis) - 1], pis, dict=True)[0]
pv = sp.Matrix([[sol3[pis[i]] for i in range(L3 + 1)]])
g3 = sp.Matrix([0] + [1] * L3)
p3 = sp.simplify((pv * g3)[0]); gt3 = g3 - p3 * sp.ones(L3 + 1, 1)
Cf = lambda m: (pv * sp.Matrix([gt3[i] * (P3 ** m * gt3)[i] for i in range(L3 + 1)]))[0]
dpi_n = sp.zeros(1, L3 + 1)                    # d pi_{n+1} = d pi_n P + pi* dP
for nn_ in range(1, 5):
    dpi_n = dpi_n * P3 + pv * dP3
    dpn = sp.simplify((dpi_n * g3)[0])
    Xn = sp.simplify(s_ * dpn / (ds_ * sp.simplify(Cf(0) - Cf(nn_))))
    checks.append((f"renewal (L=3): SET-side ratio X(n={nn_}) = 1 exactly (explicit matrices)", check_equal(Xn, 1)))

print(report(checks))
n_fail = sum(1 for _, r in checks if r.ok is not True)
print(f"\n{len(checks) - n_fail}/{len(checks)} checks PASS (refutation checks pass when the doc's claim is shown false)")
sys.exit(1 if n_fail else 0)
