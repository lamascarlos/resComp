"""N10: volatile, two-time-scale junction constrained by measured AgNW data (docs/N10_DESIGN.md).

Each junction carries two states in [0, 1]:

  x, the volatile (thin, labile) filament: rate balance of Milano et al., Nat. Mater. 21, 195 (2022), plus rupture
     by dissipated power (Diaz Schneider et al. 2024):
       dx/dt = k_p(V) [P <= P_c] (1 - x) - [k_d(V) + k_P(P)] x
       k_p = k_p0 exp(eta_p v_scale |V|),  k_d = k_d0,j exp(-eta_d v_scale |V|),  k_P = k_P0 [P > P_c],  P = G V^2
     Milano's parameters were fitted to a whole network between two electrodes (one effective memristor, threshold
     k_p = k_d0 at 0.49 V). v_scale rescales the voltage axis to a single junction of our network (threshold
     0.49 V / v_scale); v_scale = 1 is Milano's model exactly.
     Above P_c growth stops (Joule melting): the Milano k_p is astronomically large at |V| >~ 1 V (5e24 1/s at
     2 V), so a competing rupture rate alone could never limit the filament.
     At zero bias x relaxes with tau_s,j = 1 / k_d0,j, log-normal across junctions (median tau_s, spread
     tau_spread in natural-log units), fixed per network (a device property).
  y, the consolidated (thick, stable) filament (Milano et al. 2023: amplitude-dependent long-term memory):
       dy/dt = k_c x^n (1 - y) [G V^2 > P_c2] - y / tau_l
     gated on the junction power: formed junctions self-regulate near threshold (~0.07-0.1 V at any drive), so a
     voltage gate cannot tell strong from weak drive, while their power can (<= 4e-8 W at 2 V, median 2e-6 W at
     3.5 V on the network; 2026-09-27).

Random ruptures (N10-R; N11 adds state-dependent laws): a fraction rupture_frac of x is lost at Poisson events of rate
r_j per junction, r_j = r0 (constant), r0 (P_j / P0)^alpha (power) or r0 exp(P_j / P*) (thermal), P_j = G_j V_j^2.

Conductance (tunnelling through the remaining gap; exponential in a gap that shrinks with the filament):
  "log":    G = G_off (G_on / G_off)^min(1, x + y)
  "linear": G = G_off + (G_on - G_off) min(1, x + y)      (Milano's g G_max + (1 - g) G_min, for the special case)

Non-polar: the rates depend on |V| only (Juan, 2026-09-27). Integration: for a step with fixed junction voltage both
equations are linear in their own state, so each is advanced EXACTLY over the step (x first with the old G for the
power term, then y with the old x): unconditionally stable for any dt.
"""
from dataclasses import dataclass, replace

import numpy as np

EXP_CAP = 700.0             # cap of the exponent in k_p (float64 overflow guard)
RUPTURE_LAWS = ("constant", "power", "thermal")


@dataclass(frozen=True)
class VolatileParams:
    k_p0: float = 2.56e-6       # 1/s   potentiation constant (Milano 2022, via Cipollini et al. 2024 Table 1)
    eta_p: float = 34.9         # 1/V   potentiation transition rate (idem)
    tau_s: float = 1.0 / 64.9   # s     median zero-bias relaxation time 1/k_d0 (Milano 2022: k_d0 = 64.9 1/s)
    eta_d: float = 5.59         # 1/V   voltage suppression of relaxation (idem)
    v_scale: float = 1.0        #       voltage-axis scale network -> junction (1 = Milano; set in config)
    k_max: float = np.inf       # 1/s   cap on k_p (finite ion-drift speed; inf = Milano)
    tau_spread: float = 0.0     # log-normal sigma of tau_s across junctions (0 = homogeneous)
    P_c: float = 1e-3           # W     power above which the volatile filament ruptures (Diaz Schneider 2024)
    k_P0: float = 1e3           # 1/s   rupture rate above P_c (0 disables)
    k_c: float = 0.0            # 1/s   consolidation rate (0 = pure volatile)
    n_c: float = 2.0            #       consolidation order in x
    P_c2: float = 1e-6          # W     consolidation needs the junction power above this
    tau_l: float = 1e3          # s     decay time of the consolidated filament
    G_on: float = 1e-3          # S
    G_off: float = 1e-9         # S
    conductance: str = "log"    # "log" (tunnelling-like) or "linear" (Milano)
    sigma: float = 0.0          # additive noise on x per sqrt(s) (0 = deterministic)
    rupture_rate: float = 0.0   # 1/s  random rupture events of the volatile filament (Poisson, per junction)
    rupture_frac: float = 1.0   #      fraction of x lost at a rupture (1 = complete: abrupt conductance drop)
    rupture_law: str = "constant"   # N11: random-rupture rate per junction: "constant" r0, "power" r0 (P/P0)^alpha,
                                    #      "thermal" r0 exp(P/P*), with P = G V^2 the junction power (r0 = rupture_rate)
    rupture_alpha: float = 0.0      # N11: power-law exponent (>= 0; 0 = constant)
    rupture_P0: float = 1e-6        # W    N11: reference power of the power law (set from the strip in stage 2)
    rupture_Pstar: float = np.inf   # W    N11: power scale of the thermal law (inf = constant)
    dt: float = 1e-3            # s

    def __post_init__(self):
        if self.rupture_law not in RUPTURE_LAWS:
            raise ValueError(f"unknown rupture_law {self.rupture_law!r}; one of {RUPTURE_LAWS}")
        if self.rupture_alpha < 0:
            raise ValueError("rupture_alpha must be >= 0")

    def with_(self, **kw) -> "VolatileParams":
        return replace(self, **kw)


def junction_kd0(p: VolatileParams, n_junctions: int, network_seed: int) -> np.ndarray:
    """Per-junction zero-bias relaxation rates 1/tau_s,j: log-normal around 1/tau_s, fixed by the network seed."""
    if p.tau_spread <= 0:
        return np.full(n_junctions, 1.0 / p.tau_s)
    rng = np.random.default_rng(70_000 + network_seed)
    return np.exp(rng.normal(np.log(1.0 / p.tau_s), p.tau_spread, n_junctions))


def conductance(state, p: VolatileParams):
    """Junction conductance (S) for state = (x, y) stacked as an array of shape (2, n)."""
    f = np.minimum(1.0, state[0] + state[1])
    if p.conductance == "linear":
        return p.G_off + (p.G_on - p.G_off) * f
    return p.G_off * (p.G_on / p.G_off) ** f


def rates(V, G, p: VolatileParams, kd0):
    """(k_p, k_d, k_P) for junction voltage drops V and conductances G (k_p is 0 where the power exceeds P_c)."""
    a = np.abs(np.asarray(V, dtype=float))
    k_p = np.minimum(p.k_p0 * np.exp(np.minimum(p.eta_p * p.v_scale * a, EXP_CAP)), p.k_max)
    k_d = kd0 * np.exp(-p.eta_d * p.v_scale * a)
    if p.k_P0 > 0:
        hot = G * a ** 2 > p.P_c
        k_p = np.where(hot, 0.0, k_p)                       # melting: no growth while overpowered
        k_P = p.k_P0 * hot
    else:
        k_P = np.zeros_like(a)
    return k_p, k_d, k_P


def rupture_rates(P, p: VolatileParams):
    """Random-rupture rate (1/s) of each junction at junction powers P (W), docs/PLAN_N11_CALIBRATION.md section 4."""
    P = np.asarray(P, dtype=float)
    if p.rupture_law == "power":
        return p.rupture_rate * (P / p.rupture_P0) ** p.rupture_alpha
    if p.rupture_law == "thermal":
        return p.rupture_rate * np.exp(np.minimum(P / p.rupture_Pstar, EXP_CAP))
    return np.full(P.shape, p.rupture_rate)


def step_state(state, V, p: VolatileParams, kd0, rng=None):
    """Advance (x, y) by p.dt at fixed junction voltages V (exact for x and for y given x)."""
    x, y = state
    G = conductance(state, p)
    k_p, k_d, k_P = rates(V, G, p, kd0)
    k = k_p + k_d + k_P
    x_inf = k_p / k
    x_new = x_inf + (x - x_inf) * np.exp(-k * p.dt)
    if p.sigma > 0:
        if rng is None:
            raise ValueError("sigma > 0 needs an rng")
        x_new = np.clip(x_new + p.sigma * np.sqrt(p.dt) * rng.standard_normal(x.shape), 0.0, 1.0)
    if p.rupture_rate > 0:
        # abrupt random ruptures (measured relaxation shows sudden drops and plateaus, Diaz-Alvarez 2019)
        if rng is None:
            raise ValueError("rupture_rate > 0 needs an rng")
        hit = rng.random(x.shape) < rupture_rates(G * np.asarray(V, dtype=float) ** 2, p) * p.dt
        x_new = np.where(hit, x_new * (1.0 - p.rupture_frac), x_new)
    if p.k_c > 0:
        up = p.k_c * x ** p.n_c * (G * np.asarray(V, dtype=float) ** 2 > p.P_c2)
        k_y = up + 1.0 / p.tau_l
        y_inf = up / k_y
        y_new = y_inf + (y - y_inf) * np.exp(-k_y * p.dt)
    else:
        y_new = y * np.exp(-p.dt / p.tau_l)
    return np.stack([x_new, y_new])
