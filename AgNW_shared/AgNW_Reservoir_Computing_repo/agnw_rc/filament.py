"""N4: continuous-filament junction model (docs/N4_DESIGN.md).

Each junction carries a signed filament state Lambda (V s) instead of a binary ON/OFF state. The model follows the
threshold-driven bipolar memristive switch of Hochstetter et al., Nat. Commun. 12, 4008 (2021), Methods Eqs. 5-7
(with the Simmons formula as corrected in the 2024 Author Correction), plus an optional additive noise term:

  dLambda/dt = (|V| - V_set) sgn(V)            if |V| > V_set          (filament growth, polarity-driven)
             = 0                               if V_reset <= |V| <= V_set
             = b (|V| - V_reset) sgn(Lambda)   if |V| < V_reset        (spontaneous dissolution)
  |Lambda| <= Lambda_max
  (Euler-Maruyama step: Lambda += dt f(V, Lambda) + sigma sqrt(dt) xi, xi ~ N(0, 1))

  gap s = max(Lambda_crit - |Lambda|, 0) s_max / Lambda_crit
  G_t(s) = A 3 sqrt(2 m*) e^2 sqrt(e phi_V) / (2 h^2 s) exp(-4 pi s sqrt(2 m* e phi_V) / h)   (Simmons, low bias)
  G(Lambda) = 1 / (1/G_t + R_f) + G_off

Unipolar variant (option, not in the source): growth (|V| - V_set) always increases Lambda >= 0, so a filament
survives a polarity reversal, as the framework's binary junctions do (their SET/RESET depend on |V| and I^2 only).

G(Lambda) rises from G_off (gap s_max, closed filament) by tunnelling as the gap shrinks, and saturates at
1/R_f + G_off when the filament bridges the gap (|Lambda| >= Lambda_crit). In this project R_f = 1/G_on and
G_off are the framework's G_on / G_off by default, so the conductance range is the one the framework's current scale
was calibrated with (see the design document for this choice).
"""
from dataclasses import dataclass, replace

import numpy as np

E = 1.602176634e-19          # C
H = 6.62607015e-34           # J s
M_E = 9.1093837015e-31       # kg
G0 = 2 * E ** 2 / H          # conductance quantum, S


@dataclass(frozen=True)
class FilamentParams:
    """Parameters of the continuous-filament junction model. Defaults: Hochstetter et al. (2021) Methods, except
    the conductance range (R_f = 1/G_on, G_off), which defaults to the framework's (1e-3 S, 1e-9 S)."""
    V_set: float = 0.01          # V, growth threshold
    V_reset: float = 0.005       # V, dissolution threshold (V_set / 2 in the source)
    b: float = 10.0              # dissolution rate factor (dimensionless)
    lambda_crit: float = 0.01    # V s, state at which the gap closes
    lambda_max: float = 0.015    # V s, cap on |Lambda|
    s_max: float = 5e-9          # m, gap at Lambda = 0 (insulating layer thickness)
    phi: float = 0.82            # eV, tunnelling barrier height
    area: float = 0.17e-18       # m^2, tunnelling area (0.17 nm^2)
    m_eff: float = 0.99          # effective electron mass / m_e
    G_on: float = 1e-3           # S, filament conductance 1/R_f (source: G0 = 7.75e-5 S)
    G_off: float = 1e-9          # S, parallel leakage 1/R_off (source: G0/1000)
    sigma: float = 0.0           # V s^(1/2), additive noise on Lambda (0 = deterministic, as in the source)
    unipolar: bool = False       # True: growth depends on |V| only (Lambda >= 0, survives polarity reversal)
    dt: float = 1e-3             # s

    def with_(self, **kw) -> "FilamentParams":
        return replace(self, **kw)


def lambda_rate(V, lam, p: FilamentParams):
    """Deterministic dLambda/dt (V/s ... i.e. V s per s) for junction voltage drops V and states lam."""
    V = np.asarray(V, dtype=float)
    lam = np.asarray(lam, dtype=float)
    a = np.abs(V)
    if p.unipolar:
        grow = a - p.V_set                                   # always towards larger Lambda >= 0
        decay = p.b * (a - p.V_reset) * (lam > 0)
    else:
        grow = (a - p.V_set) * np.sign(V)
        decay = p.b * (a - p.V_reset) * np.sign(lam)
    return np.where(a > p.V_set, grow, np.where(a < p.V_reset, decay, 0.0))


def step_lambda(lam, V, p: FilamentParams, rng=None):
    """One Euler(-Maruyama) step. The deterministic dissolution never crosses zero (it stops at Lambda = 0), and
    |Lambda| is clipped at lambda_max."""
    lam = np.asarray(lam, dtype=float)
    new = lam + p.dt * lambda_rate(V, lam, p)
    # dissolution drives |Lambda| to 0 but not through it
    dissolving = np.abs(np.asarray(V)) < p.V_reset
    new = np.where(dissolving & (np.sign(new) != np.sign(lam)), 0.0, new)
    if p.sigma > 0:
        if rng is None:
            raise ValueError("sigma > 0 needs an rng")
        new = new + p.sigma * np.sqrt(p.dt) * rng.standard_normal(lam.shape)
    return np.clip(new, -p.lambda_max, p.lambda_max)


def gap(lam, p: FilamentParams):
    """Tunnelling gap s (m) for filament states lam."""
    return np.maximum(p.lambda_crit - np.abs(lam), 0.0) * p.s_max / p.lambda_crit


def tunnel_conductance(s, p: FilamentParams):
    """Simmons low-bias tunnelling conductance G_t(s) (S); +inf at s = 0."""
    s = np.asarray(s, dtype=float)
    m = p.m_eff * M_E
    phi_J = p.phi * E
    with np.errstate(divide="ignore", over="ignore"):
        pref = p.area * 3 * np.sqrt(2 * m * phi_J) * E ** 2 / (2 * H ** 2 * s)
        return pref * np.exp(-4 * np.pi * s * np.sqrt(2 * m * phi_J) / H)


def conductance(lam, p: FilamentParams):
    """Junction conductance G(Lambda) = 1/(R_t + R_f) + G_off (S)."""
    Gt = tunnel_conductance(gap(lam, p), p)
    with np.errstate(divide="ignore"):
        Rt = np.where(np.isinf(Gt), 0.0, 1.0 / Gt)
    return 1.0 / (Rt + 1.0 / p.G_on) + p.G_off
