"""Stochastic SET/RESET junction switching dynamics."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimParams:
    """Switching parameters.

    P0 and P_decay are RATES (multiplied by dt inside p_set/p_reset). Lamas
    et al. 2026 (Table 1) instead list the per-step PRODUCTS dt*P0 = 0.4 and
    dt*P_decay = 0.01 -- build those with SimParams.from_step_probs(), never
    by passing 0.4 / 0.01 as P0 / P_decay (that applies dt twice).

    I_unit is the size in amperes of the current unit in which P_decay is
    expressed (P_RESET = dt * P_decay * (I / I_unit)^2). The paper does not
    state it; 1.0 (amperes) is the neutral default.

    p_relax is the per-step probability of SPONTANEOUS relaxation (ON -> OFF
    independent of voltage and current; exponential lifetime dt / p_relax).
    Not in the published equations: reported as a third switching mechanism in
    a talk by Lamas (2026), details unconfirmed. 0.0 disables it.

    lifetime_steps is the alternative, deterministic reading of that mechanism:
    a junction turned ON by SET turns OFF after exactly lifetime_steps steps
    unless the current-driven RESET fires first (0 disables it). It needs a
    per-junction age, so use update_dynamics_lifetime() instead of
    update_dynamics(); run_simulation() does this automatically.
    """
    P0: float = 0.1
    alpha: float = 1.0
    V_th: float = 0.01
    P_decay: float = 0.01
    dt: float = 1e-3
    G_on: float = 1e-3
    G_off: float = 1e-9
    I_unit: float = 1.0
    p_relax: float = 0.0
    lifetime_steps: int = 0

    @classmethod
    def from_step_probs(cls, p0_step, p_decay_step, dt=1e-3, **kwargs):
        """Build params from per-step products dt*P0 and dt*P_decay
        (the convention of Lamas et al. 2026, Table 1)."""
        return cls(P0=p0_step / dt, P_decay=p_decay_step / dt, dt=dt, **kwargs)


def p_set(V_mem, params):
    """SET probability (Eq. 1). V_mem must already be |V_i - V_j|."""
    return params.dt * params.P0 * np.clip(
        1 - np.exp(-params.alpha * (V_mem - params.V_th)), 0, None
    )


def p_reset(I_mem, params):
    """RESET probability (Eq. 2)."""
    return np.clip(params.dt * params.P_decay * (I_mem / params.I_unit) ** 2, None, 1)


def update_dynamics(G, V_mem, I_mem, params, rng):
    """Vectorized Bernoulli SET/RESET update. Returns a new conductance
    array; does not mutate G in place."""
    is_off = G <= params.G_off
    is_on = ~is_off

    pset = p_set(V_mem, params)
    preset = p_reset(I_mem, params)
    if params.p_relax:
        # independent channels: OFF unless neither the current-driven RESET nor
        # the spontaneous relaxation fires
        preset = 1.0 - (1.0 - preset) * (1.0 - params.p_relax)

    set_draws = rng.random(len(G))
    reset_draws = rng.random(len(G))

    set_events = is_off & (set_draws < pset)
    reset_events = is_on & (reset_draws < preset)

    new_G = G.copy()
    new_G[set_events] = params.G_on
    new_G[reset_events] = params.G_off
    return new_G


def update_dynamics_lifetime(G, age, V_mem, I_mem, params, rng):
    """update_dynamics plus a deterministic lifetime: an ON junction turns OFF
    once `params.lifetime_steps` steps have elapsed since its SET (age counts
    steps since SET; OFF junctions have age 0). Returns (new_G, new_age).
    With lifetime_steps == 0 this is exactly update_dynamics (same RNG use)."""
    new_G = update_dynamics(G, V_mem, I_mem, params, rng)
    was_off = G <= params.G_off
    now_on = new_G > params.G_off
    new_age = np.where(now_on & ~was_off, age + 1, 0)       # SET this step -> 0; still ON -> +1
    if params.lifetime_steps:
        expired = now_on & (new_age >= params.lifetime_steps)
        new_G = new_G.copy()
        new_G[expired] = params.G_off
        new_age[expired] = 0
    return new_G, new_age
