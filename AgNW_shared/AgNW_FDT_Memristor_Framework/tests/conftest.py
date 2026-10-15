import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------------------- shared FDTest test helpers
# Plain importable helpers (`from conftest import ...`: tests/ is on sys.path in pytest's default import mode).
import numpy as np                                                                   # noqa: E402
from agnw_sim.dynamics import SimParams, p_set, p_reset, update_dynamics, update_dynamics_lifetime   # noqa: E402


def _mc_cycles(params, V0, I0, V1, I1, n_cycles, K, T_relax, T2, n_lags, T4, seed, lifetime=False):
    """Independent-junction Monte Carlo of the four-stage protocol with the package dynamics.
    Perturbed drive (V1, I1) in stages I-II, (V0, I0) in III-IV. Returns g2, g3, g4 with shape (cycle, time, K)."""
    rng = np.random.default_rng(seed)
    M = n_cycles * K
    s0 = float(p_set(np.array([V0]), params)[0])
    r0 = float(p_reset(np.array([I0]), params)[0])
    G = np.where(rng.random(M) < s0 / (s0 + r0), params.G_on, params.G_off)
    age = np.zeros(M, dtype=int)

    def step(G, age, V, I):
        Va, Ia = np.full(M, V), np.full(M, I)
        if lifetime:
            return update_dynamics_lifetime(G, age, Va, Ia, params, rng)
        return update_dynamics(G, Va, Ia, params, rng), age

    on = lambda G: (G >= params.G_on).astype(np.uint8)
    for _ in range(T_relax):
        G, age = step(G, age, V1, I1)
    g2 = np.empty((T2, M), np.uint8)
    for t in range(T2):
        g2[t] = on(G)
        G, age = step(G, age, V1, I1)
    g3 = np.empty((n_lags, M), np.uint8)
    for lag in range(n_lags):                                                # lag 0 = state at the removal instant
        g3[lag] = on(G)
        G, age = step(G, age, V0, I0)
    for _ in range(T_relax):
        G, age = step(G, age, V0, I0)
    g4 = np.empty((T4, M), np.uint8)
    for t in range(T4):
        g4[t] = on(G)
        G, age = step(G, age, V0, I0)
    shape = lambda g: g.reshape(g.shape[0], n_cycles, K).transpose(1, 0, 2)
    return shape(g2), shape(g3), shape(g4)


PRESET = SimParams.from_step_probs(0.4, 0.01, dt=1e-3, alpha=1.0, V_th=0.1, I_unit=1e-3)
V0, I0 = 0.30, 2.5e-3
N_LAGS = 30


def _ar_cycles(n_cycles, hidden, seed, f=0.06, a=0.8, c=0.5, d=0.95, T_r=300, T2=400, n_lags=60, T4=400):
    """x_{t+1} = a x_t (+ c y_t) + force + noise with a slow hidden y (only if `hidden`); constant force f in stages I-II."""
    r = np.random.default_rng(seed)
    x, y = np.zeros(n_cycles), np.zeros(n_cycles)

    def step(force):
        nonlocal x, y
        x_new = a * x + (c * y if hidden else 0.0) + force + 0.3 * r.normal(size=n_cycles)
        y = d * y + 0.3 * r.normal(size=n_cycles)
        x = x_new

    for _ in range(T_r):
        step(f)
    x2 = np.empty((n_cycles, T2))
    for t in range(T2):
        x2[:, t] = x
        step(f)
    x3 = np.empty((n_cycles, n_lags))
    for lag in range(n_lags):
        x3[:, lag] = x
        step(0.0)
    for _ in range(T_r):
        step(0.0)
    x4 = np.empty((n_cycles, T4))
    for t in range(T4):
        x4[:, t] = x
        step(0.0)
    return x2, x3, x4
