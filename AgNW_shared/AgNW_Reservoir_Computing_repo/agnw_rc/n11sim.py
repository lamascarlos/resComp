"""N11: the measured protocols of Milano et al. 2025 on the strip (docs/PLAN_N11_CALIBRATION.md sections 6-7):
a constant voltage on the driven band of a relaxed device, the network conductance G = I / V sampled every 0.6 s
(Figs. 2a, 2b and inset, 3, 5a), and relaxation at 10 mV after a drive (Fig. 1e)."""
from functools import lru_cache

import numpy as np

from . import config as C
from .multicircuit import MultiTerminalCircuit
from .strip import make_strip
from .volatile import conductance, rupture_rates, step_state

SAMPLE_DT = 0.6      # s   measured sampling (series_stats.GRID_DT)
V_RELAX = 0.01       # V   read voltage of the measured relaxation (Fig. 1e)


@lru_cache(maxsize=4)
def strip_circuit(seed: int, length_mm: float, width_mm: float):
    """(Strip, MultiTerminalCircuit) of a strip; cached per process (the 15.6 mm strip takes about 3 s to build)."""
    s = make_strip(seed, length_mm, width_mm)
    _, wd = C.network_params()
    return s, MultiTerminalCircuit(s.network, wd, s.electrodes)


def run_constant(mc, vp, kd0, V, T, state=None, rng=None, sample_dt=SAMPLE_DT, t0=0.0):
    """Hold the driven band at V (> 0) volts for T s from `state` (None = relaxed: x = y = 0). G = I / V is sampled every
    sample_dt from the first step on. Also returns the time-averaged mean random-rupture rate per junction at the samples,
    the median power of the formed junctions (x > 0.5) at the end (nan if none), and the final state."""
    if V <= 0:
        raise ValueError("run_constant needs V > 0 (G = I / V)")
    state = np.zeros((2, len(kd0))) if state is None else np.array(state, dtype=float)
    n, every = int(round(T / vp.dt)), int(round(sample_dt / vp.dt))
    Vf = np.array([V, 0.0])
    ts, Gs, rr = [], [], []
    for i in range(n):
        Gj = conductance(state, vp)
        Vn = mc.solve(Gj, Vf, method="pcg")
        dV = mc.junction_voltage_drops(Vn)
        if i % every == 0:
            ts.append(t0 + i * vp.dt)
            Gs.append(mc.electrode_currents(Vn)[mc.fixed_electrodes[0]] / V)
            rr.append(float(np.mean(rupture_rates(Gj * dV ** 2, vp))) if vp.rupture_rate > 0 else 0.0)
        state = step_state(state, dV, vp, kd0, rng)
    Gj = conductance(state, vp)
    dV = mc.junction_voltage_drops(mc.solve(Gj, Vf, method="pcg"))
    formed = state[0] > 0.5
    return {"t": np.array(ts), "G": np.array(Gs), "mean_rupture_rate": float(np.mean(rr)) if rr else 0.0,
            "P_formed_median": float(np.median((Gj * dV ** 2)[formed])) if formed.any() else float("nan"),
            "state": state}
