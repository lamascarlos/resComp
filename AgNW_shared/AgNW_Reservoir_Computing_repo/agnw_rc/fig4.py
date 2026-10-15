"""N4 calibration against Lamas et al. (2026) Fig. 4 (docs/N4_DESIGN.md section 5).

Protocol of the framework's scripts/replicate_lamas_fig4.py: one cycle of a 3.6 V sine, 50 000 steps of 1 ms,
fresh network, two-band electrodes. Scored with the framework's own envelope metrics
(docs/theory_audit/digitize_lamas_fig4.metrics, imported from the framework checkout, not copied), against the
paper values stored in the framework's docs/theory_audit/fig4_metrics.json.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

import agnw_sim

from . import config as C
from .electrodes import two_bands
from .filament import FilamentParams, conductance, step_lambda
from .multicircuit import MultiTerminalCircuit

FRAMEWORK = Path(agnw_sim.__file__).resolve().parents[1]
AUDIT = FRAMEWORK / "docs" / "theory_audit"
STEPS_PER_CYCLE = 50_000
V_AMP = 3.6
LOSS_KEYS = ("Imax", "onset_V", "area_pos", "ratio")


def metrics_fn():
    """The framework's Fig. 4 envelope metrics function (V in volts, I in mA)."""
    spec = importlib.util.spec_from_file_location("digitize_lamas_fig4", AUDIT / "digitize_lamas_fig4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metrics


def paper_targets() -> dict:
    """{2000: metrics, 3000: metrics} digitized from the paper (framework fig4_metrics.json)."""
    d = json.loads((AUDIT / "fig4_metrics.json").read_text())
    return {2000: d["paper N=2000"], 3000: d["paper N=3000"]}


def sine_protocol(fp: FilamentParams, n_wires: int = 2000, network_seed: int = 1, n_steps: int = STEPS_PER_CYCLE,
                  v_amp: float = V_AMP, dyn_seed: int = 0, record_every: int = 0):
    """Run the Fig. 4 protocol with N4 junctions. Returns (V_in, I_total in A) per step, and optionally the ON
    fraction (|Lambda| >= lambda_crit) every `record_every` integration steps. The integration step is fp.dt
    (stride = fp.dt / 1 ms, review P0.3): the cycle keeps its duration n_steps * 1 ms and V_in, I_total have
    n_steps // stride points."""
    params, wd = C.network_params()
    net = C.make_network(network_seed, n_wires=n_wires)
    mc = MultiTerminalCircuit(net, wd, two_bands(net))
    rng = np.random.default_rng(dyn_seed)
    lam = np.zeros(len(net.junctions))
    stride = int(round(fp.dt / C.DT))                        # coarse integration step (review P0.3)
    n_c = n_steps // stride
    t = np.arange(n_c) * fp.dt
    v_in = v_amp * np.sin(2 * np.pi * t / (n_steps * C.DT))
    n_steps = n_c
    i_tot = np.empty(n_steps)
    on = []
    for i, v in enumerate(v_in):
        V = mc.solve(conductance(lam, fp), [v, 0.0], method="pcg")
        i_tot[i] = mc.electrode_currents(V)[0]
        if record_every and i % record_every == 0:
            on.append(float(np.mean(np.abs(lam) >= fp.lambda_crit)))
        lam = step_lambda(lam, mc.junction_voltage_drops(V), fp, rng)
    return v_in, i_tot, np.array(on)


def loss(m: dict, target: dict, keys=LOSS_KEYS) -> float:
    """Sum of squared relative errors on the calibration metrics."""
    return float(sum(((m[k] - target[k]) / target[k]) ** 2 for k in keys))
