#!/usr/bin/env python3
"""Why does a coarse N4 step fail the Fig. 4 check (scripts/n4_stride_check.py)? Separate the two error sources:
the circuit is solved every `stride` ms, and Lambda is integrated with `sub` explicit Euler substeps at the frozen
junction voltages. If the error came from the Euler step, sub = stride would recover the 1 ms result.

    python scripts/n4_substep_check.py        # needs results/n4_stride_check/fig4_stride1.npz

Output: results/n4_substep_check.json (relative deviation of the calibration metrics from stride 1).
Result 2026-09-27: substeps change nothing (stride 10: A-/A+ +12 % with 10 substeps vs +11 % without), so the error
comes from re-solving the circuit too rarely during SET avalanches; the step must stay at <= 2 ms.
"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.electrodes import two_bands  # noqa: E402
from agnw_rc.fig4 import LOSS_KEYS, STEPS_PER_CYCLE, V_AMP, metrics_fn  # noqa: E402
from agnw_rc.filament import conductance, step_lambda  # noqa: E402
from agnw_rc.multicircuit import MultiTerminalCircuit  # noqa: E402

CASES = ((10, 10), (5, 5), (10, 2))           # (stride, Euler substeps per stride)
OUT = ROOT / "results" / "n4_substep_check.json"
def run(args):
    stride, sub = args
    fp = C.filament_params(stride=1)
    fp_sub = fp.with_(dt=stride * C.DT / sub)
    _, wd = C.network_params()
    net = C.make_network(1)
    mc = MultiTerminalCircuit(net, wd, two_bands(net))
    rng = np.random.default_rng(0)
    lam = np.zeros(len(net.junctions))
    t = np.arange(STEPS_PER_CYCLE // stride) * stride * C.DT
    v_in = V_AMP * np.sin(2 * np.pi * t / (STEPS_PER_CYCLE * C.DT))
    i_tot = np.empty(len(v_in))
    for k, v in enumerate(v_in):
        V = mc.solve(conductance(lam, fp), [v, 0.0], method="pcg")
        i_tot[k] = mc.electrode_currents(V)[0]
        dV = mc.junction_voltage_drops(V)            # frozen for the substeps
        for _ in range(sub):
            lam = step_lambda(lam, dV, fp_sub, rng)
    return (stride, sub), metrics_fn()(v_in, i_tot * 1e3)


def main():
    z = np.load(ROOT / "results" / "n4_stride_check" / "fig4_stride1.npz")
    ref = metrics_fn()(z["V_in"], z["I_total"] * 1e3)
    out = {}
    with ProcessPoolExecutor(len(CASES)) as pool:
        for (stride, sub), m in pool.map(run, CASES):
            out[f"stride{stride}_sub{sub}"] = {k: float((m[k] - ref[k]) / ref[k]) for k in LOSS_KEYS}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
