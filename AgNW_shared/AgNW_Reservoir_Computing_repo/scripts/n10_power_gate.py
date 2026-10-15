#!/usr/bin/env python3
"""N10: junction voltage and power of the formed junctions (x > 0.5) after 2 s at a constant drive (tau_s 0.3 s,
network seed 1, 1 ms step). Motivates gating consolidation on power: formed junctions self-regulate near threshold
(~0.07-0.1 V at any drive), their power separates weak from strong drive.

    python scripts/n10_power_gate.py     -> results/n10_power_gate.json
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
from agnw_rc.multicircuit import MultiTerminalCircuit  # noqa: E402
from agnw_rc.volatile import conductance, junction_kd0, step_state  # noqa: E402

OUT = ROOT / "results" / "n10_power_gate.json"
DRIVES = (2.0, 2.5, 3.5)


def run(V0):
    vp = C.volatile_params(None, tau_s=0.3, dt=C.DT)
    _, wd = C.network_params()
    net = C.make_network(1)
    mc = MultiTerminalCircuit(net, wd, two_bands(net))
    kd0 = junction_kd0(vp, len(net.junctions), 1)
    s = np.zeros((2, len(net.junctions)))
    for _ in range(2000):
        G = conductance(s, vp)
        dV = mc.junction_voltage_drops(mc.solve(G, [V0, 0.0], method="pcg"))
        s = step_state(s, dV, vp, kd0)
    on = s[0] > 0.5
    q = lambda a, qs: [float(v) for v in np.quantile(a, qs)] if on.any() else None  # noqa: E731
    return {"drive_V": V0, "n_formed": int(on.sum()), "power_W_q50_90_99_max": q((G * dV ** 2)[on], [.5, .9, .99, 1]),
            "abs_V_q50_90_max": q(np.abs(dV[on]), [.5, .9, 1])}


if __name__ == "__main__":
    with ProcessPoolExecutor(len(DRIVES)) as pool:
        rows = list(pool.map(run, DRIVES))
    OUT.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)
