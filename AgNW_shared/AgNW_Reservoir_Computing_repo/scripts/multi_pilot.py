#!/usr/bin/env python3
"""Quasi-static staircase for the multi-electrode layouts (NEXT_STEPS N5), to choose their operating windows.

All driven electrodes at the same voltage V (grounded ones at 0), V = 0.25 .. 3.0 V up and back down in 0.25 V
steps, 3000 steps per level, stochastic framework dynamics, network seed 1. Recorded per level (mean over its last
1000 steps): total conductance I_in / V (I_in = total current of the driven electrodes) and the ON fraction.
"two_bands" (the framework layout through the multi-terminal circuit) should reproduce scripts/pilot.py.

    python scripts/multi_pilot.py   ->  results/multi_pilot.npz   (~15 min, one process per layout)
"""
import os
import sys

# one BLAS thread per process: the scripts parallelize over runs with processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_sim.dynamics import update_dynamics  # noqa: E402

from agnw_rc import config as C  # noqa: E402
from agnw_rc.electrodes import make_layout  # noqa: E402
from agnw_rc.multicircuit import MultiTerminalCircuit  # noqa: E402

LAYOUTS = ("two_bands", "split_bands", "pads")
LEVELS = np.round(np.arange(0.25, 3.01, 0.25), 2)
PER, TAIL = 3000, 1000


def staircase(layout):
    params, wd = C.network_params()
    net = C.make_network(1)
    els = make_layout(layout, net)
    mc = MultiTerminalCircuit(net, wd, els)
    fixed = mc.fixed_electrodes
    driven = np.array([els[i].role == "in" for i in fixed])
    driven_all = np.array([e.role == "in" for e in els])
    rng = np.random.default_rng(zlib_seed := 1234)
    G = np.full(len(net.junctions), params.G_off)
    v_stair = np.concatenate([LEVELS, LEVELS[::-1][1:]])
    g, on = [], []
    for v in v_stair:
        Vf = np.where(driven, v, 0.0)
        gs, ons = [], []
        for s in range(PER):
            V = mc.solve(G, Vf)
            if s >= PER - TAIL:
                gs.append(mc.electrode_currents(V)[driven_all].sum() / v)
                ons.append(np.mean(G > params.G_off))
            dV = mc.junction_voltage_drops(V)
            G = update_dynamics(G, np.abs(dV), G * dV, params, rng)
        g.append(np.mean(gs)); on.append(np.mean(ons))
        print(f"{layout:11s} V={v:.2f}  G={g[-1] * 1e3:.3f} mS  ON={on[-1]:.3f}", flush=True)
    return layout, v_stair, np.array(g), np.array(on)


def main():
    with ProcessPoolExecutor(max_workers=len(LAYOUTS)) as pool:
        res = list(pool.map(staircase, LAYOUTS))
    out = {"v": res[0][1], "per": PER, "tail": TAIL}
    for layout, _, g, on in res:
        out[f"g_{layout}"] = g
        out[f"on_{layout}"] = on
    np.savez_compressed(ROOT / "results" / "multi_pilot.npz", **out, framework_commit=C.framework_commit())
    print("saved results/multi_pilot.npz")


if __name__ == "__main__":
    main()
