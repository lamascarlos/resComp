#!/usr/bin/env python3
"""Operating-point pilot of the working-model network (network seed 1), two protocols, run with the framework's
run_simulation:

1. staircase: V = 0.25 .. 3.0 V up then down in 0.25 V steps, 3000 steps per level. Mean conductance I/V and ON
   fraction over the last 1000 steps of each level (onset and hysteresis of the quasi-static response).
2. steps: 5000 steps at 1.25 V, 5000 at 2.25 V, 15000 at 1.25 V: SET (fast) and relaxation (slow) time scales.

    python scripts/pilot.py   ->  results/pilot.npz
"""
import os
import sys

# one BLAS thread per process: the scripts parallelize over runs with processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_sim.simulate import run_simulation  # noqa: E402

from agnw_rc import config as C  # noqa: E402

SNAP_EVERY = 50


def run(v_steps, seed):
    params, wd = C.network_params()
    n = len(v_steps)
    snaps = np.arange(0, n, SNAP_EVERY)
    sim = run_simulation(C.N_WIRES, C.AREA, C.LENGTH, lambda t: v_steps[min(int(round(t / C.DT)), n - 1)],
                         n, C.DT, params=params, rng=np.random.default_rng(seed), n_sampled_junctions=0,
                         wire_diameter=wd, solver=C.SOLVER, network=C.make_network(1), snapshot_steps=snaps)
    return sim.I_total, snaps, sim.G_snapshots.mean(axis=1)


def main():
    levels = np.round(np.arange(0.25, 3.01, 0.25), 2)
    stair_v = np.concatenate([levels, levels[::-1][1:]])
    per = 3000
    v1 = np.repeat(stair_v, per)
    i1, s1, on1 = run(v1, 1)
    g_level = np.array([np.mean(i1[k * per + 2000:(k + 1) * per] / stair_v[k]) for k in range(len(stair_v))])
    on_level = np.array([np.mean(on1[(s1 >= k * per + 2000) & (s1 < (k + 1) * per)])
                         for k in range(len(stair_v))])

    v2 = np.concatenate([np.full(5000, 1.25), np.full(5000, 2.25), np.full(15000, 1.25)])
    i2, s2, on2 = run(v2, 2)

    out = ROOT / "results" / "pilot.npz"
    np.savez_compressed(out, stair_v=stair_v, per=per, stair_i=i1.astype(np.float32), stair_snaps=s1,
                        stair_on=on1, g_level=g_level, on_level=on_level, step_v=v2,
                        step_i=i2.astype(np.float32), step_snaps=s2, step_on=on2,
                        framework_commit=C.framework_commit())
    for v, g, o in zip(stair_v, g_level, on_level):
        print(f"V={v:.2f}  G={g * 1e3:.3f} mS  ON={o:.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
