#!/usr/bin/env python3
"""Integration-step check for the N10 volatile junction (as review P0.3 for N4). The junction update is exact for
any step, so the only error is re-solving the circuit every `stride` ms instead of every 1 ms.

  1. Sine sweep (±5 V, 20 s, tau_s = 0.3 s, network seed 1): current trace at stride 2/5/10 vs stride 1
     (relative RMS on the common time grid; peak current; onset).
  2. Reservoir features: an N10-1-like i.i.d. run (hold 100 ms, 60 samples, tau_s = 0.3 s, 2.75 V window) at
     stride 1 vs 2/5/10: end-of-hold driven current and junction filling x + y (relative RMS, correlation).

    python scripts/n10_stride_check.py [--procs 8]   -> results/n10_stride_check.json
"""
import argparse
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
from agnw_rc.multidrive import simulate_multi  # noqa: E402
from agnw_rc.volatile import conductance, junction_kd0, step_state  # noqa: E402

OUT = ROOT / "results" / "n10_stride_check.json"
STRIDES = (1, 2, 5, 10)
TAU = 0.3


def sweep(stride):
    vp = C.volatile_params(None, tau_s=TAU, dt=stride * C.DT)
    _, wd = C.network_params()
    net = C.make_network(1)
    mc = MultiTerminalCircuit(net, wd, two_bands(net))
    kd0 = junction_kd0(vp, len(net.junctions), 1)
    state = np.zeros((2, len(net.junctions)))
    t = np.arange(20_000 // stride) * stride * C.DT
    v = 5.0 * np.sin(2 * np.pi * t / 20.0)
    I = np.empty(len(v))
    for k, vk in enumerate(v):
        Vn = mc.solve(conductance(state, vp), [vk, 0.0], method="pcg")
        I[k] = mc.electrode_currents(Vn)[0]
        state = step_state(state, mc.junction_voltage_drops(Vn), vp, kd0)
    return t, v, I


def features(stride):
    spec = C.RunSpec("iid", 100, 2.75, 1, 0, n_samples=60, layout="two_bands", encoding="direct",
                     model="volatile", tau_s=TAU, stride=stride, washout=3000, n_virtual=10)
    raw = simulate_multi(spec, save_states=True)
    n_e = int(raw["n_electrodes"])
    i_end = raw["current"][:, -n_e:][:, raw["electrode_roles"] == "in"].sum(axis=1)
    fill = np.minimum(1.0, raw["junction_x"].astype(float) + raw["junction_y"].astype(float))
    return i_end, fill


def task(args):
    kind, stride = args
    return kind, stride, (sweep(stride) if kind == "sweep" else features(stride))


def rel(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return {"rel_rms": float(np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(a ** 2)), 1e-30)),
            "corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.procs) as pool:
        res = list(pool.map(task, [(k, s) for k in ("sweep", "features") for s in STRIDES]))
    sw = {s: r for k, s, r in res if k == "sweep"}
    ft = {s: r for k, s, r in res if k == "features"}
    t1, _, I1 = sw[1]
    out = {"tau_s": TAU, "sweep": {}, "features": {}}
    for s in STRIDES:
        t, _, I = sw[s]
        ref = np.interp(t, t1, I1)
        out["sweep"][str(s)] = {**rel(ref, I), "I_max_rel": float(np.abs(I).max() / np.abs(I1).max() - 1)}
        out["features"][str(s)] = {"current_end": rel(ft[1][0], ft[s][0]), "filling": rel(ft[1][1], ft[s][1])}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
