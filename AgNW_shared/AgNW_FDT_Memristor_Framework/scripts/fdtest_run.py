"""Chained FDTest cycles on one fixed network for one dynamics seed (spec sections 3-4).

  python -m scripts.fdtest_run N --v0 V0 --eps EPS --t-relax T --t-samp T --cycles C --seed S
         [--network-seed 1] [--k 1000] [--record-every 10] [--n-lags 300] [--p-relax 1e-4] [--out validation_output/fdtest]

Saves validation_output/fdtest/run_N{N}_seed{S}.npz: I_total (every step), G_sampled (uint8 ON flags of the K
recorded junctions every record_every steps), sampled_junction_ids, G_snapshots (full states at the end of every
stage II and IV), snapshot_steps, schedule_json. Standalone driver (no unit tests; see tests/test_fdtest_smoke.py)."""
import argparse
import dataclasses
import json
import os
import time

import numpy as np

from agnw_sim.io import _git_commit_hash
from agnw_sim.simulate import run_simulation
from agnw_sim.fdtest_protocol import Schedule
from scripts.fdtest_common import build_setup, AREA, LENGTH, DT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("N", type=int)
    p.add_argument("--v0", type=float, required=True); p.add_argument("--eps", type=float, required=True)
    p.add_argument("--t-relax", type=int, required=True); p.add_argument("--t-samp", type=int, required=True)
    p.add_argument("--cycles", type=int, required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--network-seed", type=int, default=1); p.add_argument("--k", type=int, default=1000)
    p.add_argument("--record-every", type=int, default=10); p.add_argument("--n-lags", type=int, default=300)
    p.add_argument("--p-relax", type=float, default=1e-4); p.add_argument("--out", default="validation_output/fdtest")
    a = p.parse_args(argv)

    schedule = Schedule(V0=a.v0, eps=a.eps, T_relax=a.t_relax, T_samp=a.t_samp, n_cycles=a.cycles,
                        record_every=a.record_every, n_lags=a.n_lags)
    net, ids, params, wd = build_setup(a.N, a.network_seed, a.k, a.p_relax)
    t0 = time.time()
    r = run_simulation(n_wires=a.N, area=AREA, length=LENGTH, waveform_fn=schedule.waveform_fn(DT),
                       n_steps=schedule.n_steps, dt=DT, params=params, rng=np.random.default_rng(a.seed),
                       network=net, sampled_junction_ids=ids, record_every=a.record_every,
                       snapshot_steps=schedule.snapshot_steps(), solver="cached", wire_diameter=wd,
                       sample_on_flags=True)  # uint8 ON flags directly: float64 G_sample would be ~1.7 GB per process
    meta = {**dataclasses.asdict(schedule), "N": a.N, "network_seed": a.network_seed, "dynamics_seed": a.seed,
            "k": len(ids), "p_relax": a.p_relax, "wire_diameter": wd, "params": dataclasses.asdict(params),
            "git_commit": _git_commit_hash(), "wall_time_s": time.time() - t0, "solver_stats": r.solver_stats}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"run_N{a.N}_seed{a.seed}.npz")
    np.savez_compressed(path, I_total=r.I_total, G_sampled=r.G_sample,
                        sampled_junction_ids=ids, G_snapshots=r.G_snapshots, snapshot_steps=r.snapshot_steps,
                        schedule_json=json.dumps(meta))
    print(f"saved {path} ({meta['wall_time_s']:.0f} s, solver stats {r.solver_stats})")


if __name__ == "__main__":
    main()
