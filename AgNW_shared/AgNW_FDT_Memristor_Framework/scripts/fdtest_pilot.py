"""Fixed-voltage pilot for the FDTest operating point (spec section 6).

  python -m scripts.fdtest_pilot run N V [--n-steps 60000] [--network-seed 1] [--seed 1] [--k 1000]
  python -m scripts.fdtest_pilot summarize N [--block 5000] [--tail 30000]

`run` simulates one voltage (abrupt turn-on, cached solver) and saves validation_output/fdtest/pilot_N{N}_V{V}.npz;
`summarize` reads every pilot file of that N, prints stationarity and saturation per voltage, chooses V0 and
eps (agnw_sim.fdtest_protocol) and writes pilot_choice_N{N}.json. Standalone driver (no unit tests)."""
import argparse
import json
import glob
import os
import sys
import time

import numpy as np

from agnw_sim.simulate import run_simulation
from agnw_sim import fdtest_protocol as fp
from scripts.fdtest_common import build_setup, AREA, LENGTH, DT


def cmd_run(a):
    net, ids, params, wd = build_setup(a.N, a.network_seed, a.k, a.p_relax)
    t0 = time.time()
    r = run_simulation(n_wires=a.N, area=AREA, length=LENGTH, waveform_fn=lambda t: a.V, n_steps=a.n_steps, dt=DT,
                       params=params, rng=np.random.default_rng(a.seed), network=net, sampled_junction_ids=ids,
                       record_every=10, solver="cached", wire_diameter=wd)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"pilot_N{a.N}_V{a.V:g}.npz")
    np.savez_compressed(path, I_total=r.I_total, V_in=r.V_in, G_on=(r.G_sample > params.G_off).astype(np.uint8),
                        wall_time=time.time() - t0, solver_stats=json.dumps(r.solver_stats))
    print("saved", path, f"({time.time() - t0:.0f} s)")


def cmd_summarize(a):
    files = sorted(glob.glob(os.path.join(a.out, f"pilot_N{a.N}_V*.npz")))
    if not files:
        sys.exit(f"no pilot files for N={a.N} in {a.out}")
    summaries, table = {}, []
    for f in files:
        z = np.load(f)
        V = float(z["V_in"][0])
        s = fp.pilot_summary(z["I_total"], z["V_in"], z["G_on"], block=a.block, tail=a.tail)
        summaries[V] = s
        table.append({"V": V, **{k: s[k] for k in ("mean_geff", "sigma_geff", "drift", "frac_intermediate")}})
        print(f"V = {V:4.2f} V  g_eff = {s['mean_geff']:.3e} S  sigma = {s['sigma_geff']:.2e}  drift = {s['drift']:.3f}  "
              f"junctions with 0.1 < p_on < 0.9: {s['frac_intermediate']:.3f}")
    V0 = fp.choose_v0(summaries)
    if V0 is None:
        print("no stationary voltage among the pilots (drift above tolerance): lengthen the pilots or raise V")
        sys.exit(2)
    volts = sorted(summaries)
    eps = fp.choose_epsilon(volts, [summaries[v]["mean_geff"] for v in volts], summaries[V0]["sigma_geff"], V0)
    out = {"N": a.N, "V0": V0, "eps": eps, "table": table}
    path = os.path.join(a.out, f"pilot_choice_N{a.N}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"chosen V0 = {V0} V, eps = {eps:.4f} V (mean response 0.5 sigma) -> {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("N", type=int); r.add_argument("V", type=float)
    r.add_argument("--n-steps", type=int, default=60000); r.add_argument("--network-seed", type=int, default=1)
    r.add_argument("--seed", type=int, default=1); r.add_argument("--k", type=int, default=1000)
    r.add_argument("--p-relax", type=float, default=1e-4); r.add_argument("--out", default="validation_output/fdtest")
    s = sub.add_parser("summarize")
    s.add_argument("N", type=int); s.add_argument("--block", type=int, default=5000)
    s.add_argument("--tail", type=int, default=30000); s.add_argument("--out", default="validation_output/fdtest")
    a = p.parse_args(argv)
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
