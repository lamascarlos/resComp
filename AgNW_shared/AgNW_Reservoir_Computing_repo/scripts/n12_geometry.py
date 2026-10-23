#!/usr/bin/env python3
"""N12: non-uniform wire placement (agnw_rc/substrate.py, docs/N12_GEOMETRY.md) - connectivity before any simulation.

  stats    every geometry in substrate.GEOMETRIES (or --only NAME,...) x network seeds 1-3: junctions, mean degree,
           isolated wires, largest connected wire cluster, percolation between the 4x4 pads and between the two bands
           -> results/n12_geometry_stats.json (+ a table on stdout)
  figure   the networks of seed 1 for each geometry, with the 4x4 pads (orange driven, blue grounded) and the cluster
           centres (2 sigma circles) -> figures/fig36_n12_geometries.png
  sweep    onset voltage of each geometry (network seed 1, quasi-static 0 -> 3 V ramp, as N10-L)
           -> results/n12_sweep.json
  run      the N10-E reservoir on each geometry: 4x4 checkerboard pads, random input weights, 16 electrode currents,
           volatile junction tau_s 0.3 s, spread 1, 100 ms holds, 800 inputs, windows 1.05 / 1.25 x onset (width
           0.45 x onset), 3 networks x 2 replicates + sine/square, no ruptures -> results/raw/*.npz
  report   scores and G0/G1/G2 per geometry (agnw_rc.criteria, as N10-L); "uniform" is the control through this same
           pipeline, the published N10-E row a cross-check -> results/n12_criteria.json, figures/fig37_n12_scores.png.
           Rows of geometries not in --only are kept from an existing n12_criteria.json, so machines that ran different
           geometries can each report theirs and merge through git (cluster/run_n12_local.sh).

    python scripts/n12_geometry.py stats [--only clusters4,aligned_x] [--seeds 1,2,3]
    python scripts/n12_geometry.py figure
    python scripts/n12_geometry.py sweep [--only ...] [--procs N] [--layout split_bands] [--n-wires 3000]
                                                                     (-> n12_sweep[_<layout>][_N<n>].json)
    python scripts/n12_geometry.py run [--only ...] [--procs N] [--share i/n]
    python scripts/n12_geometry.py report [--only ...]
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.electrodes import make_layout  # noqa: E402
from agnw_rc.substrate import GEOMETRIES, network_stats  # noqa: E402

OUT = ROOT / "results" / "n12_geometry_stats.json"
FIG = ROOT / "figures" / "fig36_n12_geometries.png"
RAW = ROOT / "results" / "raw"
SWEEP = ROOT / "results" / "n12_sweep.json"
CRIT = ROOT / "results" / "n12_criteria.json"
FIG_SCORES = ROOT / "figures" / "fig37_n12_scores.png"
TAU, SPREAD, HOLD, LAYOUT, OBS = 0.3, 1.0, 100, "pads", "electrodes"     # the N10-E reservoir (as N10-L at 1 mm)
RC_GEOMETRIES = list(GEOMETRIES)       # "uniform" = control through this same pipeline (N10-E row: cross-check)


def stats(names, seeds):
    rows = []
    for name in names:
        for seed in seeds:
            net = C.make_network(seed, geometry=name)
            bands = network_stats(net)
            pads = network_stats(net, make_layout("pads", net))
            rows.append({"geometry": name, "network_seed": seed, **{k: bands[k] for k in (
                "n_wires", "n_junctions", "mean_degree", "isolated_fraction", "largest_cluster_fraction")},
                "bands_percolate": bands["percolates"], "pads_percolate": pads["percolates"],
                "pads_in_linked": pads["in_linked"], "pads_gnd_linked": pads["gnd_linked"]})
            r = rows[-1]
            print(f"{name:16s} seed {seed}  junctions {r['n_junctions']:5d}  degree {r['mean_degree']:5.2f}  "
                  f"isolated {r['isolated_fraction']:5.1%}  largest {r['largest_cluster_fraction']:5.1%}  "
                  f"bands {'yes' if r['bands_percolate'] else 'NO '}  pads {'yes' if r['pads_percolate'] else 'NO '} "
                  f"(in {r['pads_in_linked']}, gnd {r['pads_gnd_linked']})")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"geometries": {n: repr(GEOMETRIES[n]) for n in names}, "rows": rows}, indent=1))
    print(f"-> {OUT.relative_to(ROOT)}")


def figure(names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Rectangle
    ncol = min(3, len(names))
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5.2 * nrow), squeeze=False)
    um = 1e6
    for ax, name in zip(axes.flat, names):
        net = C.make_network(1, geometry=name)
        side = np.sqrt(net.area) * um
        ax.add_collection(LineCollection(net.wire_endpoints * um, colors="0.35", linewidths=0.3))
        for i in range(4):
            for j in range(4):
                c = ((i + 0.5) * side / 4, (j + 0.5) * side / 4)
                color = "tab:orange" if (i + j) % 2 == 0 else "tab:blue"
                ax.add_patch(Rectangle((c[0] - 25, c[1] - 25), 50, 50, color=color, alpha=0.8, zorder=3))
        for cl in GEOMETRIES[name].clusters:
            ax.add_patch(Circle((cl.x * side, cl.y * side), 2 * cl.sigma * side, fill=False, ls="--", color="tab:red"))
        s = network_stats(net, make_layout("pads", net))
        ax.set_title(f"{name}\n{s['n_junctions']} junctions, largest cluster {s['largest_cluster_fraction']:.0%}, "
                     f"pads {'percolate' if s['percolates'] else 'do NOT percolate'}", fontsize=9)
        ax.set_xlim(0, side), ax.set_ylim(0, side), ax.set_aspect("equal")
        ax.set_xlabel("x (um)"), ax.set_ylabel("y (um)")
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"-> {FIG.relative_to(ROOT)}")


# ------------------------------------------------------------------ onset per geometry (as n10l_size.sweep_one)
def sweep_path(layout=LAYOUT, n_wires=0):
    """Onsets per geometry: results/n12_sweep.json for the pads at 2000 wires (the run/report setting), else
    n12_sweep[_<layout>][_N<n_wires>].json."""
    if layout == LAYOUT and not n_wires:
        return SWEEP
    return SWEEP.with_name("n12_sweep" + (f"_{layout}" if layout != LAYOUT else "")
                           + (f"_N{n_wires}" if n_wires else "") + ".json")


def sweep_one(name, layout=LAYOUT, n_wires=0):
    from agnw_rc.multicircuit import MultiTerminalCircuit
    from agnw_rc.volatile import conductance, junction_kd0, step_state
    vp = C.volatile_params(None, tau_s=TAU, tau_spread=SPREAD, dt=C.DT)
    _, wd = C.network_params()
    net = C.make_network(1, n_wires=n_wires, geometry=name)
    els = make_layout(layout, net)
    mc = MultiTerminalCircuit(net, wd, els)
    fixed = mc.fixed_electrodes
    driven = [k for k, i in enumerate(fixed) if els[i].role == "in"]
    kd0 = junction_kd0(vp, len(net.junctions), 1)
    state = np.zeros((2, len(net.junctions)))
    n, v_max = 10_000, 3.0                                             # 0 -> 3 V in 10 s (as N10-S / N10-L)
    Vf = np.zeros(len(fixed))
    vs, gs = [], []
    for i in range(n):
        v = v_max * i / n
        Vf[driven] = v
        V = mc.solve(conductance(state, vp), Vf, method="pcg")
        if i % 10 == 0 and v > 0:
            I = mc.electrode_currents(V)
            vs.append(v)
            gs.append(sum(I[fixed[k]] for k in driven) / v)
        state = step_state(state, mc.junction_voltage_drops(V), vp, kd0)
    v, g = np.array(vs), np.array(gs)
    on = float(v[np.argmax(g > 10 * g[:5].mean())]) if np.any(g > 10 * g[:5].mean()) else None
    return name, {"onset_V": on, "n_junctions": len(net.junctions),
                  "pad_wires_mean": float(np.mean([len(e.wires) for e in els]))}


def sweep(names, procs, layout=LAYOUT, n_wires=0):
    path = sweep_path(layout, n_wires)
    res = json.loads(path.read_text()) if path.exists() else {}
    todo = [n for n in names if n not in res]
    if todo:
        with ProcessPoolExecutor(max(1, min(procs, len(todo)))) as pool:
            for name, r in pool.map(sweep_one, todo, [layout] * len(todo), [n_wires] * len(todo)):
                res[name] = r
                print(name, r, flush=True)
                path.write_text(json.dumps(res, indent=1))
    print(json.dumps({n: res[n] for n in names}, indent=1))


# ------------------------------------------------------------------ runs
def specs(names):
    sw = json.loads(SWEEP.read_text())
    wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
    out = []
    for name in names:
        on = sw[name]["onset_V"]
        if on is None:
            print(f"{name}: no onset below 3 V, skipped")
            continue
        kw = dict(n_samples=C.E4_N_SAMPLES, washout=-(-wo // HOLD) * HOLD, n_virtual=C.E4_N_VIRTUAL, layout=LAYOUT,
                  encoding="weights", model="volatile", tau_s=TAU, tau_spread=SPREAD, stride=C.N10_STRIDE,
                  geometry=name, v_width=round(0.45 * on, 2))
        for vc in (round(1.05 * on, 2), round(1.25 * on, 2)):
            for net in C.E1_SEEDS:
                out += [C.RunSpec("iid", HOLD, vc, net, rep, **kw) for rep in C.E1_REPLICATES]
                out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, **kw))
    return out


def run_one(spec):
    from agnw_rc.drive import raw_path, save_raw
    from agnw_rc.multidrive import simulate_multi
    f = raw_path(spec, RAW)
    mark = f.with_suffix(".running")
    if f.exists() or mark.exists():
        return spec.name + " (exists or running elsewhere)"
    mark.write_text(str(os.getpid()))
    save_raw(spec, simulate_multi(spec), RAW)
    mark.unlink(missing_ok=True)
    return spec.name


def run(names, procs, share=""):
    from agnw_rc.drive import raw_path
    sp = specs(names)
    if share:
        idx, n = share.split("/")
        keep = {int(i) for i in idx.split(",")}
        sp = [s for k, s in enumerate(sp) if k % int(n) in keep]
    todo = [s for s in sp if not raw_path(s, RAW).exists() and not raw_path(s, RAW).with_suffix(".running").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report
def report(names, procs):
    import dataclasses
    from agnw_rc import criteria as K
    from agnw_rc.drive import raw_path
    from n10l_size import _cons, _score                                 # same scoring as N10-L (RAW, OBS equal)
    sp = [s for s in specs(names) if raw_path(s, RAW).exists()]
    with ProcessPoolExecutor(procs) as pool:
        scored = list(pool.map(_score, sp))
        cs = list(pool.map(_cons, [(s, dataclasses.replace(s, replicate=1)) for s in sp if s.stream == "iid"
                                   and s.replicate == 0 and raw_path(dataclasses.replace(s, replicate=1), RAW).exists()]))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in C.E1_SEEDS)
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    bl = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in bl]))}
    ref = {x["setting"]: x for x in json.loads((ROOT / "results" / "n10r_criteria.json").read_text())["rows"]}
    keys = ("operating_point", "MC", "IPC2", "narma10_nmse", "accuracy", "consistency_inc", "G0", "G1", "G2", "G2_loose")
    rows = [{"geometry": "N10-E (ref.)", "reference": "results/n10r_criteria.json, uniform, onset 0.36 V",
             **{k: ref["no noise (N10-E)"].get(k) for k in keys}}]
    for name in names:
        sel = lambda s: s.geometry == name  # noqa: E731
        runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                     replicate=s.replicate, scores=sc) for s, sc, _ in scored if sel(s)]
        cons = [dict(hold=a.hold, v_center=a.v_center, consistency=c) for a, c in cs if sel(a)]
        try:
            nf = next(n for s, _, n in scored if sel(s) and n)
            strict = K.evaluate(runs, cons, OBS, nf, esn, floor)
            loose = K.evaluate(runs, cons, OBS, 16, esn, floor)
        except (ValueError, KeyError, StopIteration) as err:
            print(f"{name}: incomplete ({err})")
            continue
        rows.append({"geometry": name, **strict, "G2_loose": loose["G2"], "n_runs": len(runs)})
    if CRIT.exists():                       # keep geometries reported elsewhere (e.g. on another machine)
        done = {x["geometry"] for x in rows}
        prev = [x for x in json.loads(CRIT.read_text())["rows"] if x["geometry"] not in done]
        order = {g: k for k, g in enumerate(["N10-E (ref.)"] + list(GEOMETRIES))}
        rows = sorted(rows + prev, key=lambda x: order.get(x["geometry"], len(order)))
    for x in rows:
        acc = x["accuracy"]["mean"] if x.get("accuracy") else float("nan")
        print(f"{x['geometry']:16s} op {x['operating_point']} MC {x['MC']['mean']:.2f}+-{x['MC']['sd']:.2f} "
              f"IPC2 {x['IPC2']['mean']:.2f}+-{x['IPC2']['sd']:.2f} NARMA {x['narma10_nmse']['mean']:.3f} acc {acc:.2f} "
              f"-> G0 {x['G0']} G1 {x['G1']} G2 {x['G2']} G2 loose {x['G2_loose']}", flush=True)
    CRIT.write_text(json.dumps({"sweep": json.loads(SWEEP.read_text()), "esn_total": esn, "noise_floor": floor,
                                "rows": rows}, indent=1))
    print(f"-> {CRIT.relative_to(ROOT)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    labels = [x["geometry"] for x in rows]
    for ax, k, t, thr in zip(axes, ("MC", "IPC2", "narma10_nmse"),
                             ("memory capacity", "degree-2 capacity", "NARMA-10 NMSE (lower is better)"), (1, 1, 0.7)):
        ax.bar(range(len(rows)), [x[k]["mean"] for x in rows], yerr=[x[k]["sd"] for x in rows], capsize=3,
               color=["0.6"] + ["tab:blue"] * (len(rows) - 1))
        ax.axhline(thr, color="r", ls="--", lw=0.8, label="G1 threshold")
        ax.set_xticks(range(len(rows)), labels, rotation=30, ha="right")
        ax.set_title(t)
    axes[0].legend(fontsize=8)
    fig.suptitle("N12: N10-E reservoir (16 pads, 1 mm, no ruptures) on each wire geometry; grey = published N10-E")
    fig.tight_layout()
    fig.savefig(FIG_SCORES, dpi=150)
    print(f"-> {FIG_SCORES.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("stats", "figure", "sweep", "run", "report"))
    ap.add_argument("--only", default="", help="comma-separated geometry names (default: all)")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--layout", default=LAYOUT, help="electrode layout for sweep (run/report use the pads)")
    ap.add_argument("--n-wires", type=int, default=0, help="number of wires for sweep (0 = 2000, the working density)")
    ap.add_argument("--share", default="", help="i/n: only runs with index %% n == i (split between machines)")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else (list(GEOMETRIES) if a.cmd in ("stats", "figure") else RC_GEOMETRIES)
    unknown = set(names) - set(GEOMETRIES)
    if unknown:
        ap.error(f"unknown geometries {sorted(unknown)}; known: {list(GEOMETRIES)}")
    if a.cmd == "stats":
        stats(names, [int(s) for s in a.seeds.split(",")])
    elif a.cmd == "figure":
        figure(names)
    elif a.cmd == "sweep":
        sweep(names, a.procs, a.layout, a.n_wires)
    elif a.cmd == "run":
        run(names, a.procs, a.share)
    else:
        report(names, a.procs)


if __name__ == "__main__":
    main()
