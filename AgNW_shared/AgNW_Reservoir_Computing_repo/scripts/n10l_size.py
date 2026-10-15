#!/usr/bin/env python3
"""N10-L: does the rupture tolerance of the N10 reservoir grow with network size?

N10-C showed that a single rupture changes the conductance less in a larger network (fewer junctions with detectable
ruptures per junction), and N10-RN that the damage of N10-R comes from ruptures rerouting current inside a small network.
Here the N10-E reservoir (4x4 checkerboard pads, random input weights, 16 electrode currents, tau_s 0.3 s, spread 1,
100 ms holds, 800 inputs, 3 networks x 2 replicates + sine/square) is enlarged geometrically: network side 1.41 and 2 mm
at the working wire density (RunSpec.side), pads scaled with the side ("pads_scaled": 50 um per mm), windows 1.05 /
1.25 x the onset of each size (width 0.45 x onset), ruptures 0, 1e-4, 1e-3, 1e-2 /s per junction. Reference at 1 mm: the
N10-E / N10-R runs ("pads" = "pads_scaled" at 1 mm, tested).

    python scripts/n10l_size.py sweep [--procs N]                     -> results/n10l_sweep.json
    python scripts/n10l_size.py run [--procs N] [--share i/n]           -> results/raw/*.npz  (share: runs with index % n == i;
                                                                         i may be a comma list, e.g. 1,2/3)
    python scripts/n10l_size.py report                                  -> results/n10l_criteria.json, figures/fig28_n10l.png
"""
import argparse
import dataclasses
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
from agnw_rc.drive import load_raw, raw_path, save_raw  # noqa: E402
from agnw_rc.multidrive import simulate_multi  # noqa: E402

RAW = ROOT / "results" / "raw"
SWEEP = ROOT / "results" / "n10l_sweep.json"
OUT = ROOT / "results" / "n10l_criteria.json"
FIG = ROOT / "figures" / "fig28_n10l.png"
TAU, SPREAD, HOLD = 0.3, 1.0, 100
SIDES = (1.41, 2.0)
RUPTURES = (0.0, 1e-4, 1e-3, 1e-2)
# Second round (2026-10-07, this machine only): 2.83 mm, and 3e-3 /s at every size (pins r_75 between 1e-3 and 1e-2).
EXTRA = [(2.83, 0.0), (2.83, 3e-3), (2.83, 1e-2), (1.41, 3e-3), (2.0, 3e-3)]
GRID = [(s, r) for r in RUPTURES for s in SIDES] + EXTRA
ALL_SIDES = tuple(sorted({s for s, _ in GRID}))
OBS = "electrodes"


# ------------------------------------------------------------------ onset per size
def sweep_one(side):
    from agnw_rc.electrodes import make_layout
    from agnw_rc.multicircuit import MultiTerminalCircuit
    from agnw_rc.volatile import conductance, junction_kd0, step_state
    vp = C.volatile_params(None, tau_s=TAU, tau_spread=SPREAD, dt=C.DT)
    _, wd = C.network_params()
    net = C.make_network(1, side=side)
    els = make_layout("pads_scaled", net)
    mc = MultiTerminalCircuit(net, wd, els)
    fixed = mc.fixed_electrodes
    driven = [k for k, i in enumerate(fixed) if els[i].role == "in"]
    kd0 = junction_kd0(vp, len(net.junctions), 1)
    state = np.zeros((2, len(net.junctions)))
    n, v_max = 10_000, 3.0                                             # 0 -> 3 V in 10 s (as N10-S)
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
    return f"{side:g}", {"onset_V": on, "n_junctions": len(net.junctions),
                         "pad_wires_mean": float(np.mean([len(e.wires) for e in els]))}


def sweep(procs):
    res = json.loads(SWEEP.read_text()) if SWEEP.exists() else {}
    todo = [s for s in (1.0,) + ALL_SIDES if f"{s:g}" not in res]       # only sizes not swept yet
    with ProcessPoolExecutor(max(1, min(procs, len(todo)))) as pool:
        res.update(dict(pool.map(sweep_one, todo)))
    SWEEP.write_text(json.dumps(res, indent=1))
    print(res)


# ------------------------------------------------------------------ runs
def specs():
    sw = json.loads(SWEEP.read_text())
    wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
    out = []
    for side, r in GRID:
        if True:
            on = sw[f"{side:g}"]["onset_V"]
            kw = dict(n_samples=C.E4_N_SAMPLES, washout=-(-wo // HOLD) * HOLD, n_virtual=C.E4_N_VIRTUAL,
                      layout="pads_scaled", encoding="weights", model="volatile", tau_s=TAU, tau_spread=SPREAD,
                      stride=C.N10_STRIDE, rupture=r, side=side, v_width=round(0.45 * on, 2))
            for vc in (round(1.05 * on, 2), round(1.25 * on, 2)):
                for net in C.E1_SEEDS:
                    out += [C.RunSpec("iid", HOLD, vc, net, rep, **kw) for rep in C.E1_REPLICATES]
                    out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, **kw))
    return out


def run_one(spec):
    f = raw_path(spec, RAW)
    mark = f.with_suffix(".running")
    if f.exists() or mark.exists():
        return spec.name + " (exists or running elsewhere)"
    mark.write_text(str(os.getpid()))
    save_raw(spec, simulate_multi(spec), RAW)
    mark.unlink(missing_ok=True)
    return spec.name


def select(sp, only):
    """only = "side:rate[:seed],...": keep the runs of those (side, rupture rate[, network seed]) combinations."""
    keys = [tuple(float(x) for x in k.split(":")) for k in only.split(",")]
    return [s for s in sp if any(abs(s.side - k[0]) < 1e-9 and abs(s.rupture - k[1]) < 1e-12
                                 and (len(k) < 3 or s.network_seed == int(k[2])) for k in keys)]


def run(procs, share="", only=""):
    sp = specs()
    if only:
        sp = select(sp, only)
    if share:
        idx, n = share.split("/")
        keep = {int(i) for i in idx.split(",")}
        sp = [s for k, s in enumerate(sp) if k % int(n) in keep]
    todo = [s for s in sp if not raw_path(s, RAW).exists() and not raw_path(s, RAW).with_suffix(".running").exists()]
    todo.sort(key=lambda s: -s.side)                                      # largest networks first
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report
def _score(spec):
    from agnw_rc import evaluate as E
    raw = load_raw(spec, RAW)
    if spec.stream == "sinesquare":
        return spec, E.evaluate_sinesquare_run(raw), None
    return spec, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), E.observables(raw)[OBS].shape[1]


def _cons(pair):
    from agnw_rc import evaluate as E
    return pair[0], E.evaluate_consistency(load_raw(pair[0], RAW), load_raw(pair[1], RAW))


def _twin(spec):
    from agnw_rc import evaluate as E
    o = {}
    for tag, sp in (("rupture", spec), ("no_rupture", dataclasses.replace(spec, rupture=0.0))):
        raw = load_raw(sp, RAW)
        sc = E.evaluate_iid_run(raw, E.noise_floor(raw["u"]))[OBS]
        o[tag] = {k: float(sc[k]) for k in ("MC", "IPC2", "narma10_nmse")}
    return spec, o


def report(procs=8):
    from agnw_rc import criteria as K
    sp = [s for s in specs() if raw_path(s, RAW).exists()]
    with ProcessPoolExecutor(procs) as pool:
        scored = list(pool.map(_score, sp))
        cs = list(pool.map(_cons, [(s, dataclasses.replace(s, replicate=1)) for s in sp if s.stream == "iid"
                                   and s.replicate == 0 and raw_path(dataclasses.replace(s, replicate=1), RAW).exists()]))
        tw = list(pool.map(_twin, [s for s in sp if s.stream == "iid" and s.rupture
                                   and raw_path(dataclasses.replace(s, rupture=0.0), RAW).exists()]))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in C.E1_SEEDS)
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    bl = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in bl]))}
    rows = []
    for side in ALL_SIDES:
        for r in sorted({rr for ss, rr in GRID if ss == side}):
            sel = lambda s: s.side == side and s.rupture == r  # noqa: E731
            runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                         replicate=s.replicate, scores=sc) for s, sc, _ in scored if sel(s)]
            cons = [dict(hold=a.hold, v_center=a.v_center, consistency=c) for a, c in cs if sel(a)]
            try:
                nf = next(n for s, _, n in scored if sel(s) and n)
                strict = K.evaluate(runs, cons, OBS, nf, esn, floor)
                loose = K.evaluate(runs, cons, OBS, 16, esn, floor)
            except (ValueError, KeyError, StopIteration) as err:
                print(f"side {side} rupture {r}: incomplete ({err})")
                continue
            ratios = [(o["rupture"]["MC"] + o["rupture"]["IPC2"]) / (o["no_rupture"]["MC"] + o["no_rupture"]["IPC2"])
                      for s, o in tw if sel(s)]
            rows.append({"side_mm": side, "rupture": r, **strict, "G2_loose": loose["G2"],
                         "twin_ratio_median": float(np.median(ratios)) if ratios else None,
                         "twin_collapsed": int(np.sum(np.array(ratios) < 0.5)) if ratios else None, "twin_n": len(ratios)})
            x = rows[-1]
            print(f"side {side} rupture {r:<6g} op {x['operating_point']} MC {x['MC']['mean']:.2f}±{x['MC']['sd']:.2f} "
                  f"IPC2 {x['IPC2']['mean']:.2f}±{x['IPC2']['sd']:.2f} NARMA {x['narma10_nmse']['mean']:.3f} acc "
                  f"{x['accuracy']['mean'] if x['accuracy'] else float('nan'):.2f} cons "
                  f"{x['consistency_inc']['mean'] if x['consistency_inc'] else float('nan'):.2f} -> G0 {x['G0']} G1 {x['G1']} "
                  f"G2 loose {x['G2_loose']} | twins {x['twin_ratio_median']} collapsed {x['twin_collapsed']}/{x['twin_n']}",
                  flush=True)
    # 1 mm reference (N10-E / N10-R) and its twins
    ref = {x["setting"]: x for x in json.loads((ROOT / "results" / "n10r_criteria.json").read_text())["rows"]}
    tw1 = json.loads((ROOT / "results" / "n10r_twins.json").read_text())["summary"]
    for r in RUPTURES + (3e-3,):
        key = "no noise (N10-E)" if r == 0 else f"rupture {r:g}/s"
        if key in ref:
            t = tw1.get(f"{r:g}", {})
            rows.append({"side_mm": 1.0, "rupture": r, **{k: ref[key][k] for k in ("MC", "IPC2", "narma10_nmse", "G0", "G1")},
                         "twin_ratio_median": t.get("median_ratio"), "twin_collapsed": t.get("collapsed"),
                         "twin_n": t.get("n"), "reference": "N10-E / N10-R"})
    OUT.write_text(json.dumps({"sweep": json.loads(SWEEP.read_text()), "esn_total": esn, "noise_floor": floor,
                               "rows": rows}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for side in (1.0,) + ALL_SIDES:
        rs = sorted([x for x in rows if x["side_mm"] == side], key=lambda x: x["rupture"])
        xs = [max(x["rupture"], 3e-5) for x in rs]
        for ax, k in zip(axes[:3], ("MC", "IPC2", "narma10_nmse")):
            ax.errorbar(xs, [x[k]["mean"] for x in rs], [x[k]["sd"] for x in rs], marker="o", capsize=3,
                        label=f"{side:g} mm")
        tr = [(max(x["rupture"], 3e-5), x["twin_ratio_median"]) for x in rs if x["twin_ratio_median"] is not None]
        axes[3].plot([p[0] for p in tr], [p[1] for p in tr], "o-", label=f"{side:g} mm")
    for ax, t, thr in zip(axes, ("memory capacity", "degree-2 capacity", "NARMA-10 NMSE", "(MC + deg-2) / rupture-free twin"),
                          (1, 1, 0.7, 0.5)):
        ax.axhline(thr, color="r", ls="--", lw=0.8)
        ax.set(xscale="log", xlabel="rupture rate per junction (1/s; 3e-5 = none)", title=t)
    axes[0].legend(fontsize=8)
    fig.suptitle("N10-L: rupture tolerance vs network size (pads and network enlarged together, 16 electrode currents)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "run", "report"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--share", default="")
    ap.add_argument("--only", default="", help='e.g. "1.41:0.003,2:0.003:1" (side:rate[:network seed])')
    a = ap.parse_args()
    {"sweep": lambda: sweep(min(a.procs, 3)), "run": lambda: run(a.procs, a.share, a.only), "report": lambda: report(a.procs)}[a.stage]()
