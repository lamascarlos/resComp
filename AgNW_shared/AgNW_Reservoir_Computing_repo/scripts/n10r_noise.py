#!/usr/bin/env python3
"""N10-R: realistic junction noise and its mitigation, on the N10-E setup (4x4 pads, random input weights, 16
electrode currents, tau_s 0.3 s, spread 1, 100 ms holds, 3 networks x 2 replicates + sine/square).

Blocks:
  rupture     random complete ruptures of the volatile filament (Poisson, per junction) at 0.3, 1 and 3 per second:
              the abrupt drops and plateaus of measured relaxation (Diaz-Alvarez 2019), instead of the smooth
              Gaussian noise of N10-M
  large_pads  Gaussian noise sigma = 0.1 (breaks G1 in N10-M) with 100 um pads (4x the area): more wires per pad
  long_hold   sigma = 0.1 with inputs held 300 ms: more samples of the transient per input
Post-hoc (no new runs): "repeat" averages the electrode features of the two replicates of each run (same network,
same input, independent noise), i.e. presenting the input twice and averaging, for N10-M sigma = 0.1 and for
the rupture blocks. rupture_low (0.01, 0.03, 0.1 per second) was added after rupture at 0.3/s already broke G1; G0 cannot be tested for these (both replicates are used).
rupture_vlow (0.001, 0.003 per second, added 2026-10-04) locates the tolerated rupture rate; it still fails G1 (every run
below its rupture-free twin, 3 of 12 collapse), so rupture_vvlow (1e-5, 1e-4, 3e-4 per second) extends it.

    python scripts/n10r_noise.py sweep                     -> results/n10r_sweep.json (onset of large pads)
    python scripts/n10r_noise.py run --blocks ... [--procs N]
    python scripts/n10r_noise.py report                    -> results/n10r_criteria.json, figures/fig20_n10r.png
    python scripts/n10r_noise.py twins                     -> results/n10r_twins.json (each low-rate run vs no rupture)
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
from agnw_rc import criteria as K  # noqa: E402
from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.drive import load_raw, raw_path, save_raw  # noqa: E402
from agnw_rc.multidrive import simulate_multi  # noqa: E402

RAW = ROOT / "results" / "raw"
SWEEP = ROOT / "results" / "n10r_sweep.json"
OUT = ROOT / "results" / "n10r_criteria.json"
FIG = ROOT / "figures" / "fig20_n10r.png"
TAU, SPREAD = 0.3, 1.0
OBS = "electrodes"
BLOCKS = {
    "rupture": [dict(layout="pads", hold=100, rupture=r) for r in (0.3, 1.0, 3.0)],
    "rupture_low": [dict(layout="pads", hold=100, rupture=r) for r in (0.01, 0.03, 0.1)],   # added after the first results
    "rupture_vlow": [dict(layout="pads", hold=100, rupture=r) for r in (0.001, 0.003)],   # tolerance threshold
    "rupture_vvlow": [dict(layout="pads", hold=100, rupture=r) for r in (1e-5, 1e-4, 3e-4)],   # 0.001 still fails
    "large_pads": [dict(layout="pads_large", hold=100, sigma=0.1)],
    "long_hold": [dict(layout="pads", hold=300, sigma=0.1)],
}


def sweep():
    """Onset of the large-pad layout (same ramp as the N10-E0 sweep)."""
    import n10e_pilot as P
    P_layout = "pads_large"
    _, res = P.sweep_one(P_layout)
    SWEEP.write_text(json.dumps({P_layout: res}, indent=1))
    print(res)


def windows(layout):
    if layout == "pads":
        return C.N10M_WINDOWS
    on = json.loads(SWEEP.read_text())[layout]["onset_V"]
    return ((round(1.05 * on, 2), round(0.45 * on, 2)), (round(1.25 * on, 2), round(0.45 * on, 2)))


def specs(blocks=tuple(BLOCKS)):
    out = []
    for b in blocks:
        for cfg in BLOCKS[b]:
            cfg = dict(cfg)
            hold = cfg.pop("hold")
            wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
            common = dict(n_samples=C.E4_N_SAMPLES, washout=-(-wo // hold) * hold, n_virtual=C.E4_N_VIRTUAL,
                          encoding="weights", model="volatile", tau_s=TAU, tau_spread=SPREAD, stride=C.N10_STRIDE,
                          **cfg)
            for vc, w in windows(cfg["layout"]):
                for net in C.E1_SEEDS:
                    out += [C.RunSpec("iid", hold, vc, net, r, v_width=w, **common) for r in C.E1_REPLICATES]
                    out.append(C.RunSpec("sinesquare", hold, vc, net, 0, v_width=w, **common))
    return out


def run_one(spec):
    if not raw_path(spec, RAW).exists():
        save_raw(spec, simulate_multi(spec), RAW)
    return spec.name


def run(blocks, procs, reverse=False, limit=0):
    todo = [s for s in specs(blocks) if not raw_path(s, RAW).exists()]
    todo.sort(key=lambda s: -s.hold)
    if reverse:
        todo.reverse()
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report
def label(s):
    parts = []
    if s.rupture:
        parts.append(f"rupture {s.rupture:g}/s")
    if s.sigma:
        parts.append(f"noise σ {s.sigma:g}")
    if s.layout == "pads_large":
        parts.append("100 µm pads")
    if s.hold != 100:
        parts.append(f"hold {s.hold * C.DT:g} s")
    return ", ".join(parts)


def _score(spec):
    raw = load_raw(spec, RAW)
    if spec.stream == "sinesquare":
        return spec, E.evaluate_sinesquare_run(raw), None
    return spec, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), E.observables(raw)[OBS].shape[1]


def _cons(pair):
    a, b = pair
    return a, E.evaluate_consistency(load_raw(a, RAW), load_raw(b, RAW))


def _repeat(pair):
    """Average the electrode features of the two replicates (input presented twice) and score."""
    a, b = pair
    ra, rb = load_raw(a, RAW), load_raw(b, RAW)
    F = 0.5 * (E.observables(ra)[OBS] + E.observables(rb)[OBS])
    s = E.score_iid(F, ra["u"], E.noise_floor(ra["u"]))
    return a, {m: s[m] for m in ("MC", "IPC2", "narma10_nmse", "val_MC", "val_IPC2")}


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = [s for s in specs() if raw_path(s, RAW).exists()]
    n10m_sigma01 = [s for s in C.n10m_specs(("noise",)) if s.sigma == 0.1]
    with ProcessPoolExecutor(16) as pool:
        scored = list(pool.map(_score, sp))
        pairs = [(s, C.RunSpec(**{**s.__dict__, "replicate": 1})) for s in sp if s.stream == "iid" and s.replicate == 0
                 and raw_path(C.RunSpec(**{**s.__dict__, "replicate": 1}), RAW).exists()]
        cs = list(pool.map(_cons, pairs))
        rep_pairs = [(s, C.RunSpec(**{**s.__dict__, "replicate": 1})) for s in sp + n10m_sigma01
                     if s.stream == "iid" and s.replicate == 0 and (s.rupture or s in n10m_sigma01)]
        reps = list(pool.map(_repeat, rep_pairs))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in C.E1_SEEDS)
    base = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in base]))}
    m = json.loads((ROOT / "results" / "n10m_criteria.json").read_text())["rows"]
    rows = [{**r, "setting": r["config"].split(", ")[-1] if "noise" in r["config"] else "no noise (N10-E)"}
            for r in m if r["config"].startswith("N10-E") or "noise 0.1" in r["config"]]
    order = {}
    for s in sp:                                   # numeric order (rupture rate, then the noise settings)
        order.setdefault(label(s), (s.rupture, s.sigma, s.layout, s.hold))
    for lab in sorted(order, key=order.get):
        runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                     replicate=s.replicate, scores=sc) for s, sc, _ in scored if label(s) == lab]
        cons = [dict(hold=a.hold, v_center=a.v_center, consistency=c) for a, c in cs if label(a) == lab]
        try:
            nf = next(n for s, _, n in scored if label(s) == lab and n)
            strict = K.evaluate(runs, cons, OBS, nf, esn, floor)
            loose = K.evaluate(runs, cons, OBS, 16, esn, floor)
        except (ValueError, KeyError, StopIteration) as err:
            print(f"{lab}: incomplete ({err})")
            continue
        rows.append({"setting": lab, **strict, "G2_loose": loose["G2"]})
    # post-hoc repeat averaging: validation-selected window per configuration, mean/sd over networks
    rep_rows = []
    groups = {}
    for a, s in reps:
        groups.setdefault(label(a), {}).setdefault(a.v_center, []).append(s)
    for lab, by_v in groups.items():
        vbest = max(by_v, key=lambda v: np.mean([x["val_MC"] + x["val_IPC2"] for x in by_v[v]]))
        xs = by_v[vbest]
        rep_rows.append({"setting": lab + ", input presented twice (averaged)", "v_center": vbest, "n": len(xs),
                         **{k: {"mean": float(np.mean([x[k] for x in xs])), "sd": float(np.std([x[k] for x in xs], ddof=1))}
                            for k in ("MC", "IPC2", "narma10_nmse")}})
    OUT.write_text(json.dumps({"esn_total": esn, "noise_floor": floor, "rows": rows, "repeat_averaged": rep_rows},
                              indent=1))
    for r in rows:
        print(f"{r['setting']:40s} MC {r['MC']['mean']:.2f}±{r['MC']['sd']:.2f} IPC2 {r['IPC2']['mean']:.2f}±"
              f"{r['IPC2']['sd']:.2f} NARMA {r['narma10_nmse']['mean']:.3f} acc "
              f"{r['accuracy']['mean'] if r['accuracy'] else float('nan'):.2f} cons "
              f"{r['consistency_inc']['mean'] if r['consistency_inc'] else float('nan'):.2f} -> G0 {r['G0']} G1 {r['G1']} "
              f"G2 loose {r['G2_loose']}")
    for r in rep_rows:
        print(f"{r['setting']:60s} MC {r['MC']['mean']:.2f}±{r['MC']['sd']:.2f} IPC2 {r['IPC2']['mean']:.2f} "
              f"NARMA {r['narma10_nmse']['mean']:.3f}")
    allrows = rows + [{**r, "repeat": True} for r in rep_rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    x = np.arange(len(allrows))
    for ax, k, t in zip(axes, ("MC", "IPC2", "narma10_nmse"), ("memory capacity", "degree-2 capacity", "NARMA-10 NMSE")):
        ax.bar(x, [r[k]["mean"] for r in allrows], yerr=[r[k]["sd"] for r in allrows], capsize=3,
               color=["0.6" if "N10-E" in r["setting"] else ("C2" if r.get("repeat") else "C1") for r in allrows])
        ax.set_xticks(x, [r["setting"].replace(", input presented twice (averaged)", "\n+ repeat x2")
                          .replace(", ", "\n") for r in allrows], fontsize=7, rotation=45, ha="right")
        ax.axhline(1 if k != "narma10_nmse" else 0.7, color="r", ls="--", lw=0.8)
        ax.set_title(t)
    fig.suptitle("N10-R: junction noise and mitigation (16 pad currents, 3 networks; grey: noise-free N10-E, "
                 "green: input presented twice and averaged)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


def _twin(spec):
    """Electrode-readout scores of a rupture run and of the same run without ruptures (same network, replicate, window)."""
    import dataclasses
    z = dataclasses.replace(spec, rupture=0.0)
    out = {}
    for tag, sp in (("rupture", spec), ("no_rupture", z)):
        raw = load_raw(sp, RAW)
        sc = E.evaluate_iid_run(raw, E.noise_floor(raw["u"]))[OBS]
        out[tag] = {k: float(sc[k]) for k in ("MC", "IPC2", "narma10_nmse")}
    return spec, out


def twins():
    """Per run: rupture vs rupture-free twin (low rates), -> results/n10r_twins.json. A run 'collapses' when its
    MC + IPC2 falls below half of its twin's."""
    sp = [s for s in specs(("rupture_vvlow", "rupture_vlow", "rupture_low")) if s.stream == "iid"
          and raw_path(s, RAW).exists() and raw_path(C.RunSpec(**{**s.__dict__, "rupture": 0.0}), RAW).exists()]
    with ProcessPoolExecutor(16) as pool:
        res = list(pool.map(_twin, sp))
    rows, summ = [], {}
    for s, o in res:
        tot = o["rupture"]["MC"] + o["rupture"]["IPC2"]
        ref = o["no_rupture"]["MC"] + o["no_rupture"]["IPC2"]
        rows.append({"rupture": s.rupture, "v_center": s.v_center, "network_seed": s.network_seed,
                     "replicate": s.replicate, **o, "ratio_total": tot / ref})
        summ.setdefault(f"{s.rupture:g}", []).append(tot / ref)
    summary = {r: {"n": len(v), "median_ratio": float(np.median(v)), "collapsed": int(np.sum(np.array(v) < 0.5)),
                   "worse": int(np.sum(np.array(v) < 1.0))} for r, v in sorted(summ.items(), key=lambda kv: float(kv[0]))}
    (ROOT / "results" / "n10r_twins.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    for r, v in summary.items():
        print(f"rupture {r:>6}/s: n {v['n']}, (MC+IPC2)/twin median {v['median_ratio']:.2f}, worse {v['worse']}, "
              f"collapsed (<0.5) {v['collapsed']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "run", "report", "twins"])
    ap.add_argument("--blocks", nargs="+", default=list(BLOCKS))
    ap.add_argument("--procs", type=int, default=18)
    ap.add_argument("--reverse", action="store_true", help="work from the end of the queue (second machine)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"sweep": sweep, "run": lambda: run(tuple(a.blocks), a.procs, a.reverse, a.limit), "report": report, "twins": twins}[a.stage]()
