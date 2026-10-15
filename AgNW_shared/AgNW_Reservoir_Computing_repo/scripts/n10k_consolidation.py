#!/usr/bin/env python3
"""N10-K: can a consolidated filament backbone protect the N10 reservoir against junction ruptures?

N10 junction (docs/N10_DESIGN.md) with its consolidated filament y switched on (agnw_rc.volatile: dy/dt = k_c x^2
(1 - y) [P > P_c2] - y / tau_l; ruptures act on the volatile x only). Before the usual pre-drive, a constant forming
drive (forming_v on all driven pads for `forming` s) lets y build where current concentrates. Then the N10-E
reservoir protocol (4x4 pads, random weights, 16 electrode currents, tau_s 0.3 s, spread 1, 100 ms holds, 800 inputs,
3 networks x 2 replicates + sine/square) with random ruptures of x.

Stages:
  pilot   one network, short operation (150 inputs): forming voltage x consolidation rate k_c x power gate P_c2 ->
          fraction consolidated after forming, drift of y and of the electrode currents during operation
          -> results/n10k_pilot.json
  run     the reservoir runs for the settings chosen from the pilot (BLOCKS below) -> results/raw/*.npz
  report  G0-G2 per setting and rupture rate, twins vs the rupture-free run of the same setting
          -> results/n10k_criteria.json, figures/fig24_n10k.png

    python scripts/n10k_consolidation.py pilot [--procs N]
    python scripts/n10k_consolidation.py run [--blocks ...] [--procs N] [--reverse] [--limit K]
    python scripts/n10k_consolidation.py report
"""
import argparse
import itertools
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

from agnw_rc import config as C  # noqa: E402
from agnw_rc.multidrive import simulate_multi  # noqa: E402

RAW = ROOT / "results" / "raw"
PILOT = ROOT / "results" / "n10k_pilot.json"
TAU, SPREAD, HOLD = 0.3, 1.0, 100
ONSET_PADS = 0.36                                   # V, results/n10e_sweep.json
PILOT_V = (round(2 * ONSET_PADS, 2), round(3 * ONSET_PADS, 2))
PILOT_KC = (0.1, 1.0, 10.0)
PILOT_PC2 = (1e-6, 1e-5, 1e-4)
FORMING_S = 10.0


def common(**kw):
    wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
    return dict(washout=-(-wo // HOLD) * HOLD, n_virtual=C.E4_N_VIRTUAL, layout="pads", encoding="weights",
                model="volatile", tau_s=TAU, tau_spread=SPREAD, stride=C.N10_STRIDE, **kw)


# ------------------------------------------------------------------ pilot
def pilot_one(cfg):
    vf, kc, pc2 = cfg
    vc, w = C.N10M_WINDOWS[1]
    spec = C.RunSpec("iid", HOLD, vc, 1, v_width=w, n_samples=150,
                     **common(consolidation=kc > 0, k_c=kc, p_c2=pc2 if kc > 0 else 0.0,
                              forming=FORMING_S, forming_v=vf))
    raw = simulate_multi(spec, save_states=True)
    y0, y = raw["formed_y"].astype(float), raw["junction_y"].astype(float)
    n_e = int(raw["n_electrodes"])
    cur = np.abs(raw["current"].reshape(len(raw["u"]), -1, n_e)).mean(axis=1)    # per sample, per electrode
    first, last = cur[:20].mean(axis=0), cur[-20:].mean(axis=0)
    return {"forming_v": vf, "k_c": kc, "p_c2": pc2,
            "formed_frac_y_gt_0.5": float((y0 > 0.5).mean()), "formed_frac_y_gt_0.1": float((y0 > 0.1).mean()),
            "formed_y_mean": float(y0.mean()),
            "op_frac_y_gt_0.5_first": float((y[0] > 0.5).mean()), "op_frac_y_gt_0.5_last": float((y[-1] > 0.5).mean()),
            "op_y_mean_first": float(y[0].mean()), "op_y_mean_last": float(y[-1].mean()),
            "current_drift_rel": float(np.median(np.abs(last - first) / np.maximum(first, 1e-12))),
            "current_signal_rel": float(np.median(cur.std(axis=0) / np.maximum(cur.mean(axis=0), 1e-12))),
            "on_fraction_mean": float(raw["on_fraction"].mean()), "seconds": float(raw["seconds"])}


def pilot(procs):
    cfgs = [(v, 0.0, 0.0) for v in PILOT_V[:1]] + list(itertools.product(PILOT_V, PILOT_KC, PILOT_PC2))
    out = []
    with ProcessPoolExecutor(procs) as pool:
        for f in as_completed([pool.submit(pilot_one, c) for c in cfgs]):
            r = f.result()
            out.append(r)
            print(f"V {r['forming_v']} kc {r['k_c']:g} pc2 {r['p_c2']:g}: formed y>0.5 {r['formed_frac_y_gt_0.5']:.3f} "
                  f"(>0.1 {r['formed_frac_y_gt_0.1']:.3f}) | operation y>0.5 {r['op_frac_y_gt_0.5_first']:.3f} -> "
                  f"{r['op_frac_y_gt_0.5_last']:.3f}, current drift {r['current_drift_rel']:.3f}, signal "
                  f"{r['current_signal_rel']:.3f}, on {r['on_fraction_mean']:.3f}", flush=True)
    out.sort(key=lambda r: (r["forming_v"], r["k_c"], r["p_c2"]))
    PILOT.write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------ reservoir runs
# Settings from the pilot (results/n10k_pilot.json; k_c saturates above 1): consolidated fraction after forming
# A 1.4 %, B 8 %, C 12.5 % (C's low power gate also lets y grow in operation). Rupture rates per junction: 0 (twin
# reference), 1e-3 (every N10-R run degraded) and 1e-2 (inside the N10-C range for real networks).
SETTINGS = {"A": dict(forming_v=PILOT_V[0], p_c2=1e-5), "B": dict(forming_v=PILOT_V[1], p_c2=1e-5),
            "C": dict(forming_v=PILOT_V[0], p_c2=1e-6)}
RUPTURES = (0.0, 1e-3, 1e-2)
BLOCKS = {k: [dict(setting=k, rupture=r) for r in RUPTURES] for k in SETTINGS}


def specs(blocks=tuple(BLOCKS)):
    out = []
    for b in blocks:
        for cfg in BLOCKS[b]:
            kw = common(consolidation=True, k_c=1.0, forming=FORMING_S, rupture=cfg["rupture"], **SETTINGS[cfg["setting"]])
            for vc, w in C.N10M_WINDOWS:
                for net in C.E1_SEEDS:
                    out += [C.RunSpec("iid", HOLD, vc, net, r, v_width=w, n_samples=C.E4_N_SAMPLES, **kw)
                            for r in C.E1_REPLICATES]
                    out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, v_width=w, n_samples=C.E4_N_SAMPLES, **kw))
    return out


def setting_of(spec):
    return next(k for k, v in SETTINGS.items() if spec.forming_v == v["forming_v"] and spec.p_c2 == v["p_c2"])


def run_one(spec):
    from agnw_rc.drive import raw_path, save_raw
    f = raw_path(spec, RAW)
    mark = f.with_suffix(".running")
    if f.exists() or mark.exists():
        return spec.name + " (exists or running elsewhere)"
    mark.write_text(str(os.getpid()))
    save_raw(spec, simulate_multi(spec), RAW)
    mark.unlink(missing_ok=True)
    return spec.name


def run(blocks, procs, reverse=False, limit=0):
    from agnw_rc.drive import raw_path
    todo = [s for s in specs(blocks) if not raw_path(s, RAW).exists() and not raw_path(s, RAW).with_suffix(".running").exists()]
    if reverse:
        todo.reverse()
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report
OBS = "electrodes"
OUT = ROOT / "results" / "n10k_criteria.json"
FIG = ROOT / "figures" / "fig24_n10k.png"


def label(spec):
    return f"{setting_of(spec)}, rupture {spec.rupture:g}/s"


def _score(spec):
    from agnw_rc import evaluate as E
    from agnw_rc.drive import load_raw
    raw = load_raw(spec, RAW)
    extra = {"formed_y_gt_0.5": float((raw["formed_y"] > 0.5).mean()), "on_fraction": float(raw["on_fraction"].mean()),
             "y_drift": (float(np.abs(raw["junction_y"][-1].astype(float) - raw["junction_y"][0].astype(float)).mean())
                         if "junction_y" in raw else None)}           # None: states slimmed (full files archived)
    if spec.stream == "sinesquare":
        return spec, E.evaluate_sinesquare_run(raw), None, extra
    return spec, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), E.observables(raw)[OBS].shape[1], extra


def _cons(pair):
    from agnw_rc import evaluate as E
    from agnw_rc.drive import load_raw
    return pair[0], E.evaluate_consistency(load_raw(pair[0], RAW), load_raw(pair[1], RAW))


def _twin(spec):
    import dataclasses
    from agnw_rc import evaluate as E
    from agnw_rc.drive import load_raw
    o = {}
    for tag, sp in (("rupture", spec), ("no_rupture", dataclasses.replace(spec, rupture=0.0))):
        raw = load_raw(sp, RAW)
        sc = E.evaluate_iid_run(raw, E.noise_floor(raw["u"]))[OBS]
        o[tag] = {k: float(sc[k]) for k in ("MC", "IPC2", "narma10_nmse")}
    return spec, o


def report():
    import dataclasses
    from agnw_rc import criteria as K
    from agnw_rc.drive import raw_path
    sp = [s for s in specs() if raw_path(s, RAW).exists()]
    with ProcessPoolExecutor(16) as pool:
        scored = list(pool.map(_score, sp))
        pairs = [(s, dataclasses.replace(s, replicate=1)) for s in sp if s.stream == "iid" and s.replicate == 0
                 and raw_path(dataclasses.replace(s, replicate=1), RAW).exists()]
        cs = list(pool.map(_cons, pairs))
        tw = list(pool.map(_twin, [s for s in sp if s.stream == "iid" and s.rupture
                                   and raw_path(dataclasses.replace(s, rupture=0.0), RAW).exists()]))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in C.E1_SEEDS)
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    bl = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in bl]))}
    ref = {r["setting"]: r for r in json.loads((ROOT / "results" / "n10r_criteria.json").read_text())["rows"]}
    rows = []
    for lab in sorted({label(s) for s in sp}, key=lambda l: (l[0], float(l.split()[2][:-2]))):
        runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                     replicate=s.replicate, scores=sc) for s, sc, _, _ in scored if label(s) == lab]
        cons = [dict(hold=a.hold, v_center=a.v_center, consistency=c) for a, c in cs if label(a) == lab]
        ex = [e for s, _, _, e in scored if label(s) == lab]
        try:
            nf = next(n for s, _, n, _ in scored if label(s) == lab and n)
            strict = K.evaluate(runs, cons, OBS, nf, esn, floor)
            loose = K.evaluate(runs, cons, OBS, 16, esn, floor)
        except (ValueError, KeyError, StopIteration) as err:
            print(f"{lab}: incomplete ({err})")
            continue
        ratios = [(o["rupture"]["MC"] + o["rupture"]["IPC2"]) / (o["no_rupture"]["MC"] + o["no_rupture"]["IPC2"])
                  for s, o in tw if label(s) == lab]
        rows.append({"setting": lab, **strict, "G2_loose": loose["G2"],
                     "consolidated_frac": float(np.mean([e["formed_y_gt_0.5"] for e in ex])),
                     "on_fraction": float(np.mean([e["on_fraction"] for e in ex])),
                     "y_drift": (float(np.mean([e["y_drift"] for e in ex])) if all(e["y_drift"] is not None for e in ex)
                                 else None),
                     "twin_ratio_median": float(np.median(ratios)) if ratios else None,
                     "twin_collapsed": int(np.sum(np.array(ratios) < 0.5)) if ratios else None,
                     "twin_n": len(ratios)})
    OUT.write_text(json.dumps({"esn_total": esn, "noise_floor": floor, "settings": SETTINGS, "rows": rows}, indent=1))
    for r in rows:
        print(f"{r['setting']:22s} cons.frac {r['consolidated_frac']:.3f} MC {r['MC']['mean']:.2f}±{r['MC']['sd']:.2f} "
              f"IPC2 {r['IPC2']['mean']:.2f}±{r['IPC2']['sd']:.2f} NARMA {r['narma10_nmse']['mean']:.3f} acc "
              f"{r['accuracy']['mean'] if r['accuracy'] else float('nan'):.2f} cons "
              f"{r['consistency_inc']['mean'] if r['consistency_inc'] else float('nan'):.2f} -> G0 {r['G0']} G1 {r['G1']} "
              f"G2 loose {r['G2_loose']} | twins median {r['twin_ratio_median']} collapsed {r['twin_collapsed']}/{r['twin_n']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, k, t, thr in zip(axes, ("MC", "IPC2", "narma10_nmse"), ("memory capacity", "degree-2 capacity",
                                                                     "NARMA-10 NMSE"), (1, 1, 0.7)):
        for name, mk in (("none (N10-R)", "s"),) + tuple((key, "o") for key in SETTINGS):
            if name.startswith("none"):
                pts = [(0.0, ref["no noise (N10-E)"])] + [(r, ref[f"rupture {r:g}/s"]) for r in RUPTURES[1:]
                                                         if f"rupture {r:g}/s" in ref]
                lab = "no consolidation (N10-E / N10-R)"
            else:
                pts = [(float(r["setting"].split()[2][:-2]), r) for r in rows if r["setting"].startswith(name + ",")]
                lab = f"backbone {name} ({SETTINGS[name]['forming_v']} V, P_c2 {SETTINGS[name]['p_c2']:g} W)"
            if pts:
                xs = [max(p[0], 1e-4) for p in pts]
                ax.errorbar(xs, [p[1][k]["mean"] for p in pts], [p[1][k]["sd"] for p in pts], marker=mk, capsize=3,
                            label=lab)
        ax.axhline(thr, color="r", ls="--", lw=0.8)
        ax.set(xscale="log", xlabel="rupture rate per junction (1/s; 1e-4 = none)", title=t)
    axes[0].legend(fontsize=7)
    fig.suptitle("N10-K: consolidated backbone vs random ruptures (16 pad currents, 3 networks)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "run", "report"])
    ap.add_argument("--blocks", nargs="+", default=list(BLOCKS))
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"pilot": lambda: pilot(a.procs), "run": lambda: run(tuple(a.blocks), a.procs, a.reverse, a.limit),
     "report": report}[a.stage]()
