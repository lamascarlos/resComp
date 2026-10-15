#!/usr/bin/env python3
"""N10-B: operate the N10 reservoir deep in the conducting state, with a constant bias plus a small input (the scheme of
Milano et al. 2025, who run NARMA at a 3.6 V bias = 1.5 x their 2.4 V onset with a 50 mV input), and test it against
junction ruptures.

N10-E setup (4x4 pads, 8 driven with random weights + 8 grounded, 16 electrode currents, tau_s 0.3 s, spread 1, 100 ms
holds, 800 inputs, 3 networks x 2 replicates + sine/square), but the input window is a bias V_b well above the pad onset
(0.36 V) with a narrow width:
    V_b = 1.5, 2, 3 x onset (0.54, 0.72, 1.08 V); full input width 3 % (Milano: 2.8 %) or 15 % of V_b;
    random complete ruptures of the volatile filament at 0, 1e-3, 1e-2 /s per junction (N10-R tolerance ~1e-4;
    N10-C estimate for real networks 0.005-0.03).
The operating point (bias, width) is chosen on validation data per rupture rate, as everywhere (agnw_rc.criteria).
Report extras: twins (each rupture run vs its rupture-free twin) and the measured network-level readout noise of
N10-RN (Milano 2025 OU + jumps, x1, per electrode) added to every run, because a small input gives a small signal.

    python scripts/n10b_bias.py run [--procs N] [--reverse] [--limit K]
    python scripts/n10b_bias.py report          -> results/n10b_criteria.json, figures/fig25_n10b.png
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
OUT = ROOT / "results" / "n10b_criteria.json"
FIG = ROOT / "figures" / "fig25_n10b.png"
TAU, SPREAD, HOLD = 0.3, 1.0, 100
ONSET_PADS = 0.36
BIAS_X = (1.5, 2.0, 3.0)
WIDTH_FRAC = (0.03, 0.15)
RUPTURES = (0.0, 1e-3, 1e-2)
OBS = "electrodes"


def windows():
    return [(round(b * ONSET_PADS, 2), max(0.01, round(f * b * ONSET_PADS, 2))) for b in BIAS_X for f in WIDTH_FRAC]


def specs():
    wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
    out = []
    for r in RUPTURES:                                   # rupture-free first: the noise-free answer comes early
        kw = dict(n_samples=C.E4_N_SAMPLES, washout=-(-wo // HOLD) * HOLD, n_virtual=C.E4_N_VIRTUAL, layout="pads",
                  encoding="weights", model="volatile", tau_s=TAU, tau_spread=SPREAD, stride=C.N10_STRIDE, rupture=r)
        for vc, w in windows():
            for net in C.E1_SEEDS:
                out += [C.RunSpec("iid", HOLD, vc, net, rep, v_width=w, **kw) for rep in C.E1_REPLICATES]
                out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, v_width=w, **kw))
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


def run(procs, reverse=False, limit=0):
    todo = [s for s in specs() if not raw_path(s, RAW).exists() and not raw_path(s, RAW).with_suffix(".running").exists()]
    if reverse:
        todo.reverse()
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        futs = [pool.submit(run_one, s) for s in todo]                 # submitted in order: rupture-free first
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report
def _raw(spec, noisy):
    if noisy:
        import n10e_readout_noise as RN
        return RN.noisy_raw(spec, 1, "ou", "indep")
    return load_raw(spec, RAW)


def _score(job):
    from agnw_rc import evaluate as E
    spec, noisy = job
    raw = _raw(spec, noisy)
    if spec.stream == "sinesquare":
        return job, E.evaluate_sinesquare_run(raw), None
    return job, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), E.observables(raw)[OBS].shape[1]


def _cons(job):
    from agnw_rc import evaluate as E
    (a, b), noisy = job
    return job, E.evaluate_consistency(_raw(a, noisy), _raw(b, noisy))


def _twin(spec):
    from agnw_rc import evaluate as E
    o = {}
    for tag, sp in (("rupture", spec), ("no_rupture", dataclasses.replace(spec, rupture=0.0))):
        raw = load_raw(sp, RAW)
        sc = E.evaluate_iid_run(raw, E.noise_floor(raw["u"]))[OBS]
        o[tag] = {k: float(sc[k]) for k in ("MC", "IPC2", "narma10_nmse")}
    return spec, o


def report():
    from agnw_rc import criteria as K
    sp = [s for s in specs() if raw_path(s, RAW).exists()]
    jobs = [(s, n) for s in sp for n in (False, True)]
    pairs = [((s, dataclasses.replace(s, replicate=1)), n) for s in sp if s.stream == "iid" and s.replicate == 0
             and raw_path(dataclasses.replace(s, replicate=1), RAW).exists() for n in (False, True)]
    with ProcessPoolExecutor(8) as pool:
        scored = list(pool.map(_score, jobs, chunksize=2))
        cs = list(pool.map(_cons, pairs, chunksize=2))
        tw = list(pool.map(_twin, [s for s in sp if s.stream == "iid" and s.rupture
                                   and raw_path(dataclasses.replace(s, rupture=0.0), RAW).exists()]))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in C.E1_SEEDS)
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    bl = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in bl]))}
    groups = []                                           # (label, selector): all windows per rupture rate, and per bias
    for r in RUPTURES:
        groups.append((f"rupture {r:g}/s, best window", lambda s, r=r: s.rupture == r))
        for b in BIAS_X:
            vb = round(b * ONSET_PADS, 2)
            groups.append((f"rupture {r:g}/s, bias {vb} V", lambda s, r=r, vb=vb: s.rupture == r and s.v_center == vb))
    rows = []
    for noisy in (False, True):
        for lab, sel in groups:
            runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                         replicate=s.replicate, scores=sc, v_width=s.v_width) for (s, n), sc, _ in scored
                    if n == noisy and sel(s)]
            # criteria select on (hold, v_center): fold the width into the key by encoding it in v_center
            for x in runs:
                x["v_center"] = x["v_center"] + 1e-4 * x.pop("v_width")
            cons = [dict(hold=p[0].hold, v_center=p[0].v_center + 1e-4 * p[0].v_width, consistency=c)
                    for (p, n), c in cs if n == noisy and sel(p[0])]
            try:
                nf = next(f for (s, n), _, f in scored if n == noisy and sel(s) and f)
                strict = K.evaluate(runs, cons, OBS, nf, esn, floor)
                loose = K.evaluate(runs, cons, OBS, 16, esn, floor)
            except (ValueError, KeyError, StopIteration) as err:
                print(f"{lab}: incomplete ({err})")
                continue
            op = strict.get("operating_point", {})
            if "v_center" in op:
                vc = op["v_center"]
                op["v_center"], op["v_width"] = round(vc, 2), round((vc - round(vc, 2)) * 1e4, 2)
            ratios = [(o["rupture"]["MC"] + o["rupture"]["IPC2"]) / (o["no_rupture"]["MC"] + o["no_rupture"]["IPC2"])
                      for s, o in tw if sel(s)] if not noisy else []
            rows.append({"setting": lab, "readout_noise": "measured x1" if noisy else "none", **strict,
                         "G2_loose": loose["G2"], "twin_ratio_median": float(np.median(ratios)) if ratios else None,
                         "twin_collapsed": int(np.sum(np.array(ratios) < 0.5)) if ratios else None,
                         "twin_n": len(ratios)})
            r = rows[-1]
            print(f"{lab:32s} noise {r['readout_noise']:11s} op {op} MC {r['MC']['mean']:.2f}±{r['MC']['sd']:.2f} "
                  f"IPC2 {r['IPC2']['mean']:.2f}±{r['IPC2']['sd']:.2f} NARMA {r['narma10_nmse']['mean']:.3f} acc "
                  f"{r['accuracy']['mean'] if r['accuracy'] else float('nan'):.2f} cons "
                  f"{r['consistency_inc']['mean'] if r['consistency_inc'] else float('nan'):.2f} -> G0 {r['G0']} "
                  f"G1 {r['G1']} G2 loose {r['G2_loose']} | twins {r['twin_ratio_median']} "
                  f"collapsed {r['twin_collapsed']}/{r['twin_n']}", flush=True)
    ref = {x["setting"]: x for x in json.loads((ROOT / "results" / "n10r_criteria.json").read_text())["rows"]}
    OUT.write_text(json.dumps({"windows": windows(), "esn_total": esn, "noise_floor": floor, "rows": rows}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    base = [(0.0, ref["no noise (N10-E)"])] + [(r, ref[f"rupture {r:g}/s"]) for r in RUPTURES[1:] if f"rupture {r:g}/s" in ref]
    for ax, k, t, thr in zip(axes, ("MC", "IPC2", "narma10_nmse"), ("memory capacity", "degree-2 capacity",
                                                                     "NARMA-10 NMSE"), (1, 1, 0.7)):
        ax.errorbar([max(p[0], 1e-4) for p in base], [p[1][k]["mean"] for p in base], [p[1][k]["sd"] for p in base],
                    marker="s", capsize=3, label="near onset (N10-E / N10-R)")
        for noisy, mk in (("none", "o"), ("measured x1", "^")):
            pts = [r for r in rows if r["readout_noise"] == noisy and r["setting"].endswith("best window")]
            ax.errorbar([max(float(r["setting"].split()[1].rstrip(",").replace("/s", "")), 1e-4) for r in pts], [r[k]["mean"] for r in pts],
                        [r[k]["sd"] for r in pts], marker=mk, capsize=3,
                        label="bias + small input" + (", measured readout noise" if noisy != "none" else ""))
        ax.axhline(thr, color="r", ls="--", lw=0.8)
        ax.set(xscale="log", xlabel="rupture rate per junction (1/s; 1e-4 = none)", title=t)
    axes[0].legend(fontsize=7)
    fig.suptitle("N10-B: bias deep in the conducting state + small input (validation-chosen window), vs ruptures")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"run": lambda: run(a.procs, a.reverse, a.limit), "report": report}[a.stage]()
