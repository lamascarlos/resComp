#!/usr/bin/env python3
"""E4 report (review 2026-09-27, decisive plan): time-scale-matched inputs scored against the go/no-go criteria.

Reads results/metrics.json (run scripts/evaluate_all.py first) and the E4 raw files, and writes
  results/e4_criteria.json   G0-G2 per model and readout, operating point chosen on validation, drift per run
  figures/fig12_e4.png       scores vs hold time per model, with baselines and the G1 thresholds

    python scripts/e4_report.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc import criteria as K  # noqa: E402
from agnw_rc.drive import load_raw  # noqa: E402
from agnw_rc.evaluate import observables  # noqa: E402

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "e4_criteria.json"
FIG = ROOT / "figures" / "fig12_e4.png"
OBS = ("current", "probes", "junctions")
MODELS = ("stochastic", "filament")
DRIFT_MAX = 0.02


def is_e4(r):
    return r["n_samples"] == C.E4_N_SAMPLES and r["n_virtual"] == C.E4_N_VIRTUAL and bool(r["washout"])


def model_of(r):
    return r["model"] or "stochastic"


def n_channels(model):
    spec = C.e4_specs((model,))[0]
    raw = load_raw(spec, RAW)
    return {k: v.shape[1] for k, v in observables(raw).items()}


def n4_state_diagnostic():
    """How much the N4 network state moves under E4 drive (i.i.d. runs, replicate 0): junctions pinned at
    |Lambda| >= 0.99 Lambda_crit or at 0, junctions changing by > 1e-3 Lambda_crit per input sample, and the relative
    variation of the end-of-hold conductance and its correlation with the current and previous input."""
    rows = []
    for spec in C.e4_specs(("filament",)):
        if spec.stream != "iid" or spec.replicate != 0:
            continue
        raw = load_raw(spec, RAW)
        lam = raw["junction_lambda"].astype(float)
        n_e = int(raw["n_electrodes"])
        cur = raw["current"].reshape(len(lam), -1, n_e)[:, -1, raw["electrode_roles"] == "in"].sum(axis=1)
        g = cur / raw["v_electrodes"][:, 0]
        change = np.abs(np.diff(lam, axis=0)) > 1e-3
        rows.append({"hold": spec.hold, "v_center": spec.v_center, "network_seed": spec.network_seed,
                     "pinned_fraction": float(np.mean((np.abs(lam) >= 0.99) | (lam == 0))),
                     "changing_per_sample": float(change.sum(axis=1).mean()),
                     "ever_changing": int(change.any(axis=0).sum()), "n_junctions": int(lam.shape[1]),
                     "G_rel_std": float(g.std() / g.mean()),
                     "corr_G_u": float(np.corrcoef(g, raw["u"])[0, 1]),
                     "corr_G_u_prev": float(np.corrcoef(g[1:], raw["u"][:-1])[0, 1])})
    return rows


def main():
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    runs = [r for r in d["runs"] if is_e4(r)]
    cons = [c for c in d["consistency"] if is_e4(c)]
    seeds = C.E1_SEEDS
    floor = max(d["noise_floor_short"][f"{s}_{C.E4_N_SAMPLES}"] for s in seeds)
    base = [d["baselines_short"][f"{s}_{C.E4_N_SAMPLES}"] for s in seeds]
    esn = {n: float(np.mean([b["iid"][f"esn_{n}"]["MC"] + b["iid"][f"esn_{n}"]["IPC2"] for b in base]))
           for n in (32, 200)}
    ref = {name: {m: float(np.mean([b["iid"][name][m] for b in base])) for m in ("MC", "IPC2", "narma10_nmse")}
           for name in ("input_only", "delay_line_32", "esn_32")}
    out = {"noise_floor": floor, "esn_total": esn, "baselines": ref, "models": {}}
    for model in MODELS:
        rm = [r for r in runs if model_of(r) == model]
        cm = [c for c in cons if model_of(c) == model]
        if not rm:
            continue
        nch = n_channels(model)
        res = {obs: K.evaluate(rm, cm, obs, nch[obs], esn, floor) for obs in OBS}
        drifts = [{"hold": r["hold"], "v_center": r["v_center"], "network_seed": r["network_seed"],
                   "stream": r["stream"], "replicate": r["replicate"], "drift": r["drift"]} for r in rm]
        out["models"][model] = {"criteria": res, "n_channels": nch, "n_runs": len(rm),
                                "drift_max_abs": float(max(abs(x["drift"]) for x in drifts)),
                                "n_drifting": int(sum(abs(x["drift"]) > DRIFT_MAX for x in drifts)),
                                "drift": drifts}
    if "filament" in out["models"]:
        diag = n4_state_diagnostic()
        out["models"]["filament"]["state_diagnostic"] = diag
        keys = ("pinned_fraction", "changing_per_sample", "G_rel_std", "corr_G_u", "corr_G_u_prev")
        print("N4 state (range over runs):", {k: (round(min(r[k] for r in diag), 4), round(max(r[k] for r in diag), 4))
                                              for k in keys})
    OUT.write_text(json.dumps(out, indent=1))
    for model, m in out["models"].items():
        print(f"{model}: {m['n_runs']} runs, max |drift| {m['drift_max_abs']:.3f} ({m['n_drifting']} > 2 %)")
        for obs, r in m["criteria"].items():
            op = r["operating_point"]
            print(f"  {obs:9s} hold {op['hold']:5d} v {op['v_center']:.2f}: MC {r['MC']['mean']:.2f}±"
                  f"{r['MC']['sd']:.2f} IPC2 {r['IPC2']['mean']:.2f}±{r['IPC2']['sd']:.2f} NARMA "
                  f"{r['narma10_nmse']['mean']:.3f} cons_inc "
                  f"{r['consistency_inc']['mean'] if r['consistency_inc'] else float('nan'):.2f} acc "
                  f"{r['accuracy']['mean'] if r['accuracy'] else float('nan'):.2f} -> G0 {r['G0']} G1 {r['G1']} "
                  f"G2 {r['G2']}")
    figure(runs, cons, out)


def figure(runs, cons, out):
    models = [m for m in MODELS if m in out["models"]]
    panels = (("MC", "memory capacity"), ("IPC2", "degree-2 capacity"), ("narma10_nmse", "NARMA-10 NMSE"),
              ("cons", "increment consistency"))
    fig, axes = plt.subplots(len(models), 4, figsize=(15, 3.4 * len(models)), squeeze=False)
    colors = dict(zip(OBS, ("C0", "C1", "C2")))
    for i, model in enumerate(models):
        rm = [r for r in runs if model_of(r) == model and r["stream"] == "iid"]
        cm = [c for c in cons if model_of(c) == model]
        holds = sorted({r["hold"] for r in rm})
        centers = sorted({r["v_center"] for r in rm})
        for j, (key, label) in enumerate(panels):
            ax = axes[i, j]
            for obs in OBS:
                for vc, ls, wname in zip(centers, ("-", "--", ":"), ("lower", "upper")):
                    y, e = [], []
                    for h in holds:
                        if key == "cons":
                            v = [c["consistency"][obs + "_state_inc"] for c in cm
                                 if c["hold"] == h and c["v_center"] == vc]
                        else:
                            v = [r["scores"][obs][key] for r in rm if r["hold"] == h and r["v_center"] == vc]
                        y.append(np.mean(v) if v else np.nan)
                        e.append(np.std(v, ddof=1) if len(v) > 1 else 0)
                    ax.errorbar(np.array(holds) * C.DT, y, yerr=e, color=colors[obs], ls=ls, marker="o", ms=3,
                                capsize=2, label=f"{obs}, {wname} window")
            if key == "narma10_nmse":
                for name, c in (("input_only", "0.4"), ("delay_line_32", "k"), ("esn_32", "m")):
                    ax.axhline(out["baselines"][name][key], color=c, lw=0.8, ls="-.", label=name)
            elif key != "cons":                        # capacities: baselines far off scale, quoted in the panel
                b = out["baselines"]
                ax.text(0.02, 0.97, f"delay line 32: {b['delay_line_32'][key]:.1f}\nESN-32: {b['esn_32'][key]:.1f}",
                        transform=ax.transAxes, va="top", fontsize=8, color="m")
                ax.set_ylim(-0.1, 2.0)
            thr = {"MC": K.G1_MC, "IPC2": K.G1_IPC2, "narma10_nmse": K.G1_NARMA, "cons": K.G0_MIN}[key]
            ax.axhline(thr, color="r", lw=1, ls=(0, (1, 1)), label="G0/G1 threshold")
            ax.set_xscale("log")
            ax.set_xticks([h * C.DT for h in holds], [f"{h * C.DT:g}" for h in holds])
            ax.minorticks_off()
            ax.set_xlabel("input hold time (s)")
            ax.set_title(f"{model} ({' / '.join(f'{v:g}' for v in centers)} V): {label}", fontsize=10)
            if key == "narma10_nmse":
                ax.set_ylim(0, 1.2)
    h, lab = axes[0, 2].get_legend_handles_labels()          # the NARMA panel has every entry
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("E4: time-scale-matched inputs (800 samples, 50 virtual nodes, washout 3 forgetting times)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
