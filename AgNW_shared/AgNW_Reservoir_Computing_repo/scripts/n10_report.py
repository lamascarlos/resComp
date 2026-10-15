#!/usr/bin/env python3
"""N10-1 report: the E4 protocol on the N10 volatile junction (docs/N10_DESIGN.md), scored against G0-G2.

Reads results/metrics.json (run scripts/evaluate_all.py first) and writes
  results/n10_1_criteria.json   G0-G2 per (tau_s, tau_spread) and readout, operating point (hold, window) chosen on
                                validation (agnw_rc.criteria), plus per-hold scores at the validation-best window
  figures/fig15_n10_1.png       scores vs hold for each tau_s and spread, with E4 baselines and thresholds

    python scripts/n10_report.py [--consolidation]
"""
import argparse
import json
import sys
from collections import defaultdict
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
OBS = ("current", "probes", "junctions")
FIG_OBS = ("current", "probes")                 # physical readouts shown per hold (junctions = oracle)


def mean(v):
    return float(np.mean(v)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidation", action="store_true")
    a = ap.parse_args()
    tag = "n10_1_con" if a.consolidation else "n10_1"
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    sel = lambda r: (r["model"] == "volatile" and bool(r.get("consolidation")) == a.consolidation  # noqa: E731
                     and r["n_samples"] == C.E4_N_SAMPLES)
    runs = [r for r in d["runs"] if sel(r)]
    cons = [c for c in d["consistency"] if sel(c)]
    seeds = C.E1_SEEDS
    floor = max(d["noise_floor_short"][f"{s}_{C.E4_N_SAMPLES}"] for s in seeds)
    base = [d["baselines_short"][f"{s}_{C.E4_N_SAMPLES}"] for s in seeds]
    esn = {n: mean([b["iid"][f"esn_{n}"]["MC"] + b["iid"][f"esn_{n}"]["IPC2"] for b in base]) for n in (32, 200)}
    ref = {name: {m: mean([b["iid"][name][m] for b in base]) for m in ("MC", "IPC2", "narma10_nmse")}
           for name in ("input_only", "delay_line_32", "esn_32")}
    groups = defaultdict(list)
    for r in runs:
        groups[(r["tau_s"], r["tau_spread"])].append(r)
    spec0 = next(s for s in C.n10_1_specs(a.consolidation) if (RAW / f"{s.name}.npz").exists())
    nch = {k: v.shape[1] for k, v in observables(load_raw(spec0, RAW)).items()}
    out = {"noise_floor": floor, "esn_total": esn, "baselines": ref, "n_channels": nch, "groups": []}
    for (tau, spread), rs in sorted(groups.items()):
        cs = [c for c in cons if c["tau_s"] == tau and c["tau_spread"] == spread]
        g = {"tau_s": tau, "tau_spread": spread, "n_runs": len(rs), "criteria": {},
             "per_hold": {o: {} for o in FIG_OBS}}
        for obs in OBS:
            try:
                g["criteria"][obs] = K.evaluate(rs, cs, obs, nch[obs], esn, floor)
            except (ValueError, StopIteration, KeyError):
                continue
        for obs, hold in [(o, h) for o in FIG_OBS for h in sorted({r["hold"] for r in rs})]:
            best = None
            for vc in sorted({r["v_center"] for r in rs}):
                iid = [r["scores"][obs] for r in rs if r["hold"] == hold and r["v_center"] == vc
                       and r["stream"] == "iid"]
                if not iid:
                    continue
                val = mean([s["val_MC"] + s["val_IPC2"] for s in iid])
                if best is None or val > best[0]:
                    ss = [r["scores"][obs]["accuracy"] for r in rs if r["hold"] == hold
                          and r["v_center"] == vc and r["stream"] == "sinesquare"]
                    ci = [c["consistency"][obs + "_state_inc"] for c in cs if c["hold"] == hold
                          and c["v_center"] == vc]
                    best = (val, {"v_center": vc, "n": len(iid),
                                  **{m: mean([s[m] for s in iid]) for m in ("MC", "IPC2", "narma10_nmse")},
                                  **{m + "_sd": float(np.std([s[m] for s in iid], ddof=1)) if len(iid) > 1 else 0.0
                                     for m in ("MC", "IPC2", "narma10_nmse")},
                                  "accuracy": mean(ss), "consistency_inc": mean(ci),
                                  "drift_max": float(max(abs(r["drift"]) for r in rs if r["hold"] == hold))})
            g["per_hold"][obs][str(hold)] = best[1] if best else None
        out["groups"].append(g)
    (ROOT / "results" / f"{tag}_criteria.json").write_text(json.dumps(out, indent=1))
    print(f"floor {floor:.3f}; ESN totals {esn}; input-only {ref['input_only']}")
    for g in out["groups"]:
        print(f"tau_s {g['tau_s']:g} s, spread {g['tau_spread']:g} ({g['n_runs']} runs)")
        for obs, r in g["criteria"].items():
            op = r["operating_point"]
            print(f"  {obs:9s} hold {op['hold']:5d} v {op['v_center']:.2f}: MC {r['MC']['mean']:.2f}±{r['MC']['sd']:.2f} "
                  f"IPC2 {r['IPC2']['mean']:.2f}±{r['IPC2']['sd']:.2f} NARMA {r['narma10_nmse']['mean']:.3f} "
                  f"cons {r['consistency_inc']['mean'] if r['consistency_inc'] else float('nan'):.2f} "
                  f"acc {r['accuracy']['mean'] if r['accuracy'] else float('nan'):.2f} -> G0 {r['G0']} G1 {r['G1']} "
                  f"G2 {r['G2']}")
    figure(out, tag)


def figure(out, tag):
    panels = (("MC", "memory capacity"), ("IPC2", "degree-2 capacity"), ("narma10_nmse", "NARMA-10 NMSE"),
              ("accuracy", "sine/square accuracy"), ("consistency_inc", "increment consistency"))
    fig, axes2 = plt.subplots(len(FIG_OBS), 5, figsize=(20, 3.6 * len(FIG_OBS)), squeeze=False)
    colors = {0.03: "C0", 0.3: "C1", 3.0: "C2"}
    for row, obs in zip(axes2, FIG_OBS):
        for g in out["groups"]:
            ph = g["per_hold"][obs]
            hs = sorted(int(h) for h, v in ph.items() if v)
            for ax, (key, label) in zip(row, panels):
                y = [ph[str(h)][key] for h in hs]
                e = [ph[str(h)].get(key + "_sd", 0) for h in hs]
                ax.errorbar(np.array(hs) * C.DT, y, yerr=e, color=colors.get(g["tau_s"], "k"),
                            ls="-" if g["tau_spread"] == 0 else "--", marker="o", ms=3, capsize=2,
                            label=f"τ_s {g['tau_s']:g} s, spread {g['tau_spread']:g}")
        for ax, (key, label) in zip(row, panels):
            ax.set_title(f"{obs}: {label}", fontsize=10)
    axes = axes2[0]
    b = out["baselines"]
    for ax, (key, label) in [(a, pk) for row in axes2 for a, pk in zip(row, panels)]:
        ax.set_xscale("log")
        ax.set(xlabel="input hold time (s)")
        if key in ("MC", "IPC2", "narma10_nmse"):
            ax.axhline(b["input_only"][key], color="0.5", ls="-.", lw=0.8, label="input only")
        thr = {"MC": K.G1_MC, "IPC2": K.G1_IPC2, "narma10_nmse": K.G1_NARMA, "accuracy": K.G2_ACCURACY,
               "consistency_inc": K.G0_MIN}[key]
        ax.axhline(thr, color="r", ls=":", lw=1, label="criterion")
    axes[0].text(0.02, 0.97, f"ESN-32: MC {b['esn_32']['MC']:.1f}, degree-2 {b['esn_32']['IPC2']:.1f}",
                 transform=axes[0].transAxes, va="top", fontsize=8, color="m")
    h, lab = axes2[0][2].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle(f"N10-1{' (consolidation on)' if 'con' in tag else ' (pure volatile)'}: current (top) and 32 probes "
                 "(bottom) at the validation-best window, 3 networks x 2 replicates")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = ROOT / "figures" / ("fig15_n10_1.png" if tag == "n10_1" else f"fig15_{tag}.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
