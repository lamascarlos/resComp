#!/usr/bin/env python3
"""N10-M report: memory study on the N10 pads (config.N10M_BLOCKS), scored like N10-E (agnw_rc.criteria, G2 strict
and loose). Readout: the 16 electrode currents ("electrodes"). Reference: N10-E (results/n10e_criteria.json, pads).
The "delays" block feeds input copies delayed by 0-7 samples to the 8 driven pads, so memory is partly supplied by
the interface: it is compared with 8-tap linear and quadratic delay lines on the same streams.

    python scripts/n10m_report.py        -> results/n10m_criteria.json, figures/fig18_n10m.png
"""
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

from agnw_rc import baselines as B  # noqa: E402
from agnw_rc import config as C  # noqa: E402
from agnw_rc import criteria as K  # noqa: E402
from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.drive import input_stream, load_raw, raw_path  # noqa: E402

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "n10m_criteria.json"
FIG = ROOT / "figures" / "fig18_n10m.png"
OBS = "electrodes"


def label(s):
    tag = f"τ_s {s.tau_s:g} s, spread {s.tau_spread:g}, hold {s.hold * C.DT:g} s"
    if s.encoding == "delays":
        tag += ", delays"
    if s.sigma:
        tag += f", noise {s.sigma:g}"
    return tag


def short(config):
    """Axis label: what changes relative to N10-E (tau_s 0.3 s, spread 1, hold 0.1 s, weights, no noise)."""
    if config.startswith("N10-E"):
        return "N10-E (ref.)"
    parts = [p for p in config.split(", ") if p not in ("τ_s 0.3 s", "spread 1", "hold 0.1 s")]
    return ", ".join(p for p in parts if not p.startswith("hold")) or config


def score(spec):
    raw = load_raw(spec, RAW)
    if spec.stream == "sinesquare":
        return spec, E.evaluate_sinesquare_run(raw), None
    return spec, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), E.observables(raw)[OBS].shape[1]


def cons(pair):
    a, b = pair
    return a, E.evaluate_consistency(load_raw(a, RAW), load_raw(b, RAW))


def delay_baselines():
    out = {}
    for name, fn in (("delay_line_8", lambda u: B.delay_line(u, 8)), ("quadratic_delay_8",
                                                                        lambda u: B.quadratic_delay_line(u, 8))):
        vals = []
        for seed in C.E1_SEEDS:
            u = input_stream(C.RunSpec("iid", 10, 1.0, seed, n_samples=C.E4_N_SAMPLES))[0]
            s = E.score_iid(fn(u), u, E.noise_floor(u))
            vals.append({m: s[m] for m in ("MC", "IPC2", "narma10_nmse")})
        out[name] = {m: float(np.mean([v[m] for v in vals])) for m in ("MC", "IPC2", "narma10_nmse")}
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = [s for s in C.n10m_specs() if raw_path(s, RAW).exists()]
    missing = len(C.n10m_specs()) - len(sp)
    with ProcessPoolExecutor(16) as pool:
        scored = list(pool.map(score, sp))
        pairs = []
        for s in sp:
            if s.stream == "iid" and s.replicate == 0:
                t = C.RunSpec(**{**s.__dict__, "replicate": 1})
                if raw_path(t, RAW).exists():
                    pairs.append((s, t))
        cs = list(pool.map(cons, pairs))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in C.E1_SEEDS)
    base = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in base]))}
    groups = {}
    for s, sc, nf in scored:
        groups.setdefault(label(s), {"runs": [], "cons": [], "nfeat": None})
        g = groups[label(s)]
        g["runs"].append(dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                              replicate=s.replicate, scores=sc))
        g["nfeat"] = g["nfeat"] or nf
    for a, c in cs:
        groups[label(a)]["cons"].append(dict(hold=a.hold, v_center=a.v_center, consistency=c))
    rows = []
    for name, g in groups.items():
        try:
            strict = K.evaluate(g["runs"], g["cons"], OBS, g["nfeat"], esn, floor)
            loose = K.evaluate(g["runs"], g["cons"], OBS, 16, esn, floor)
        except (ValueError, KeyError, StopIteration) as err:
            print(f"{name}: incomplete ({err})")
            continue
        rows.append({"config": name, **strict, "G2_loose": loose["G2"], "esn_total_loose": loose["esn_total"],
                     "n_runs": len(g["runs"])})
    ref = json.loads((ROOT / "results" / "n10e_criteria.json").read_text())["layouts"]["pads"][OBS]
    rows.insert(0, {"config": "N10-E reference: τ_s 0.3 s, spread 1, hold 0.1 s", **ref,
                    "G2_loose": ref["G2_loose"], "n_runs": 18})
    out = {"noise_floor": floor, "esn_total": esn, "delay_baselines": delay_baselines(), "missing_runs": missing,
           "rows": rows}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"missing runs: {missing}; delay baselines: {out['delay_baselines']}")
    for r in rows:
        print(f"{r['config']:50s} MC {r['MC']['mean']:.2f}±{r['MC']['sd']:.2f} IPC2 {r['IPC2']['mean']:.2f}±"
              f"{r['IPC2']['sd']:.2f} NARMA {r['narma10_nmse']['mean']:.3f}±{r['narma10_nmse']['sd']:.3f} "
              f"acc {r['accuracy']['mean'] if r['accuracy'] else float('nan'):.2f} "
              f"cons {r['consistency_inc']['mean'] if r['consistency_inc'] else float('nan'):.2f} "
              f"V {r['operating_point']['v_center']} -> G0 {r['G0']} G1 {r['G1']} G2 strict {r['G2']} loose {r['G2_loose']}")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    x = np.arange(len(rows))
    for ax, m, t in zip(axes, ("MC", "IPC2", "narma10_nmse", "total_capacity"),
                        ("memory capacity", "degree-2 capacity", "NARMA-10 NMSE", "MC + degree-2")):
        ax.bar(x, [r[m]["mean"] for r in rows], yerr=[r[m]["sd"] for r in rows], capsize=3,
               color=["0.6"] + ["C1"] * (len(rows) - 1))
        ax.set_xticks(x, [short(r["config"]) for r in rows], fontsize=8, rotation=45, ha="right")
        ax.set_title(t)
        if m in ("MC", "IPC2"):
            ax.axhline(1, color="r", ls="--", lw=0.8)
        if m == "narma10_nmse":
            ax.axhline(0.7, color="r", ls="--", lw=0.8)
        if m == "total_capacity":
            ax.axhline(esn[16] / 2, color="m", ls=":", label="G2 loose")
            ax.axhline(esn[200] / 2, color="m", ls="--", label="G2 strict")
            ax.legend(fontsize=7)
        if m == "MC":
            ax.axhline(out["delay_baselines"]["delay_line_8"]["MC"], color="k", ls="-.", lw=0.8,
                       label="8-tap delay line")
            ax.legend(fontsize=7)
    fig.suptitle("N10-M: memory study on the N10 pads (16 electrode currents, 3 networks x 2 replicates; grey: N10-E)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
