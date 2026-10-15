#!/usr/bin/env python3
"""N10-LS: ten networks per point for the size result (plan: docs/PLAN_N10L_SEEDS.md, analysis fixed in advance).

Runs: network seeds 4-10 (seeds 1-3 exist) at 1 mm ("pads", the N10-E / N10-R protocol), 2 mm and 2.83 mm ("pads_scaled",
the N10-L protocol), rupture rates 0 and 1e-2 /s per junction; 2 windows x (2 i.i.d. replicates + 1 sine/square).
Report, per size and rate over seeds 1-10: pooled G0-G2 (agnw_rc.criteria), networks passing G1 individually
(k of 10, Clopper-Pearson 95 %), sd between networks of the per-network mean MC + degree-2 (bootstrap 95 %) and the
1 mm / 2.83 mm sd ratio, median capacity kept rho vs the rupture-free twin (bootstrap 95 %) and collapses (rho < 0.5).

    python scripts/n10l_seeds.py run [--procs N] [--only SIZE:RATE,...]   -> results/raw/*.npz
    python scripts/n10l_seeds.py report [--seeds 1-10]                    -> results/n10l_seeds.json, figures/fig31_n10l_seeds.png
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

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "n10l_seeds.json"
FIG = ROOT / "figures" / "fig31_n10l_seeds.png"
TAU, SPREAD, HOLD = 0.3, 1.0, 100
SIDES = (1.0, 2.0, 2.83)
RATES = (1e-2, 0.0)                      # queue order: realistic noise first
NEW_SEEDS = tuple(range(4, 11))
# Weekend extension (2026-10-08, both machines Fri night - Mon night), in priority order after the seeds:
#   combined noise at 2.83 mm: ruptures 1e-2 /s + Gaussian junction noise sigma 0.03 (all 10 networks; the measured
#   readout noise is added post hoc in the report);  4 mm at 0 and 1e-2 /s (networks 1-3; 4-10 added 2026-10-09,
#   N10-LS4: confirms or not the pooled G1 that 3 networks gave).
SIGMA_COMB = 0.03
# group key: (side, rate, sigma, seeds to run)
GROUPS = ([(s, r, 0.0, NEW_SEEDS) for s in sorted(SIDES, reverse=True) for r in RATES]
          + [(2.83, 1e-2, SIGMA_COMB, tuple(range(1, 11)))]
          + [(4.0, r, 0.0, tuple(range(1, 11))) for r in RATES])   # 4 mm: networks 1-3 (2026-10-08), 4-10 (N10-LS4, 2026-10-09)
OBS = "electrodes"
G1_MC, G1_IPC2, G1_NARMA, G1_LAGS = 1.0, 1.0, 0.70, 2


def specs_for(side, rate, seeds, sigma=0.0):
    wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
    kw = dict(n_samples=C.E4_N_SAMPLES, washout=-(-wo // HOLD) * HOLD, n_virtual=C.E4_N_VIRTUAL, encoding="weights",
              model="volatile", tau_s=TAU, tau_spread=SPREAD, stride=C.N10_STRIDE, rupture=rate, sigma=sigma)
    if side == 1.0:                                     # the N10-E / N10-R runs (layout "pads", 1 mm working model)
        windows, kw = C.N10M_WINDOWS, {**kw, "layout": "pads"}
    else:                                               # the N10-L runs
        on = json.loads((ROOT / "results" / "n10l_sweep.json").read_text())[f"{side:g}"]["onset_V"]
        w = round(0.45 * on, 2)
        windows, kw = [(round(1.05 * on, 2), w), (round(1.25 * on, 2), w)], {**kw, "layout": "pads_scaled", "side": side}
    out = []
    for vc, w in windows:
        for net in seeds:
            out += [C.RunSpec("iid", HOLD, vc, net, rep, v_width=w, **kw) for rep in C.E1_REPLICATES]
            out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, v_width=w, **kw))
    return out


def run_one(spec):
    from agnw_rc.multidrive import simulate_multi
    import socket
    f = raw_path(spec, RAW)
    mark = f.with_suffix(".running")
    if f.exists():
        return spec.name + " (exists)"
    try:                                                # atomic claim: several node jobs share one run list (cluster)
        fd = os.open(mark, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return spec.name + " (running elsewhere)"
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{socket.gethostname()} {os.getpid()}")
    save_raw(spec, simulate_multi(spec), RAW)
    mark.unlink(missing_ok=True)
    return spec.name


def run(procs, only=""):
    """only = "side:rate[:sigma],..." restricts to those groups (in GROUPS order: seeds first)."""
    groups = GROUPS
    if only:
        keep = [tuple(float(x) for x in k.split(":")) for k in only.split(",")]
        groups = [g for g in GROUPS if any(g[0] == k[0] and g[1] == k[1] and g[2] == (k[2] if len(k) > 2 else 0.0)
                                           for k in keep)]
    todo = [s for side, r, sig, seeds in groups for s in specs_for(side, r, seeds, sig)
            if not raw_path(s, RAW).exists() and not raw_path(s, RAW).with_suffix(".running").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report (analysis of docs/PLAN_N10L_SEEDS.md)
def _score(spec):
    from agnw_rc import evaluate as E
    raw = load_raw(spec, RAW)
    if spec.stream == "sinesquare":
        return spec, E.evaluate_sinesquare_run(raw), None
    return spec, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), E.observables(raw)[OBS].shape[1]


def _cons(pair):
    from agnw_rc import evaluate as E
    return pair[0], E.evaluate_consistency(load_raw(pair[0], RAW), load_raw(pair[1], RAW))


def _floor(spec):
    from agnw_rc import evaluate as E
    return E.noise_floor(load_raw(spec, RAW)["u"])


def clopper_pearson(k, n, a=0.05):
    from scipy.stats import beta
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def boot(values, fn, n=10_000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    st = np.array([fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)])
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5)), st


def sweep4():
    """Onset of the 4 mm network (as N10-L: network 1, pads_scaled) -> added to results/n10l_sweep.json."""
    import n10l_size as L
    p = ROOT / "results" / "n10l_sweep.json"
    sw = json.loads(p.read_text())
    if "4" not in sw:
        k, v = L.sweep_one(4.0)
        sw[k] = v
        p.write_text(json.dumps(sw, indent=1))
    print(sw["4"])


def analyse(groups, seeds, procs=8):
    """Analysis of PLAN_N10L_SEEDS.md for {(side, rate, sigma): [existing RunSpecs]} -> {"seeds", "noise_floor",
    "esn_total", "groups": [...]} (also used by n10sb_layout.py for other layouts)."""
    from agnw_rc import criteria as K
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    bl = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in bl]))}
    allsp = [s for g in groups.values() for s in g]
    with ProcessPoolExecutor(procs) as pool:
        scored = dict(((s.name, (sc, n)) for s, sc, n in pool.map(_score, allsp, chunksize=2)))
        cons = dict((a.name, c) for a, c in pool.map(_cons, [(s, dataclasses.replace(s, replicate=1)) for s in allsp
                    if s.stream == "iid" and s.replicate == 0 and raw_path(dataclasses.replace(s, replicate=1), RAW).exists()]))
        floors = list(pool.map(_floor, [s for s in allsp if s.stream == "iid" and s.replicate == 0]))
    floor = float(max(floors))
    out = {"seeds": list(seeds), "noise_floor": floor, "esn_total": esn, "groups": []}
    for (side, r, sig), sp in groups.items():
        if not sp:
            continue
        runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                     replicate=s.replicate, scores=scored[s.name][0]) for s in sp]
        cc = [dict(hold=s.hold, v_center=s.v_center, consistency=cons[s.name]) for s in sp if s.name in cons]
        nf = next(scored[s.name][1] for s in sp if scored[s.name][1])
        pooled = K.evaluate(runs, cc, OBS, nf, esn, floor)
        loose = K.evaluate(runs, cc, OBS, 16, esn, floor)
        vc = pooled["operating_point"]["v_center"]
        at = [s for s in sp if s.stream == "iid" and abs(s.v_center - vc) < 1e-9]
        per_net = {}
        for s in at:
            sc = scored[s.name][0][OBS]
            per_net.setdefault(s.network_seed, []).append(sc)
        nets = []
        for net, scs in sorted(per_net.items()):
            mc, ipc = np.mean([x["MC"] for x in scs]), np.mean([x["IPC2"] for x in scs])
            narma = np.mean([x["narma10_nmse"] for x in scs])
            lags = min(int(np.sum(np.asarray(x["memory_function"][1:]) > floor)) for x in scs)
            nets.append({"seed": net, "MC": float(mc), "IPC2": float(ipc), "narma10_nmse": float(narma), "lags": lags,
                         "G1": bool(mc >= G1_MC and ipc >= G1_IPC2 and narma <= G1_NARMA and lags >= G1_LAGS)})
        k, n = sum(x["G1"] for x in nets), len(nets)
        tot = [x["MC"] + x["IPC2"] for x in nets]
        sd_lo, sd_hi, _ = boot(tot, lambda v: v.std(ddof=1))
        g = {"side_mm": side, "rupture": r, "sigma": sig, "n_networks": n, "pooled": {k2: pooled[k2] for k2 in
             ("operating_point", "MC", "IPC2", "narma10_nmse", "accuracy", "consistency_inc", "G0", "G1", "G2")},
             "G2_loose": loose["G2"], "networks": nets, "networks_pass_G1": k, "networks_pass_G1_CI95": clopper_pearson(k, n),
             "spread_sd_total": float(np.std(tot, ddof=1)) if n > 1 else None, "spread_sd_CI95": [sd_lo, sd_hi],
             "per_network_total": tot}
        if r > 0:
            rho = []
            for s in at:
                twin = dataclasses.replace(s, rupture=0.0, sigma=0.0)
                if raw_path(twin, RAW).exists() and twin.name in scored:
                    a, b = scored[s.name][0][OBS], scored[twin.name][0][OBS]
                    rho.append((a["MC"] + a["IPC2"]) / (b["MC"] + b["IPC2"]))
            if rho:
                lo, hi, _ = boot(rho, np.median)
                g.update(rho_median=float(np.median(rho)), rho_CI95=[lo, hi], collapsed=int(np.sum(np.array(rho) < 0.5)),
                         rho_n=len(rho), rho_values=[float(x) for x in rho])
        if sig > 0:                                     # + measured network-level readout noise (N10-RN, x1)
            import n10e_readout_noise as RN
            from agnw_rc import evaluate as E
            nr = []
            for s in sp:
                raw = RN.noisy_raw(s, 1, "ou", "indep")
                sc = E.evaluate_sinesquare_run(raw) if s.stream == "sinesquare" else E.evaluate_iid_run(raw, E.noise_floor(raw["u"]))
                nr.append(dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                               replicate=s.replicate, scores=sc))
            ncc = [dict(hold=s.hold, v_center=s.v_center, consistency=E.evaluate_consistency(
                   RN.noisy_raw(s, 1, "ou", "indep"), RN.noisy_raw(dataclasses.replace(s, replicate=1), 1, "ou", "indep")))
                   for s in sp if s.stream == "iid" and s.replicate == 0]
            pn = K.evaluate(nr, ncc, OBS, nf, esn, floor)
            g["with_readout_noise"] = {k2: pn[k2] for k2 in ("operating_point", "MC", "IPC2", "narma10_nmse", "G0", "G1")}
        out["groups"].append(g)
        print(f"{side:g} mm, r {r:g}, sigma {sig:g}: pooled G1 {pooled['G1']} (MC {pooled['MC']['mean']:.2f}±{pooled['MC']['sd']:.2f}, IPC2 "
              f"{pooled['IPC2']['mean']:.2f}±{pooled['IPC2']['sd']:.2f}, NARMA {pooled['narma10_nmse']['mean']:.3f}); "
              f"networks passing {k}/{n}; sd(MC+IPC2) {g['spread_sd_total']}"
              + (f"; rho {g['rho_median']:.2f} [{g['rho_CI95'][0]:.2f}, {g['rho_CI95'][1]:.2f}], collapsed {g['collapsed']}/{g['rho_n']}"
                 if "rho_median" in g else ""), flush=True)
    return out


def report(seeds, procs=8):
    keys = [(side, r, 0.0) for side in SIDES for r in RATES] + [(2.83, 1e-2, SIGMA_COMB)] + [(4.0, r, 0.0) for r in RATES]
    sw = json.loads((ROOT / "results" / "n10l_sweep.json").read_text())
    keys = [k for k in keys if k[0] != 4.0 or "4" in sw]
    groups = {k: [s for s in specs_for(k[0], k[1], seeds, k[2]) if raw_path(s, RAW).exists()] for k in keys}
    out = analyse(groups, seeds, procs)
    # 1 mm / 2.83 mm spread ratio (bootstrap over networks, independently per size)
    for r in RATES:
        a = next((g for g in out["groups"] if g["side_mm"] == 1.0 and g["rupture"] == r), None)
        b = next((g for g in out["groups"] if g["side_mm"] == 2.83 and g["rupture"] == r), None)
        if a and b and a["n_networks"] > 2 and b["n_networks"] > 2:
            _, _, sa = boot(a["per_network_total"], lambda v: v.std(ddof=1), seed=1)
            _, _, sb = boot(b["per_network_total"], lambda v: v.std(ddof=1), seed=2)
            ok = (sa > 0) & (sb > 0)                    # drop degenerate resamples (one network drawn n times)
            ratio = sa[ok] / sb[ok]
            out.setdefault("spread_ratio_1mm_over_2.83mm", {})[f"{r:g}"] = {
                "ratio": a["spread_sd_total"] / b["spread_sd_total"],
                "CI95": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))]}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out.get("spread_ratio_1mm_over_2.83mm", {}), indent=1))
    figure(out)


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sides = sorted({g["side_mm"] for g in out["groups"]})
    fig, axes = plt.subplots(1, len(sides), figsize=(5.7 * len(sides), 4.8))
    for ax, side in zip(axes, sides):
        for r, mk, col in ((0.0, "o", "0.5"), (1e-2, "^", "C3")):
            g = next((g for g in out["groups"] if g["side_mm"] == side and g["rupture"] == r), None)
            if not g:
                continue
            x = [n["MC"] for n in g["networks"]]
            y = [n["IPC2"] for n in g["networks"]]
            ok = [n["G1"] for n in g["networks"]]
            ax.scatter(x, y, marker=mk, c=[col if o else "none" for o in ok], edgecolors=col, s=60,
                       label=f"{'no ruptures' if r == 0 else 'ruptures 1e-2 /s'}: {g['networks_pass_G1']}/{g['n_networks']} pass G1")
        ax.axvline(G1_MC, color="r", ls="--", lw=0.8)
        ax.axhline(G1_IPC2, color="r", ls="--", lw=0.8)
        ax.set(xlabel="memory capacity (mean of 2 replicates)", ylabel="degree-2 capacity", title=f"{side:g} mm",
               xlim=(0, None), ylim=(0, None))
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"N10-LS: each point one network (seeds {out['seeds'][0]}-{out['seeds'][-1]}), at the validation-chosen window; "
                 "filled = passes G1 individually (open = fails;\nopen points above both dashed lines fail on NARMA-10 NMSE > 0.70)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report", "sweep4"])
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--only", default="")
    ap.add_argument("--seeds", default="1-10")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    {"run": lambda: run(a.procs, a.only), "report": lambda: report(tuple(range(lo, hi + 1)), min(a.procs, 8)),
     "sweep4": sweep4}[a.stage]()
