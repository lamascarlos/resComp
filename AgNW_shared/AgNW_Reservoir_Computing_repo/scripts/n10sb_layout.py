#!/usr/bin/env python3
"""N10-SB: the noise and size story for the split-band layout (b), on equal footing with the checkerboard pads (a)
(plan and decision rules: docs/PLAN_SPLIT_BANDS.md, fixed before the runs).

Layout "split_bands": the framework's two 5 % edge bands cut into 8 driven segments (left) and 8 grounded segments
(right), random fixed input weights, readout = the 16 electrode currents. Larger networks are the 1 mm device enlarged
geometrically (the bands are 5 % of the side, so segments grow with the side, as pads_scaled does for the pads).
Protocol unchanged from N10-E / N10-L / N10-LS: tau_s 0.3 s, spread 1, 100 ms per input, 800 inputs, 50 virtual nodes,
windows 1.05 / 1.25 x the onset with width 0.45 x onset, 2 i.i.d. replicates + 1 sine/square per window and network.

Groups (side mm, rupture /s per junction, networks):
  B1/B3  1, 2, 2.83 mm at 0 and 1e-2, networks 1-10      (N10-E + N10-LS for the pads)
  B2     1 mm at 1e-4, 1e-3, 3e-3, networks 1-3          (N10-R / N10-L rupture ladder for the pads)
Network 1 at 1 mm, no ruptures = the N10-E0 pilot runs (same specs, reused).

    python scripts/n10sb_layout.py sweep [--sides 1,2,2.83]          -> results/n10sb_sweep.json
    python scripts/n10sb_layout.py run [--procs N] [--only SIDE:RATE,... | bias | win15]
    python scripts/n10sb_layout.py report_bias | report_win15 [--procs 2]  -> results/n10sb_bias.json, n10sb_win15.json
    python scripts/n10sb_layout.py report [--procs 2]                -> results/n10sb.json, figures/fig32_n10sb.png

Report: the N10-LS analysis (n10l_seeds.analyse) on every group with data, then (b) against the pads (a) of
results/n10l_seeds.json at the same size and rate (PLAN_SPLIT_BANDS.md rules): bootstrap 95 % interval of the
difference of network means of MC + degree-2, (b) - (a); median rho intervals side by side; and the 1 mm rupture
ladder of networks 1-3 -> r_75 against the pads' (results/n10l_tolerance.json).
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
from agnw_rc.drive import raw_path  # noqa: E402

RAW = ROOT / "results" / "raw"
SWEEP = ROOT / "results" / "n10sb_sweep.json"
OUT = ROOT / "results" / "n10sb.json"
FIG = ROOT / "figures" / "fig32_n10sb.png"
LAYOUT = "split_bands"
TAU, SPREAD, HOLD = 0.3, 1.0, 100
RAMP = 0.5                               # V/s, the N10-E0 pilot ramp (5 V in 10 s, 1 ms step)
SIDES = (1.0, 2.0, 2.83)
SEEDS10, SEEDS3 = tuple(range(1, 11)), (1, 2, 3)
# queue order: 1 mm first (cheap, here), realistic noise first within a size
GROUPS = ([(1.0, r, SEEDS10) for r in (1e-2, 0.0)] + [(1.0, r, SEEDS3) for r in (1e-4, 1e-3, 3e-3)]
          + [(s, r, SEEDS10) for s in (2.83, 2.0) for r in (1e-2, 0.0)])   # cluster: 2.83 mm first


# ------------------------------------------------------------------ onset per size
def sweep_one(side):
    """Up-ramp at RAMP V/s from 0 (network 1, all driven segments at the same voltage, 1 ms step): onset = first
    voltage where the driven conductance exceeds 10 x its start. At 1 mm this is the up half of the N10-E0 pilot sweep
    (scripts/n10e_pilot.py), so it must reproduce its 2.08 V."""
    from agnw_rc.electrodes import make_layout
    from agnw_rc.multicircuit import MultiTerminalCircuit
    from agnw_rc.volatile import conductance, junction_kd0, step_state
    vp = C.volatile_params(None, tau_s=TAU, tau_spread=SPREAD, dt=C.DT)
    _, wd = C.network_params()
    net = C.make_network(1, side=side)
    els = make_layout(LAYOUT, net)
    mc = MultiTerminalCircuit(net, wd, els)
    fixed = mc.fixed_electrodes
    driven = [k for k, i in enumerate(fixed) if els[i].role == "in"]
    kd0 = junction_kd0(vp, len(net.junctions), 1)
    state = np.zeros((2, len(net.junctions)))
    v_max = 5.0 * side                                              # onset ~ 2 V per mm (N10-C two bands)
    n = int(round(v_max / RAMP / C.DT))
    Vf = np.zeros(len(fixed))
    rec_v, rec_g = [], []
    onset = None
    for i in range(n):
        v = v_max * i / n
        Vf[driven] = v
        V = mc.solve(conductance(state, vp), Vf, method="pcg")
        if i % 20 == 0 and v > 0:
            I = mc.electrode_currents(V)
            rec_v.append(v)
            rec_g.append(sum(I[fixed[k]] for k in driven) / v)
            if len(rec_g) > 5 and rec_g[-1] > 10 * np.mean(rec_g[:5]):
                onset = float(v)
                break                                                # only the onset is needed
        state = step_state(state, mc.junction_voltage_drops(V), vp, kd0)
    return f"{side:g}", {"onset_V": onset, "G_start_S": float(np.mean(rec_g[:5])), "n_junctions": len(net.junctions),
                         "segment_wires_mean": float(np.mean([len(e.wires) for e in els])), "ramp_V_per_s": RAMP}


def sweep(sides, procs):
    res = json.loads(SWEEP.read_text()) if SWEEP.exists() else {}
    todo = [s for s in sides if f"{s:g}" not in res]
    with ProcessPoolExecutor(max(1, min(procs, len(todo)))) as pool:
        for k, v in pool.map(sweep_one, todo):
            res[k] = v
            SWEEP.write_text(json.dumps(res, indent=1))
            print(k, v, flush=True)


# ------------------------------------------------------------------ runs
def specs_for(side, rate, seeds, sigma=0.0):
    on = json.loads(SWEEP.read_text())[f"{side:g}"]["onset_V"]
    w = round(0.45 * on, 2)
    windows = [(round(1.05 * on, 2), w), (round(1.25 * on, 2), w)]
    wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
    kw = dict(n_samples=C.E4_N_SAMPLES, washout=-(-wo // HOLD) * HOLD, n_virtual=C.E4_N_VIRTUAL, encoding="weights",
              model="volatile", tau_s=TAU, tau_spread=SPREAD, stride=C.N10_STRIDE, rupture=rate, sigma=sigma,
              layout=LAYOUT)
    if side != 1.0:
        kw["side"] = side
    out = []
    for vc, wd in windows:
        for net in seeds:
            out += [C.RunSpec("iid", HOLD, vc, net, rep, v_width=wd, **kw) for rep in C.E1_REPLICATES]
            out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, v_width=wd, **kw))
    return out


# B5: bias + small input on (b), the N10-B grid (scripts/n10b_bias.py) scaled to the split-band onset:
# bias 1.5 / 2 / 3 x onset, full input width 3 % or 15 % of the bias, ruptures 0 / 1e-3 / 1e-2, networks 1-3.
BIAS_X, WIDTH_FRAC, BIAS_RATES = (1.5, 2.0, 3.0), (0.03, 0.15), (0.0, 1e-3, 1e-2)


def bias_specs(side=1.0):
    on = json.loads(SWEEP.read_text())[f"{side:g}"]["onset_V"]
    wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
    out = []
    for r in BIAS_RATES:
        kw = dict(n_samples=C.E4_N_SAMPLES, washout=-(-wo // HOLD) * HOLD, n_virtual=C.E4_N_VIRTUAL, encoding="weights",
                  model="volatile", tau_s=TAU, tau_spread=SPREAD, stride=C.N10_STRIDE, rupture=r, layout=LAYOUT)
        for b in BIAS_X:
            for f in WIDTH_FRAC:
                vc, w = round(b * on, 2), max(0.01, round(f * b * on, 2))
                for net in SEEDS3:
                    out += [C.RunSpec("iid", HOLD, vc, net, rep, v_width=w, **kw) for rep in C.E1_REPLICATES]
                    out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, v_width=w, **kw))
    return out


# Window check (2026-10-09): validation picked the upper of the two protocol windows (1.25 x onset) for both layouts;
# a third window at 1.5 x onset, same width (0.45 x onset), for (a) pads and (b) split bands, networks 1-3, 1 mm,
# ruptures 0 and 1e-2. Pads: onset 0.36 V (the N10-E / N10-LS windows 0.38 / 0.45, width 0.16).
WIN15_RATES = (0.0, 1e-2)


def window15_specs():
    import n10l_seeds as S
    out = []
    for r in WIN15_RATES:
        ref = S.specs_for(1.0, r, SEEDS3)[0]                     # pads protocol exactly as N10-E / N10-LS
        out += [dataclasses.replace(s, v_center=0.54) for s in S.specs_for(1.0, r, SEEDS3) if s.v_center == ref.v_center]
        ref = specs_for(1.0, r, SEEDS3)[0]
        on = json.loads(SWEEP.read_text())["1"]["onset_V"]
        out += [dataclasses.replace(s, v_center=round(1.5 * on, 2)) for s in specs_for(1.0, r, SEEDS3)
                if s.v_center == ref.v_center]
    return out


def all_specs(only=""):
    if only == "bias":
        return bias_specs()
    if only == "win15":
        return window15_specs()
    groups = GROUPS
    if only:
        keep = [tuple(float(x) for x in k.split(":")) for k in only.split(",")]
        groups = [g for g in GROUPS if any(g[0] == k[0] and g[1] == k[1] for k in keep)]
    return [s for side, r, seeds in groups for s in specs_for(side, r, seeds)]


def run(procs, only=""):
    import n10l_seeds as S                              # run_one: atomic .running claims (safe on the cluster)
    todo = [s for s in all_specs(only)
            if not raw_path(s, RAW).exists() and not raw_path(s, RAW).with_suffix(".running").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(S.run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report
def report(procs=2):
    import n10l_seeds as S
    from n10l_figures import r_level
    groups = {}
    for side, r, seeds in GROUPS:
        sp = [s for s in specs_for(side, r, seeds) if raw_path(s, RAW).exists()]
        if sp:
            groups[(side, r, 0.0)] = sp
    if (1.0, 1e-2, 0.0) in groups:                       # ladder twin of networks 1-3 at 1e-2 (r_75 on the same networks)
        groups[(1.0, 1e-2, -1.0)] = [s for s in groups[(1.0, 1e-2, 0.0)] if s.network_seed in SEEDS3]
    out = S.analyse(groups, SEEDS10, procs)
    for g in out["groups"]:                              # sigma -1 marks the networks 1-3 subset (no noise added)
        if g["sigma"] == -1.0:
            g["sigma"], g["subset"] = 0.0, "networks 1-3"
    pads = json.loads((ROOT / "results" / "n10l_seeds.json").read_text())["groups"]
    rng = np.random.default_rng(0)
    cmp = []
    for g in out["groups"]:
        if g.get("subset"):
            continue
        a = next((x for x in pads if x["side_mm"] == g["side_mm"] and x["rupture"] == g["rupture"] and x["sigma"] == 0), None)
        if not a:
            continue
        ta, tb = np.array(a["per_network_total"]), np.array(g["per_network_total"])
        diff = np.array([rng.choice(tb, len(tb)).mean() - rng.choice(ta, len(ta)).mean() for _ in range(10_000)])
        lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
        c = {"side_mm": g["side_mm"], "rupture": g["rupture"], "n_networks_b": g["n_networks"],
             "n_networks_a": a["n_networks"], "total_b": float(tb.mean()), "total_a": float(ta.mean()),
             "diff_b_minus_a": float(tb.mean() - ta.mean()), "diff_CI95": [lo, hi],
             "verdict": "better" if lo > 0 else "worse" if hi < 0 else "on par",
             "pass_G1_b": g["networks_pass_G1"], "pass_G1_a": a["networks_pass_G1"],
             "pooled_G1_b": g["pooled"]["G1"], "pooled_G1_a": a["pooled"]["G1"]}
        if g["rupture"] > 0 and "rho_CI95" in g and "rho_CI95" in a:
            c.update(rho_b=g["rho_median"], rho_b_CI95=g["rho_CI95"], rho_a=a["rho_median"], rho_a_CI95=a["rho_CI95"],
                     robustness="more robust" if g["rho_CI95"][0] > a["rho_CI95"][1]
                     else "less robust" if g["rho_CI95"][1] < a["rho_CI95"][0] else "on par")
        cmp.append(c)
    out["b_vs_a"] = cmp
    for r in (1e-2, 0.0):                                # size rule 2 on (b): 1 mm / 2.83 mm spread ratio, as n10l_seeds
        a = next((g for g in out["groups"] if g["side_mm"] == 1.0 and g["rupture"] == r and not g.get("subset")), None)
        b = next((g for g in out["groups"] if g["side_mm"] == 2.83 and g["rupture"] == r), None)
        if a and b and a["n_networks"] > 2 and b["n_networks"] > 2:
            _, _, sa = S.boot(a["per_network_total"], lambda v: v.std(ddof=1), seed=1)
            _, _, sb = S.boot(b["per_network_total"], lambda v: v.std(ddof=1), seed=2)
            ok = (sa > 0) & (sb > 0)
            out.setdefault("spread_ratio_1mm_over_2.83mm", {})[f"{r:g}"] = {
                "ratio": a["spread_sd_total"] / b["spread_sd_total"],
                "CI95": [float(np.percentile(sa[ok] / sb[ok], 2.5)), float(np.percentile(sa[ok] / sb[ok], 97.5))]}
    pts = sorted((g["rupture"], g["rho_median"]) for g in out["groups"]
                 if g["side_mm"] == 1.0 and g["rupture"] > 0 and "rho_median" in g
                 and (g.get("subset") or g["rupture"] < 1e-2))
    if pts:
        tol = json.loads((ROOT / "results" / "n10l_tolerance.json").read_text())["sides"]["1"]
        out["tolerance_1mm"] = {"b_points": pts, "b_r75": r_level([p[0] for p in pts], [p[1] for p in pts]),
                                "a_points": tol["points"], "a_r75": tol["r_level"]}
    OUT.write_text(json.dumps(out, indent=1))
    print("spread ratio 1 / 2.83 mm (b):", json.dumps(out.get("spread_ratio_1mm_over_2.83mm", {})))
    for c in cmp:
        print(f"{c['side_mm']:g} mm, r {c['rupture']:g}: (b) {c['total_b']:.2f} vs (a) {c['total_a']:.2f}, diff "
              f"{c['diff_b_minus_a']:+.2f} [{c['diff_CI95'][0]:+.2f}, {c['diff_CI95'][1]:+.2f}] -> {c['verdict']}; G1 networks "
              f"{c['pass_G1_b']}/{c['n_networks_b']} vs {c['pass_G1_a']}/{c['n_networks_a']}"
              + (f"; rho {c['rho_b']:.2f} vs {c['rho_a']:.2f} -> {c['robustness']}" if "rho_b" in c else ""), flush=True)
    if "tolerance_1mm" in out:
        print("r75 1 mm: (b)", out["tolerance_1mm"]["b_r75"], " (a)", out["tolerance_1mm"]["a_r75"])
    figure(out, pads)


def figure(out, pads):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(SIDES) + 1, figsize=(19, 4.8))
    for ax, side in zip(axes, SIDES):
        for j, r in enumerate((0.0, 1e-2)):
            for k, (lay, src, col) in enumerate((("(a) pads", pads, "0.45"), ("(b) split bands", out["groups"], "C0"))):
                g = next((x for x in src if x["side_mm"] == side and x["rupture"] == r and x["sigma"] == 0
                          and not x.get("subset")), None)
                if not g:
                    continue
                x0 = 2 * j + k
                y = g["per_network_total"]
                ok = [n["G1"] for n in g["networks"]]
                ax.scatter(x0 + 0.08 * rng_jitter(len(y)), y, c=[col if o else "none" for o in ok], edgecolors=col, s=40)
                ax.hlines(np.mean(y), x0 - 0.3, x0 + 0.3, colors=col)
        ax.set_xticks(range(4), ["(a) r=0", "(b) r=0", "(a) 1e-2", "(b) 1e-2"])
        ax.set(ylabel="MC + degree-2 per network", title=f"{side:g} mm", ylim=(0, None))
    ax = axes[-1]
    if "tolerance_1mm" in out:
        tl = out["tolerance_1mm"]
        for pts, col, lab in ((tl["a_points"], "0.45", "(a) pads"), (tl["b_points"], "C0", "(b) split bands")):
            ax.semilogx([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, label=lab)
        ax.axhline(0.75, color="r", ls="--", lw=0.8)
        ax.legend(fontsize=8)
    ax.set(xlabel="rupture rate (/s per junction)", ylabel="median capacity kept rho (networks 1-3)", title="1 mm tolerance",
           ylim=(0, 1.1))
    fig.suptitle("N10-SB: split bands (b) vs checkerboard pads (a); dots = networks (filled: passes G1), bars = mean")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


def rng_jitter(n, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, n)


# ------------------------------------------------------------------ B5 and window-check reports
def _summary(g):
    p = g["pooled"]
    return {"operating_point": p["operating_point"], "MC": p["MC"]["mean"], "MC_sd": p["MC"]["sd"], "IPC2": p["IPC2"]["mean"],
            "IPC2_sd": p["IPC2"]["sd"], "NARMA": p["narma10_nmse"]["mean"], "NARMA_sd": p["narma10_nmse"]["sd"],
            "consistency": p["consistency_inc"]["mean"], "G1": p["G1"], "networks_pass_G1": g["networks_pass_G1"],
            "n_networks": g["n_networks"], "rho_median": g.get("rho_median"), "collapsed": g.get("collapsed"),
            "rho_n": g.get("rho_n")}


def report_bias(procs=2):
    """B5: N10-B on split bands. Per rupture rate: operating point (bias, width) chosen on validation among the 6
    windows (as N10-B), and per bias level (best width); next to N10-B on the pads (results/n10b_criteria.json)."""
    import n10l_seeds as S
    sp = [s for s in bias_specs() if raw_path(s, RAW).exists()]
    groups = {}
    for r in BIAS_RATES:
        groups[(1.0, r, 0.0)] = [s for s in sp if s.rupture == r]
        for k, b in enumerate(sorted({s.v_center for s in sp})):
            groups[(1.0, r, -(k + 1.0))] = [s for s in sp if s.rupture == r and s.v_center == b]
    out = S.analyse({k: v for k, v in groups.items() if v}, SEEDS3, procs)
    on = json.loads(SWEEP.read_text())["1"]["onset_V"]
    rows = []
    for g in out["groups"]:
        lab = "best window" if g["sigma"] == 0 else f"bias {g['pooled']['operating_point']['v_center']:g} V"
        rows.append({"rupture": g["rupture"], "setting": lab, **_summary(g)})
    pads = json.loads((ROOT / "results" / "n10b_criteria.json").read_text())["rows"]
    res = {"onset_V": on, "bias_x": BIAS_X, "width_frac": WIDTH_FRAC, "rows": rows,
           "pads_N10B": [{k: r[k] for k in ("setting", "operating_point")} | {q: r[q]["mean"] for q in ("MC", "IPC2", "narma10_nmse")}
                         | {"G1": r.get("G1")} for r in pads if r["readout_noise"] == "none"]}
    (ROOT / "results" / "n10sb_bias.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"r {r['rupture']:g}, {r['setting']}: op {r['operating_point']}, MC {r['MC']:.2f}, IPC2 {r['IPC2']:.2f}, "
              f"NARMA {r['NARMA']:.2f}, cons {r['consistency']:.2f}, G1 {r['G1']} ({r['networks_pass_G1']}/{r['n_networks']})"
              + (f", rho {r['rho_median']:.2f}, collapsed {r['collapsed']}/{r['rho_n']}" if r["rho_median"] is not None else ""))
    print("pads N10-B (best window): " + "; ".join(f"{r['setting']}: MC {r['MC']:.2f} IPC2 {r['IPC2']:.2f} NARMA {r['narma10_nmse']:.2f}"
                                             for r in res["pads_N10B"] if "best" in r["setting"]))


def report_win15(procs=2):
    """Window check: per layout and rupture rate, the validation-chosen operating point among the two protocol windows
    (networks 1-3) and among the three (+ 1.5 x onset); scores at each window."""
    import n10l_seeds as S
    w15 = window15_specs()
    groups = {}
    for lay, two in (("pads", S.specs_for), ("split_bands", specs_for)):
        for r in WIN15_RATES:
            base = [s for s in two(1.0, r, SEEDS3) if raw_path(s, RAW).exists()]
            new = [s for s in w15 if s.layout == lay and s.rupture == r and raw_path(s, RAW).exists()]
            tag = 1.0 if lay == "pads" else 2.0
            groups[(1.0, r, -tag)] = base                     # two protocol windows
            groups[(1.0, r, -tag - 0.5)] = base + new         # + 1.5 x onset
            for vc in sorted({s.v_center for s in base + new}):
                groups[(1.0, r, -tag - 0.01 * (1 + sorted({s.v_center for s in base + new}).index(vc)))] = \
                    [s for s in base + new if s.v_center == vc]
    out = S.analyse({k: v for k, v in groups.items() if v}, SEEDS3, procs)
    rows = []
    for g in out["groups"]:
        s = -g["sigma"]
        lay = "pads" if s < 2 else "split_bands"
        frac = round(s - (1 if lay == "pads" else 2), 2)
        kind = "2 windows" if frac == 0 else "3 windows" if frac == 0.5 else "single window"
        rows.append({"layout": lay, "rupture": g["rupture"], "choice": kind, **_summary(g)})
    (ROOT / "results" / "n10sb_win15.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['layout']:11s} r {r['rupture']:g} {r['choice']:13s} op {r['operating_point']['v_center']:5.2f}: MC {r['MC']:.2f} "
              f"IPC2 {r['IPC2']:.2f} NARMA {r['NARMA']:.2f} G1 {r['G1']} ({r['networks_pass_G1']}/{r['n_networks']})"
              + (f" rho {r['rho_median']:.2f}" if r["rho_median"] is not None else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "run", "report", "report_bias", "report_win15"])
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--only", default="")
    ap.add_argument("--sides", default="1,2,2.83")
    a = ap.parse_args()
    if a.stage == "sweep":
        sweep(tuple(float(x) for x in a.sides.split(",")), a.procs)
    elif a.stage == "run":
        run(a.procs, a.only)
    elif a.stage == "report":
        report(a.procs)
    else:
        {"report_bias": report_bias, "report_win15": report_win15}[a.stage](a.procs)
