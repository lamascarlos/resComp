#!/usr/bin/env python3
"""N10-E0 pilot: the N10 volatile junction on 16-electrode layouts, read out through the electrodes a chip has
(docs/NEXT_STEPS.md, N10-E). Cheap screen before the full N10-E (3 networks, G0-G2).

Stages:
  sweep   0 -> V_MAX -> 0 triangular ramp (20 s, all driven electrodes, 1 ms step, tau_s 0.3 s, spread 1, network 1)
          per layout: network onset (driven conductance x 10 above its start) -> results/n10e_sweep.json
  run     E4-like runs (network 1, 800 samples, hold 100 ms, 50 virtual nodes, 2 ms step, tau_s 0.3 s, spread 1,
          pure volatile), for LAYOUTS x ENCODINGS x two windows scaled to each layout's onset (the two-band window
          that worked, 2.25-3.25 V around a 2.2 V onset, is centre 1.25 x onset, width 0.45 x onset; plus a lower
          one at 1.05 x onset); i.i.d. replicates 0 and 1 (consistency) and sine/square
  report  scores per configuration and readout (window chosen on validation) -> results/n10e_pilot.json,
          figures/fig16_n10e_pilot.png

Readouts: "current" = currents of all electrodes (the two-band N10-1 "current" had 2), "electrodes" = currents of
driven/grounded electrodes + voltages of floating ones (pads_float), "probes" = 32 internal probes (reference).

    python scripts/n10e_pilot.py sweep [--procs 4]
    python scripts/n10e_pilot.py run [--procs 18]
    python scripts/n10e_pilot.py report
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

from agnw_rc import config as C  # noqa: E402
from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.drive import load_raw, raw_path, save_raw  # noqa: E402
from agnw_rc.electrodes import make_layout  # noqa: E402
from agnw_rc.multicircuit import MultiTerminalCircuit  # noqa: E402
from agnw_rc.multidrive import simulate_multi  # noqa: E402
from agnw_rc.volatile import conductance, junction_kd0, step_state  # noqa: E402

RAW = ROOT / "results" / "raw"
SWEEP = ROOT / "results" / "n10e_sweep.json"
OUT = ROOT / "results" / "n10e_pilot.json"
FIG = ROOT / "figures" / "fig16_n10e_pilot.png"
SWEEP_LAYOUTS = ("two_bands", "split_bands", "pads", "pads_float")
LAYOUTS = ("split_bands", "pads", "pads_float")
ENCODINGS = ("direct", "weights")
V_MAX = 5.0
TAU, SPREAD, HOLD, STRIDE = 0.3, 1.0, 100, 2
READOUTS = ("current", "electrodes", "probes")


# ------------------------------------------------------------------ sweep
def sweep_one(layout):
    vp = C.volatile_params(None, tau_s=TAU, tau_spread=SPREAD, dt=C.DT)
    _, wd = C.network_params()
    net = C.make_network(1)
    els = make_layout(layout, net)
    mc = MultiTerminalCircuit(net, wd, els)
    fixed = mc.fixed_electrodes
    driven = [k for k, i in enumerate(fixed) if els[i].role == "in"]
    kd0 = junction_kd0(vp, len(net.junctions), 1)
    state = np.zeros((2, len(net.junctions)))
    n = 20_000
    v_t = V_MAX * (1 - np.abs(np.arange(n) / (n / 2) - 1))                     # 0 -> V_MAX -> 0
    Vf = np.zeros(len(fixed))
    rec_v, rec_g, rec_on = [], [], []
    for i, v in enumerate(v_t):
        Vf[driven] = v
        V = mc.solve(conductance(state, vp), Vf, method="pcg")
        if i % 20 == 0 and v > 0:
            I = mc.electrode_currents(V)
            rec_v.append(v)
            rec_g.append(sum(I[fixed[k]] for k in driven) / v)
            rec_on.append(float(np.mean(np.minimum(1, state.sum(axis=0)) >= 0.5)))
        state = step_state(state, mc.junction_voltage_drops(V), vp, kd0)
    v, g = np.array(rec_v), np.array(rec_g)
    up = np.arange(len(v)) < len(v) // 2
    g0 = g[up][:5].mean()
    onset = float(v[up][np.argmax(g[up] > 10 * g0)]) if np.any(g[up] > 10 * g0) else None
    return layout, {"onset_V": onset, "G_start_S": float(g0), "G_max_S": float(g.max()),
                    "on_max": float(max(rec_on)), "n_in": len(driven),
                    "n_float": sum(e.role == "float" for e in els)}


def sweep(procs):
    with ProcessPoolExecutor(procs) as pool:
        res = dict(pool.map(sweep_one, SWEEP_LAYOUTS))
    SWEEP.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v)


# ------------------------------------------------------------------ runs
def windows(onset):
    width = round(0.45 * onset, 2)
    return [(round(1.05 * onset, 2), width), (round(1.25 * onset, 2), width)]


def specs(layouts=LAYOUTS, encodings=ENCODINGS, networks=(1,)):
    onsets = json.loads(SWEEP.read_text())
    out = []
    for layout in layouts:
        for enc in encodings:
            for vc, w in windows(onsets[layout]["onset_V"]):
                common = dict(v_width=w, n_samples=C.E4_N_SAMPLES, n_virtual=C.E4_N_VIRTUAL, layout=layout,
                              encoding=enc, model="volatile", tau_s=TAU, tau_spread=SPREAD, stride=STRIDE,
                              washout=-(-C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD) // HOLD) * HOLD)
                for net in networks:
                    out += [C.RunSpec("iid", HOLD, vc, net, r, **common) for r in (0, 1)]
                    out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, **common))
    return out


FULL = dict(layouts=("pads", "pads_float"), encodings=("weights",), networks=(1, 2, 3))


def run_one(spec):
    if not raw_path(spec, RAW).exists():
        save_raw(spec, simulate_multi(spec), RAW)
    return spec.name


def run(procs, full=False):
    todo = [s for s in (specs(**FULL) if full else specs()) if not raw_path(s, RAW).exists()]
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report
def score(spec_group):
    iid0, iid1, ss = spec_group
    r0, r1, rs = load_raw(iid0, RAW), load_raw(iid1, RAW), load_raw(ss, RAW)
    floor = E.noise_floor(r0["u"])
    sc0, sc1 = E.evaluate_iid_run(r0, floor), E.evaluate_iid_run(r1, floor)
    acc = E.evaluate_sinesquare_run(rs)
    cons = E.evaluate_consistency(r0, r1)
    nch = {k: v.shape[1] for k, v in E.observables(r0).items()}
    out = {}
    for obs in READOUTS:
        if obs not in sc0:
            continue
        out[obs] = {m: float(np.mean([sc0[obs][m], sc1[obs][m]])) for m in ("MC", "IPC2", "narma10_nmse",
                                                                             "val_MC", "val_IPC2")}
        out[obs].update(accuracy=acc[obs]["accuracy"], consistency_inc=cons[obs + "_state_inc"],
                        n_features=nch[obs])
    return out


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = specs()
    rows = []
    for i in range(0, len(sp), 3):
        g = sp[i:i + 3]
        rows.append({"layout": g[0].layout, "encoding": g[0].encoding, "v_center": g[0].v_center,
                     "v_width": g[0].v_width, "scores": score(g)})
    best = {}
    for r in rows:
        for obs, s in r["scores"].items():
            k = (r["layout"], r["encoding"], obs)
            if k not in best or s["val_MC"] + s["val_IPC2"] > best[k]["val_MC"] + best[k]["val_IPC2"]:
                best[k] = {**s, "v_center": r["v_center"]}
    ref = json.loads((ROOT / "results" / "n10_1_criteria.json").read_text())
    n10 = next(g for g in ref["groups"] if g["tau_s"] == TAU and g["tau_spread"] == SPREAD)["per_hold"]
    out = {"rows": rows, "best": [{"layout": k[0], "encoding": k[1], "readout": k[2], **v} for k, v in best.items()],
           "reference_two_bands": {o: n10[o][str(HOLD)] for o in ("current", "probes")}}
    OUT.write_text(json.dumps(out, indent=1))
    print("reference two bands (N10-1, 3 networks):", {o: {m: round(v[m], 2) for m in ("MC", "IPC2", "narma10_nmse")}
                                                        for o, v in out["reference_two_bands"].items()})
    for b in out["best"]:
        print(f"{b['layout']:11s} {b['encoding']:8s} {b['readout']:10s} ({b['n_features']:4d} f) V {b['v_center']:.2f}: "
              f"MC {b['MC']:.2f} IPC2 {b['IPC2']:.2f} NARMA {b['narma10_nmse']:.2f} acc {b['accuracy']:.2f} "
              f"cons {b['consistency_inc']:.2f}")
    # figure: MC, IPC2, NARMA per configuration
    labels = [f"{b['layout']}\n{b['encoding']}" for b in out["best"] if b["readout"] == "current"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(labels))
    for k, obs in enumerate(READOUTS):
        vals = [b for b in out["best"] if b["readout"] == obs]
        idx = [labels.index(f"{b['layout']}\n{b['encoding']}") for b in vals]
        for ax, m in zip(axes, ("MC", "IPC2", "narma10_nmse")):
            ax.bar(np.array(idx) + (k - 1) * 0.27, [b[m] for b in vals], 0.27, label=obs, color=f"C{k}")
    refc, refp = out["reference_two_bands"]["current"], out["reference_two_bands"]["probes"]
    for ax, m, title in zip(axes, ("MC", "IPC2", "narma10_nmse"), ("memory capacity", "degree-2 capacity",
                                                                    "NARMA-10 NMSE")):
        ax.axhline(refc[m], color="C0", ls=":", lw=1, label="two bands, current (N10-1)")
        ax.axhline(refp[m], color="C2", ls=":", lw=1, label="two bands, 32 probes (N10-1)")
        ax.axhline({"MC": 1, "IPC2": 1, "narma10_nmse": 0.7}[m], color="r", ls="--", lw=0.8, label="G1")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"N10-E0 pilot: 16-electrode layouts, N10 junction (tau_s {TAU} s, spread {SPREAD}, "
                 f"{HOLD} ms holds, network 1)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


def _score_raw(spec):
    raw = load_raw(spec, RAW)
    if spec.stream == "sinesquare":
        return spec, E.evaluate_sinesquare_run(raw), None
    return spec, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), {k: v.shape[1] for k, v in E.observables(raw).items()}


def _cons(pair):
    a, b = pair
    return a, E.evaluate_consistency(load_raw(a, RAW), load_raw(b, RAW))


def full_report():
    """N10-E: G0-G2 per layout and readout over 3 networks x 2 replicates (agnw_rc.criteria). G2 is reported two
    ways (Juan, 2026-09-29): strict (headline) = an ESN with as many units as readout features, virtual-node
    samples included (the largest available, ESN-200, when there are more features); loose = one unit per physical
    channel (16 electrodes, 32 probes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from agnw_rc import criteria as K
    sp = specs(**FULL)
    with ProcessPoolExecutor(12) as pool:
        scored = list(pool.map(_score_raw, sp))
        pairs = [(s, next(t for t in sp if t.stream == "iid" and t.replicate == 1 and t.layout == s.layout
                          and t.v_center == s.v_center and t.network_seed == s.network_seed))
                 for s in sp if s.stream == "iid" and s.replicate == 0]
        cons = list(pool.map(_cons, pairs))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in FULL["networks"])
    base = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in FULL["networks"]]
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in base]))}
    physical = {"electrodes": 16, "current": 16, "probes": 32}
    out = {"noise_floor": floor, "esn_total": esn, "layouts": {}}
    for layout in FULL["layouts"]:
        runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                     replicate=s.replicate, scores=sc) for s, sc, _ in scored if s.layout == layout]
        cs = [dict(hold=a.hold, v_center=a.v_center, consistency=c) for a, c in cons if a.layout == layout]
        nfeat = next(n for s, _, n in scored if s.layout == layout and n)
        res = {}
        for obs in ("electrodes", "current", "probes"):
            strict = K.evaluate(runs, cs, obs, nfeat[obs], esn, floor)
            loose = K.evaluate(runs, cs, obs, physical[obs], esn, floor)
            res[obs] = {**strict, "n_features": nfeat[obs], "G2_loose": loose["G2"],
                        "esn_units_loose": loose["esn_units"], "esn_total_loose": loose["esn_total"]}
        out["layouts"][layout] = res
    (ROOT / "results" / "n10e_criteria.json").write_text(json.dumps(out, indent=1))
    print(f"floor {floor:.3f}; ESN totals {esn}")
    for layout, res in out["layouts"].items():
        for obs, r in res.items():
            op = r["operating_point"]
            print(f"{layout:10s} {obs:10s} ({r['n_features']:3d} f) {op['v_center']:.2f} V: "
                  f"MC {r['MC']['mean']:.2f}±{r['MC']['sd']:.2f} IPC2 {r['IPC2']['mean']:.2f}±{r['IPC2']['sd']:.2f} "
                  f"NARMA {r['narma10_nmse']['mean']:.3f}±{r['narma10_nmse']['sd']:.3f} "
                  f"acc {r['accuracy']['mean']:.2f} cons {r['consistency_inc']['mean']:.2f} lags {min(r['lags_above_floor'])}"
                  f" -> G0 {r['G0']} G1 {r['G1']} G2 strict {r['G2']} (ESN-{r['esn_units']} {r['esn_total']:.1f}) "
                  f"loose {r['G2_loose']} (ESN-{r['esn_units_loose']} {r['esn_total_loose']:.1f})")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    labels, k = [], 0
    for layout, res in out["layouts"].items():
        for obs in ("electrodes", "probes"):
            r = res[obs]
            for ax, m in zip(axes, ("MC", "IPC2", "narma10_nmse", "total_capacity")):
                ax.bar(k, r[m]["mean"], yerr=r[m]["sd"], color="C1" if obs == "electrodes" else "C2", capsize=3)
            labels.append(f"{layout}\n{obs}")
            k += 1
    for ax, m, t in zip(axes, ("MC", "IPC2", "narma10_nmse", "total_capacity"),
                        ("memory capacity", "degree-2 capacity", "NARMA-10 NMSE", "MC + degree-2")):
        ax.set_xticks(range(k), labels, fontsize=8)
        ax.set_title(t)
        if m in ("MC", "IPC2"):
            ax.axhline(1, color="r", ls="--", lw=0.8, label="G1")
        if m == "narma10_nmse":
            ax.axhline(0.7, color="r", ls="--", lw=0.8, label="G1")
        if m == "total_capacity":
            ax.axhline(esn[16] / 2, color="m", ls=":", label="G2 loose (half ESN-16)")
            ax.axhline(esn[200] / 2, color="m", ls="--", label="G2 strict (half ESN-200)")
        ax.legend(fontsize=7)
    fig.suptitle("N10-E: N10 junction on 16 electrodes, random input weights (3 networks x 2 replicates, "
                 "tau_s 0.3 s, spread 1, 100 ms holds)")
    fig.tight_layout()
    path = ROOT / "figures" / "fig17_n10e.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "run", "report", "full", "full_report"])
    ap.add_argument("--procs", type=int, default=18)
    a = ap.parse_args()
    {"sweep": lambda: sweep(min(a.procs, 4)), "run": lambda: run(a.procs), "report": report,
     "full": lambda: run(a.procs, full=True), "full_report": full_report}[a.stage]()
