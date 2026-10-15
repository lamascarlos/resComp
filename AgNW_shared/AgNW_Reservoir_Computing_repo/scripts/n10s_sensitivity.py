#!/usr/bin/env python3
"""N10-S: do the N10-E conclusions depend on the two model settings we chose rather than fitted?
(voltage scale v_scale = 5 and growth-rate cap k_max = 100 1/s; docs/RESULTS.md sections 15-17)

N10-E best setup (4x4 pads, 8 driven with random weights + 8 grounded, 16 electrode currents, tau_s 0.3 s,
spread 1, 100 ms holds, 3 networks x 2 replicates + sine/square), one setting changed at a time:
    v_scale 3 and 8 (k_max 100), k_max 30 and 300 1/s (v_scale 5; k_max 300 at a 1 ms step, since k dt must stay
    well below 1, section 15). Reference: N10-E itself (results/n10e_criteria.json).
The onset voltage moves with v_scale, so each setting gets its own sweep and windows (1.05 x and 1.25 x onset,
width 0.45 x onset, as for N10-E).

    python scripts/n10s_sensitivity.py sweep           -> results/n10s_sweep.json
    python scripts/n10s_sensitivity.py run --part a    (v_scale block)   / --part b (k_max block) [--procs N]
    python scripts/n10s_sensitivity.py report          -> results/n10s_criteria.json, figures/fig19_n10s.png
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
from agnw_rc import criteria as K  # noqa: E402
from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.drive import load_raw, raw_path, save_raw  # noqa: E402
from agnw_rc.electrodes import make_layout  # noqa: E402
from agnw_rc.multicircuit import MultiTerminalCircuit  # noqa: E402
from agnw_rc.multidrive import simulate_multi  # noqa: E402
from agnw_rc.volatile import conductance, junction_kd0, step_state  # noqa: E402

RAW = ROOT / "results" / "raw"
SWEEP = ROOT / "results" / "n10s_sweep.json"
OUT = ROOT / "results" / "n10s_criteria.json"
FIG = ROOT / "figures" / "fig19_n10s.png"
TAU, SPREAD, HOLD = 0.3, 1.0, 100
SETTINGS = {"a": [dict(v_scale=3.0, k_max=100.0), dict(v_scale=8.0, k_max=100.0)],
            "b": [dict(v_scale=5.0, k_max=30.0), dict(v_scale=5.0, k_max=300.0)]}
OBS = "electrodes"


def key(st):
    return f"vs{st['v_scale']:g}_km{st['k_max']:g}"


def stride(st):
    return 1 if st["k_max"] > 100 else C.N10_STRIDE


def sweep_one(st):
    vp = C.volatile_params(None, tau_s=TAU, tau_spread=SPREAD, dt=C.DT, **st)
    _, wd = C.network_params()
    net = C.make_network(1)
    els = make_layout("pads", net)
    mc = MultiTerminalCircuit(net, wd, els)
    fixed = mc.fixed_electrodes
    driven = [k for k, i in enumerate(fixed) if els[i].role == "in"]
    kd0 = junction_kd0(vp, len(net.junctions), 1)
    state = np.zeros((2, len(net.junctions)))
    n, v_max = 10_000, 3.0                                           # 0 -> 3 V ramp in 10 s
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
    g0 = g[:5].mean()
    return key(st), float(v[np.argmax(g > 10 * g0)]) if np.any(g > 10 * g0) else None


def sweep():
    allst = SETTINGS["a"] + SETTINGS["b"] + [dict(v_scale=5.0, k_max=100.0)]
    with ProcessPoolExecutor(len(allst)) as pool:
        res = dict(pool.map(sweep_one, allst))
    SWEEP.write_text(json.dumps(res, indent=1))
    print(res)


def specs(parts=("a", "b")):
    onsets = json.loads(SWEEP.read_text())
    out = []
    for part in parts:
        for st in SETTINGS[part]:
            on = onsets[key(st)]
            wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
            common = dict(n_samples=C.E4_N_SAMPLES, washout=-(-wo // HOLD) * HOLD, n_virtual=C.E4_N_VIRTUAL,
                          layout="pads", encoding="weights", model="volatile", tau_s=TAU, tau_spread=SPREAD,
                          stride=stride(st), v_width=round(0.45 * on, 2), **st)
            for vc in (round(1.05 * on, 2), round(1.25 * on, 2)):
                for net in C.E1_SEEDS:
                    out += [C.RunSpec("iid", HOLD, vc, net, r, **common) for r in C.E1_REPLICATES]
                    out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, **common))
    return out


def run_one(spec):
    if not raw_path(spec, RAW).exists():
        save_raw(spec, simulate_multi(spec), RAW)
    return spec.name


def run(part, procs):
    todo = [s for s in specs((part,)) if not raw_path(s, RAW).exists()]
    todo.sort(key=lambda s: s.stride)                           # 1 ms-step runs (slowest) first
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


def _score(spec):
    raw = load_raw(spec, RAW)
    if spec.stream == "sinesquare":
        return spec, E.evaluate_sinesquare_run(raw), None
    return spec, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), E.observables(raw)[OBS].shape[1]


def _cons(pair):
    a, b = pair
    return a, E.evaluate_consistency(load_raw(a, RAW), load_raw(b, RAW))


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = specs()
    with ProcessPoolExecutor(16) as pool:
        scored = list(pool.map(_score, sp))
        pairs = [(s, C.RunSpec(**{**s.__dict__, "replicate": 1})) for s in sp if s.stream == "iid" and s.replicate == 0]
        cs = list(pool.map(_cons, pairs))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in C.E1_SEEDS)
    base = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in base]))}
    onsets = json.loads(SWEEP.read_text())
    rows = [{"setting": "v_scale 5, k_max 100 (N10-E)", "onset_V": onsets.get("vs5_km100"),
             **json.loads((ROOT / "results" / "n10e_criteria.json").read_text())["layouts"]["pads"][OBS]}]
    for part in ("a", "b"):
        for st in SETTINGS[part]:
            mine = lambda s: s.v_scale == st["v_scale"] and s.k_max == st["k_max"]  # noqa: E731
            runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                         replicate=s.replicate, scores=sc) for s, sc, _ in scored if mine(s)]
            cons = [dict(hold=a.hold, v_center=a.v_center, consistency=c) for a, c in cs if mine(a)]
            nf = next(n for s, _, n in scored if mine(s) and n)
            strict = K.evaluate(runs, cons, OBS, nf, esn, floor)
            loose = K.evaluate(runs, cons, OBS, 16, esn, floor)
            rows.append({"setting": f"v_scale {st['v_scale']:g}, k_max {st['k_max']:g}", "onset_V": onsets[key(st)],
                         **strict, "G2_loose": loose["G2"]})
    OUT.write_text(json.dumps({"esn_total": esn, "noise_floor": floor, "rows": rows}, indent=1))
    for r in rows:
        print(f"{r['setting']:30s} onset {r['onset_V']} V, window {r['operating_point']['v_center']}: "
              f"MC {r['MC']['mean']:.2f}±{r['MC']['sd']:.2f} IPC2 {r['IPC2']['mean']:.2f}±{r['IPC2']['sd']:.2f} "
              f"NARMA {r['narma10_nmse']['mean']:.3f}±{r['narma10_nmse']['sd']:.3f} "
              f"acc {r['accuracy']['mean']:.2f} cons {r['consistency_inc']['mean']:.2f} -> G0 {r['G0']} G1 {r['G1']} "
              f"G2 strict {r['G2']} loose {r['G2_loose']}")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(rows))
    for ax, m, t in zip(axes, ("MC", "IPC2", "narma10_nmse", "total_capacity"),
                        ("memory capacity", "degree-2 capacity", "NARMA-10 NMSE", "MC + degree-2")):
        ax.bar(x, [r[m]["mean"] for r in rows], yerr=[r[m]["sd"] for r in rows], capsize=3,
               color=["0.6"] + ["C1"] * (len(rows) - 1))
        ax.set_xticks(x, [r["setting"].replace(" (N10-E)", "\n(N10-E)").replace(", ", "\n") for r in rows],
                      fontsize=8)
        ax.set_title(t)
        if m in ("MC", "IPC2"):
            ax.axhline(1, color="r", ls="--", lw=0.8)
        if m == "narma10_nmse":
            ax.axhline(0.7, color="r", ls="--", lw=0.8)
        if m == "total_capacity":
            ax.axhline(esn[16] / 2, color="m", ls=":", label="G2 loose")
            ax.axhline(esn[200] / 2, color="m", ls="--", label="G2 strict")
            ax.legend(fontsize=7)
    fig.suptitle("N10-S: sensitivity to the chosen settings (16 pad currents, 3 networks x 2 replicates; grey: N10-E)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "run", "report"])
    ap.add_argument("--part", choices=["a", "b"])
    ap.add_argument("--procs", type=int, default=18)
    a = ap.parse_args()
    if a.stage == "sweep":
        sweep()
    elif a.stage == "run":
        run(a.part, a.procs)
    else:
        report()
