#!/usr/bin/env python3
"""N10-V: N10 with the voltage scale matched to the measured network (Milano et al. 2025).

N10-C found that N10 junctions (v_scale 5) switch at ~12x the voltage of the measured network: extrapolated to a 7 mm
gap, N10 turns on at ~14 V vs the measured 2.4 V, so v_scale ~ 30-60 would match. Here v_scale = 40 (k_max 100 1/s):
  sweep    pads onset (N10-S ramp)                            -> results/n10v_sweep.json
  check    integration step at v_scale 40: 60-input pad run at 1 ms vs 2 ms (electrode currents, junction filling)
                                                              -> results/n10v_stride_check.json
  run      N10-E reservoir protocol (windows 1.05 / 1.25 x onset, width 0.45 x onset) with ruptures 0, 1e-3, 1e-2 /s
  csweep / crun   the N10-C stationary protocol (two bands, n L^2 9.8, sides 1 / 1.41 / 2 mm, 2 seeds, bias 1.5 x onset,
           0.6 s sampling) at r = 0, 1e-4, 1e-3 /s: does N_det (junctions with detectable ruptures) change when junctions
           switch at the measured voltage?                    -> results/raw/n10c/*_vs40_*.npz
  report   -> results/n10v_criteria.json, figures/fig26_n10v.png

    python scripts/n10v_vscale.py sweep | check | run [--procs N] | csweep | crun [--procs N] | report
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
VS, KMAX = 40.0, 100.0
TAU, SPREAD, HOLD = 0.3, 1.0, 100
SWEEP = ROOT / "results" / "n10v_sweep.json"
CHECK = ROOT / "results" / "n10v_stride_check.json"
OUT = ROOT / "results" / "n10v_criteria.json"
FIG = ROOT / "figures" / "fig26_n10v.png"
RUPTURES = (0.0, 1e-3, 1e-2)
C_SIDES, C_RUPTURES = (1.0, 1.41, 2.0), (0.0, 1e-4, 1e-3)
OBS = "electrodes"


# ------------------------------------------------------------------ pads onset + step check
def sweep():
    import n10s_sensitivity as S
    k, on = S.sweep_one(dict(v_scale=VS, k_max=KMAX))
    SWEEP.write_text(json.dumps({k: on}, indent=1))
    print(k, on)


def onset():
    return json.loads(SWEEP.read_text())[f"vs{VS:g}_km{KMAX:g}"]


def windows():
    on = onset()
    return [(round(1.05 * on, 3), round(0.45 * on, 3)), (round(1.25 * on, 3), round(0.45 * on, 3))]


def base_kw(stride=C.N10_STRIDE, **kw):
    wo = C.model_washout_steps("volatile", tau_s=TAU, tau_spread=SPREAD)
    return dict(washout=-(-wo // HOLD) * HOLD, n_virtual=C.E4_N_VIRTUAL, layout="pads", encoding="weights",
                model="volatile", tau_s=TAU, tau_spread=SPREAD, v_scale=VS, k_max=KMAX, stride=stride, **kw)


def _check_one(stride):
    vc, w = windows()[1]
    raw = simulate_multi(C.RunSpec("iid", HOLD, vc, 1, v_width=w, n_samples=60, **base_kw(stride)), save_states=True)
    return stride, raw["current"], np.minimum(1, raw["junction_x"].astype(float) + raw["junction_y"].astype(float))


def check():
    with ProcessPoolExecutor(2) as pool:
        res = dict((s, (c, j)) for s, c, j in pool.map(_check_one, (1, 2)))
    (c1, j1), (c2, j2) = res[1], res[2]
    out = {"current_rel_rms": float(np.sqrt(np.mean((c2 - c1) ** 2)) / np.sqrt(np.mean(c1 ** 2))),
           "current_corr": float(np.corrcoef(c1.ravel(), c2.ravel())[0, 1]),
           "filling_rel_rms": float(np.sqrt(np.mean((j2 - j1) ** 2)) / np.sqrt(np.mean(j1 ** 2))),
           "filling_corr": float(np.corrcoef(j1.ravel(), j2.ravel())[0, 1])}
    CHECK.write_text(json.dumps(out, indent=1))
    print(out)


# ------------------------------------------------------------------ reservoir runs
def specs():
    out = []
    for r in RUPTURES:
        for vc, w in windows():
            for net in C.E1_SEEDS:
                kw = base_kw(rupture=r)
                out += [C.RunSpec("iid", HOLD, vc, net, rep, v_width=w, n_samples=C.E4_N_SAMPLES, **kw)
                        for rep in C.E1_REPLICATES]
                out.append(C.RunSpec("sinesquare", HOLD, vc, net, 0, v_width=w, n_samples=C.E4_N_SAMPLES, **kw))
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


def run(procs):
    todo = [s for s in specs() if not raw_path(s, RAW).exists() and not raw_path(s, RAW).with_suffix(".running").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ N10-C stationary protocol at v_scale 40
CSWEEP = ROOT / "results" / "n10v_csweep.json"


def _csweep_one(net_key):
    import n10c_sim as S
    side, seed = net_key
    net, mc = S.setup(side, seed, S.NL2)
    vp = S.vparams(v_scale=VS)
    from agnw_rc.volatile import conductance, junction_kd0, step_state
    kd0 = junction_kd0(vp, len(net.junctions), seed)
    state = np.zeros((2, len(net.junctions)))
    v_max = 6.0 * side ** 1.5 * 5.0 / VS
    n = int(20.0 / vp.dt)
    v, g = [], []
    for i in range(n):
        vi = v_max * (i + 1) / n
        V = mc.solve(conductance(state, vp), np.array([vi, 0.0]), method="pcg")
        if i % 10 == 0:
            v.append(vi)
            g.append(mc.electrode_currents(V)[mc.fixed_electrodes[0]] / vi)
        state = step_state(state, mc.junction_voltage_drops(V), vp, kd0)
    v, g = np.array(v), np.array(g)
    on = float(v[np.argmax(g > 10 * g[:5].mean())]) if np.any(g > 10 * g[:5].mean()) else None
    return f"{side}_{seed}", on


def csweep(procs):
    import n10c_sim as S
    with ProcessPoolExecutor(procs) as pool:
        res = dict(pool.map(_csweep_one, [(s, n) for s in C_SIDES for n in S.SEEDS]))
    CSWEEP.write_text(json.dumps(res, indent=1))
    print(res)


def cfile(side, seed, r):
    import n10c_sim as S
    return S.RAW / f"side{side:g}_n{seed}_nL{S.NL2:.1f}_vs{VS:g}_rupture{r:g}.npz"


def _crun_one(job):
    import n10c_sim as S
    from agnw_rc.volatile import conductance, junction_kd0, step_state
    side, seed, r = job
    f = cfile(side, seed, r)
    mark = f.with_suffix(".running")
    if f.exists() or mark.exists():
        return f.name + " (exists or running)"
    mark.write_text(str(os.getpid()))
    on = json.loads(CSWEEP.read_text())[f"{side}_{seed}"]
    vb = S.BIAS_X_ONSET * on
    net, mc = S.setup(side, seed, S.NL2)
    vp = S.vparams(v_scale=VS, rupture_rate=r)
    kd0 = junction_kd0(vp, len(net.junctions), seed)
    rng = np.random.default_rng(1000 * seed + int(1e5 * r) + 40)
    state = np.zeros((2, len(net.junctions)))
    n_wo, n_rec, every = int(S.WASHOUT_S / vp.dt), int(S.RECORD_S / vp.dt), int(round(S.SAMPLE_S / vp.dt))
    t, G = [], []
    for i in range(n_wo + n_rec):
        V = mc.solve(conductance(state, vp), np.array([vb, 0.0]), method="pcg")
        if i >= n_wo and (i - n_wo) % every == 0:
            t.append((i - n_wo) * vp.dt)
            G.append(mc.electrode_currents(V)[mc.fixed_electrodes[0]] / vb)
        state = step_state(state, mc.junction_voltage_drops(V), vp, kd0, rng)
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t=np.array(t), G=np.array(G), bias_V=vb, onset_V=on, n_junctions=len(net.junctions),
                        x_final=state[0].astype(np.float32))
    tmp.rename(f)
    mark.unlink(missing_ok=True)
    return f.name


def crun(procs):
    import n10c_sim as S
    jobs = [(s, n, r) for s in sorted(C_SIDES, reverse=True) for n in S.SEEDS for r in C_RUPTURES]
    jobs = [j for j in jobs if not cfile(*j).exists()]
    print(f"{len(jobs)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(_crun_one, j) for j in jobs]), 1):
            print(f"[{i}/{len(jobs)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report
def _score(spec):
    from agnw_rc import evaluate as E
    raw = load_raw(spec, RAW)
    if spec.stream == "sinesquare":
        return spec, E.evaluate_sinesquare_run(raw), None
    return spec, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), E.observables(raw)[OBS].shape[1]


def _cons(pair):
    from agnw_rc import evaluate as E
    return pair[0], E.evaluate_consistency(load_raw(pair[0], RAW), load_raw(pair[1], RAW))


def _twin(spec):
    from agnw_rc import evaluate as E
    o = {}
    for tag, sp in (("rupture", spec), ("no_rupture", dataclasses.replace(spec, rupture=0.0))):
        raw = load_raw(sp, RAW)
        sc = E.evaluate_iid_run(raw, E.noise_floor(raw["u"]))[OBS]
        o[tag] = {k: float(sc[k]) for k in ("MC", "IPC2", "narma10_nmse")}
    return spec, o


def report(procs=8):
    from agnw_rc import criteria as K
    import n10c_sim as S
    sp = [s for s in specs() if raw_path(s, RAW).exists()]
    with ProcessPoolExecutor(procs) as pool:
        scored = list(pool.map(_score, sp))
        cs = list(pool.map(_cons, [(s, dataclasses.replace(s, replicate=1)) for s in sp if s.stream == "iid"
                                   and s.replicate == 0 and raw_path(dataclasses.replace(s, replicate=1), RAW).exists()]))
        tw = list(pool.map(_twin, [s for s in sp if s.stream == "iid" and s.rupture]))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in C.E1_SEEDS)
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    bl = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in bl]))}
    rows = []
    for r in RUPTURES:
        runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                     replicate=s.replicate, scores=sc) for s, sc, _ in scored if s.rupture == r]
        cons = [dict(hold=a.hold, v_center=a.v_center, consistency=c) for a, c in cs if a.rupture == r]
        nf = next(n for s, _, n in scored if s.rupture == r and n)
        strict = K.evaluate(runs, cons, OBS, nf, esn, floor)
        loose = K.evaluate(runs, cons, OBS, 16, esn, floor)
        ratios = [(o["rupture"]["MC"] + o["rupture"]["IPC2"]) / (o["no_rupture"]["MC"] + o["no_rupture"]["IPC2"])
                  for s, o in tw if s.rupture == r]
        rows.append({"setting": f"v_scale {VS:g}, rupture {r:g}/s", **strict, "G2_loose": loose["G2"],
                     "twin_ratio_median": float(np.median(ratios)) if ratios else None,
                     "twin_collapsed": int(np.sum(np.array(ratios) < 0.5)) if ratios else None, "twin_n": len(ratios)})
        x = rows[-1]
        print(f"{x['setting']:28s} op {x['operating_point']} MC {x['MC']['mean']:.2f}±{x['MC']['sd']:.2f} IPC2 "
              f"{x['IPC2']['mean']:.2f}±{x['IPC2']['sd']:.2f} NARMA {x['narma10_nmse']['mean']:.3f} acc "
              f"{x['accuracy']['mean'] if x['accuracy'] else float('nan'):.2f} cons "
              f"{x['consistency_inc']['mean'] if x['consistency_inc'] else float('nan'):.2f} -> G0 {x['G0']} G1 {x['G1']} "
              f"G2 loose {x['G2_loose']} | twins {x['twin_ratio_median']} collapsed {x['twin_collapsed']}/{x['twin_n']}")
    # calibration: N_det at v_scale 40 vs v_scale 5 (same networks, sides, threshold)
    thr = json.loads((ROOT / "results" / "n10c_data.json").read_text())["stationary_3p6V"]["threshold_rel_per_sample"]
    cal = []
    for side in C_SIDES:
        for seed in S.SEEDS:
            for r in C_RUPTURES:
                for vs, f in ((VS, cfile(side, seed, r)), (5.0, S.raw_file((side, seed, S.NL2, "rupture", r)))):
                    if not f.exists():
                        continue
                    with np.load(f) as z:
                        if "G" not in z.files:
                            continue
                        G = z["G"]
                        rel = np.diff(G) / G[:-1]
                        lam = float(np.sum(np.abs(rel) > thr) / (z["t"][-1] - z["t"][0]))
                        cal.append({"v_scale": vs, "side_mm": side, "seed": seed, "rupture": r, "jumps_per_s": lam,
                                    "N_det": lam / r if r else None, "bias_V": float(z["bias_V"]),
                                    "rel_sd_robust": float(1.4826 * np.median(np.abs(rel - np.median(rel))))})
    for c in sorted(cal, key=lambda c: (c["rupture"], c["side_mm"], c["seed"], c["v_scale"])):
        print(f"cal v_scale {c['v_scale']:>4g} side {c['side_mm']:4} n{c['seed']} r {c['rupture']:<6g} bias {c['bias_V']:.3f} V "
              f"jumps {c['jumps_per_s']:.3f}/s N_det {c['N_det'] if c['N_det'] is None else round(c['N_det'], 1)}")
    OUT.write_text(json.dumps({"v_scale": VS, "onset_pads_V": onset(), "windows": windows(), "esn_total": esn,
                               "noise_floor": floor, "rows": rows, "calibration": cal,
                               "stride_check": json.loads(CHECK.read_text())}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = {x["setting"]: x for x in json.loads((ROOT / "results" / "n10r_criteria.json").read_text())["rows"]}
    base = [(0.0, ref["no noise (N10-E)"])] + [(r, ref[f"rupture {r:g}/s"]) for r in RUPTURES[1:] if f"rupture {r:g}/s" in ref]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    for ax, k, t, thr_ in zip(axes[:3], ("MC", "IPC2", "narma10_nmse"),
                              ("memory capacity", "degree-2 capacity", "NARMA-10 NMSE"), (1, 1, 0.7)):
        ax.errorbar([max(p[0], 1e-4) for p in base], [p[1][k]["mean"] for p in base], [p[1][k]["sd"] for p in base],
                    marker="s", capsize=3, label="v_scale 5 (N10-E / N10-R)")
        ax.errorbar([max(r, 1e-4) for r in RUPTURES], [x[k]["mean"] for x in rows], [x[k]["sd"] for x in rows],
                    marker="o", capsize=3, label=f"v_scale {VS:g} (matched)")
        ax.axhline(thr_, color="r", ls="--", lw=0.8)
        ax.set(xscale="log", xlabel="rupture rate per junction (1/s; 1e-4 = none)", title=t)
    axes[0].legend(fontsize=8)
    for vs, mk in ((5.0, "s"), (VS, "o")):
        for r, col in ((1e-4, "C0"), (1e-3, "C1")):
            pts = [c for c in cal if c["v_scale"] == vs and c["rupture"] == r and c["N_det"]]
            axes[3].loglog([c["side_mm"] for c in pts], [c["N_det"] for c in pts], mk, color=col, mfc="none" if vs == 5 else col,
                           label=f"v_scale {vs:g}, r = {r:g}/s")
    axes[3].set(xlabel="network side (mm)", ylabel=r"$N_\mathrm{det} = \lambda / r$", title="detectable ruptures (N10-C protocol)")
    axes[3].legend(fontsize=7)
    fig.suptitle(f"N10-V: junction voltage scale matched to the measured network (v_scale {VS:g}, pad onset {onset()} V)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "check", "run", "csweep", "crun", "report"])
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    {"sweep": sweep, "check": check, "run": lambda: run(a.procs), "csweep": lambda: csweep(a.procs),
     "crun": lambda: crun(a.procs), "report": lambda: report(max(a.procs, 8))}[a.stage]()
