#!/usr/bin/env python3
"""N10-C stage B: the Milano et al. 2025 stationary protocol on N10 networks, to turn their network-level jump rate
(0.08 detectable jumps per second at 3.6 V, 7 mm gap; scripts/n10c_data.py) into a per-junction rupture rate.

Protocol, per network: two facing band electrodes (framework `two_bands`), N10-E junction (tau_s 0.3 s, spread 1,
v_scale 5, k_max 100 1/s, 2 ms step), constant bias at 1.5 x the network onset (their 3.6 V / 2.4 V), washout 20 s,
then 300 s recorded; the conductance is sampled every 0.6 s as in the measurement. Junction noise: random complete
ruptures (rate r per junction, as N10-R) or Gaussian noise on x (sigma, as N10-M), one at a time.
Analysis (report): relative change per sample Delta G / G, jumps = samples above the measured relative threshold
(1.2e-3, n10c_data.json), detected rate lambda(r), N_det = lambda / r (junctions whose rupture is detectable), up/down
balance, Gaussian sd per sample for sigma runs. Networks: our density n L^2 = 9.8 (the N10-R networks), sides
1-2.8 mm, 2 seeds; the size dependence of N_det and of the noise is extrapolated to their gap / L = 200. Their
mass-based density 5.8 is only 3 % above percolation (5.64): at 2-2.8 mm such networks barely conduct (first sweep,
results/n10c_sweep_nL5.8_first.json), so it is run as a 1 mm check only.

    python scripts/n10c_sim.py sweep [--procs N]   -> results/n10c_sweep.json (onset per network)
    python scripts/n10c_sim.py run [--procs N] [--reverse] [--limit K]   -> results/raw/n10c/*.npz
    python scripts/n10c_sim.py report              -> results/n10c_sim.json, figures/fig22_n10c_sim.png
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

from agnw_sim.geometry import build_network  # noqa: E402

from agnw_rc import config as C  # noqa: E402
from agnw_rc.electrodes import make_layout  # noqa: E402
from agnw_rc.multicircuit import MultiTerminalCircuit  # noqa: E402
from agnw_rc.volatile import conductance, junction_kd0, step_state  # noqa: E402

RAW = ROOT / "results" / "raw" / "n10c"
SWEEP = ROOT / "results" / "n10c_sweep.json"
OUT = ROOT / "results" / "n10c_sim.json"
FIG = ROOT / "figures" / "fig22_n10c_sim.png"
DATA = ROOT / "results" / "n10c_data.json"

TAU, SPREAD, STRIDE = 0.3, 1.0, 2
NL2 = C.N_WIRES / C.AREA * C.LENGTH ** 2   # 9.8, our density (the N10-R networks)
NL2_THEIRS = 5.8                           # their mass-based estimate (n10c_data.json): too close to percolation
                                           # (5.64) for a size study (2.83 mm networks barely conduct); 1 mm check only
SIDES_MM = (1.0, 1.41, 2.0, 2.83)          # areas 1, 2, 4, 8 mm^2
SEEDS = (1, 2)
NETS = [(s, n, NL2) for s in SIDES_MM for n in SEEDS] + [(1.0, n, NL2_THEIRS) for n in SEEDS]
BIAS_X_ONSET = 1.5                         # 3.6 V / 2.4 V
WASHOUT_S, RECORD_S, SAMPLE_S = 20.0, 300.0, 0.6
RUPTURES = (0.0, 1e-4, 0.001, 0.01, 0.1, 1.0)
SIGMAS = (0.01, 0.03, 0.1)
# Reduced grid for sides > 1 mm (2026-10-05): at r >= 0.01 /s and sigma >= 0.03 nearly every 0.6 s sample is already a
# jump at 1 mm (detection saturates), so only the low rates carry N_det; 1e-4 checks that 0.001 is in the linear range.
RUPTURES_BIG = (0.0, 1e-4, 0.001)
SIGMAS_BIG = (0.01,)
THRESHOLD_REL = 1.2e-3                      # overwritten from n10c_data.json when present


def n_wires(side_mm, nl2=NL2):
    return int(round(nl2 * (side_mm * 1e-3) ** 2 / C.LENGTH ** 2))


def setup(side_mm, seed, nl2):
    net = build_network(n_wires(side_mm, nl2), (side_mm * 1e-3) ** 2, C.LENGTH, np.random.default_rng(seed))
    _, wd = C.network_params()
    els = make_layout("two_bands", net)
    return net, MultiTerminalCircuit(net, wd, els)


def vparams(**kw):
    return C.volatile_params(None, tau_s=TAU, tau_spread=SPREAD, dt=STRIDE * C.DT, **kw)


# ------------------------------------------------------------------ onset sweep
def key(side, seed, nl2):
    return f"{side}_{seed}_{nl2:.1f}"


def sweep_one(net_key):
    side, seed, nl2 = net_key
    net, mc = setup(side, seed, nl2)
    vp = vparams()
    kd0 = junction_kd0(vp, len(net.junctions), seed)
    state = np.zeros((2, len(net.junctions)))
    v_max = 6.0 * side ** 1.5                                  # onset grows ~ side^1.5 (first sweep at n L^2 5.8)
    n = int(20.0 / vp.dt)
    v_t = v_max * np.arange(1, n + 1) / n                        # 0 -> v_max in 20 s (up ramp only)
    rec_v, rec_g = [], []
    for i, v in enumerate(v_t):
        V = mc.solve(conductance(state, vp), np.array([v, 0.0]), method="pcg")
        if i % 10 == 0:
            rec_v.append(v)
            rec_g.append(mc.electrode_currents(V)[mc.fixed_electrodes[0]] / v)
        state = step_state(state, mc.junction_voltage_drops(V), vp, kd0)
    v, g = np.array(rec_v), np.array(rec_g)
    g0 = g[:5].mean()
    onset = float(v[np.argmax(g > 10 * g0)]) if np.any(g > 10 * g0) else None
    return key(side, seed, nl2), {"side_mm": side, "seed": seed, "nL2": nl2, "n_wires": n_wires(side, nl2),
                                  "n_junctions": len(net.junctions), "onset_V": onset, "G_start_S": float(g0),
                                  "G_max_S": float(g.max()), "v_max": v_max}


def sweep(procs):
    with ProcessPoolExecutor(procs) as pool:
        res = dict(pool.map(sweep_one, NETS))
    SWEEP.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v)


# ------------------------------------------------------------------ stationary runs
def specs(full=False):
    """Runs to do: the full grid at 1 mm, the reduced grid above (full=True: the full grid everywhere, for the report,
    which also scores full-grid runs that were completed before the cut)."""
    out = []
    for side, seed, nl2 in NETS:
        big = side > 1.0 and not full
        out += [(side, seed, nl2, "rupture", r) for r in (RUPTURES_BIG if big else RUPTURES)]
        out += [(side, seed, nl2, "sigma", s) for s in (SIGMAS_BIG if big else SIGMAS)]
    return out


def done(spec):
    """Completed run (placeholders written to cancel queued runs of the cut grid do not count)."""
    f = raw_file(spec)
    if not f.exists():
        return False
    with np.load(f) as d:
        return "placeholder" not in d.files


def raw_file(spec):
    side, seed, nl2, kind, val = spec
    return RAW / f"side{side:g}_n{seed}_nL{nl2:.1f}_{kind}{val:g}.npz"


def run_one(spec):
    side, seed, nl2, kind, val = spec
    f = raw_file(spec)
    mark = f.with_suffix(".running")
    if f.exists() or mark.exists():
        return f.name + " (exists or running elsewhere)"
    mark.write_text(str(os.getpid()))
    onset = json.loads(SWEEP.read_text())[key(side, seed, nl2)]["onset_V"]
    if onset is None:
        mark.unlink(missing_ok=True)
        return f.name + " (no onset, skipped)"
    vb = BIAS_X_ONSET * onset
    net, mc = setup(side, seed, nl2)
    vp = vparams(**({"rupture_rate": val} if kind == "rupture" else {"sigma": val}))
    kd0 = junction_kd0(vp, len(net.junctions), seed)
    rng = np.random.default_rng(1000 * seed + int(1e4 * val) + (7 if kind == "sigma" else 0))
    state = np.zeros((2, len(net.junctions)))
    n_wo, n_rec, every = int(WASHOUT_S / vp.dt), int(RECORD_S / vp.dt), int(round(SAMPLE_S / vp.dt))
    Vf = np.array([vb, 0.0])
    t, G = [], []
    for i in range(n_wo + n_rec):
        V = mc.solve(conductance(state, vp), Vf, method="pcg")
        if i >= n_wo and (i - n_wo) % every == 0:
            t.append((i - n_wo) * vp.dt)
            G.append(mc.electrode_currents(V)[mc.fixed_electrodes[0]] / vb)
        state = step_state(state, mc.junction_voltage_drops(V), vp, kd0, rng)
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t=np.array(t), G=np.array(G), bias_V=vb, onset_V=onset,
                        n_junctions=len(net.junctions), x_final=state[0].astype(np.float32))
    tmp.rename(f)
    mark.unlink(missing_ok=True)
    return f.name


def run(procs, reverse=False, limit=0):
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs() if not raw_file(s).exists() and not raw_file(s).with_suffix(".running").exists()]
    todo.sort(key=lambda s: -s[0])                              # largest (slowest) networks first
    if reverse:
        todo.reverse()
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


# ------------------------------------------------------------------ report
def analyse(spec, thr):
    d = np.load(raw_file(spec))
    G = d["G"]
    rel = np.diff(G) / G[:-1]
    jumps = rel[np.abs(rel) > thr]
    T = float(d["t"][-1] - d["t"][0])
    return {"side_mm": spec[0], "seed": spec[1], "nL2": spec[2], "kind": spec[3], "value": spec[4],
            "bias_V": float(d["bias_V"]),
            "n_junctions": int(d["n_junctions"]), "G_mean_S": float(G.mean()), "rel_sd": float(rel.std()),
            "rel_sd_robust": float(1.4826 * np.median(np.abs(rel - np.median(rel)))),
            "jumps": int(len(jumps)), "jumps_up": int((jumps > 0).sum()), "jump_rate_per_s": len(jumps) / T,
            "jump_rel_median": float(np.median(np.abs(jumps))) if len(jumps) else None,
            "x_on_frac": float(np.mean(d["x_final"] > 0.5))}


def report():
    thr = json.loads(DATA.read_text())["stationary_3p6V"]["threshold_rel_per_sample"] if DATA.exists() else THRESHOLD_REL
    meas = json.loads(DATA.read_text())["stationary_3p6V"] if DATA.exists() else {}
    rows = [analyse(s, thr) for s in specs(full=True) if done(s)]
    main = [x for x in rows if x["nL2"] == NL2]                  # size study at our density; nL2 5.8 = check
    sw = json.loads(SWEEP.read_text())
    # N_det = lambda / r per network and rupture rate; size scaling of N_det at each r (log-log fit vs gap/L)
    gap_L_theirs = 7e-3 / 35e-6
    scal = {}
    for r in RUPTURES[1:]:
        pts = [(x["side_mm"] * 1e-3 / C.LENGTH, x["jump_rate_per_s"] / r) for x in main
               if x["kind"] == "rupture" and x["value"] == r and x["jumps"] > 0]
        if len({p[0] for p in pts}) >= 2:
            b, a = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)
            ndet = float(np.exp(a) * gap_L_theirs ** b)
            scal[f"{r:g}"] = {"exponent": float(b), "N_det_theirs": ndet,
                              "r_needed_per_s": meas.get("jump_rate_per_s_fig3g", 0.08) / ndet, "points": pts}
    sig = {}
    for s in SIGMAS:
        pts = [(x["n_junctions"], x["rel_sd_robust"]) for x in main if x["kind"] == "sigma" and x["value"] == s]
        if len({p[0] for p in pts}) >= 2:
            b, a = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)
            sig[f"{s:g}"] = {"exponent_vs_junctions": float(b), "points": pts}
    # junction count of their network (random sticks: 2 n L^2 / pi crossings per wire, each shared by two wires)
    dd = json.loads(DATA.read_text())["density"] if DATA.exists() else None
    nj_theirs = dd["their_wires_est"] * dd["their_nL2"] / np.pi if dd else None
    if nj_theirs and "0.01" in sig:
        b = sig["0.01"]["exponent_vs_junctions"]
        pts = sig["0.01"]["points"]
        a0 = float(np.mean([np.log(y) - b * np.log(x) for x, y in pts]))
        sd_theirs = float(np.exp(a0) * nj_theirs ** b)
        sig["0.01"].update({"junctions_theirs_est": nj_theirs, "rel_sd_at_theirs": sd_theirs,
                            "sigma_equivalent_linear": 0.01 * meas["gauss_sd_rel_per_sample"] / sd_theirs})
    out = {"threshold_rel": thr, "sweep": sw, "rows": rows, "rupture_scaling": scal, "sigma_scaling": sig,
           "gap_over_L_theirs": gap_L_theirs}
    OUT.write_text(json.dumps(out, indent=1))
    for x in rows:
        print(f"side {x['side_mm']:4} n{x['seed']} nL2 {x['nL2']:.1f} {x['kind']:7s} {x['value']:<6g} bias {x['bias_V']:.2f} V "
              f"G {x['G_mean_S']:.2e} sd {x['rel_sd_robust']:.1e} jumps {x['jumps']:4d} (up {x['jumps_up']}) "
              f"rate {x['jump_rate_per_s']:.3f}/s on {x['x_on_frac']:.2f}")
    if "sigma_equivalent_linear" in sig.get("0.01", {}):
        v = sig["0.01"]
        print(f"sigma 0.01: sd ~ junctions^{v['exponent_vs_junctions']:.2f}; at their ~{v['junctions_theirs_est']:.2g} junctions "
              f"{v['rel_sd_at_theirs']:.2g} -> measured sd needs sigma ~ {v['sigma_equivalent_linear']:.3f} (linear, upper bound)")
    for r, v in scal.items():
        print(f"r {r}: N_det ~ (gap/L)^{v['exponent']:.2f}, theirs {v['N_det_theirs']:.3g} -> r needed "
              f"{v['r_needed_per_s']:.3g} /s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for r in RUPTURES[1:]:
        xs = [x for x in main if x["kind"] == "rupture" and x["value"] == r]
        ax[0].loglog([x["side_mm"] * 1e-3 / C.LENGTH for x in xs], [max(x["jump_rate_per_s"], 1e-4) for x in xs],
                     "o", label=f"r = {r:g} /s")
    ax[0].axhline(meas.get("jump_rate_per_s_fig3g", 0.08), color="k", ls="--", lw=0.8, label="measured (7 mm)")
    ax[0].set(xlabel="gap / wire length", ylabel="detected jumps per s", title="detected network jumps")
    ax[0].legend(fontsize=8)
    for r in RUPTURES[1:]:
        xs = [x for x in main if x["kind"] == "rupture" and x["value"] == r and x["jumps"] > 0]
        ax[1].loglog([x["side_mm"] * 1e-3 / C.LENGTH for x in xs], [x["jump_rate_per_s"] / r for x in xs], "o",
                     label=f"r = {r:g} /s")
    ax[1].axvline(gap_L_theirs, color="k", ls="--", lw=0.8)
    ax[1].set(xlabel="gap / wire length", ylabel=r"$N_\mathrm{det} = \lambda / r$", title="junctions with detectable ruptures")
    ax[1].legend(fontsize=8)
    for s in SIGMAS:
        xs = [x for x in main if x["kind"] == "sigma" and x["value"] == s]
        ax[2].loglog([x["n_junctions"] for x in xs], [x["rel_sd_robust"] for x in xs], "o", label=rf"$\sigma$ = {s:g}")
    if meas:
        ax[2].axhline(meas["gauss_sd_rel_per_sample"], color="k", ls="--", lw=0.8, label="measured (7 mm)")
    ax[2].set(xlabel="junctions", ylabel=r"sd of $\Delta G/G$ per 0.6 s", title="Gaussian junction noise at network level")
    ax[2].legend(fontsize=8)
    fig.suptitle("N10-C: Milano 2025 stationary protocol on N10 (bias 1.5 x onset, sampled every 0.6 s)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sweep", "run", "report"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"sweep": lambda: sweep(a.procs), "run": lambda: run(a.procs, a.reverse, a.limit), "report": report}[a.stage]()
