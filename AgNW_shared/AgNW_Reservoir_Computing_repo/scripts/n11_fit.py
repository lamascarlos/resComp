#!/usr/bin/env python3
"""N11 stages 1-2: fit of the N10 volatile junction to the measured statistics of Milano et al. 2025
(results/n11_data.json) on the strip at the measured gap (docs/PLAN_N11_CALIBRATION.md sections 6-7; choices in 12).

Stage 1 (deterministic, junction noise off), each run = a relaxed strip held at one voltage, G sampled every 0.6 s:
  stage1 vscale [--procs N]   v_scale root find on the onset (tau_s 0.3 s, spread 1, k_max 100), seeds 1-2
  stage1 grid [--procs N]     tau_s x spread x k_max grid (v_scale rescaled per tau_s), 11 voltages + relaxation
  stage1 refine [--procs N]   3 x 3 x 3 around the best grid point
  stage1 checks [--procs N]   at the best point: widths 0.25 / 1 mm, 7.8 mm strip, 1 ms step, seed 2
  stage1 report               -> results/n11_fit_stage1.json, figures/fig34_n11_stage1.png
Stage 2 (noise; stage-1 best fixed): stage2 plan | run [--refine] | refine | report  (see stage-2 section)
  bench                       ms per step of the full strip -> budget
--mini: the 1.5 x 0.3 mm strip with second-long runs (tests, smoke runs), files under results/raw/n11_mini/.
"""
import argparse
import dataclasses
import json
import os
import random
import sys
import time
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agnw_rc import config as C  # noqa: E402
from agnw_rc import n11loss as L  # noqa: E402
from agnw_rc import series_stats as S  # noqa: E402
from agnw_rc.n11sim import V_RELAX, run_constant, strip_circuit  # noqa: E402
from agnw_rc.volatile import VolatileParams, junction_kd0  # noqa: E402

DATA_FILE = ROOT / "results" / "n11_data.json"


@dataclasses.dataclass(frozen=True)
class Setup:
    length_mm: float = 15.6        # free gap 0.9 x 15.6 = 14.0 mm = 200 wire lengths (spec section 12)
    width_mm: float = 0.5
    t_run_min: float = 200.0       # s  run length clip(20 tau_s, min, max) from a relaxed device
    t_run_max: float = 1000.0
    t_theta: float = 80.0          # s  theta fitted on the first 80 s (Fig. 2a)
    t_tail: float = 100.0          # s  steady state = mean of the last 100 s
    t_relax: float = 250.0         # s  relaxation at 10 mV (Fig. 1e)
    t_noise_wash: float = 100.0    # s  stage 2: noisy washout from the deterministic steady state
    t_record: float = 500.0        # s  stage 2: recorded stationary state (Fig. 2b inset)
    dt: float = 2e-3
    raw: Path = ROOT / "results" / "raw" / "n11"
    out: Path = ROOT / "results"
    fig: Path = ROOT / "figures"


FULL = Setup()
MINI = Setup(length_mm=1.5, width_mm=0.3, t_run_min=3.0, t_run_max=3.0, t_theta=2.4, t_tail=1.2, t_relax=6.0,
             t_noise_wash=0.6, t_record=6.0, raw=ROOT / "results" / "raw" / "n11_mini", out=ROOT / "results" / "n11_mini",
             fig=ROOT / "results" / "n11_mini")

REF = {"tau": 0.3, "sp": 1.0, "km": 100.0}               # the N10 values: reference point of the v_scale root find
GRID = {"tau": (0.3, 3.0, 30.0), "sp": (0.0, 1.0, 2.0), "km": (10.0, 100.0, 1000.0)}
V_ONSET = (2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0, 5.5, 6.6)  # bracket the onset; >= 5 V for G_sat
V_GRID = (2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0, 3.6, 4.5, 5.5, 6.6)
V_DRIVE_RELAX = 3.7                                     # 1.5 x the measured onset
CHECK_VOLTS = (3.0, 4.5)                                # step-size check
VS_TOL = 0.02
FIELDS = ("kind", "seed", "L", "W", "vs", "tau", "sp", "km", "V", "dt", "law", "r0", "alpha", "pstar", "P0", "frac",
          "sigma")


# ------------------------------------------------------------------ runs and files
def name(r):
    """Run key: every field that changes the simulation (Review Focus 4)."""
    parts = []
    for k in FIELDS:
        v = r.get(k)
        if v is None or v == "":
            continue
        parts.append(f"{k}{v:.4g}" if isinstance(v, float) else f"{k}{v}")
    return "_".join(parts)


def t_run(setup, tau):
    return float(np.clip(20 * tau, setup.t_run_min, setup.t_run_max))


def point(seed, vs, tau, sp, km, setup, dt=None, L=None, W=None):
    return {"seed": seed, "L": L or setup.length_mm, "W": W or setup.width_mm, "vs": float(f"{vs:.4g}"), "tau": tau,
            "sp": sp, "km": km, "dt": dt or setup.dt}


def scale_vs(vs_ref, tau):
    """v_scale keeping the junction threshold k_p = k_d at the reference voltage: v ~ ln(1 / (k_p0 tau_s))."""
    k = VolatileParams().k_p0
    return float(f"{vs_ref * np.log(1 / (k * tau)) / np.log(1 / (k * REF['tau'])):.4g}")


def vparams(r):
    kw = dict(tau_s=r["tau"], tau_spread=r["sp"], v_scale=r["vs"], k_max=r["km"], dt=r["dt"])
    if r.get("law"):
        kw.update(rupture_law=r["law"], rupture_rate=r["r0"], rupture_alpha=r.get("alpha") or 0.0,
                  rupture_P0=r["P0"], rupture_Pstar=(r["pstar"] * r["P0"]) if r.get("pstar") else np.inf,
                  rupture_frac=r["frac"], sigma=r.get("sigma") or 0.0)
    return C.volatile_params(None, **kw)


BASE_KEYS = ("seed", "L", "W", "vs", "tau", "sp", "km", "V", "dt")


def base_of(r):
    return {k: r[k] for k in BASE_KEYS} | {"kind": "base"}


def raw_file(r, setup):
    return setup.raw / (name(r) + ".npz")


STALE_S = 24 * 3600.0     # a .running marker older than this belongs to a killed job (longest run: about 4 h)


def claim(mark):
    """Atomically create the .running marker (O_EXCL: safe for several nodes on the shared /home). A marker older than
    STALE_S (killed job: SLURM time limit, scancel) is taken over."""
    for _ in range(2):
        try:
            fd = os.open(mark, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                born = float(mark.read_text().split()[2])
            except (OSError, IndexError, ValueError):
                born = mark.stat().st_mtime if mark.exists() else 0.0
            if time.time() - born < STALE_S:
                return False
            mark.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{os.uname().nodename} {os.getpid()} {time.time()}")
        return True
    return False


def run_one(r, setup):
    f = raw_file(r, setup)
    mark = f.with_suffix(".running")
    if f.exists():
        return f.name + " (exists)"
    if r["kind"] == "noise" and not raw_file(base_of(r), setup).exists():
        return f.name + " (base state not ready, skipped)"
    setup.raw.mkdir(parents=True, exist_ok=True)
    if not claim(mark):
        return f.name + " (running elsewhere)"
    try:
        s, mc = strip_circuit(r["seed"], r["L"], r["W"])
        vp = vparams(r)
        kd0 = junction_kd0(vp, len(s.network.junctions), r["seed"])
        T = t_run(setup, r["tau"])
        if r["kind"] == "step":
            a = run_constant(mc, vp, kd0, r["V"], T)
            data = {"t": a["t"], "G": a["G"], "P_formed_median": a["P_formed_median"]}
        elif r["kind"] == "relax":
            a = run_constant(mc, vp, kd0, r["V"], T)
            b = run_constant(mc, vp, kd0, V_RELAX, setup.t_relax, state=a["state"], t0=T)
            data = {"t": b["t"], "G": b["G"]}
        elif r["kind"] == "base":
            a = run_constant(mc, vp, kd0, r["V"], T)
            data = {"t": a["t"], "G": a["G"], "state": a["state"], "P_formed_median": a["P_formed_median"]}
        elif r["kind"] == "noise":
            state = np.load(raw_file(base_of(r), setup))["state"]
            rng = np.random.default_rng(zlib.crc32(name(r).encode()))
            a = run_constant(mc, vp, kd0, r["V"], setup.t_noise_wash, state=state, rng=rng)
            b = run_constant(mc, vp, kd0, r["V"], setup.t_record, state=a["state"], rng=rng)
            data = {"t": b["t"], "G": b["G"], "mean_rupture_rate": b["mean_rupture_rate"]}
        else:
            raise ValueError(f"unknown run kind {r['kind']!r}")
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.rename(f)
    finally:
        mark.unlink(missing_ok=True)
    return f.name


def _run_one_star(args):
    return run_one(*args)


def run_many(rs, procs, setup):
    todo = [r for r in rs if not raw_file(r, setup).exists()]
    random.Random(os.environ.get("SLURM_JOB_ID", os.getpid())).shuffle(todo)   # nodes start on different runs
    if procs <= 1:
        for r in todo:
            print(run_one(r, setup), flush=True)
    else:
        with ProcessPoolExecutor(procs) as pool:
            for msg in pool.map(_run_one_star, [(r, setup) for r in todo]):
                print(msg, flush=True)
    return len(todo)


def load(r, setup):
    f = raw_file(r, setup)
    if not f.exists():
        return None
    with np.load(f) as d:
        return {k: d[k] for k in d.files}


# ------------------------------------------------------------------ stage 1
def steps(p, volts):
    return [p | {"kind": "step", "V": V} for V in volts]


def relax_run(p):
    return p | {"kind": "relax", "V": V_DRIVE_RELAX}


def point_runs(p, volts):
    return steps(p, volts) + [relax_run(p)]


def point_stats(p, volts, setup, data, need_relax=True):
    """Simulated statistics of one parameter point (None if a run is missing)."""
    ss, th, P, conv = {}, [], {}, []
    for V in volts:
        d = load(p | {"kind": "step", "V": V}, setup)
        if d is None:
            return None
        t, G = d["t"], d["G"]
        tail = G[t >= t[-1] - setup.t_tail]
        prev = G[(t >= t[-1] - 2 * setup.t_tail) & (t < t[-1] - setup.t_tail)]
        ss[V] = float(tail.mean())
        conv.append(bool(prev.size == 0 or abs(tail.mean() / prev.mean() - 1) < 0.02))
        m = t <= setup.t_theta
        try:
            f = S.fit_theta(t[m], G[m], n_boot=50)
            th.append({"V": V, "theta_per_s": f["theta_per_s"], "unresolved": f["unresolved"]})
        except (RuntimeError, ValueError):
            th.append({"V": V, "theta_per_s": 0.0, "unresolved": True})
        P[V] = float(d["P_formed_median"])
    relax_tau = None
    if need_relax:
        r = load(relax_run(p), setup)
        if r is None:
            return None
        try:
            relax_tau = S.fit_relaxation(r["t"], r["G"])["one"]["tau_s"]
        except (RuntimeError, ValueError):
            relax_tau = None                                          # no fit: the relaxation term is left out
    Vs = np.array(volts, dtype=float)
    Gss = np.array([ss[V] for V in volts])
    hi = Vs >= 5.0
    gsat = float(np.median(Gss[hi])) if hi.any() else float(Gss.max())
    return {"onset_V": S.onset(Vs, Gss, gsat), "G_sat_S": gsat, "G_ss": {str(V): ss[V] for V in volts},
            "shape": [{"V": V, "ratio": ss[V] / gsat} for V in volts if V > data["onset_V"]],
            "theta": th, "relax_tau_s": relax_tau, "P_formed_median": {str(V): P[V] for V in volts},
            "converged": all(conv)}


def next_vscale(vs, onset, onset_data, lowest=V_ONSET[0]):
    """One root-find step on onset(v_scale) ~ 1 / v_scale (Review Focus 1)."""
    if onset is None:
        return float(f"{vs * 1.5:.4g}")
    if onset <= lowest + 1e-9:
        return float(f"{vs * 0.7:.4g}")
    return float(f"{vs * onset / onset_data:.4g}")


def stage1_vscale(procs, setup, data, vs0=40.0, max_iter=6, volts=V_ONSET):
    """Iterate v_scale at the reference point (seeds 1-2) until the mean onset is within VS_TOL of the measured one."""
    f = setup.out / "n11_stage1_vscale.json"
    hist = json.loads(f.read_text())["history"] if f.exists() else []
    vs = hist[-1]["next"] if hist else vs0
    for _ in range(max_iter):
        if hist and abs(hist[-1]["onset_mean"] / data["onset_V"] - 1) <= VS_TOL:
            break
        ons = []
        for seed in (1, 2):
            p = point(seed, vs, setup=setup, **REF)
            run_many(steps(p, volts), procs, setup)
            st = point_stats(p, volts, setup, data, need_relax=False)
            if st is None:
                raise SystemExit(f"vscale: runs missing for v_scale {vs} seed {seed} (running elsewhere, or a killed "
                                 f"job's marker younger than {STALE_S / 3600:g} h): run vscale on one node only")
            ons.append(st["onset_V"])
        valid = [o for o in ons if o is not None]
        mean = float(np.mean(valid)) if len(valid) == len(ons) else None
        nxt = next_vscale(vs, mean if mean is None or mean > volts[0] + 1e-9 else volts[0], data["onset_V"], volts[0])
        hist.append({"v_scale": vs, "onsets_V": ons, "onset_mean": mean if mean is not None else float("nan"),
                     "next": nxt})
        print(hist[-1], flush=True)
        setup.out.mkdir(parents=True, exist_ok=True)
        ok = [h for h in hist if np.isfinite(h["onset_mean"])]
        best = min(ok, key=lambda h: abs(h["onset_mean"] / data["onset_V"] - 1)) if ok else None
        out = {"history": hist, "v_scale_ref": best["v_scale"] if best else hist[-1]["v_scale"]}
        f.write_text(json.dumps(out, indent=1))
        vs = nxt
    return json.loads(f.read_text())


def _vs_ref(setup):
    return json.loads((setup.out / "n11_stage1_vscale.json").read_text())["v_scale_ref"]


def grid_points(setup, grid=GRID, seed=1):
    v = _vs_ref(setup)
    return [point(seed, scale_vs(v, tau), tau, sp, km, setup)
            for tau in grid["tau"] for sp in grid["sp"] for km in grid["km"]]


def stage1_grid(procs, setup, data, grid=GRID, volts=V_GRID):
    pts = grid_points(setup, grid)
    run_many([r for p in pts for r in point_runs(p, volts)], procs, setup)
    return pts


def _report_json(setup):
    return setup.out / "n11_fit_stage1.json"


def refine_grid(best):
    f = np.sqrt(10.0)
    return {"tau": (best["tau"] / f, best["tau"], best["tau"] * f),
            "sp": tuple(sorted({max(0.0, best["sp"] - 0.5), best["sp"], best["sp"] + 0.5})),
            "km": (best["km"] / f, best["km"], best["km"] * f)}


def _refines_file(setup):
    return setup.out / "n11_stage1_refines.json"


def stage1_refine(procs, setup, data, volts=V_GRID, recenter=False):
    """3 x 3 x 3 around the current best; the grid is recorded so the report keeps it when the best moves. Once a
    refine grid exists it is reused (nodes started later must not open a new, unbudgeted grid); a new one is centred
    on the current best only with recenter=True (--recenter)."""
    f = _refines_file(setup)
    grids = json.loads(f.read_text()) if f.exists() else []
    if grids and not recenter:
        return stage1_grid(procs, setup, data, grid={k: tuple(v) for k, v in grids[-1].items()}, volts=volts)
    best = stage1_report(setup, data, volts=volts)["best"]["params"]
    rg = {k: tuple(float(f"{x:.4g}") for x in v) for k, v in refine_grid(best).items()}
    if {k: list(v) for k, v in rg.items()} not in grids:
        grids.append({k: list(v) for k, v in rg.items()})
        f.write_text(json.dumps(grids, indent=1))
    return stage1_grid(procs, setup, data, grid=rg, volts=volts)


def check_points(setup, best):
    b = dict(best)
    pts = {"width_0.25": point(1, b["vs"], b["tau"], b["sp"], b["km"], setup, W=0.25),
           "width_1": point(1, b["vs"], b["tau"], b["sp"], b["km"], setup, W=1.0),
           "length_half": point(1, b["vs"], b["tau"], b["sp"], b["km"], setup, L=setup.length_mm / 2),
           "seed_2": point(2, b["vs"], b["tau"], b["sp"], b["km"], setup),
           "dt_1ms": point(1, b["vs"], b["tau"], b["sp"], b["km"], setup, dt=setup.dt / 2)}
    return pts


def stage1_checks(procs, setup, data, volts=V_GRID):
    best = stage1_report(setup, data, volts=volts)["best"]["params"]     # refreshed: includes refined points
    pts = check_points(setup, best)
    runs = []
    for k, p in pts.items():
        runs += steps(p, CHECK_VOLTS) if k == "dt_1ms" else point_runs(p, volts)
    run_many(runs, procs, setup)
    return list(pts.values())


def _evaluate(p, volts, setup, data):
    st = point_stats(p, volts, setup, data)
    if st is None:
        return None
    return {"params": {k: p[k] for k in ("vs", "tau", "sp", "km")}, "stats": st,
            "terms": L.stage1_terms(st, data), "verdict": L.calibration_verdict(st, data)}


def stage1_report(setup, data, grid=GRID, volts=V_GRID):
    pts, incomplete = [], 0
    rf = _report_json(setup)
    cands = grid_points(setup, grid)
    if _refines_file(setup).exists():
        for rg in json.loads(_refines_file(setup).read_text()):
            cands += grid_points(setup, {k: tuple(v) for k, v in rg.items()})
    seen = set()
    for p in cands:
        if name(p) in seen:
            continue
        seen.add(name(p))
        e = _evaluate(p, volts, setup, data)
        if e is None:
            incomplete += 1
            continue
        e["loss"] = e["terms"]["loss"]
        pts.append(e)
    best = min(pts, key=lambda e: e["loss"]) if pts else None
    if best is not None:
        best = dict(best)
        best["P_formed_median_at_1p5_onset"] = best["stats"]["P_formed_median"].get(
            str(min(volts, key=lambda V: abs(V - 1.5 * data["onset_V"]))))
        best["checks"] = _checks(setup, data, best, volts)
    out = {"points": pts, "incomplete": incomplete, "best": best, "v_scale_ref": _vs_ref(setup)}
    setup.out.mkdir(parents=True, exist_ok=True)
    rf.write_text(json.dumps(out, indent=1, default=float))
    if best is not None:
        _figure1(setup, data, out)
    print(f"stage 1: {len(pts)} points, {incomplete} incomplete; best {best['params'] if best else None}, "
          f"loss {best['loss'] if best else None}, verdict {best['verdict'] if best else None}")
    return out


def _checks(setup, data, best, volts):
    """Width, length, seed and step-size checks at the best point (None where not run yet)."""
    pts = check_points(setup, best["params"] | {"vs": best["params"]["vs"]})
    out = {}
    ref = best["stats"]
    for k in ("width_0.25", "width_1"):
        st = point_stats(pts[k], volts, setup, data)
        if st is None or st["onset_V"] is None or ref["onset_V"] is None:
            out[k] = None
            continue
        sh = [abs(a["ratio"] / b["ratio"] - 1) for a, b in zip(st["shape"], ref["shape"])]
        out[k] = {"onset_rel_diff": abs(st["onset_V"] / ref["onset_V"] - 1), "shape_max_rel_diff": max(sh, default=0.0)}
        out[k]["within_5pct"] = bool(out[k]["onset_rel_diff"] <= 0.05 and out[k]["shape_max_rel_diff"] <= 0.05)
    for k in ("length_half", "seed_2"):
        st = point_stats(pts[k], volts, setup, data)
        out[k] = None if st is None else {"onset_V": st["onset_V"], "loss": L.stage1_terms(st, data)["loss"]}
    d1 = [load(pts["dt_1ms"] | {"kind": "step", "V": V}, setup) for V in CHECK_VOLTS]
    d2 = [load(point(1, best["params"]["vs"], best["params"]["tau"], best["params"]["sp"], best["params"]["km"], setup)
               | {"kind": "step", "V": V}, setup) for V in CHECK_VOLTS]
    if all(x is not None for x in d1 + d2):
        g = [abs(a["G"][-10:].mean() / b["G"][-10:].mean() - 1) for a, b in zip(d1, d2)]
        th = []
        for a, b in zip(d1, d2):
            try:
                ta = S.fit_theta(a["t"][a["t"] <= setup.t_theta], a["G"][a["t"] <= setup.t_theta], n_boot=5)["theta_per_s"]
                tb = S.fit_theta(b["t"][b["t"] <= setup.t_theta], b["G"][b["t"] <= setup.t_theta], n_boot=5)["theta_per_s"]
                th.append(abs(ta / tb - 1) if tb > 0 else np.inf)
            except (RuntimeError, ValueError):
                th.append(np.inf)
        out["dt_1ms"] = {"G_rel_diff_max": max(g), "theta_rel_diff_max": max(th),
                         "within_2pct": bool(max(g) <= 0.02 and max(th) <= 0.02)}
    else:
        out["dt_1ms"] = None
    return out


def _figure1(setup, data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = out["best"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ds = data["shape"]
    ax[0].semilogy([r["V"] for r in ds], [r["ratio"] for r in ds], "o", ms=3, label="measured (Fig. 2b)")
    ax[0].semilogy([r["V"] for r in best["stats"]["shape"]], [r["ratio"] for r in best["stats"]["shape"]], "s-",
                   label="best fit")
    ax[0].axvline(data["onset_V"], color="r", ls="--", lw=0.8)
    ax[0].set(xlabel="applied voltage (V)", ylabel=r"$G/G_\mathrm{sat}$", title="steady state")
    ax[0].legend(fontsize=8)
    tg = L.theta_targets(data["theta"])
    ax[1].plot([t["V"] for t in tg], [t["theta"] for t in tg], "o", ms=3, label="measured, smoothed")
    ok = [r for r in best["stats"]["theta"] if not r["unresolved"]]
    ax[1].plot([r["V"] for r in ok], [r["theta_per_s"] for r in ok], "s", label="best fit")
    ax[1].set(yscale="log", xlabel="applied voltage (V)", ylabel=r"$\theta$ (1/s)", title="potentiation rate")
    ax[1].legend(fontsize=8)
    loss = np.array([e["loss"] for e in out["points"]])
    ax[2].plot(np.sort(np.minimum(loss, 1e6)), "o", ms=3)
    ax[2].set(yscale="log", xlabel="parameter point (sorted)", ylabel="stage-1 loss", title="all points")
    fig.suptitle(f"N11 stage 1: best $\\tau_s$ {best['params']['tau']:g} s, spread {best['params']['sp']:g}, "
                 f"$k_\\mathrm{{max}}$ {best['params']['km']:g} /s, v_scale {best['params']['vs']:g}")
    fig.tight_layout()
    setup.fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(setup.fig / "fig34_n11_stage1.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "bench"])
    ap.add_argument("phase", nargs="?", default="")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--vs0", type=float, default=40.0)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--recenter", action="store_true", help="stage1 refine: new grid around the current best")
    a = ap.parse_args()
    setup = MINI if a.mini else FULL
    data = json.loads(DATA_FILE.read_text())
    if a.stage == "bench":
        return bench(setup)
    if a.stage == "stage1":
        {"vscale": lambda: stage1_vscale(a.procs, setup, data, vs0=a.vs0),
         "grid": lambda: stage1_grid(a.procs, setup, data),
         "refine": lambda: stage1_refine(a.procs, setup, data, recenter=a.recenter),
         "checks": lambda: stage1_checks(a.procs, setup, data),
         "report": lambda: stage1_report(setup, data)}[a.phase]()
    else:
        stage2_cli(a, setup, data)


def bench(setup, n=200):
    """Seconds per 2 ms step on the strip (one process), and the stage 1-2 budget it implies."""
    import time
    s, mc = strip_circuit(1, setup.length_mm, setup.width_mm)
    p = point(1, 40.0, setup=setup, **REF)
    vp = vparams(p)
    kd0 = junction_kd0(vp, len(s.network.junctions), 1)
    a = run_constant(mc, vp, kd0, 3.6, 0.2)                     # warm-up (factorisation caches, imports)
    t0 = time.time()
    run_constant(mc, vp, kd0, 3.6, n * vp.dt, state=a["state"])
    sec = (time.time() - t0) / n
    steps_per_s = 1 / vp.dt
    h = lambda sim_s: sim_s * steps_per_s * sec / 3600           # core-hours per run of sim_s seconds
    n_grid = len(GRID["tau"]) * len(GRID["sp"]) * len(GRID["km"])
    t_mean = np.mean([t_run(setup, tau) for tau in GRID["tau"]])
    budget = {
        "ms_per_step": 1e3 * sec, "junctions": len(s.network.junctions),
        "stage1_vscale": 6 * 2 * len(V_ONSET) * h(setup.t_run_min),
        "stage1_grid": n_grid * (len(V_GRID) * h(t_mean) + h(t_mean + setup.t_relax)),
        "stage1_refine": n_grid * (len(V_GRID) * h(t_mean) + h(t_mean + setup.t_relax)),
        "stage1_checks": 4 * (len(V_GRID) + 1) * h(t_mean) + len(CHECK_VOLTS) * 2 * h(t_mean),
        "stage2_base": 2 * N_STAGE2_V * h(t_mean),
        "stage2_coarse": len(coarse_configs()) * 2 * N_STAGE2_V * h(setup.t_noise_wash + setup.t_record),
        "stage2_refine": 6 * 2 * N_STAGE2_V * h(setup.t_noise_wash + setup.t_record)}
    budget["total_core_h"] = sum(v for k, v in budget.items() if k.startswith("stage"))
    print(json.dumps(budget, indent=1))
    (setup.out / "n11_budget.json").write_text(json.dumps(budget, indent=1))
    return budget


# ------------------------------------------------------------------ stage 2
R0 = (1e-4, 1e-3, 1e-2, 1e-1)           # 1/s per junction, 4 decades around N10-C's 0.005-0.03
ALPHAS = (0.5, 1.0, 2.0)
PSTARS = (0.3, 1.0, 3.0)                 # P* in units of P0
FRACS = (1.0, 0.5)
SIGMAS = (0.01, 0.03, 0.1)               # Gaussian junction noise, only if the best law's sd < data sd / 2
N_STAGE2_V = 6


def coarse_configs():
    out = []
    for r0 in R0:
        for frac in FRACS:
            out.append({"law": "constant", "r0": r0, "frac": frac})
            out += [{"law": "power", "r0": r0, "alpha": a, "frac": frac} for a in ALPHAS]
            out += [{"law": "thermal", "r0": r0, "pstar": ps, "frac": frac} for ps in PSTARS]
    return out


def _plan_file(setup):
    return setup.out / "n11_stage2_plan.json"


def stage2_plan(setup, data, s1):
    """Stage-1 best parameters, 6 measured voltages spread over those with >= 20 jumps, P0 from stage 1, coarse grid."""
    P0 = s1["best"].get("P_formed_median_at_1p5_onset")
    if P0 is None or not np.isfinite(P0) or P0 <= 0:
        raise SystemExit(f"stage2 plan: P0 undefined ({P0}): no formed junctions in the best stage-1 run at 1.5 x onset")
    cand = data["stage2_voltages"]
    idx = np.unique(np.round(np.linspace(0, len(cand) - 1, N_STAGE2_V)).astype(int))
    plan = {"params": s1["best"]["params"], "voltages": [cand[i] for i in idx],
            "P0_W": s1["best"]["P_formed_median_at_1p5_onset"], "coarse": coarse_configs(), "refine": []}
    setup.out.mkdir(parents=True, exist_ok=True)
    _plan_file(setup).write_text(json.dumps(plan, indent=1))
    return plan


def _noise_run(plan, cfg, seed, V, setup):
    b = plan["params"]
    return point(seed, b["vs"], b["tau"], b["sp"], b["km"], setup) | {"kind": "noise", "V": V, "P0": plan["P0_W"]} | cfg


def stage2_runs(plan, setup, refine=False):
    cfgs = plan["coarse"] + (plan["refine"] if refine else [])
    noise = [_noise_run(plan, c, seed, V, setup) for c in cfgs for seed in (1, 2) for V in plan["voltages"]]
    base = list({name(base_of(r)): base_of(r) for r in noise}.values())
    return base, noise


def config_stats(plan, cfg, setup):
    """Per voltage (mean over seeds): events, duration, Gaussian sd, rate, sizes, mean rupture rate per junction.
    None if a run is missing; status 'disconnected' if a record hits G <= 0 (Review Focus 3)."""
    by_V = {}
    for V in plan["voltages"]:
        rows = []
        for seed in (1, 2):
            d = load(_noise_run(plan, cfg, seed, V, setup), setup)
            if d is None:
                return None
            if np.any(~np.isfinite(d["G"])) or np.any(d["G"] <= 0):
                return {"status": "disconnected", "by_V": {}}
            js = S.jump_stats(d["t"], d["G"])
            rows.append((js, float(d["mean_rupture_rate"])))
        by_V[V] = {"events": float(np.mean([j["events"] for j, _ in rows])),
                   "duration_s": float(np.mean([j["duration_s"] for j, _ in rows])),
                   "sd_rel": float(np.mean([j["sd_rel"] for j, _ in rows])),
                   "rate_per_s": float(np.mean([j["rate_per_s"] for j, _ in rows])),
                   "sizes": np.concatenate([j["sizes"] for j, _ in rows]),
                   "mean_rupture_rate": float(np.mean([m for _, m in rows]))}
    return {"status": "ok", "by_V": by_V}


def stage2_report(setup, data):
    plan = json.loads(_plan_file(setup).read_text())
    rows = {r["V"]: r for r in data["inset"] if "jumps" in r}
    data_rows = {V: rows[V] for V in plan["voltages"]}
    data_sizes = np.concatenate([np.asarray(rows[V]["sizes"], dtype=float) for V in plan["voltages"]])
    v_op = min(plan["voltages"], key=lambda V: abs(V - 1.5 * data["onset_V"]))
    configs, incomplete = [], 0
    for cfg in plan["coarse"] + plan["refine"]:
        st = config_stats(plan, cfg, setup)
        if st is None:
            incomplete += 1
            continue
        c = dict(cfg) | {"status": st["status"]}
        if st["status"] == "ok":
            t = L.stage2_terms(st["by_V"], data_rows, data_sizes)
            c |= {"terms": t, "loss": t["loss"], "aic": L.aic(t["loss"], cfg["law"]),
                  "rate_by_V": {V: s["rate_per_s"] for V, s in st["by_V"].items()},
                  "sd_by_V": {V: s["sd_rel"] for V, s in st["by_V"].items()},
                  "mean_rupture_rate_at_op": st["by_V"][v_op]["mean_rupture_rate"]}
        else:
            c |= {"loss": np.inf, "aic": np.inf}
        configs.append(c)
    best_by_law = {}
    for law in L.K_PARAMS:
        ok = [c for c in configs if c["law"] == law and np.isfinite(c["aic"])]
        if ok:
            best_by_law[law] = min(ok, key=lambda c: c["aic"])
    chosen, reason = (L.choose_law(best_by_law, data_rows) if "constant" in best_by_law
                      else (None, "no complete constant-law configuration"))
    out = {"plan": plan, "configs": configs, "incomplete": incomplete, "best_by_law": best_by_law,
           "chosen_law": chosen, "reason": reason, "operating_voltage": v_op}
    if chosen is not None:
        law_rows = [c for c in configs if c["law"] == chosen and np.isfinite(c["aic"])]
        lo, hi = L.r0_interval(law_rows)
        b = best_by_law[chosen]
        ends = [c for c in law_rows if c["r0"] in (lo, hi)]
        out |= {"r0_best": b["r0"], "r0_interval": [lo, hi], "mean_rate_per_junction_best": b["mean_rupture_rate_at_op"],
                "mean_rate_per_junction_range": [min(c["mean_rupture_rate_at_op"] for c in ends),
                                                 max(c["mean_rupture_rate_at_op"] for c in ends)],
                "needs_sigma": bool(np.median([b["sd_by_V"][V] / data_rows[V]["jumps"]["sd_rel"]
                                               for V in plan["voltages"]]) < 0.5)}
    setup.out.mkdir(parents=True, exist_ok=True)
    (setup.out / "n11_fit_stage2.json").write_text(json.dumps(out, indent=1, default=_json_default))
    if best_by_law:
        _figure2(setup, data_rows, out)
    print(f"stage 2: {len(configs)} configs, {incomplete} incomplete; law {chosen} ({reason})")
    return out


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return float(o)


def stage2_refine(setup):
    """Around each law's best coarse point: r0 x 1/3 and x 3 (other settings kept); appended to the plan."""
    rep = json.loads((setup.out / "n11_fit_stage2.json").read_text())
    plan = json.loads(_plan_file(setup).read_text())
    keys = ("law", "r0", "alpha", "pstar", "frac")
    new = []
    for b in rep["best_by_law"].values():
        for f in (1 / 3, 3.0):
            c = {k: b[k] for k in keys if k in b}
            c["r0"] = float(f"{b['r0'] * f:.4g}")
            new.append(c)
    have = {json.dumps(c, sort_keys=True) for c in plan["coarse"] + plan["refine"]}
    plan["refine"] += [c for c in new if json.dumps(c, sort_keys=True) not in have]
    _plan_file(setup).write_text(json.dumps(plan, indent=1))
    return plan["refine"]


def _figure2(setup, data_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    Vs = sorted(data_rows)
    rate = [data_rows[V]["jumps"]["rate_per_s"] for V in Vs]
    ci = np.array([data_rows[V]["jumps"]["rate_ci_per_s"] for V in Vs])
    ax[0].errorbar(Vs, rate, yerr=[np.maximum(np.array(rate) - ci[:, 0], 0), np.maximum(ci[:, 1] - np.array(rate), 0)],
                   fmt="ko", ms=4, label="measured")
    for law, b in out["best_by_law"].items():
        ax[0].plot(Vs, [b["rate_by_V"][V] for V in Vs], "s-", ms=3, label=f"{law} (AIC {b['aic']:.1f})")
    ax[0].set(xlabel="applied voltage (V)", ylabel="jumps per s", title="jump rate, best of each law")
    ax[0].legend(fontsize=8)
    ax[1].plot(Vs, [data_rows[V]["jumps"]["sd_rel"] for V in Vs], "ko", ms=4, label="measured")
    for law, b in out["best_by_law"].items():
        ax[1].plot(Vs, [b["sd_by_V"][V] for V in Vs], "s-", ms=3, label=law)
    ax[1].set(yscale="log", xlabel="applied voltage (V)", ylabel=r"Gaussian sd of $\Delta G/G$", title="noise")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"N11 stage 2: chosen law {out['chosen_law']} ({out['reason']})")
    fig.tight_layout()
    setup.fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(setup.fig / "fig35_n11_stage2.png", dpi=150)
    plt.close(fig)


def wait_for(files, poll=60.0, timeout=12 * 3600.0):
    """Block until every file exists (base states being computed by another node)."""
    t0 = time.time()
    while not all(f.exists() for f in files):
        if time.time() - t0 > timeout:
            raise SystemExit(f"stage2 run: base states still missing after {timeout / 3600:g} h")
        time.sleep(poll)


def stage2_cli(a, setup, data):
    if a.phase == "plan":
        stage2_plan(setup, data, json.loads(_report_json(setup).read_text()))
    elif a.phase == "run":
        plan = json.loads(_plan_file(setup).read_text())
        base, noise = stage2_runs(plan, setup, refine=a.refine)
        run_many(base, a.procs, setup)             # base states first: noise runs start from them
        wait_for([raw_file(r, setup) for r in base])
        run_many(noise, a.procs, setup)
    elif a.phase == "refine":
        print(stage2_refine(setup))
    elif a.phase == "report":
        stage2_report(setup, data)
    elif a.phase == "clean-markers":
        n = 0
        for m in setup.raw.glob("*.running"):
            m.unlink(missing_ok=True)
            n += 1
        print(f"removed {n} .running markers (only when no job is running)")
    else:
        raise SystemExit("stage2 phase: plan | run [--refine] | refine | report")


if __name__ == "__main__":
    main()
