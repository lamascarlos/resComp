"""Monte-Carlo calibration of the FDTest decision rule on independent junctions with a known answer.

  python -m scripts.fdtest_interval_calibration [--reps 40] [--Ms 120,240,480] [--procs 3]
                                                [--out docs/results/interval_calibration.md]

For every scenario, cycle count M and replica it simulates the four-stage protocol (relaxation, stage II with the perturbation,
stage III relaxation after removal, stage IV stationary) for K iid two-state junctions, builds ONE class with all valid junctions
(binary_stage_stats -> binary_curves -> class_delta2 / bootstrap over cycles / split-half floor) over a FIXED window (lags 0..n_lags,
no data-dependent cutoff) and applies five decision rules:

  percentile, reverse, sqrt_reverse   violation_summary(method=...): the interval's lower end must exceed the floor
  studentized                         bootstrap-t with a nested bootstrap (n_boot outer x n_inner inner resamples of the cycles)
  plain3x                             delta2_hat > 3 * floor

Scenarios (S1 and S2 use the RESET-side perturbation of tests/test_fdtest.py: V1 = V0, I1 = 3.3e-3 against I0 = 2.5e-3):
  S0  no perturbation (p_eps = p_0 exactly, Z is built from noise): the false-positive regime of a network whose junctions do not respond
  S1  Markov junctions with a real response: true delta2 = 0
  S2  fixed-lifetime junctions (L = 8, hidden age): true delta2 from benchmarks.renewal_fdtest over the same lags

Per (scenario, M, method) the table reports the excludes-floor rate (false positives for S0 and S1, power for S2), the coverage of the true
delta2 by the interval (S0: the fraction of intervals containing 0), the median interval width and the median delta2_hat / floor.
Seeds are deterministic per (scenario, M, replica), so the numbers do not depend on --procs. The simulator is a copy of the minimal
Monte-Carlo of tests/conftest.py (scripts do not import from tests/). The outer bootstrap draws are shared by all interval methods.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                     # one BLAS thread per worker: the machine is shared and the arrays are small

import argparse
import dataclasses
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from agnw_sim import fdtest as ft
from agnw_sim.benchmarks import renewal_fdtest
from agnw_sim.dynamics import SimParams, p_set, p_reset, update_dynamics, update_dynamics_lifetime

SEED = 20260920
PRESET = SimParams.from_step_probs(0.4, 0.01, dt=1e-3, alpha=1.0, V_th=0.1, I_unit=1e-3)      # as tests/conftest.py
V0, I0 = 0.30, 2.5e-3
LIFETIME = 8
DEFAULTS = dict(K=40, t_relax=60, t2=300, t4=300, n_lags=30, n_boot=100, n_inner=15, n_splits=10, stud_max_M=10 ** 9)
SCENARIOS = {                                          # name -> perturbed drive of stages I-II and the junction model
    "S0": dict(index=0, V1=V0, I1=I0, lifetime=False, text="no perturbation (p_eps = p_0), Z from noise"),
    "S1": dict(index=1, V1=V0, I1=3.3e-3, lifetime=False, text="Markov, real response, true delta2 = 0"),
    "S2": dict(index=2, V1=V0, I1=3.3e-3, lifetime=True, text=f"fixed lifetime L = {LIFETIME}, exact delta2"),
}
METHODS = ("percentile", "reverse", "sqrt_reverse", "studentized", "plain3x")
INTERVAL = ("percentile", "reverse", "sqrt_reverse", "studentized")


# ----------------------------------------------------------------------------- simulator (copy of tests/conftest._mc_cycles)
def mc_cycles(params, V0, I0, V1, I1, n_cycles, K, T_relax, T2, n_lags, T4, seed, lifetime=False):
    """Independent-junction Monte Carlo of the four-stage protocol with the package dynamics. Perturbed drive (V1, I1) in
    stages I-II, (V0, I0) in III-IV. Returns g2, g3, g4 with shape (cycle, time, K); lag 0 of g3 is the state at the removal."""
    rng = np.random.default_rng(seed)
    M = n_cycles * K
    s0 = float(p_set(np.array([V0]), params)[0])
    r0 = float(p_reset(np.array([I0]), params)[0])
    G = np.where(rng.random(M) < s0 / (s0 + r0), params.G_on, params.G_off)
    age = np.zeros(M, dtype=int)

    def step(G, age, V, I):
        Va, Ia = np.full(M, V), np.full(M, I)
        if lifetime:
            return update_dynamics_lifetime(G, age, Va, Ia, params, rng)
        return update_dynamics(G, Va, Ia, params, rng), age

    on = lambda G: (G >= params.G_on).astype(np.uint8)
    for _ in range(T_relax):
        G, age = step(G, age, V1, I1)
    g2 = np.empty((T2, M), np.uint8)
    for t in range(T2):
        g2[t] = on(G)
        G, age = step(G, age, V1, I1)
    g3 = np.empty((n_lags, M), np.uint8)
    for lag in range(n_lags):
        g3[lag] = on(G)
        G, age = step(G, age, V0, I0)
    for _ in range(T_relax):
        G, age = step(G, age, V0, I0)
    g4 = np.empty((T4, M), np.uint8)
    for t in range(T4):
        g4[t] = on(G)
        G, age = step(G, age, V0, I0)
    shape = lambda g: g.reshape(g.shape[0], n_cycles, K).transpose(1, 0, 2)
    return shape(g2), shape(g3), shape(g4)


def _rates(V, I):
    return float(p_set(np.array([V]), PRESET)[0]), float(p_reset(np.array([I]), PRESET)[0])


def true_delta2(name, cfg):
    """True delta2 over the lags 0..n_lags: None for S0 (undefined: no response), 0 for the Markov junction, the exact
    renewal value for S2."""
    if name == "S0":
        return None
    if name == "S1":
        return 0.0
    s0, r0 = _rates(V0, I0)
    s1, r1 = _rates(SCENARIOS[name]["V1"], SCENARIOS[name]["I1"])
    return float(renewal_fdtest(s0, r0, s1, r1, LIFETIME, cfg["n_lags"])["delta2"])


def replica_seeds(scenario_index, M, rep, base=SEED):
    """Four independent seeds (simulation, bootstrap, floor splits, inner bootstrap) of one (scenario, M, replica)."""
    return tuple(int(x) for x in np.random.SeedSequence([base, scenario_index, M, rep]).generate_state(4))


def simulate_stats(scenario, M, cfg, seed):
    """Stage statistics (binary_stage_stats) of M cycles of K iid junctions and the (1, K) mask of the valid junctions."""
    params = dataclasses.replace(PRESET, lifetime_steps=LIFETIME) if scenario["lifetime"] else PRESET
    g2, g3, g4 = mc_cycles(params, V0, I0, scenario["V1"], scenario["I1"], M, cfg["K"], cfg["t_relax"], cfg["t2"],
                           cfg["n_lags"] + 1, cfg["t4"], seed, scenario["lifetime"])
    stats = ft.binary_stage_stats(g2, g3, g4, cfg["n_lags"])
    return stats, ft.binary_curves(stats)["valid"][None, :]


def nested_bootstrap_se(stats, masks, W, n_inner, cutoff, rng):
    """Standard error of delta2 for every outer bootstrap resample (rows of W: cycle weights): `n_inner` resamples OF the
    resampled cycles (multinomial with the outer weights as probabilities), delta2 over the fixed window, std with ddof = 1."""
    n = W.shape[1]
    se = np.full(len(W), np.nan)
    for b, w in enumerate(W):
        inner = rng.multinomial(n, w / w.sum(), size=n_inner) / n
        sz, sc, cnt = ft.bootstrap_class_sums(stats, inner, masks)
        d = np.array([ft.delta2_from_sums(sz[i], sc[i], cnt[i], cutoff=cutoff)[0] for i in range(n_inner)])
        d = d[np.isfinite(d)]
        if d.size > 1:
            se[b] = d.std(ddof=1)
    return se


def run_replica(task):
    """One (scenario, M, replica): the record with delta2_hat, floor, the true value and every method's decision / interval."""
    name, M, rep, cfg = task
    scen = SCENARIOS[name]
    s_sim, s_boot, s_floor, s_inner = replica_seeds(scen["index"], M, rep)
    stats, masks = simulate_stats(scen, M, cfg, s_sim)
    cut = cfg["n_lags"]                                           # fixed window: every lag
    cur = ft.binary_curves(stats)
    d2 = float(ft.class_delta2(cur["z_mean"], cur["c_zz"], masks, cutoff=cut)[0][0])
    floor = float(ft.split_half_floor(stats, masks, np.random.default_rng(s_floor), n_splits=cfg["n_splits"], cutoff=cut)[0])
    W = ft.bootstrap_weights(M, cfg["n_boot"], np.random.default_rng(s_boot))
    sz, sc, cnt = ft.bootstrap_class_sums(stats, W, masks)
    boot = np.array([ft.delta2_from_sums(sz[b], sc[b], cnt[b], cutoff=cut)[0] for b in range(cfg["n_boot"])])
    methods = {}
    for m in ("percentile", "reverse", "sqrt_reverse"):
        v = ft.violation_summary(d2, boot, floor, method=m)
        methods[m] = {"excl": v["excludes_floor"], "lo": v["ci95"][0], "hi": v["ci95"][1]}
    if M <= cfg["stud_max_M"]:
        se = nested_bootstrap_se(stats, masks, W, cfg["n_inner"], cut, np.random.default_rng(s_inner))
        lo, hi = ft.studentized_interval(d2, boot, se)
        methods["studentized"] = {"excl": bool(np.isfinite(lo) and np.isfinite(floor) and lo > floor), "lo": lo, "hi": hi}
    methods["plain3x"] = {"excl": bool(np.isfinite(d2) and np.isfinite(floor) and d2 > 3 * floor)}
    return {"scenario": name, "M": M, "rep": rep, "d2": d2, "floor": floor, "true": true_delta2(name, cfg),
            "n_valid": int(masks.sum()), "methods": methods}


# ----------------------------------------------------------------------------- aggregation and output
def aggregate(records):
    """One cell per (scenario, M, method): n, rate (excludes-floor fraction) and its binomial SE, n_ci and coverage of the true
    delta2 by the intervals that are finite (S0: containment of 0), median width, median delta2/floor, median delta2 and floor."""
    cells = []
    keys = sorted({(r["scenario"], r["M"]) for r in records}, key=lambda k: (SCENARIOS[k[0]]["index"], k[1]))
    for scen, M in keys:
        rs = [r for r in records if r["scenario"] == scen and r["M"] == M]
        d2, fl = np.array([r["d2"] for r in rs], dtype=float), np.array([r["floor"] for r in rs], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = d2 / fl
        for m in METHODS:
            ms = [r["methods"][m] for r in rs if m in r["methods"]]
            if not ms:
                continue
            excl = np.array([x["excl"] for x in ms], dtype=float)
            lo = np.array([x.get("lo", np.nan) for x in ms], dtype=float)
            hi = np.array([x.get("hi", np.nan) for x in ms], dtype=float)
            true = 0.0 if rs[0]["true"] is None else rs[0]["true"]
            fin = np.isfinite(lo) & np.isfinite(hi)
            n = len(ms)
            rate = float(excl.mean())
            cells.append({"scenario": scen, "M": M, "method": m, "n": n, "rate": rate, "rate_se": float(np.sqrt(rate * (1 - rate) / n)),
                          "n_ci": int(fin.sum()),
                          "coverage": float(np.mean((lo[fin] <= true) & (true <= hi[fin]))) if fin.any() else float("nan"),
                          "median_width": float(np.median(hi[fin] - lo[fin])) if fin.any() else float("nan"),
                          "median_ratio": float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else float("nan"),
                          "median_d2": float(np.nanmedian(d2)) if np.isfinite(d2).any() else float("nan"),
                          "median_floor": float(np.nanmedian(fl)) if np.isfinite(fl).any() else float("nan")})
    return cells


def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.generic):
        return _sanitize(o.item())
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


def _f(v, nd=3):
    return "-" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def markdown(cells, cfg, reps, Ms, truths, runtime):
    lines = ["# FDTest decision-rule calibration (Monte Carlo, independent junctions)", "",
             f"- reps = {reps} replicas per (scenario, M); M in {list(Ms)} cycles; K = {cfg['K']} iid junctions, one class, fixed window (lags 0..{cfg['n_lags']})",
             f"- protocol: T_relax {cfg['t_relax']}, T2 {cfg['t2']}, n_lags {cfg['n_lags']}, T4 {cfg['t4']}; bootstrap n_boot = {cfg['n_boot']}, "
             f"studentized inner = {cfg['n_inner']}, floor = mean of {cfg['n_splits']} split-halves",
             f"- seeds: numpy SeedSequence([{SEED}, scenario index, M, replica]) -> (simulation, bootstrap, floor, inner); independent of --procs",
             f"- runtime: {runtime:.0f} s wall", "",
             "Scenarios: " + "; ".join(f"**{k}** {v['text']}" + (f" (true delta2 = {_f(truths[k], 4)})" if truths[k] is not None else "")
                                       for k, v in SCENARIOS.items()) + ".", "",
             "`rate` = fraction of replicas whose interval lower end exceeds the floor (S0, S1: false-positive rate; S2: power); "
             "plain3x: delta2_hat > 3 floor. `coverage` = fraction of finite intervals containing the true delta2 (S0: containing 0). "
             "`width` and `ratio` are medians (interval width; delta2_hat / floor).", "",
             "| scenario | M | method | rate (+- SE) | coverage | median width | median delta2/floor | median delta2 | median floor |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        lines.append(f"| {c['scenario']} | {c['M']} | {c['method']} | {_f(c['rate'], 2)} +- {_f(c['rate_se'], 2)} | {_f(c['coverage'], 2)} "
                     f"| {_f(c['median_width'])} | {_f(c['median_ratio'], 2)} | {_f(c['median_d2'])} | {_f(c['median_floor'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reps", type=int, default=40); p.add_argument("--Ms", default="120,240,480")
    p.add_argument("--procs", type=int, default=3); p.add_argument("--out", default="docs/results/interval_calibration.md")
    p.add_argument("--K", type=int, default=DEFAULTS["K"]); p.add_argument("--t-relax", type=int, default=DEFAULTS["t_relax"])
    p.add_argument("--t2", type=int, default=DEFAULTS["t2"]); p.add_argument("--t4", type=int, default=DEFAULTS["t4"])
    p.add_argument("--n-lags", type=int, default=DEFAULTS["n_lags"])
    p.add_argument("--n-boot", type=int, default=DEFAULTS["n_boot"]); p.add_argument("--n-inner", type=int, default=DEFAULTS["n_inner"])
    p.add_argument("--n-splits", type=int, default=DEFAULTS["n_splits"])
    p.add_argument("--stud-max-M", type=int, default=DEFAULTS["stud_max_M"], help="skip the (slow) studentized interval above this M")
    a = p.parse_args(argv)
    Ms = [int(x) for x in a.Ms.split(",") if x]
    if a.reps < 1 or not Ms or min(Ms) < 4 or a.procs < 1 or a.n_lags >= a.t4:
        sys.exit("--reps >= 1, --Ms >= 4 cycles, --procs >= 1 and --n-lags < --t4 are required")
    cfg = dict(K=a.K, t_relax=a.t_relax, t2=a.t2, t4=a.t4, n_lags=a.n_lags, n_boot=a.n_boot, n_inner=a.n_inner,
               n_splits=a.n_splits, stud_max_M=a.stud_max_M)
    tasks = [(name, M, rep, cfg) for name in SCENARIOS for M in Ms for rep in range(a.reps)]
    t0 = time.time()
    records = []
    if a.procs == 1:
        for i, t in enumerate(tasks, 1):
            records.append(run_replica(t))
            print(f"\r{i}/{len(tasks)} replicas", end="", file=sys.stderr, flush=True)
    else:
        with Pool(a.procs) as pool:
            for i, rec in enumerate(pool.imap_unordered(run_replica, tasks, chunksize=1), 1):
                records.append(rec)
                print(f"\r{i}/{len(tasks)} replicas", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    records.sort(key=lambda r: (SCENARIOS[r["scenario"]]["index"], r["M"], r["rep"]))
    runtime = time.time() - t0
    truths = {k: true_delta2(k, cfg) for k in SCENARIOS}
    cells = aggregate(records)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(markdown(cells, cfg, a.reps, Ms, truths, runtime))
    with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
        json.dump(_sanitize({"config": {**cfg, "reps": a.reps, "Ms": Ms, "seed": SEED}, "runtime_s": runtime, "true_delta2": truths,
                             "cells": cells, "replicas": records}), fh, indent=1, allow_nan=False)
    print(f"wrote {a.out} ({runtime:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
