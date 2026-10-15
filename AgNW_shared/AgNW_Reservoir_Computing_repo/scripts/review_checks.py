#!/usr/bin/env python3
"""Quantitative checks behind the 2026-09-27 project review (docs/REVIEW_2026-09-27.md).

A. Consistency without shared trends. The consistency metric (mean Pearson correlation of whole feature series
   across two replicates) also rewards a slow trend common to both replicates (e.g. both relaxing from the
   pre-drive), which is not an input-driven response. Recomputed on (i) first differences of the features and
   (ii) high-passed features (series minus a centred 51-sample moving average), for representative runs of each
   junction model. Also: the share of each feature's variance explained by a cubic time trend.
B. Input-explained variance: in-sample R^2 of each state feature regressed on the inputs u[t-k], k = 0..30 (plus a
   constant), averaged over features; chance level for pure noise is ~31/n.
C. Stationarity: ON (closed) fraction in the first vs the last 20 % of the scored stream.
D. The "relaxational memory cap": memory capacity of an ideal leaky-integrator bank as a function of the ridge
   floor (default grid 1e-6..1e4 vs 1e-14..1e4) and of additive feature noise (relative SNR).
E. Ridge-grid boundary hits for representative feature sets (share of memory targets whose validation-selected
   alpha is the smallest or largest grid value).
F. Noise floor for high-dimensional features (Gaussian features with the junction-state dimension).

    python scripts/review_checks.py [--procs 6]   ->  results/review_checks.json   (~10 min)
"""
import argparse
import json
import os
import sys

# one BLAS thread per process: the scripts parallelize over runs with processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import baselines as B  # noqa: E402
from agnw_rc import config as C  # noqa: E402
from agnw_rc.drive import input_stream, load_raw  # noqa: E402
from agnw_rc.evaluate import MEAS_RES, observables, state_observables  # noqa: E402
from agnw_rc.metrics import capacity, consistency  # noqa: E402
from agnw_rc.readout import ALPHAS, fit_predict  # noqa: E402
from agnw_rc.tasks import MAX_LAG, delayed, memory_targets, split_indices  # noqa: E402

RAW = ROOT / "results" / "raw"
CASES = {   # label: kwargs of RunSpec (iid stream, hold 30) at a representative operating point of each model
    "binary stochastic (E1, 1.25-2.25 V)": dict(v_center=1.75),
    "mean-field (M1, 5.0-6.0 V)": dict(v_center=5.5, model="meanfield"),
    "continuous filament (N4-1, 1.75-2.75 V)": dict(v_center=2.25, layout="two_bands", encoding="direct",
                                                    model="filament"),
    "binary stochastic, pads (N5a, 0-1 V)": dict(v_center=0.5, layout="pads", encoding="weights"),
}
WIDE_ALPHAS = np.logspace(-14, 4, 19)


def highpass(F, w=51):
    k = np.ones(w) / w
    pad = np.pad(F, ((w // 2, w // 2), (0, 0)), mode="edge")
    trend = np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(F.shape[1])], axis=1)
    return F - trend


def trend_share(F):
    """Mean over features of the variance fraction explained by a cubic polynomial in time."""
    t = np.linspace(-1, 1, len(F))
    X = np.stack([t ** p for p in range(4)], axis=1)
    keep = F.std(axis=0) > MEAS_RES
    if not keep.any():
        return float("nan")
    Fk = F[:, keep]
    coef, *_ = np.linalg.lstsq(X, Fk, rcond=None)
    res = Fk - X @ coef
    return float(np.mean(1 - res.var(axis=0) / Fk.var(axis=0)))


def input_explained(F, u):
    """Mean over features of the in-sample R^2 of F[t] on u[t-k], k = 0..MAX_LAG (+ constant)."""
    X = np.hstack([np.nan_to_num(np.stack([delayed(u, k) for k in range(MAX_LAG + 1)], axis=1)),
                   np.ones((len(u), 1))])[MAX_LAG:]
    Fk = F[MAX_LAG:]
    keep = Fk.std(axis=0) > MEAS_RES
    if not keep.any():
        return float("nan")
    Fk = Fk[:, keep]
    coef, *_ = np.linalg.lstsq(X, Fk, rcond=None)
    res = Fk - X @ coef
    return float(np.mean(1 - res.var(axis=0) / Fk.var(axis=0)))


def alpha_hits(F, u, alphas=ALPHAS):
    mem = memory_targets(u)
    tr, va, te = split_indices(len(u))
    _, best, _ = fit_predict(F, mem, tr, va, te, alphas=alphas, min_std=MEAS_RES)
    return {"at_min": float(np.mean(best == alphas[0])), "at_max": float(np.mean(best == alphas[-1]))}


def check_case(item):
    label, kw = item
    out = {"per_seed": {}}
    for seed in C.E1_SEEDS:
        specs = [C.RunSpec("iid", 30, network_seed=seed, replicate=r, **kw) for r in (0, 1)]
        if not all((RAW / f"{s.name}.npz").exists() for s in specs):
            continue
        raws = [load_raw(s, RAW) for s in specs]
        S = [state_observables(r) for r in raws]
        u = raws[0]["u"]
        row = {}
        for obs in ("current", "probes", "junctions"):
            a, b = S[0][obs][MAX_LAG:], S[1][obs][MAX_LAG:]
            row[obs] = {
                "consistency_raw": consistency(a, b, min_std=MEAS_RES),
                "consistency_increments": consistency(np.diff(a, axis=0), np.diff(b, axis=0), min_std=MEAS_RES),
                "consistency_highpass": consistency(highpass(a), highpass(b), min_std=MEAS_RES),
                "trend_share": trend_share(a),
                "input_explained_R2": input_explained(S[0][obs], u),
            }
        on = raws[0]["on_fraction"]
        n5 = len(on) // 5
        row["on_fraction_first20"] = float(on[:n5].mean())
        row["on_fraction_last20"] = float(on[-n5:].mean())
        if seed == 1:
            row["alpha_hits_junctions"] = alpha_hits(observables(raws[0])["junctions"], u)
        out["per_seed"][str(seed)] = row
    return label, out


def leaky_bank(u, taus):
    a = np.exp(-1.0 / np.asarray(taus, dtype=float))
    x = np.zeros(len(a))
    X = np.empty((len(u), len(a)))
    for t, ut in enumerate(u):
        x = a * x + (1 - a) * ut
        X[t] = x
    return X


def bank_mc(args):
    K, snr, alphas_name = args
    u, _ = input_stream(C.RunSpec("iid", 10, 1.0, 1))
    X = leaky_bank(u, np.logspace(0, 2, K))
    if snr:
        X = X + np.random.default_rng(K).standard_normal(X.shape) * X.std(axis=0) / np.sqrt(snr)
    alphas = WIDE_ALPHAS if alphas_name == "wide" else ALPHAS
    mem = memory_targets(u)
    tr, va, te = split_indices(len(u))
    pred, best, _ = fit_predict(X, mem, tr, va, te, alphas=alphas, min_std=1e-14)
    c = capacity(pred, mem[te])
    return (K, snr, alphas_name), {"MC": float(c[1:].sum()), "lag0": float(c[0]),
                                    "alpha_at_min": float(np.mean(best == alphas[0]))}


def noise_floor_highdim(dim):
    u, _ = input_stream(C.RunSpec("iid", 10, 1.0, 1))
    mem = memory_targets(u)
    tr, va, te = split_indices(len(u))
    worst = []
    for s in (0, 1):
        pred, _, _ = fit_predict(B.noise_features(len(u), dim, s), mem, tr, va, te, min_std=MEAS_RES)
        worst.append(float(capacity(pred, mem[te]).max()))
    return dim, max(worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    res = {}
    with ProcessPoolExecutor(max_workers=a.procs) as pool:
        res["cases"] = dict(pool.map(check_case, CASES.items()))
        bank_jobs = [(K, snr, al) for K in (32, 200) for snr in (0, 1e8, 1e4, 1e2) for al in ("default", "wide")]
        res["leaky_bank"] = {f"K={K}, SNR={snr:g}, alphas={al}": v for (K, snr, al), v in
                             pool.map(bank_mc, bank_jobs)}
        res["noise_floor_highdim"] = {str(d): v for d, v in pool.map(noise_floor_highdim, (32, 1000, 6000))}
    (ROOT / "results" / "review_checks.json").write_text(json.dumps(res, indent=1))
    for label, r in res["cases"].items():
        print(label)
        for seed, row in r["per_seed"].items():
            print(f"  seed {seed}: " + " | ".join(
                f"{o}: raw {row[o]['consistency_raw']:.2f} inc {row[o]['consistency_increments']:.2f} "
                f"hp {row[o]['consistency_highpass']:.2f} trend {row[o]['trend_share']:.2f} "
                f"inR2 {row[o]['input_explained_R2']:.3f}" for o in ("current", "junctions"))
                + f" | ON {row['on_fraction_first20']:.3f}->{row['on_fraction_last20']:.3f}"
                + (f" | alpha hits {row['alpha_hits_junctions']}" if "alpha_hits_junctions" in row else ""))
    for k, v in res["leaky_bank"].items():
        print(f"{k}: MC {v['MC']:.2f} (alpha at grid min for {v['alpha_at_min']:.0%} of lags)")
    print("noise floor (max held-out capacity of Gaussian features):", res["noise_floor_highdim"])
    print("saved results/review_checks.json")


if __name__ == "__main__":
    main()
