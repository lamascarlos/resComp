"""Go/no-go criteria of the decisive plan (docs/NEXT_STEPS.md, review 2026-09-27).

The operating point (hold, voltage window) is chosen on VALIDATION scores only (review P0.2), and the criteria are
applied to the TEST scores of that single point:

  G0 echo-state : state increment consistency >= 0.8 (mean over networks; readout channels actually used)
  G1 computes   : MC >= 1 with >= 2 lags above the noise floor, degree-2 capacity >= 1 and NARMA-10 NMSE <= 0.70,
                  each with a margin of 2 standard deviations over the runs (3 networks x 2 replicates)
  G2 useful     : MC + degree-2 capacity >= 50 % of an ESN with as many units as readout channels, and sine/square
                  accuracy >= 0.95

A readout "passes" a level when all its criteria hold; G1 and G2 presuppose G0.
"""
from collections import defaultdict

import numpy as np

G0_MIN = 0.8
G1_MC, G1_IPC2, G1_NARMA, G1_LAGS = 1.0, 1.0, 0.70, 2
G2_FRACTION, G2_ACCURACY = 0.5, 0.95
CONFIG_KEYS = ("hold", "v_center")


def _cfg(r):
    return tuple(r[k] for k in CONFIG_KEYS)


def select_operating_point(runs: list[dict], obs: str) -> tuple:
    """Configuration with the largest mean validation capacity (val_MC + val_IPC2) over the i.i.d. runs."""
    score = defaultdict(list)
    for r in runs:
        if r["stream"] == "iid":
            s = r["scores"][obs]
            score[_cfg(r)].append(s["val_MC"] + s["val_IPC2"])
    if not score:
        raise ValueError("no i.i.d. runs to select from")
    return max(score, key=lambda k: np.mean(score[k]))


def _stats(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}


def evaluate(runs: list[dict], consistency: list[dict], obs: str, n_channels: int, esn_totals: dict,
             floor: float) -> dict:
    """Apply G0-G2 to one readout (`obs`) of one experiment group (runs / consistency already filtered to it).

    n_channels: number of readout features of `obs` (sets the ESN size of G2); esn_totals: {n_units: MC + IPC2}
    of the ESN baselines on streams of the same length (the closest size >= n_channels is used, else the largest);
    floor: noise floor of the capacities (streams of the same length)."""
    cfg = select_operating_point(runs, obs)
    iid = [r["scores"][obs] for r in runs if r["stream"] == "iid" and _cfg(r) == cfg]
    ss = [r["scores"][obs]["accuracy"] for r in runs if r["stream"] == "sinesquare" and _cfg(r) == cfg]
    cons = [c["consistency"][obs + "_state_inc"] for c in consistency if _cfg(c) == cfg]
    mc, ipc, narma = (_stats([s[k] for s in iid]) for k in ("MC", "IPC2", "narma10_nmse"))
    acc = _stats(ss) if ss else None
    g0v = _stats(cons) if cons else None
    sizes = sorted(esn_totals)
    esn_n = next((n for n in sizes if n >= n_channels), sizes[-1])
    total = _stats([s["MC"] + s["IPC2"] for s in iid])
    lags = [int(np.sum(np.asarray(s["memory_function"][1:]) > floor)) for s in iid]
    g0 = bool(g0v and g0v["mean"] >= G0_MIN)
    g1 = bool(g0 and mc["mean"] - 2 * mc["sd"] >= G1_MC and ipc["mean"] - 2 * ipc["sd"] >= G1_IPC2
              and narma["mean"] + 2 * narma["sd"] <= G1_NARMA and min(lags) >= G1_LAGS)
    g2 = bool(g1 and total["mean"] >= G2_FRACTION * esn_totals[esn_n] and acc and acc["mean"] >= G2_ACCURACY)
    return {"operating_point": dict(zip(CONFIG_KEYS, cfg)), "selected_on": "validation (val_MC + val_IPC2)",
            "consistency_inc": g0v, "MC": mc, "IPC2": ipc, "narma10_nmse": narma, "accuracy": acc,
            "total_capacity": total, "lags_above_floor": lags, "esn_units": esn_n, "esn_total": esn_totals[esn_n],
            "G0": g0, "G1": g1, "G2": g2}


def drift(on_fraction) -> float:
    """Relative change of the ON (closed) fraction between the first and the last 20 % of the scored stream
    (review P0.1: runs with |drift| > 2 % are not stationary)."""
    on = np.asarray(on_fraction, dtype=float)
    n5 = max(1, len(on) // 5)
    ref = max(float(on.mean()), 1e-12)
    return float((on[-n5:].mean() - on[:n5].mean()) / ref)
