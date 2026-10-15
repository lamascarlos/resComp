"""N11 losses and decision rules (docs/PLAN_N11_CALIBRATION.md sections 6, 7, 9; choices in section 12).

Stage 1: sum of squared normalised log errors, (log s_sim - log s_data)^2 / delta^2 with delta >= 0.05, over the onset,
the steady-state shape G / G_sat above the measured onset, theta(V) (smoothed targets, section 12) and the relaxation time.
Stage 2: Poisson deviance of the jump counts per voltage + squared normalised log-sd errors + 10 x the KS distance of the
pooled |jump sizes|, all on the -2 log L scale; AIC = loss + 2 k.
"""
import numpy as np
from scipy.stats import ks_2samp

DELTA_FLOOR = 0.05
THETA_WINDOW = 0.3          # V   neighbourhood of the smoothed theta target
KS_WEIGHT = 10.0
K_PARAMS = {"constant": 2, "power": 3, "thermal": 3}
ONSET_TOL, SHAPE_RMS_MAX, THETA_FACTOR, THETA_SHARE = 0.10, 0.2, 2.0, 0.75


def theta_targets(theta_rows, window=THETA_WINDOW):
    """Resolved measured theta, smoothed: median of the resolved values within +-window V; delta = max(bootstrap log-sd,
    log-sd of those neighbours, floor)."""
    ok = [r for r in theta_rows if not r["unresolved"] and r["theta_per_s"] > 0]
    V = np.array([r["V"] for r in ok])
    th = np.array([r["theta_per_s"] for r in ok])
    out = []
    for r in ok:
        m = np.abs(V - r["V"]) <= window + 1e-9
        spread = float(np.std(np.log(th[m]))) if m.sum() > 1 else 0.0
        boot = r["log_sd"] if np.isfinite(r.get("log_sd", np.nan)) else 0.0
        out.append({"V": r["V"], "theta": float(np.median(th[m])), "delta": max(boot, spread, DELTA_FLOOR)})
    return out


def relax_target(relaxation):
    """One-exponential relaxation time; delta = half the log ratio to the fast time of the two-exponential fit."""
    tau = relaxation["one"]["tau_s"]
    two = relaxation.get("two")
    d = np.log(tau / min(two["tau1_s"], two["tau2_s"])) / 2 if two else 0.0
    return float(tau), float(max(abs(d), DELTA_FLOOR))


def _sq(sim, data, delta):
    return float((np.log(sim) - np.log(data)) ** 2 / max(delta, DELTA_FLOOR) ** 2)


def _data_shape_at(V, shape):
    Vd = np.array([r["V"] for r in shape])
    if V < Vd.min() - 1e-9 or V > Vd.max() + 1e-9:
        return None
    lr = np.interp(V, Vd, np.log([r["ratio"] for r in shape]))
    dl = np.interp(V, Vd, [r["log_delta"] for r in shape])
    return float(np.exp(lr)), float(dl)


def _theta_pairs(sim, data):
    tg = theta_targets(data["theta"])
    pairs = []
    for r in sim["theta"]:
        near = [t for t in tg if abs(t["V"] - r["V"]) <= 0.15 + 1e-9]
        if near:
            t = min(near, key=lambda t: abs(t["V"] - r["V"]))
            pairs.append((r, t))
    return pairs


def stage1_terms(sim, data):
    if sim.get("onset_V") is None:
        return {"onset": np.inf, "shape": [], "theta": [], "relax": None, "loss": np.inf}
    onset = _sq(sim["onset_V"], data["onset_V"], 0.0)
    shape = []
    for r in sim["shape"]:
        d = _data_shape_at(r["V"], data["shape"])
        if d is not None and r["ratio"] > 0:
            shape.append(_sq(r["ratio"], d[0], d[1]))
    theta = [_sq(r["theta_per_s"], t["theta"], t["delta"]) if r["theta_per_s"] > 0 else 25.0
             for r, t in _theta_pairs(sim, data)]
    relax = None
    if sim.get("relax_tau_s"):
        tau, d = relax_target(data["relaxation"])
        relax = _sq(sim["relax_tau_s"], tau, d)
    loss = onset + sum(shape) + sum(theta) + (relax or 0.0)
    return {"onset": onset, "shape": shape, "theta": theta, "relax": relax, "loss": float(loss)}


def calibration_verdict(sim, data):
    """Spec section 9 rule 1."""
    if sim.get("onset_V") is None:
        return {"onset_ok": False, "shape_rms": np.inf, "shape_ok": False, "theta_share": 0.0, "theta_ok": False,
                "pass": False}
    onset_ok = abs(sim["onset_V"] / data["onset_V"] - 1) <= ONSET_TOL
    errs = [np.log(r["ratio"]) - np.log(d[0]) for r in sim["shape"]
            if r["ratio"] > 0 and (d := _data_shape_at(r["V"], data["shape"])) is not None]
    rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else np.inf
    pairs = _theta_pairs(sim, data)
    within = [r["theta_per_s"] > 0 and abs(np.log(r["theta_per_s"] / t["theta"])) <= np.log(THETA_FACTOR)
              for r, t in pairs]
    share = float(np.mean(within)) if within else 0.0
    out = {"onset_ok": bool(onset_ok), "shape_rms": rms, "shape_ok": bool(rms < SHAPE_RMS_MAX), "theta_share": share,
           "theta_ok": bool(share >= THETA_SHARE)}
    out["pass"] = out["onset_ok"] and out["shape_ok"] and out["theta_ok"]
    return out


def poisson_deviance(n, mu):
    """2 [n log(n / mu) - (n - mu)], mean floored at 0.5."""
    mu = max(float(mu), 0.5)
    return float(2 * ((n * np.log(n / mu) if n > 0 else 0.0) - (n - mu)))


def stage2_terms(sim_by_V, data_rows, data_sizes):
    dev, sd = 0.0, 0.0
    for V, s in sim_by_V.items():
        j = data_rows[V]["jumps"]
        dev += poisson_deviance(j["events"], s["events"] * j["duration_s"] / s["duration_s"])
        sd += _sq(s["sd_rel"], j["sd_rel"], data_rows[V].get("sd_log_delta") or DELTA_FLOOR)
    sim_sizes = np.concatenate([np.asarray(s["sizes"], dtype=float) for s in sim_by_V.values()])
    ks = float(ks_2samp(np.abs(sim_sizes), np.abs(np.asarray(data_sizes, dtype=float))).statistic) if len(sim_sizes) else 1.0
    return {"deviance": float(dev), "sd": float(sd), "ks": ks, "loss": float(dev + sd + KS_WEIGHT * ks)}


def aic(loss, law):
    return float(loss + 2 * K_PARAMS[law])


def lambda_misses(rate_by_V, data_rows):
    """Voltages where the simulated jump rate lies outside the measured 95 % interval."""
    out = set()
    for V, rate in rate_by_V.items():
        lo, hi = data_rows[V]["jumps"]["rate_ci_per_s"]
        if not lo <= rate <= hi:
            out.add(V)
    return out


def choose_law(best_by_law, data_rows):
    """Spec section 7: a state-dependent law replaces the constant one only if its AIC is lower by more than 4 AND its
    rates lie inside the data intervals at every voltage where the constant law's do not (and there is at least one)."""
    c = best_by_law["constant"]
    miss_c = lambda_misses(c["rate_by_V"], data_rows)
    ok = []
    for law in ("power", "thermal"):
        s = best_by_law.get(law)
        if s is None:
            continue
        if c["aic"] - s["aic"] > 4 and miss_c and not (miss_c & lambda_misses(s["rate_by_V"], data_rows)):
            ok.append((s["aic"], law))
    if ok:
        law = min(ok)[1]
        return law, f"{law}: AIC {c['aic'] - best_by_law[law]['aic']:.1f} lower, covers {sorted(miss_c)}"
    reason = "constant misses no voltage" if not miss_c else "no state-dependent law meets both conditions"
    return "constant", reason


def r0_interval(rows):
    """Range of r0 with AIC within 4 of the best (profile over the law's runs)."""
    best = min(r["aic"] for r in rows)
    r0 = [r["r0"] for r in rows if r["aic"] <= best + 4]
    return float(min(r0)), float(max(r0))
