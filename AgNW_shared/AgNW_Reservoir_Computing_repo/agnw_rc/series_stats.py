"""N11 stage 0: statistics of conductance time series, applied identically to the measured records of Milano et al.
2025 (scripts/n11_data.py) and to simulated conductance sampled at the measured times (stage 1-2 fits).
docs/PLAN_N11_CALIBRATION.md section 5; the estimators are validated on synthetic series (tests/test_series_stats.py).

Jumps: relative change per sample r = Delta G / G on a common 0.6 s grid; Gaussian sd from a sigma-clipped estimate
(iterated on |r| < k sd, corrected for the truncation); a jump event = a run of consecutive same-sign samples with
|r| > k sd (k = 2.6, the measured Gaussian/jump boundary of Fig. 3, 2.55 sd); rate = (events - expected Gaussian
false positives N 2 Phi(-k)) / duration, with an exact Poisson interval.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2, norm

GRID_DT = 0.6        # s   common sampling grid (Figs. 3 and 5a; the inset is sampled at 0.1-0.6 s)
JUMP_K = 2.6         #     jump threshold in Gaussian sd
BLOCK_S = 10.0       # s   bootstrap block length
ONSET_FRAC = 1e-3    #     onset: G reaches ONSET_FRAC x the saturated conductance


def clean(t, G):
    """Drop NaN padding, sort by time, keep the first of repeated time stamps."""
    t, G = np.asarray(t, dtype=float), np.asarray(G, dtype=float)
    m = ~(np.isnan(t) | np.isnan(G))
    t, G = t[m], G[m]
    o = np.argsort(t, kind="stable")
    t, G = t[o], G[o]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], G[keep]


def to_grid(t, G, dt=GRID_DT):
    """The measured sample nearest to each point of a regular grid of step dt (no interpolation). Grid points with no
    sample within dt/2 are dropped, and no sample is used twice."""
    grid = np.arange(t[0], t[-1] + 1e-9, dt)
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    i = np.where(np.abs(t[i - 1] - grid) <= np.abs(t[i] - grid), i - 1, i)
    i = i[np.abs(t[i] - grid) <= dt / 2]
    i = i[np.r_[True, np.diff(i) > 0]]
    return t[i], G[i]


def interpolation_index(G):
    """Share of interior samples on the straight line through their neighbours (|second difference| < 1e-3 x |first
    difference|): near 1 for linearly interpolated records, near 0 for raw noisy samples."""
    d1, d2 = np.diff(G), np.diff(G, 2)
    return float(np.mean(np.abs(d2) <= 1e-3 * np.abs(d1[:-1])))


def clipped_sd(x, k=JUMP_K, iters=50):
    """Gaussian sd of x with outliers (jumps) removed: start from the MAD, iterate the sd of the samples within
    k sd of the median, corrected for the truncation of a Gaussian at +-k."""
    x = np.asarray(x, dtype=float) - np.median(x)
    c = np.sqrt(1 - 2 * k * norm.pdf(k) / (2 * norm.cdf(k) - 1))
    sd = 1.4826 * np.median(np.abs(x))
    for _ in range(iters):
        new = x[np.abs(x) < k * sd].std() / c
        if abs(new - sd) <= 1e-9 * sd:
            break
        sd = new
    return float(sd)


def poisson_ci(n, level=0.95):
    """Exact (Garwood) interval for a Poisson count n."""
    a = (1 - level) / 2
    lo = 0.0 if n == 0 else chi2.ppf(a, 2 * n) / 2
    return float(lo), float(chi2.ppf(1 - a, 2 * n + 2) / 2)


def jump_stats(t, G, k=JUMP_K):
    """Gaussian noise and jumps of a stationary record sampled at times t (s): see the module docstring. The
    returned sizes are the summed relative changes of each event."""
    t, G = np.asarray(t, dtype=float), np.asarray(G, dtype=float)
    if np.any(G <= 0):
        raise ValueError("jump_stats needs a conducting record (G > 0); records at the instrument floor are excluded")
    rel = np.diff(G) / G[:-1]
    rel = rel - np.median(rel)
    sd = clipped_sd(rel, k)
    above = np.abs(rel) > k * sd
    sgn = np.sign(rel) * above
    start = above & ~np.r_[False, (sgn[1:] == sgn[:-1]) & above[:-1]]
    run = np.cumsum(start)
    n = int(start.sum())
    sizes = np.bincount(run[above], weights=rel[above], minlength=n + 1)[1:] if n else np.zeros(0)
    T = float(t[-1] - t[0])
    fp = len(rel) * 2 * norm.sf(k)
    lo, hi = poisson_ci(n)
    return {"duration_s": T, "dt_s": float(np.median(np.diff(t))), "n_samples": int(len(rel)), "sd_rel": sd,
            "events": n, "false_expected": float(fp), "rate_per_s": (n - fp) / T,
            "rate_ci_per_s": [max(lo - fp, 0.0) / T, max(hi - fp, 0.0) / T],
            "up_share": float(np.mean(sizes > 0)) if n else float("nan"),
            "size_max": float(np.abs(sizes).max()) if n else 0.0,
            "share_above_1pct": float(np.mean(np.abs(sizes) > 0.01)) if n else 0.0, "sizes": sizes}


def block_bootstrap(x, stat, block, n_boot=200, rng=None):
    """Moving-block bootstrap of stat(x) with blocks of `block` samples: 95 % percentile interval and the sd of
    log stat (the delta of the stage-1/2 losses; nan if stat is not positive)."""
    rng = np.random.default_rng(0) if rng is None else rng
    x = np.asarray(x)
    n = len(x)
    b = max(1, min(int(block), n))
    nb = int(np.ceil(n / b))
    vals = np.array([stat(np.concatenate([x[s:s + b] for s in rng.integers(0, n - b + 1, nb)])[:n])
                     for _ in range(n_boot)])
    pos = vals[vals > 0]
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "log_sd": float(np.std(np.log(pos))) if len(pos) == len(vals) else float("nan")}


def synthetic_series(rng, t, G_det, sd_rel, rate, alpha, size_min):
    """Validation series sampled at times t: deterministic G_det x cumulative Poisson jumps (rate per s, relative
    sizes Pareto with exponent alpha above size_min, random sign) x white noise, so that Delta G / G per sample has
    Gaussian sd sd_rel between jumps."""
    t = np.asarray(t, dtype=float)
    dt = np.diff(t, prepend=t[0])
    hit = rng.poisson(rate * dt) > 0
    s = np.zeros(len(t))
    m = int(hit.sum())
    s[hit] = size_min * (1 - rng.random(m)) ** (-1 / (alpha - 1)) * rng.choice([-1.0, 1.0], m)
    return np.asarray(G_det, dtype=float) * np.cumprod(1 + s) * (1 + rng.normal(0, sd_rel / np.sqrt(2), len(t)))


def onset(V, G, G_ref, frac=ONSET_FRAC):
    """Lowest voltage at which G reaches frac x G_ref, interpolated in log G between neighbouring points (values at
    or below the floor count as frac x G_ref x 1e-6). None if never reached."""
    V, G = np.asarray(V, dtype=float), np.asarray(G, dtype=float)
    thr = frac * G_ref
    above = np.nonzero(G >= thr)[0]
    if not len(above):
        return None
    i = int(above[0])
    if i == 0:
        return float(V[0])
    g0 = max(G[i - 1], thr * 1e-6)
    return float(V[i - 1] + (V[i] - V[i - 1]) * np.log(thr / g0) / np.log(G[i] / g0))


def approach(t, G_inf, G_0, theta):
    """Exponential approach to a plateau: G_inf - (G_inf - G_0) exp(-theta t)."""
    return G_inf - (G_inf - G_0) * np.exp(-theta * t)


def _fit_approach(t, G, theta_max, p0):
    popt, _ = curve_fit(approach, t, G, p0=p0, bounds=([0, 0, 0], [np.inf, np.inf, theta_max]), maxfev=20000)
    return popt


def fit_theta(t, G, n_boot=200, rng=None):
    """Potentiation rate theta (1/s) of a transient (Fig. 2a): least-squares approach() fit, theta bounded by
    10 / (sampling step). Flagged `unresolved` at that bound, with R^2 < 0.5, or when no plateau is reached in the
    record (theta < 1 / record length, or G_inf > 3 x the largest measured G: a drift, not an approach). 95 % interval
    and log-sd from a
    moving-block bootstrap of the residuals (blocks of BLOCK_S)."""
    rng = np.random.default_rng(0) if rng is None else rng
    t, G = np.asarray(t, dtype=float), np.asarray(G, dtype=float)
    t = t - t[0]
    dt = float(np.median(np.diff(t)))
    theta_max = 10.0 / dt
    tail = G[-max(1, len(G) // 10):].mean()
    p0 = (max(tail, 1e-30), max(G[0], 0.0), min(5.0 / t[-1], theta_max / 2))
    popt = _fit_approach(t, G, theta_max, p0)
    fit = approach(t, *popt)
    res = G - fit
    r2 = float(1 - res.var() / G.var()) if G.var() > 0 else 0.0
    b = max(1, int(round(BLOCK_S / dt)))
    n = len(res)
    th = []
    for _ in range(n_boot):
        rb = np.concatenate([res[s:s + b] for s in rng.integers(0, max(1, n - b + 1), int(np.ceil(n / b)))])[:n]
        try:
            th.append(_fit_approach(t, fit + rb, theta_max, popt)[2])
        except RuntimeError:
            continue
    th = np.array(th) if th else np.array([popt[2]])
    return {"theta_per_s": float(popt[2]), "G_inf_S": float(popt[0]), "G_0_S": float(popt[1]), "r2": r2,
            "unresolved": bool(popt[2] >= 0.99 * theta_max or r2 < 0.5 or popt[2] < 1.0 / t[-1]
                               or popt[0] > 3 * G.max()),
            "lo": float(min(np.percentile(th, 2.5), popt[2])), "hi": float(max(np.percentile(th, 97.5), popt[2])),
            "log_sd": float(np.std(np.log(th[th > 0]))) if np.any(th > 0) else float("nan")}


def _aic(res, k):
    n = len(res)
    return float(n * np.log(max(np.mean(res ** 2), 1e-300)) + 2 * k)


def fit_relaxation(t, G):
    """Relaxation after a pulse (Fig. 1e): one vs two exponentials plus offset, Gaussian residuals,
    AIC = n log(RSS / n) + 2 k; `best` = two only if it converged and lowers the AIC by more than 4."""
    t, G = np.asarray(t, dtype=float), np.asarray(G, dtype=float)
    t = t - t[0]
    d = G[0] - G[-1]
    T = t[-1]

    def f1(t, c, a, tau):
        return c + a * np.exp(-t / tau)

    def f2(t, c, a1, t1, a2, t2):
        return c + a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2)

    p1, _ = curve_fit(f1, t, G, p0=(G[-1], d, T / 3), bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, 1e3 * T]),
                      maxfev=20000)
    out = {"one": {"tau_s": float(p1[2]), "aic": _aic(G - f1(t, *p1), 3)}, "two": None, "best": "one"}
    try:
        p2, _ = curve_fit(f2, t, G, p0=(G[-1], d / 2, T / 30, d / 2, T / 2),
                          bounds=([-np.inf, -np.inf, 1e-3, -np.inf, 1e-3], [np.inf, np.inf, 1e3 * T, np.inf, 1e3 * T]),
                          maxfev=20000)
        out["two"] = {"tau1_s": float(p2[2]), "tau2_s": float(p2[4]), "aic": _aic(G - f2(t, *p2), 5)}
        if out["two"]["aic"] < out["one"]["aic"] - 4:
            out["best"] = "two"
    except (RuntimeError, ValueError):
        pass
    return out


STATIONARY_TOL = 0.10    # a 500 s record is stationary if its mean is within 10 % of the steady-state point


def is_stationary(g_point, g_mean, tol=STATIONARY_TOL):
    """The record's mean agrees with the independently reported steady-state value (Fig. 2b point) within tol."""
    return bool(g_mean > 0 and abs(g_point / g_mean - 1) <= tol)


def loglog_fit(x, y):
    """Least-squares slope of log y against log x (positive entries only) and its standard error."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0)
    (slope, _), cov = np.polyfit(np.log(x[m]), np.log(y[m]), 1, cov="unscaled") if m.sum() > 2 else ((np.nan, 0), None)
    if cov is None:
        return float(slope), float("nan")
    res = np.log(y[m]) - np.polyval(np.polyfit(np.log(x[m]), np.log(y[m]), 1), np.log(x[m]))
    return float(slope), float(np.sqrt(cov[0, 0] * np.sum(res ** 2) / (m.sum() - 2)))


def loglog_slope(x, y):
    """Least-squares slope of log y against log x (positive entries only)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = (x > 0) & (y > 0)
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])
