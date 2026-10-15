"""FDTest estimator (pure functions, numpy/scipy only; no I/O, no plotting).

Given samples of an observed variable x in four protocol stages (Geli et al., FDTest):
  stage II  (perturbation on, stationary)   -> P_eps
  stage III (perturbation just removed)     -> mean relaxation <Z(tau)>
  stage IV  (off, stationary)               -> P_0 and the autocorrelation C_ZZ(tau)
build Z = P_eps / P_0 - 1 and the violation
  delta2 = sum_{tau <= tau_c} (C_ZZ - <Z>)^2 / sum_{tau <= tau_c} <Z>^2,
which is 0 for a Markov variable. Lags are in units of the recording interval.

Two observables are supported:
  * binary (one junction: ON/OFF): Z takes two values, no binning. The per-cycle sufficient
    statistics (binary_stage_stats) make bootstrap over cycles exact and cheap, and class averages
    over junctions additive across chunks of junctions (class_sums).
  * continuous (network g_eff): Z from a histogram or a Gaussian KDE (z_map / z_eval).
All stage arrays have the cycle as first axis.
"""
import warnings

import numpy as np
from scipy.stats import gaussian_kde

# ----------------------------------------------------------------------------- core
def select_cutoff(z_mean, frac=0.05, run=3):
    """tau_c: first lag from which |<Z>| stays below frac * |<Z>(0)| for `run` consecutive lags (a single
    noisy lag does not trigger it). Last lag if that never happens or <Z>(0) is 0/NaN."""
    z = np.asarray(z_mean, dtype=float)
    n = len(z)
    if n == 0 or not np.isfinite(z[0]) or z[0] == 0:
        return max(n - 1, 0)
    below = np.abs(z) < frac * abs(z[0])
    for i in range(n - run + 1):
        if below[i:i + run].all():
            return i
    return n - 1


def noise_level(z3, smooth=7):
    """(sigma_noise, z_mean, z_smooth) of per-cycle curves z3 (n_cycles, n_lags), NaN = missing. z_mean = nanmean over the
    cycles; SE(tau) = nanstd(ddof=1) / sqrt(number of valid cycles) (NaN with fewer than 2); sigma_noise = MEDIAN of SE over
    the lags (robust to the few lags with a huge scatter, where the mean would be inflated); z_smooth = centred moving
    average of z_mean over `smooth` lags that ignores missing lags and shrinks at the edges (edge-safe)."""
    z3 = np.asarray(z3, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z_mean = np.nanmean(z3, axis=0)
        n = np.isfinite(z3).sum(axis=0)
        se = np.where(n >= 2, np.nanstd(z3, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
        sigma = float(np.nanmedian(se)) if np.isfinite(se).any() else float("nan")
    ok = np.isfinite(z_mean)
    kernel = np.ones(min(max(int(smooth), 1), max(len(z_mean), 1)))
    num = np.convolve(np.where(ok, z_mean, 0.0), kernel, mode="same")
    den = np.convolve(ok.astype(float), kernel, mode="same")
    with np.errstate(divide="ignore", invalid="ignore"):
        z_smooth = np.where(den > 0, num / den, np.nan)
    return sigma, z_mean, z_smooth


def noise_cutoff(z3, smooth=7, sustain=3):
    """Noise-based analysis cutoff (Engbring et al.: stop where <z(tau)> reaches the standard error of the measurement
    noise). Returns (cutoff, hit_end): the first lag from which the smoothed |<z>| stays below sigma_noise (noise_level:
    median over the lags of the standard error of <z>) for `sustain` consecutive lags. hit_end is True, and the cutoff is the
    LAST lag, when that never happens or when it is already the case at lag 0 (no resolvable signal, so no honest cut can
    be made; a window of length 1 must not be returned) or when the noise level is undefined (no valid data). NaN smoothed
    values never count as below the noise. Choose it ONCE on the full sample and hold it fixed (cutoff= argument of the
    estimate, its bootstrap and its floor): it is data-dependent and jumps between resamples otherwise."""
    z3 = np.asarray(z3, dtype=float)
    n = z3.shape[1]
    sigma, _, sm = noise_level(z3, smooth)
    if not np.isfinite(sigma):
        return n - 1, True
    below = np.abs(sm) < sigma                                # NaN compares False
    for i in range(n - sustain + 1):
        if below[i:i + sustain].all():
            return (n - 1, True) if i == 0 else (i, False)
    return n - 1, True


def _fixed_end(cutoff, n):
    """Inclusive last lag of a FIXED analysis window (the `cutoff` argument of the delta2 functions), clipped to the last
    of the n lags. A fixed window bypasses select_cutoff, so the estimate, its bootstrap and its noise floor share one
    window that does not depend on the data."""
    if cutoff < 0:
        raise ValueError("cutoff must be a non-negative lag index")
    return int(min(cutoff, max(n - 1, 0)))


def _fixed_bin(edges, cutoff):
    """Last bin of a fixed window on binned curves: every bin that STARTS at a lag <= cutoff is used, i.e. the bin
    containing the cutoff lag is used whole (its reported tau_c is its last lag)."""
    if cutoff < 0:
        raise ValueError("cutoff must be a non-negative lag index")
    return int(min(np.searchsorted(np.asarray(edges, dtype=int), cutoff, side="right") - 1, len(edges) - 2))


def delta_squared(c_zz, z_mean, tau_c):
    """Relative squared difference of the two FDT terms over lags 0..tau_c (NaN if <Z> vanishes)."""
    c = np.asarray(c_zz, dtype=float)[:tau_c + 1]
    z = np.asarray(z_mean, dtype=float)[:tau_c + 1]
    den = np.sum(z ** 2)
    if not np.isfinite(den) or den == 0:
        return float("nan")
    return float(np.sum((c - z) ** 2) / den)


def masked_autocorr(z, max_lag):
    """Raw autocorrelation <z_t z_{t+l}>, l = 0..max_lag, pooled over the rows of z (n_series, T);
    NaN samples are missing (only pairs with both samples present count)."""
    z = np.asarray(z, dtype=float)
    _, T = z.shape
    present = np.isfinite(z).astype(float)
    filled = np.where(present > 0, z, 0.0)
    nfft = 1 << int(np.ceil(np.log2(2 * T)))

    def corr(a):
        F = np.fft.rfft(a, n=nfft, axis=1)
        return np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :max_lag + 1].sum(axis=0)

    num, den = corr(filled), corr(present)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0.5, num / den, np.nan)


# ----------------------------------------------------------------------------- lag binning
def lag_bin_edges(n_lags, n_first=8, ratio=1.15):
    """Bin edges (int array, edges[0] = 0, strictly increasing, edges[-1] = n_lags) of the lag-binned violation: the first
    n_first lags are single-lag bins, then every edge is max(previous + 1, floor(previous * ratio + 0.5)), the last one
    clipped to n_lags. The scheme is fixed in advance (not tuned to data); see docs/lag_binning_design.md."""
    if n_lags < 1 or n_first < 1 or not (np.isfinite(ratio) and ratio > 1):
        raise ValueError("n_lags >= 1, n_first >= 1 and a finite ratio > 1 are required")
    edges = list(range(min(n_first, n_lags) + 1))
    while edges[-1] < n_lags:
        edges.append(min(max(edges[-1] + 1, int(np.floor(edges[-1] * ratio + 0.5))), n_lags))
    return np.array(edges, dtype=int)


def bin_lags(curve, edges):
    """Mean of the last axis over each lag bin [edges[b], edges[b+1]) -> (..., n_bins). NaN-aware: NaN lags are left out of
    their bin's mean and an all-NaN bin is NaN (no warning). The bin widths n_b are np.diff(edges); edges must start at 0,
    increase strictly and end at the number of lags."""
    x = np.asarray(curve, dtype=float)
    edges = np.asarray(edges, dtype=int)
    if edges.ndim != 1 or edges.size < 2 or edges[0] != 0 or edges[-1] != x.shape[-1] or not (np.diff(edges) >= 1).all():
        raise ValueError("edges must start at 0, increase strictly and end at the number of lags")
    fin = np.isfinite(x)
    total = np.add.reduceat(np.where(fin, x, 0.0), edges[:-1], axis=-1)
    count = np.add.reduceat(fin.astype(float), edges[:-1], axis=-1)
    return np.where(count > 0, total / np.where(count > 0, count, 1.0), np.nan)


def _bin_joint(curves, edges):
    """Bin several per-lag curves over the lags where ALL of them are finite (a lag missing in one curve is dropped from the
    others too, so every bin average uses the same set of lags). Returns (list of binned curves, n_b) with n_b the number of
    jointly valid lags of each bin (the weight of the bin; 0 and NaN averages for a bin without any)."""
    stack = np.stack([np.asarray(c, dtype=float) for c in curves])
    fin = np.isfinite(stack).all(axis=0)
    binned = bin_lags(np.where(fin, stack, np.nan), edges)
    return list(binned), np.add.reduceat(fin.astype(float), np.asarray(edges, dtype=int)[:-1])


def _binned_delta2(c_zz, z_mean, edges, frac=0.05, run=3, cutoff=None):
    """(delta2, cutoff bin) of the lag-binned violation sum_b n_b (Cb - Zb)^2 / sum_b n_b Zb^2 over the bins up to the
    cutoff bin, selected on the binned Z curve with select_cutoff (bins play the role of lags) or, with `cutoff` (a lag),
    the last bin starting at a lag <= cutoff. Z and C are averaged over the same (jointly finite) lags of a bin and n_b
    counts those lags; bins without any are left out; NaN if Zb vanishes."""
    (zb, cb), n = _bin_joint([z_mean, c_zz], edges)
    bc = select_cutoff(np.where(np.isfinite(zb), zb, 0.0), frac, run) if cutoff is None else _fixed_bin(edges, cutoff)
    use = n[:bc + 1] > 0
    w, z, c = n[:bc + 1][use], zb[:bc + 1][use], cb[:bc + 1][use]
    den = np.sum(w * z ** 2)
    if not np.isfinite(den) or den == 0:
        return float("nan"), bc
    return float(np.sum(w * (c - z) ** 2) / den), bc


LAG_NOISE_MAX_LAG = 50


def lag_noise_correlation(z_a, z_b, max_lag=LAG_NOISE_MAX_LAG):
    """Correlation of the scatter of <Z> across the LAG index: autocorrelation of d = z_a - z_b (two independent halves of
    the cycles; NaN lags are missing, only pairs with both lags present count; the finite mean is removed) normalised by
    its lag-0 value. Returns (rho[0..max_lag], corr_length) with corr_length = 1 + 2 * sum of rho[k > 0] up to the first
    non-positive value (>= 1; about 1 for white scatter, so binning n_b lags reduces the noise variance by about
    n_b / corr_length when n_b is larger than corr_length). NaN if fewer than 2 finite lags."""
    d = np.asarray(z_a, dtype=float) - np.asarray(z_b, dtype=float)
    fin = np.isfinite(d)
    m = int(min(max_lag, len(d) - 1))
    if fin.sum() < 2 or m < 0:
        return np.full(max(m, 0) + 1, np.nan), float("nan")
    raw = masked_autocorr((d - d[fin].mean())[None, :], m)
    if not raw[0] > 0:
        return np.full(m + 1, np.nan), float("nan")
    rho = raw / raw[0]
    length = 1.0
    for k in range(1, m + 1):
        if not np.isfinite(rho[k]) or rho[k] <= 0:
            break
        length += 2.0 * rho[k]
    return rho, float(length)


# ----------------------------------------------------------------------------- binary observable
def binary_stage_stats(g2, g3, g4, max_lag):
    """Per-cycle sufficient statistics for binary observables (junction ON/OFF), junctions on the last axis.

    g2: (n_cycles, T2, K) stage II samples; g3: (n_cycles, max_lag + 1, K) stage III (lag 0 = state at the
    removal instant); g4: (n_cycles, T4, K) stage IV samples. Returns dict(m2, m4: (n_cycles, K) means,
    pair: (n_cycles, max_lag + 1, K) time averages of g_t g_{t+l} in stage IV, g3: float32 copy)."""
    g2, g3, g4 = np.asarray(g2), np.asarray(g3), np.asarray(g4)
    n_cycles, T4, K = g4.shape
    if g3.shape[1] != max_lag + 1 or g2.shape[0] != n_cycles or g3.shape[0] != n_cycles \
            or g2.shape[2] != K or g3.shape[2] != K:
        raise ValueError("inconsistent stage array shapes")
    if max_lag >= T4:
        raise ValueError("max_lag must be smaller than the stage IV length")
    nfft = 1 << int(np.ceil(np.log2(2 * T4)))
    pair = np.empty((n_cycles, max_lag + 1, K), dtype=np.float32)
    norm = (T4 - np.arange(max_lag + 1))[:, None]
    for c in range(n_cycles):
        F = np.fft.rfft(g4[c].astype(float), n=nfft, axis=0)
        pair[c] = np.fft.irfft(F * np.conj(F), n=nfft, axis=0)[:max_lag + 1] / norm
    return {"m2": g2.mean(axis=1), "m4": g4.mean(axis=1), "pair": pair, "g3": g3.astype(np.float32)}


def binary_curves(stats, weights=None, p_min=0.02):
    """<Z(tau)> and C_ZZ(tau) of every junction for a (re)weighted set of cycles.

    weights: (n_cycles,) non-negative, summing to 1 (default uniform); bootstrap resamples use counts / n.
    Z(g) = alpha + beta g with alpha = Z(OFF), beta = Z(ON) - Z(OFF). A junction is valid only if p_eps and p_0 lie in
    [p_min, 1 - p_min]; invalid junctions get NaN. Returns dict(valid, alpha, beta, z_mean (L, K), c_zz (L, K))."""
    n = stats["m2"].shape[0]
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    L, K = stats["pair"].shape[1], stats["pair"].shape[2]
    p_eps, p_0 = w @ stats["m2"], w @ stats["m4"]
    pair = (w @ stats["pair"].reshape(n, -1)).reshape(L, K)
    g3 = (w @ stats["g3"].reshape(n, -1)).reshape(L, K)
    valid = (p_0 >= p_min) & (p_0 <= 1 - p_min) & (p_eps >= p_min) & (p_eps <= 1 - p_min)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = (1 - p_eps) / (1 - p_0) - 1.0
        beta = (p_eps / p_0 - 1.0) - alpha
        z_mean = alpha + beta * g3
        c_zz = alpha ** 2 + 2 * alpha * beta * p_0 + beta ** 2 * pair       # raw <Z_t Z_{t+l}>, as in the paper
    return {"valid": valid, "alpha": alpha, "beta": beta,
            "z_mean": np.where(valid, z_mean, np.nan), "c_zz": np.where(valid, c_zz, np.nan)}


def binary_fdtest(stats, p_min=0.02, frac=0.05, run=3):
    """Per-junction FDTest (uniform weights): dict from binary_curves plus delta2 (K,) with each junction's own tau_c."""
    out = binary_curves(stats, None, p_min)
    delta2 = np.full(out["valid"].shape, np.nan)
    for k in np.nonzero(out["valid"])[0]:
        z = out["z_mean"][:, k]
        delta2[k] = delta_squared(out["c_zz"][:, k], z, select_cutoff(z, frac, run))
    out["delta2"] = delta2
    return out


def class_sums(z_mean, c_zz, masks):
    """Sums of the junction curves over the valid junctions of each class (masks: (C, K) bool).
    Returns (sum_z (C, L), sum_c (C, L), count (C,)); sums from different chunks of junctions simply add."""
    masks = np.atleast_2d(np.asarray(masks, dtype=bool))
    ok = np.isfinite(z_mean[0])
    C, L = masks.shape[0], z_mean.shape[0]
    sum_z, sum_c, count = np.zeros((C, L)), np.zeros((C, L)), np.zeros(C)
    for c in range(C):
        m = masks[c] & ok
        sum_z[c] = z_mean[:, m].sum(axis=1)
        sum_c[c] = c_zz[:, m].sum(axis=1)
        count[c] = m.sum()
    return sum_z, sum_c, count


def delta2_from_sums(sum_z, sum_c, count, frac=0.05, run=3, return_tau=False, edges=None, cutoff=None):
    """Class-averaged delta2 (C,): average the curves over the junctions of a class first, then one tau_c per class.
    With return_tau=True returns (delta2, tau_c) where tau_c is an int array (-1 for an empty class); a tau_c equal to
    the last lag means the relaxation never fell below `frac` and delta2 is truncated. With `edges` (see lag_bin_edges) the
    class curves are lag-binned first: delta2 is the lag-binned violation and tau_c is the LAST LAG of the cutoff bin.
    With `cutoff` (inclusive lag index, default None = select_cutoff per class) the window is FIXED: lags 0..cutoff (clipped
    to the last lag), or with `edges` the bins starting at a lag <= cutoff; tau_c is then the last lag actually used."""
    out = np.full(len(count), np.nan)
    tau = np.full(len(count), -1, dtype=int)
    for c in range(len(count)):
        if count[c] > 0:
            z, cc = sum_z[c] / count[c], sum_c[c] / count[c]
            if edges is None:
                tau[c] = select_cutoff(z, frac, run) if cutoff is None else _fixed_end(cutoff, len(z))
                out[c] = delta_squared(cc, z, tau[c])
            else:
                out[c], bc = _binned_delta2(cc, z, edges, frac, run, cutoff)
                tau[c] = edges[bc + 1] - 1
    return (out, tau) if return_tau else out


def class_delta2(z_mean, c_zz, masks, frac=0.05, run=3, edges=None, cutoff=None):
    """Class-averaged delta2 and junction counts from junction curves (wrapper of class_sums)."""
    sz, sc, n = class_sums(z_mean, c_zz, masks)
    return delta2_from_sums(sz, sc, n, frac, run, edges=edges, cutoff=cutoff), n.astype(int)


def bootstrap_class_sums(stats, weights, masks, p_min=0.02):
    """Class sums (see class_sums) for every row of `weights` (R, n_cycles), each row a resampling of the cycles
    (non-negative, summing to 1). Returns (sum_z (R, C, L), sum_c (R, C, L), count (R, C)). Sums over different
    chunks of junctions add, so a long analysis can process junctions in chunks and finish with delta2_from_sums."""
    masks = np.atleast_2d(np.asarray(masks, dtype=bool))
    R, C, L = len(weights), masks.shape[0], stats["pair"].shape[1]
    sum_z, sum_c, count = np.zeros((R, C, L)), np.zeros((R, C, L)), np.zeros((R, C))
    for r in range(R):
        cur = binary_curves(stats, weights[r], p_min)
        sum_z[r], sum_c[r], count[r] = class_sums(cur["z_mean"], cur["c_zz"], masks)
    return sum_z, sum_c, count


def bootstrap_weights(n_cycles, n_boot, rng):
    """(n_boot, n_cycles) bootstrap resampling weights (multiplicity counts / n_cycles)."""
    return np.stack([np.bincount(rng.integers(0, n_cycles, n_cycles), minlength=n_cycles) / n_cycles
                     for _ in range(n_boot)])


def bootstrap_class_delta2(stats, masks, n_boot, rng, p_min=0.02, frac=0.05, run=3, return_z0=False, edges=None, cutoff=None):
    """(n_boot, C) class-averaged delta2 of bootstrap resamples of the CYCLES (P_eps, P_0, Z, <Z>, C_ZZ are
    recomputed for every resample). With return_z0=True also the (n_boot, C) class-averaged <Z(0)> (for the SNR).
    With `cutoff` every resample is evaluated over the same fixed window (see delta2_from_sums)."""
    w = bootstrap_weights(stats["m2"].shape[0], n_boot, rng)
    sum_z, sum_c, count = bootstrap_class_sums(stats, w, masks, p_min)
    d2 = np.stack([delta2_from_sums(sum_z[b], sum_c[b], count[b], frac, run, edges=edges, cutoff=cutoff)
                   for b in range(n_boot)])
    if not return_z0:
        return d2
    with np.errstate(divide="ignore", invalid="ignore"):
        return d2, sum_z[:, :, 0] / count


def split_half_class_sums(stats, masks, half_a, half_b, p_min=0.02):
    """Chunk-additive ingredients of the noise floor: class sums of <Z> from all cycles, from the cycles `half_a`
    and from `half_b` (indices), all with the Z mapping of ALL cycles. Returns (sum_full, sum_a, sum_b (C, L), count (C,))."""
    full = binary_curves(stats, None, p_min)
    masks = np.atleast_2d(np.asarray(masks, dtype=bool))
    with np.errstate(invalid="ignore"):                       # invalid junctions carry NaN alpha / beta
        z_a = full["alpha"] + full["beta"] * stats["g3"][half_a].mean(axis=0)
        z_b = full["alpha"] + full["beta"] * stats["g3"][half_b].mean(axis=0)
    C, L = masks.shape[0], z_a.shape[0]
    sf, sa, sb, n = np.zeros((C, L)), np.zeros((C, L)), np.zeros((C, L)), np.zeros(C)
    for c in range(C):
        m = masks[c] & full["valid"]
        sf[c], sa[c], sb[c], n[c] = full["z_mean"][:, m].sum(axis=1), z_a[:, m].sum(axis=1), z_b[:, m].sum(axis=1), m.sum()
    return sf, sa, sb, n


def floor_from_sums(sum_full, sum_a, sum_b, count, frac=0.05, run=3, edges=None, cutoff=None):
    """Finite-M noise floor of the class-averaged delta2 (C,). C_ZZ is a long time average and precise, so the bias
    comes from <Z>; two independent halves of the cycles (same Z mapping) give <Z>_A and <Z>_B and the floor is
    1/4 sum (<Z>_A - <Z>_B)^2 / sum <Z>^2 (each half has twice the variance of the full estimate, the difference
    twice that again). With `edges` the three curves are lag-binned and the floor is
    1/4 sum_b n_b (Z_A,b - Z_B,b)^2 / sum_b n_b Z_b^2 over the bins up to the binned cutoff (as in delta2_from_sums).
    With `cutoff` the window is the same fixed one as in delta2_from_sums (lags 0..cutoff, or the bins starting at a lag
    <= cutoff) instead of the data-dependent cutoff."""
    out = np.full(len(count), np.nan)
    for c in range(len(count)):
        if count[c] > 0:
            zf, za, zb = sum_full[c] / count[c], sum_a[c] / count[c], sum_b[c] / count[c]
            if edges is None:
                tc = select_cutoff(zf, frac, run) if cutoff is None else _fixed_end(cutoff, len(zf))
                out[c] = 0.25 * float(np.sum((za[:tc + 1] - zb[:tc + 1]) ** 2) / np.sum(zf[:tc + 1] ** 2))
            else:
                (zf, za, zb), n = _bin_joint([zf, za, zb], edges)
                bc = select_cutoff(np.where(np.isfinite(zf), zf, 0.0), frac, run) if cutoff is None else _fixed_bin(edges, cutoff)
                use = n[:bc + 1] > 0
                w = n[:bc + 1][use]
                den = np.sum(w * zf[:bc + 1][use] ** 2)
                out[c] = 0.25 * float(np.sum(w * (za[:bc + 1][use] - zb[:bc + 1][use]) ** 2) / den) if den > 0 else np.nan
    return out


def split_half_floor(stats, masks, rng, p_min=0.02, frac=0.05, run=3, n_splits=1, edges=None, cutoff=None):
    """Noise floor of the class-averaged delta2: the mean over `n_splits` independent random splits of the cycles into
    two halves (wrapper of split_half_class_sums / floor_from_sums). One split is a noisy estimate of the floor
    (it varied 5x across replicas); n_splits = 1 reproduces the single-split value for the same rng."""
    n = stats["m2"].shape[0]
    floors = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        half = len(perm) // 2
        floors.append(floor_from_sums(*split_half_class_sums(stats, masks, perm[:half], perm[half:], p_min), frac, run, edges, cutoff))
    return np.mean(floors, axis=0)


def binary_shift_sigma(stats):
    """Achieved perturbation of every junction in units of the binomial spread of its stage-IV state:
    |p_eps - p_0| / sqrt(p_0 (1 - p_0)) with p_eps, p_0 the cycle-averaged ON probabilities (K,); NaN if p_0 is 0 or 1."""
    p_eps, p_0 = stats["m2"].mean(axis=0), stats["m4"].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.abs(p_eps - p_0) / np.sqrt(p_0 * (1.0 - p_0))
    return np.where((p_0 > 0) & (p_0 < 1), shift, np.nan)


INTERVAL_METHODS = ("percentile", "reverse", "sqrt_reverse")


def _reverse_interval(d, boot, alpha):
    """Reverse-percentile (basic) interval [2 d - q_{1-alpha/2}, 2 d - q_{alpha/2}] of the finite draws `boot`; NaN NaN if
    d is not finite or there are no finite draws (no warning)."""
    b = boot[np.isfinite(boot)]
    if not np.isfinite(d) or b.size == 0:
        return [float("nan"), float("nan")]
    q_lo, q_hi = (float(v) for v in np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    return [2 * d - q_hi, 2 * d - q_lo]


def violation_summary(delta2_hat, boot, floor, alpha=0.05, method="reverse"):
    """Decision summary of a delta2 estimate against its finite-M noise floor. A percentile interval of the bootstrap
    of a biased estimator double counts the bias (about 12 % false positives on a Markov junction at M = 480), so
    the default violation call uses the REVERSE (basic) interval [2 d - q_hi, 2 d - q_lo]: excludes_floor is True only if
    its lower bound exceeds the floor. Non-finite inputs give excludes_floor False (fail closed).
    `method` chooses the interval that decides: "reverse" (default), "percentile" ([q_lo, q_hi] of the bootstrap) or
    "sqrt_reverse" (the reverse interval computed on s = sqrt(delta2), where the bootstrap distribution is much less skewed:
    [2 s_hat - q_hi(s*), 2 s_hat - q_lo(s*)] clipped at 0 and squared back; negative draws are clipped to 0, a negative or
    non-finite delta2_hat gives a NaN interval).
    Returns dict(delta2, ci95_percentile, ci95_reverse, ci95_sqrt_reverse (always reported), ci95 = the interval used for the
    decision, method, excess = delta2 - floor, excludes_floor)."""
    if method not in INTERVAL_METHODS:
        raise ValueError(f"unknown interval method {method!r} (use one of {INTERVAL_METHODS})")
    b = np.asarray(boot, dtype=float).ravel()
    b = b[np.isfinite(b)]
    d = float(delta2_hat)
    q_lo, q_hi = (float(v) for v in np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])) if b.size else (np.nan, np.nan)
    pct = [q_lo, q_hi]
    rev = _reverse_interval(d, b, alpha)
    if np.isfinite(d) and d >= 0 and b.size:
        s_iv = _reverse_interval(np.sqrt(d), np.sqrt(np.maximum(b, 0.0)), alpha)
        sqrt_rev = [float(np.clip(v, 0.0, None) ** 2) for v in s_iv]
    else:
        sqrt_rev = [float("nan"), float("nan")]
    ci = {"percentile": pct, "reverse": rev, "sqrt_reverse": sqrt_rev}[method]
    floor = float(floor)
    return {"delta2": d, "ci95_percentile": pct, "ci95_reverse": rev, "ci95_sqrt_reverse": sqrt_rev, "ci95": ci, "method": method,
            "excess": d - floor, "excludes_floor": bool(np.isfinite(ci[0]) and np.isfinite(floor) and ci[0] > floor)}


def studentized_interval(d_hat, d_boot, se_boot, alpha=0.05):
    """Bootstrap-t interval of delta2 (for calibration studies: it needs an inner bootstrap for every outer draw).
    t*_b = (d_b - d_hat) / se_b over the draws with a finite d_b and a finite, positive se_b; se_hat = std(d_boot, ddof=1)
    over the finite d_b; interval [d_hat - q_{1-alpha/2}(t*) se_hat, d_hat - q_{alpha/2}(t*) se_hat] clipped at 0. Returns
    [nan, nan] (which never excludes a floor) if d_hat is not finite, fewer than 2 draws are usable, or se_hat is not > 0."""
    d_boot, se_boot = np.asarray(d_boot, dtype=float).ravel(), np.asarray(se_boot, dtype=float).ravel()
    fin = np.isfinite(d_boot)
    se_hat = float(np.std(d_boot[fin], ddof=1)) if fin.sum() > 1 else float("nan")
    ok = fin & np.isfinite(se_boot) & (se_boot > 0)
    if not np.isfinite(d_hat) or ok.sum() < 2 or not (np.isfinite(se_hat) and se_hat > 0):
        return [float("nan"), float("nan")]
    t = (d_boot[ok] - d_hat) / se_boot[ok]
    q_lo, q_hi = np.percentile(t, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(max(d_hat - q_hi * se_hat, 0.0)), float(max(d_hat - q_lo * se_hat, 0.0))]


# ----------------------------------------------------------------------------- continuous observable
def z_map(x_eps, x_0, n_bins=60, q=(0.5, 99.5), n_min=20, estimator="hist", n_grid=1000):
    """Conjugate variable as (grid, z_grid); z_grid is NaN where it cannot be estimated (bins with fewer than
    n_min samples in either stage, or KDE density of P_0 below 1e-3 of its maximum)."""
    xe = np.asarray(x_eps, dtype=float).ravel()
    x0 = np.asarray(x_0, dtype=float).ravel()
    xe, x0 = xe[np.isfinite(xe)], x0[np.isfinite(x0)]
    if len(xe) == 0 or len(x0) == 0:
        return np.array([0.0, 1.0]), np.array([np.nan, np.nan])
    lo, hi = np.percentile(np.r_[xe, x0], q)
    if not hi > lo:
        return np.array([lo, lo + 1.0]), np.array([np.nan, np.nan])
    if estimator == "hist":
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ce, c0 = np.histogram(xe, edges)[0], np.histogram(x0, edges)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (ce / len(xe)) / (c0 / len(x0)) - 1.0
        return centers, np.where((ce >= n_min) & (c0 >= n_min), z, np.nan)
    if estimator == "kde":
        grid = np.linspace(lo, hi, n_grid)
        try:
            # strided subsample across ALL cycles (the ravel is cycle-major: a head slice would keep only the first ones)
            p_e = gaussian_kde(xe[::max(1, len(xe) // 50000)])(grid)
            p_0 = gaussian_kde(x0[::max(1, len(x0) // 50000)])(grid)
        except (np.linalg.LinAlgError, ValueError):              # constant or single-sample stage: no density
            return grid, np.full(n_grid, np.nan)
        good = p_0 > 1e-3 * p_0.max()
        return grid, np.where(good, p_e / np.where(good, p_0, 1.0) - 1.0, np.nan)
    raise ValueError(f"unknown estimator {estimator!r} (use 'hist' or 'kde')")


def z_eval(x, grid, z_grid):
    """Z(x) by linear interpolation between the valid grid points; NaN outside their range."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(z_grid)
    if ok.sum() < 2:
        return np.full_like(x, np.nan)
    g, v = grid[ok], z_grid[ok]
    return np.where((x >= g[0]) & (x <= g[-1]), np.interp(x, g, v), np.nan)


def continuous_fdtest(x2, x3, x4, n_bins=60, q=(0.5, 99.5), n_min=20, estimator="hist", frac=0.05, run=3, edges=None, cutoff=None):
    """FDTest of one continuous observable. x2 (n_cycles, T2), x3 (n_cycles, n_lags) from the removal instant,
    x4 (n_cycles, T4). Returns dict(z_mean, c_zz, tau_c, delta2, excluded_fraction, grid, z_grid, z3 (per-cycle)).
    With `edges` (lag_bin_edges(n_lags)) delta2 is the lag-binned violation, tau_c is the LAST LAG of the cutoff bin and
    the result also holds cutoff_bin and cutoff_at_end (z_mean and c_zz stay per lag). With `cutoff` (inclusive lag index)
    the window is fixed instead of chosen from the data: tau_c = cutoff (clipped to the last lag), or with `edges` the
    bins starting at a lag <= cutoff (tau_c = the last lag of the last of them)."""
    x2, x3, x4 = (np.asarray(a, dtype=float) for a in (x2, x3, x4))
    grid, z_grid = z_map(x2, x4, n_bins, q, n_min, estimator)
    z3, z4 = z_eval(x3, grid, z_grid), z_eval(x4, grid, z_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # lags where no cycle has a valid Z
        z_mean = np.nanmean(z3, axis=0)
    c_zz = masked_autocorr(z4, x3.shape[1] - 1)
    tau_c = select_cutoff(np.where(np.isfinite(z_mean), z_mean, 0.0), frac, run) if cutoff is None else _fixed_end(cutoff, len(z_mean))
    out = {"z_mean": z_mean, "c_zz": c_zz, "tau_c": tau_c, "delta2": delta_squared(c_zz, z_mean, tau_c),
           "excluded_fraction": float(np.mean(~np.isfinite(z4))), "grid": grid, "z_grid": z_grid, "z3": z3}
    if edges is not None:
        out["delta2"], bc = _binned_delta2(c_zz, z_mean, edges, frac, run, cutoff)
        out.update(tau_c=int(edges[bc + 1] - 1), cutoff_bin=int(bc), cutoff_at_end=bool(bc == len(edges) - 2))
    return out


def bootstrap_continuous_delta2(x2, x3, x4, n_boot, rng, return_z0=False, **kwargs):
    """(n_boot,) delta2 of bootstrap resamples of the cycles (the Z mapping is re-estimated each time); with
    return_z0=True also the (n_boot,) values of <Z(0)> (for the SNR). Other keyword arguments (edges, cutoff, ...) go to
    continuous_fdtest."""
    x2, x3, x4 = (np.asarray(a) for a in (x2, x3, x4))
    n = x2.shape[0]
    d2, z0 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        res = continuous_fdtest(x2[idx], x3[idx], x4[idx], **kwargs)
        d2[b], z0[b] = res["delta2"], res["z_mean"][0]
    return (d2, z0) if return_z0 else d2


def continuous_split_half_floor(x2, x3, x4, rng, frac=0.05, run=3, n_splits=1, edges=None, return_halves=False, cutoff=None, **kwargs):
    """Noise floor of the continuous delta2 (same 1/4 * half-difference rule as floor_from_sums), averaged over
    `n_splits` random splits of the cycles (n_splits = 1 reproduces the single-split value for the same rng). Only lags
    where the full and both half-sample curves are defined contribute (with few cycles a half can lack valid samples
    at a lag); a split with none gives NaN (and so does the mean). With `edges` the curves are lag-binned and the floor is
    1/4 sum_b n_b (Z_A,b - Z_B,b)^2 / sum_b n_b Z_b^2 over the bins up to the binned cutoff, the three curves averaged over
    the same (jointly finite) lags of a bin with n_b their count. With return_halves=True
    returns (floor, z_a, z_b) with the per-lag half curves of the FIRST split (for lag_noise_correlation). With `cutoff` the
    window is the fixed one of continuous_fdtest(cutoff=...)."""
    full = continuous_fdtest(x2, x3, x4, frac=frac, run=run, edges=edges, cutoff=cutoff, **kwargs)
    n = full["z3"].shape[0]
    zf = full["z_mean"]
    floors = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            z_a = np.nanmean(full["z3"][perm[:n // 2]], axis=0)
            z_b = np.nanmean(full["z3"][perm[n // 2:]], axis=0)
        if not floors:
            first = (z_a, z_b)
        if edges is None:
            ok = np.isfinite(z_a) & np.isfinite(z_b) & np.isfinite(zf)
            ok[full["tau_c"] + 1:] = False
            w, fa, fb, ff = np.ones(len(zf)), z_a, z_b, zf
        else:
            (fa, fb, ff), w = _bin_joint([z_a, z_b, zf], edges)                    # w = jointly valid lags per bin
            ok = w > 0
            ok[full["cutoff_bin"] + 1:] = False
        if not ok.any() or np.sum(w[ok] * ff[ok] ** 2) == 0:
            floors.append(float("nan"))
        else:
            floors.append(0.25 * float(np.sum(w[ok] * (fa[ok] - fb[ok]) ** 2) / np.sum(w[ok] * ff[ok] ** 2)))
    floor = float(np.mean(floors))
    return (floor, *first) if return_halves else floor
