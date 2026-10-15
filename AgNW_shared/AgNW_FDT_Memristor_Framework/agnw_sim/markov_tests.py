"""Memory diagnostics for a scalar time series and exact FDTest quantities of a finite Markov chain.

Series statistics (test 4):
  regression_memory_gain   cross-validated gain of adding past lags to a polynomial predictor of g(t+h) from g(t)
  cmi_ksg / cmi_test   kNN conditional mutual information I(g(t+h); g(t-h) | g(t)) with a Markov surrogate null
Exact chain (test 2):
  transition_matrix, stationary_distribution, conjugate_state_map, exact_curves, exact_delta2
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

import agnw_sim.fdtest as ft


# ------------------------------------------------------------------------------------------ linear memory gain
def _design(g, h, k, degree, thin):
    """Rows [g^1..g^degree at t, then g(t-jh), g(t-jh)^2 for j = 1..k] and targets g(t+h), t = k*h .. n-h-1, every `thin`-th."""
    g = np.asarray(g, dtype=float)
    t = np.arange(k * h, len(g) - h)[::thin]
    cols = [g[t] ** p for p in range(1, degree + 1)]
    for j in range(1, k + 1):
        cols += [g[t - j * h], g[t - j * h] ** 2]
    return np.stack(cols, axis=1), g[t + h]


def _ridge_fit(X, y, lams=np.logspace(-8, 2, 21)):
    """Ridge regression on standardized columns; the penalty minimizes the generalized cross-validation score of the
    training data (closed form from the eigendecomposition of X'X). Returns a predict(X_new) function."""
    xm, xs = X.mean(axis=0), X.std(axis=0)
    xs[xs == 0] = 1.0
    Z = (X - xm) / xs
    ym = y.mean()
    yc = y - ym
    w, V = np.linalg.eigh(Z.T @ Z)
    w = np.clip(w, 0.0, None)
    b = V.T @ (Z.T @ yc)
    n = len(y)
    wpos = np.where(w > 0, w, 1.0)
    best_lam, best_score = lams[0], np.inf
    for lam in lams:
        f = w / (w + lam)
        rss = yc @ yc - np.sum((2 * f - f ** 2) * b ** 2 / wpos)
        score = n * rss / (n - f.sum()) ** 2
        if score < best_score:
            best_lam, best_score = lam, score
    beta = V @ (b / (w + best_lam))
    return lambda Xn: ((Xn - xm) / xs) @ beta + ym


def regression_memory_gain(runs, h, k=3, degree=7, thin=5, n_folds=5, n_boot=200, rng=None):
    """Cross-validated memory gain of a series ensemble (list of 1-D arrays, e.g. one per run; standardize them first).

    Model A predicts g(t+h) from a polynomial of g(t) (degree `degree`); model B adds g(t-jh) and g(t-jh)^2, j = 1..k.
    The polynomial in A absorbs a NONLINEAR Markov dependence on g(t), so for a Markov series the gain is 0; a purely
    linear A would report memory for a nonlinear Markov series. Folds are whole runs. gain = 1 - SSE_B / SSE_A over the
    held-out runs; ci95 is a percentile bootstrap over runs (n_boot resamples). Returns dict(gain, ci95, n_runs, n_rows)."""
    rng = np.random.default_rng() if rng is None else rng
    R = len(runs)
    if R < 2 or n_folds < 2:
        raise ValueError("need at least 2 runs and 2 folds")
    n_folds = min(n_folds, R)
    des = [_design(g, h, k, degree, thin) for g in runs]
    sse = np.zeros((R, 2))
    for fold in np.array_split(rng.permutation(R), n_folds):
        test = set(int(i) for i in fold)
        train = [i for i in range(R) if i not in test]
        Xtr = np.concatenate([des[i][0] for i in train])
        ytr = np.concatenate([des[i][1] for i in train])
        fit_b, fit_a = _ridge_fit(Xtr, ytr), _ridge_fit(Xtr[:, :degree], ytr)
        for i in test:
            X, y = des[i]
            sse[i] = [np.sum((y - fit_a(X[:, :degree])) ** 2), np.sum((y - fit_b(X)) ** 2)]
    gain = 1.0 - sse[:, 1].sum() / sse[:, 0].sum()
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, R, R)
        boots[b] = 1.0 - sse[idx, 1].sum() / sse[idx, 0].sum()
    return {"gain": float(gain), "ci95": [float(v) for v in np.percentile(boots, [2.5, 97.5])],
            "n_runs": R, "n_rows": int(sum(len(d[1]) for d in des))}


# ------------------------------------------------------------------------------------------ conditional mutual information
def cmi_ksg(x, y, z, k=5):
    """Kraskov-Stoegbauer-Grassberger / Frenzel-Pompe estimator of I(X; Y | Z) (nats), max-norm, k nearest neighbours.
    Tied values bias it: add a tiny jitter to discrete data before calling."""
    pts = np.column_stack([x, y, z]).astype(float)
    d, _ = cKDTree(pts).query(pts, k=k + 1, p=np.inf)
    eps = np.nextafter(d[:, k], 0.0)                       # count neighbours STRICTLY inside the k-th neighbour distance

    def count(cols):
        sub = pts[:, cols]
        return cKDTree(sub).query_ball_point(sub, r=eps, p=np.inf, return_length=True)   # includes the point itself
    return float(digamma(k) - np.mean(digamma(count([0, 2])) + digamma(count([1, 2])) - digamma(count([2]))))


def markov_surrogate(x, z, rng, k_nn=20):
    """Resample x among the triples whose z is one of the k_nn nearest neighbours of this triple's z: x depends on z
    (and on nothing else), the marginal of x and the z -> x relation are preserved."""
    _, idx = cKDTree(np.asarray(z, float)[:, None]).query(np.asarray(z, float)[:, None], k=k_nn)
    return np.asarray(x)[idx[np.arange(len(z)), rng.integers(0, k_nn, len(z))]]


def cmi_test(x, y, z, rng, k=5, n_surr=100, k_nn=20):
    """I(X;Y|Z) against the Markov surrogate null. detected = observed > the 97.5th percentile of n_surr surrogates."""
    obs = cmi_ksg(x, y, z, k)
    sur = np.array([cmi_ksg(markov_surrogate(x, z, rng, k_nn), y, z, k) for _ in range(n_surr)])
    q975 = float(np.percentile(sur, 97.5))
    return {"cmi": obs, "surrogate_mean": float(sur.mean()), "surrogate_q975": q975,
            "excess": float(obs - sur.mean()), "detected": bool(obs > q975)}


def triples(runs, h, stride, n_max, rng, jitter=1e-9):
    """(g(t+h), g(t-h), g(t)) from a list of series: t on a grid of `stride` steps in every run (h <= t < n-h), at most n_max triples
    (random subsample), each series jittered by jitter * std to break ties."""
    xs, ys, zs = [], [], []
    for g in runs:
        g = np.asarray(g, float)
        g = g + jitter * g.std() * rng.normal(size=len(g))
        t = np.arange(h, len(g) - h, stride)
        xs.append(g[t + h]); ys.append(g[t - h]); zs.append(g[t])
    x, y, z = map(np.concatenate, (xs, ys, zs))
    if len(x) > n_max:
        keep = rng.choice(len(x), n_max, replace=False)
        x, y, z = x[keep], y[keep], z[keep]
    return x, y, z


def correlation_time(runs, max_lag=2000):
    """First lag at which the mean (over runs) normalized autocorrelation drops below 1/e (max_lag if it never does)."""
    acf = np.zeros(max_lag + 1)
    for g in runs:
        g = np.asarray(g, float) - np.mean(g)
        n = len(g)
        f = np.fft.rfft(g, 2 * n)
        c = np.fft.irfft(f * np.conj(f))[: max_lag + 1] / (n - np.arange(max_lag + 1))
        acf += c / c[0]
    acf /= len(runs)
    below = np.nonzero(acf < 1 / np.e)[0]
    return int(below[0]) if below.size else max_lag


# ------------------------------------------------------------------------------------------ exact chain (test 2)
def state_bits(n_junctions):
    """(2**J, J) uint8 array; entry [s, j] is bit j of the state index s (1 = ON)."""
    return ((np.arange(2 ** n_junctions)[:, None] >> np.arange(n_junctions)[None, :]) & 1).astype(np.uint8)


def transition_matrix(p_on, p_off):
    """Dense one-step matrix T[s, s'] of independent junction flips. p_on[s, j]: probability that junction j, OFF in state s, is
    ON next step; p_off[s, j]: probability that junction j, ON in state s, is OFF next step."""
    S, J = p_on.shape
    bits = state_bits(J)
    T = np.ones((S, S))
    for j in range(J):
        q1 = np.where(bits[:, j] == 1, 1.0 - p_off[:, j], p_on[:, j])       # P(bit j = 1 next | state s)
        T *= np.where(bits[None, :, j] == 1, q1[:, None], 1.0 - q1[:, None])
    return T


def stationary_distribution(T):
    """Stationary distribution of a chain with a single recurrent class: solve (T' - I) pi = 0 with the last equation replaced by
    sum(pi) = 1 (the rows of T' - I sum to zero, so one of them is redundant)."""
    S = T.shape[0]
    A = T.T - np.eye(S)
    A[-1, :] = 1.0
    rhs = np.zeros(S)
    rhs[-1] = 1.0
    pi = np.linalg.solve(A, rhs)
    if pi.min() < -1e-9 or abs(pi.sum() - 1) > 1e-9:
        raise ValueError(f"no valid stationary distribution (min {pi.min():.3e}, sum {pi.sum():.12f})")
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def conjugate_state_map(labels, pi_eps, pi_0, p_min=1e-6):
    """z of every state for an observable given as integer `labels` per state (-1 = outside the analysed range).
    z(label) = P_eps(label) / P_0(label) - 1 where P_0(label) >= p_min; states with an excluded label get z = 0.
    Returns (z_state, excluded_mass_0, excluded_mass_eps)."""
    labels = np.asarray(labels)
    ok_state = labels >= 0
    safe = np.where(ok_state, labels, 0)
    n_lab = int(safe.max()) + 1
    P0 = np.bincount(safe[ok_state], weights=pi_0[ok_state], minlength=n_lab)
    Pe = np.bincount(safe[ok_state], weights=pi_eps[ok_state], minlength=n_lab)
    ok_lab = P0 >= p_min
    z_lab = np.where(ok_lab, Pe / np.where(ok_lab, P0, 1.0) - 1.0, 0.0)
    used = ok_state & ok_lab[safe]
    return np.where(used, z_lab[safe], 0.0), float(1.0 - pi_0[used].sum()), float(1.0 - pi_eps[used].sum())


def exact_curves(T0, pi_eps, pi_0, Z, n_lags):
    """<Z(n)> = pi_eps T0^n z and C_ZZ(n) = sum_s pi_0(s) z(s) (T0^n z)(s) for n = 0..n_lags and every column z of Z (S, m).
    Returns (z_mean, c_zz), each (n_lags + 1, m)."""
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z[:, None]
    z_mean = np.empty((n_lags + 1, Z.shape[1]))
    c_zz = np.empty_like(z_mean)
    v, U, W = pi_eps.copy(), Z.copy(), pi_0[:, None] * Z
    for n in range(n_lags + 1):
        z_mean[n] = v @ Z
        c_zz[n] = np.sum(W * U, axis=0)
        if n < n_lags:
            v, U = v @ T0, T0 @ U
    return z_mean, c_zz


def exact_delta2(z_mean, c_zz, frac=0.05, run=3):
    """(tau_c, delta2 at tau_c, delta2 at tau_c // 2, delta2 at 2 tau_c or NaN if beyond the computed lags, hit_end) for one curve."""
    tau = ft.select_cutoff(z_mean, frac, run)
    n = len(z_mean)
    d = lambda t: ft.delta_squared(c_zz, z_mean, t)
    return tau, d(tau), d(tau // 2), (d(2 * tau) if 2 * tau <= n - 1 else float("nan")), bool(tau == n - 1)
