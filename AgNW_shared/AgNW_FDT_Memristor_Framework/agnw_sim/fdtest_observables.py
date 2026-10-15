"""Alternative observables of the FDTest (pure numpy, no I/O; docs/alt_observables_design.md).

  * on_fraction: the ON fraction m_K of the K recorded junctions (a plain average of the states);
  * fit_lda_direction / crossfit_scores: the linear combination s = w . g of the recorded junction states that best
    separates the perturbed stage II from the unperturbed stage IV (regularised linear discriminant), with the weights
    always applied to cycles other than those they were fitted on (cross-fitting), because fitting and evaluating on
    the same cycles manufactures a response out of noise.

State arrays have the shape (n_cycles, T, K): cycle, recorded time index, junction. Only strided samples enter the
covariance (about every 30th record: the samples are autocorrelated and K is large), accumulated in chunks of cycles so
that no float copy of a whole stage is ever made (stage arrays are uint8 and can be several GB).
"""
import numpy as np

from agnw_sim.fdtest_protocol import achieved_response

_ROWS_PER_CHUNK = 20000            # strided samples (rows of K junctions) converted to float at a time


def on_fraction(g):
    """m_K: mean over the last (junction) axis of a (..., K) array, in float64 (no float copy of an integer array)."""
    return np.mean(g, axis=-1, dtype=np.float64)


def _moments(g, stride, ref):
    """(n, sum, gram) of the strided samples of g (n_cycles, T, K) shifted by the reference row `ref` (K,): n rows,
    sum of (x - ref), and the sum of the outer products (x - ref)(x - ref)^T. The shift keeps constant columns exactly
    zero and the cancellation in the covariance small. Chunks of cycles bound the memory."""
    n_c, K = g.shape[0], g.shape[2]
    per_cycle = len(range(0, g.shape[1], stride))
    step = max(1, _ROWS_PER_CHUNK // max(per_cycle, 1))
    n, s, gram = 0, np.zeros(K), np.zeros((K, K))
    for lo in range(0, n_c, step):
        x = np.asarray(g[lo:lo + step, ::stride, :], dtype=np.float64).reshape(-1, K) - ref
        n += x.shape[0]
        s += x.sum(axis=0)
        gram += x.T @ x
    return n, s, gram


def _project(g, w):
    """w . g for g (n_cycles, T, K): (n_cycles, T) float64, one cycle at a time (no float copy of the whole array)."""
    out = np.empty(g.shape[:2])
    for c in range(g.shape[0]):
        out[c] = np.asarray(g[c], dtype=np.float64) @ w
    return out


def fit_lda_direction(g2, g4, ridge_frac=0.1, stride=30):
    """Regularised linear discriminant of stage II (perturbed) against stage IV (unperturbed), g2 and g4 (n_cycles, T, K).

    w is proportional to (Sigma + lam I)^-1 (mu_eps - mu_0) with mu_eps, mu_0 the mean state vectors of the STRIDED samples
    (every `stride`-th record) of stages II and IV, Sigma their pooled within-stage covariance and lam = ridge_frac *
    trace(Sigma) / K (lam = 1 if Sigma vanishes, so constant columns never make the system singular and get weight 0).
    w is scaled to unit Euclidean norm and (Sigma + lam I) is positive definite, so the mean response (mu_eps - mu_0) . w
    is positive by construction. Returns (w, mu0, sd0) with mu0 and sd0 the mean and standard deviation of w . g over ALL
    samples of g4, so that the standardised score is (w . g - mu0) / sd0. With no information (mu_eps = mu_0) w = 0,
    sd0 = 0 and the score is undefined."""
    g2, g4 = np.asarray(g2), np.asarray(g4)
    if stride < 1 or ridge_frac < 0:
        raise ValueError("stride >= 1 and ridge_frac >= 0 are required")
    K = g2.shape[2]
    ref = np.asarray(g2[0, 0], dtype=np.float64)
    n2, s2, q2 = _moments(g2, stride, ref)
    n4, s4, q4 = _moments(g4, stride, ref)
    m2, m4 = s2 / n2, s4 / n4
    sigma = (q2 - n2 * np.outer(m2, m2) + q4 - n4 * np.outer(m4, m4)) / max(n2 + n4 - 2, 1)
    trace = float(np.trace(sigma))
    lam = ridge_frac * trace / K if trace > 0 else 1.0
    w = np.linalg.solve(sigma + lam * np.eye(K), m2 - m4)
    norm = np.linalg.norm(w)
    w = w / norm if norm > 0 else np.zeros(K)
    s = _project(g4, w)
    return w, float(s.mean()), float(s.std())


def _score(g, w, mu0, sd0):
    """Standardised score (w . g - mu0) / sd0, (n_cycles, T); NaN everywhere if sd0 is 0 (no information in w)."""
    if not sd0 > 0:
        return np.full(g.shape[:2], np.nan)
    return (_project(g, w) - mu0) / sd0


def crossfit_scores(g2, g3, g4, rng, ridge_frac=0.1, stride=30):
    """Cross-fitted scores of the optimal linear observable. The cycles are split at random (rng.permutation) into fold 0
    (the first n // 2 of the permutation) and fold 1 (the rest, one more cycle if n is odd); the weights fitted on fold f
    are applied to the cycles of the OTHER fold. Returns (x2, x3, x4, info): the score arrays of stages II, III
    (from the removal instant) and IV in the ORIGINAL cycle order, shapes (n_cycles, T2), (n_cycles, n_lags),
    (n_cycles, T4).

    Standardisation: the scores of every evaluation fold (stages II, III and IV of its cycles) are standardised with that
    fold's OWN stage-IV mean and standard deviation of w . g, where w comes from the other fold (those cycles were not used
    to fit w). Standardising with the fit fold's stage-IV statistics would be wrong: the weights are fitted on those very
    samples, so they are inflated in-sample and the held-out stage-IV scores would come out with a mean far from 0 and a std
    above 1, differently for the two folds (pooling the folds would then mix two affine maps). With the fold-own statistics
    the stage-IV scores of each fold, and of the pooled series, have mean 0 and std 1 exactly.

    info: w (2, K): the weights fitted on fold 0 and on fold 1; mu0, sd0 (2,): the in-fit stage-IV standardisation of those
    weights (fit fold f; as returned by fit_lda_direction, NOT applied to the scores); mu_eval, sd_eval (2,): the
    standardisation actually applied to the cycles of fold f (weights of fold 1 - f); fold (n_cycles,): the fold of every cycle;
    response_in_sample (2,): per fold, the mean stage II minus stage IV score of ITS OWN cycles with ITS OWN weights, in units of
    the stage IV standard deviation (inflated by the fit: the leakage the cross-fitting removes); response_crossfit: the same
    for the pooled cross-fitted scores (the honest response). A fold whose evaluation stage-IV scores have no spread (or whose
    weights are 0) gets NaN scores."""
    g2, g3, g4 = np.asarray(g2), np.asarray(g3), np.asarray(g4)
    n = g2.shape[0]
    if n < 2:
        raise ValueError("cross-fitting needs at least 2 cycles")
    perm = np.asarray(rng.permutation(n))
    folds = (perm[: n // 2], perm[n // 2:])
    fold = np.empty(n, dtype=int)
    for f, idx in enumerate(folds):
        fold[idx] = f
    fits = [fit_lda_direction(g2[idx], g4[idx], ridge_frac, stride) for idx in folds]
    x2, x3, x4 = (np.empty(g.shape[:2]) for g in (g2, g3, g4))
    mu_eval, sd_eval = np.empty(2), np.empty(2)
    for f, idx in enumerate(folds):
        w = fits[1 - f][0]                                             # the weights of the other fold
        s4 = _project(g4[idx], w)
        mu_eval[f], sd_eval[f] = s4.mean(), s4.std()
        for x, g, proj in ((x2, g2, None), (x3, g3, None), (x4, g4, s4)):
            if not sd_eval[f] > 0:
                x[idx] = np.nan
            else:
                x[idx] = ((proj if proj is not None else _project(g[idx], w)) - mu_eval[f]) / sd_eval[f]
    in_sample = [achieved_response(_score(g2[idx], *fits[f]), _score(g4[idx], *fits[f]))["achieved_response_sigma"]
                 for f, idx in enumerate(folds)]
    info = {"w": np.stack([f[0] for f in fits]), "mu0": np.array([f[1] for f in fits]), "sd0": np.array([f[2] for f in fits]),
            "mu_eval": mu_eval, "sd_eval": sd_eval, "fold": fold, "response_in_sample": np.array(in_sample),
            "response_crossfit": achieved_response(x2, x4)["achieved_response_sigma"]}
    return x2, x3, x4, info
