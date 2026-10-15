"""Input streams (values in [0, 1]) and task targets.

Convention: sample t of a feature matrix is the reservoir state after input u[t] has been applied, so a target
at row t may depend on u[0..t] only (MC: u[t-k]; NARMA-10: y[t+1], which depends on s[0..t]).
"""
import numpy as np

MAX_LAG = 30            # longest lag scored; the first MAX_LAG samples are never scored
IPC_P2_LAGS = 11        # P2(u[t-k]) for k = 0..10
IPC_CROSS_MAX = 5       # u[t-k] u[t-l] for 0 <= k < l <= 5
SINESQUARE_PERIOD = 8


# ---------------------------------------------------------------- streams
def iid_stream(n: int, seed: int) -> np.ndarray:
    """i.i.d. U(0, 1) inputs."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, n)


def sinesquare_stream(n: int, seed: int, period: int = SINESQUARE_PERIOD):
    """Random concatenation of whole sine and square periods (Paquot et al. 2012). Returns (u in [0, 1],
    labels with 0 = sine, 1 = square). Sine samples sin(2 pi k/period); square: +1 for the first half period,
    -1 for the second."""
    rng = np.random.default_rng(seed)
    k = np.arange(period)
    sine = np.sin(2 * np.pi * k / period)
    square = np.where(k < period // 2, 1.0, -1.0)
    n_periods = -(-n // period)
    kinds = rng.integers(0, 2, n_periods)
    x = np.concatenate([square if c else sine for c in kinds])[:n]
    labels = np.repeat(kinds, period)[:n]
    return (x + 1) / 2, labels


# ---------------------------------------------------------------- targets
def delayed(u: np.ndarray, k: int) -> np.ndarray:
    """Row t holds u[t-k] (NaN where t < k)."""
    out = np.full(len(u), np.nan)
    out[k:] = u[:len(u) - k] if k else u
    return out


def legendre_input(u: np.ndarray) -> np.ndarray:
    """Map U(0, 1) inputs to x = 2u - 1, uniform on [-1, 1] (the Legendre measure)."""
    return 2 * np.asarray(u) - 1


def p1(x):
    return np.sqrt(3.0) * x


def p2(x):
    return np.sqrt(5.0) * (3 * x ** 2 - 1) / 2


def memory_targets(u: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """(n, max_lag + 1): column k holds u[t-k]."""
    return np.stack([delayed(u, k) for k in range(max_lag + 1)], axis=1)


def ipc2_targets(u: np.ndarray):
    """Degree-2 information-processing-capacity targets (Dambre et al. 2012) on x = 2u - 1: P2(x[t-k]) for
    k < IPC_P2_LAGS and P1(x[t-k]) P1(x[t-l]) for 0 <= k < l <= IPC_CROSS_MAX. Orthonormal under U(-1, 1).
    Returns (targets (n, m), names)."""
    x = legendre_input(u)
    cols, names = [], []
    for k in range(IPC_P2_LAGS):
        cols.append(p2(delayed(x, k)))
        names.append(f"P2[{k}]")
    for k in range(IPC_CROSS_MAX + 1):
        for l in range(k + 1, IPC_CROSS_MAX + 1):
            cols.append(p1(delayed(x, k)) * p1(delayed(x, l)))
            names.append(f"P1[{k}]P1[{l}]")
    return np.stack(cols, axis=1), names


EMA_LAMBDAS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995)


def ema_targets(u: np.ndarray, lambdas=EMA_LAMBDAS) -> np.ndarray:
    """(n, len(lambdas)): column j holds z[t] = (1 - lam) sum_{k>=0} lam^k u[t-k] (exponential moving average,
    memory time 1/(1 - lam) samples), started from 0.5 (the mean of U(0, 1)). A leaky integrator of time constant tau
    reconstructs the column with lam = exp(-1/tau) perfectly, while its memory function per lag stays small."""
    u = np.asarray(u, dtype=float)
    out = np.empty((len(u), len(lambdas)))
    for j, lam in enumerate(lambdas):
        z = 0.5
        for t, ut in enumerate(u):
            z = lam * z + (1 - lam) * ut
            out[t, j] = z
    return out


def narma10(s: np.ndarray) -> np.ndarray:
    """NARMA-10 driven by s in [0, 0.5]:
    y[t+1] = 0.3 y[t] + 0.05 y[t] sum_{i=0}^{9} y[t-i] + 1.5 s[t-9] s[t] + 0.1, with y = 0 before t = 10.
    Returns y of length len(s) + 1 (y[t+1] uses s[0..t])."""
    s = np.asarray(s, dtype=float)
    y = np.zeros(len(s) + 1)
    for t in range(9, len(s)):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9:t + 1]) + 1.5 * s[t - 9] * s[t] + 0.1
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("NARMA-10 diverged for this input")
    return y


def narma10_target(u: np.ndarray) -> np.ndarray:
    """Row t holds y[t+1] of NARMA-10 driven by s = 0.5 u (depends on u[0..t])."""
    return narma10(0.5 * np.asarray(u))[1:]


# ---------------------------------------------------------------- split
def split_indices(n: int, fractions=(0.6, 0.2, 0.2), start: int = MAX_LAG):
    """Train / validation / test index arrays, in time order, over rows start..n-1."""
    idx = np.arange(start, n)
    a = int(round(fractions[0] * len(idx)))
    b = a + int(round(fractions[1] * len(idx)))
    return idx[:a], idx[a:b], idx[b:]
