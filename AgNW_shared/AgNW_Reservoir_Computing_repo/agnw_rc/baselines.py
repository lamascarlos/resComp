"""Reference feature sets with no nanowires, computed from the same input stream."""
import numpy as np

from .tasks import delayed


def input_only(u: np.ndarray) -> np.ndarray:
    """The current input alone: what a readout gets 'for free' from any observable proportional to V_in."""
    return np.asarray(u, dtype=float)[:, None]


def delay_line(u: np.ndarray, n_taps: int = 32) -> np.ndarray:
    """u[t], u[t-1], ..., u[t-n_taps+1] (zeros before the start): perfect linear memory of n_taps samples and
    no nonlinearity."""
    return np.nan_to_num(np.stack([delayed(u, k) for k in range(n_taps)], axis=1))


def quadratic_delay_line(u: np.ndarray, n_taps: int = 8) -> np.ndarray:
    """delay_line plus all pairwise products of its taps: a memory + degree-2 nonlinearity reference."""
    D = delay_line(u, n_taps)
    i, j = np.triu_indices(n_taps)
    return np.hstack([D, D[:, i] * D[:, j]])


def esn_states(u: np.ndarray, n_units: int, spectral_radius: float, input_scaling: float, seed: int,
               bias_scaling: float = 0.2) -> np.ndarray:
    """Echo-state network x[t] = tanh(W x[t-1] + w_in (2u[t]-1) + b), W dense Gaussian rescaled to the given
    spectral radius. Row t is the state after input u[t]."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_units, n_units))
    W *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W)))
    w_in = input_scaling * rng.uniform(-1, 1, n_units)
    b = bias_scaling * rng.uniform(-1, 1, n_units)
    x = np.zeros(n_units)
    X = np.empty((len(u), n_units))
    for t, ut in enumerate(2 * np.asarray(u, dtype=float) - 1):
        x = np.tanh(W @ x + w_in * ut + b)
        X[t] = x
    return X


ESN_GRID = [(rho, sigma) for rho in (0.5, 0.9, 1.1) for sigma in (0.1, 0.5, 1.5)]


def noise_features(n: int, n_features: int, seed: int) -> np.ndarray:
    """Gaussian noise: estimates how much capacity a readout finds in features unrelated to the input."""
    return np.random.default_rng(seed).standard_normal((n, n_features))
