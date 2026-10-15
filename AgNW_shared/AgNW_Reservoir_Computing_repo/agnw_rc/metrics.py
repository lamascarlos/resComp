"""Scores: NMSE, capacity, accuracy, consistency, capacity noise floor."""
import numpy as np


def nmse(pred, target) -> float:
    pred = np.asarray(pred, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    var = np.var(target)
    if not var > 0:
        raise ValueError("target has zero variance; NMSE is undefined")
    return float(np.mean((pred - target) ** 2) / var)


def capacity(pred, target) -> np.ndarray:
    """Per-column capacity C = max(0, 1 - NMSE) on held-out data (Dambre et al. 2012). 1 = perfect
    reconstruction, 0 = no better than the mean."""
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if pred.ndim == 1:
        pred, target = pred[:, None], target[:, None]
    mse = np.mean((pred - target) ** 2, axis=0)
    var = np.var(target, axis=0)
    return np.clip(1 - mse / var, 0, None)


def thresholded_sum(caps, floor: float) -> float:
    """Sum of the capacities that exceed the noise floor (the rest count as 0)."""
    caps = np.asarray(caps, dtype=float)
    return float(np.sum(np.where(caps > floor, caps, 0.0)))


def accuracy(pred_labels, labels) -> float:
    return float(np.mean(np.asarray(pred_labels) == np.asarray(labels)))


def consistency(F_a: np.ndarray, F_b: np.ndarray, min_std: float = 1e-9) -> float:
    """Mean over features of the Pearson correlation between two replicate responses to the SAME input (different
    switching noise and initial state). 1 = the response is a deterministic function of the input history
    (echo-state property); 0 = unrelated. Features that are constant in either replicate are skipped."""
    F_a = np.asarray(F_a, dtype=float)
    F_b = np.asarray(F_b, dtype=float)
    sa, sb = F_a.std(axis=0), F_b.std(axis=0)
    ok = (sa > min_std) & (sb > min_std)
    if not ok.any():
        return float("nan")
    za = (F_a[:, ok] - F_a[:, ok].mean(axis=0)) / sa[ok]
    zb = (F_b[:, ok] - F_b[:, ok].mean(axis=0)) / sb[ok]
    return float(np.mean(np.mean(za * zb, axis=0)))
