"""Scores of a feature matrix on every task, for reservoir observables and baselines alike."""
import numpy as np

from . import baselines as B
from .drive import unpack_flags
from .metrics import accuracy, capacity, consistency, nmse, thresholded_sum
from .readout import fit_predict
from .tasks import EMA_LAMBDAS, MAX_LAG, ema_targets, ipc2_targets, memory_targets, narma10_target, split_indices

OBSERVABLES = ("current", "probes", "junctions")
# Features varying less than this over the training segment are ignored. Units are those of observables()
# (mA, V, ON flags), so this is a 10 nA / 10 uV measurement resolution. It matters for probes on regions
# that are connected only through OFF junctions (~nV) during training but become connected later.
MEAS_RES = 1e-5


def observables(raw: dict) -> dict:
    """Feature matrices of the three readout scenarios, in O(1) units (mA, V, ON flags or ON probabilities)."""
    if "junction_x" in raw:                                                   # N10: filling x + y (conductance exponent)
        junctions = np.minimum(1.0, raw["junction_x"].astype(float) + raw["junction_y"].astype(float))
    elif "junction_lambda" in raw:                                            # N4: filament state / lambda_crit
        junctions = raw["junction_lambda"].astype(float)
    elif "junction_p" in raw:                                                 # mean-field: ON probabilities
        junctions = raw["junction_p"].astype(float)
    elif "flags_packed" in raw:
        junctions = unpack_flags(raw).astype(float)
    else:                                                                     # states not stored (default since
        junctions = None                                                      # 2026-10-08) or slimmed after archiving
    out = {
        "current": raw["current"] * 1e3,
        "probes": raw["probe_v"],
    }
    if junctions is not None:
        out["junctions"] = junctions
    if "electrode_roles" in raw:
        # electrode readout of a chip (N10-E): currents of the driven / grounded electrodes and voltages of the
        # floating ones, at every virtual node
        n_e = int(raw["n_electrodes"])
        cols = np.tile(raw["electrode_roles"] != "float", raw["current"].shape[1] // n_e)
        parts = [raw["current"][:, cols] * 1e3] + ([raw["v_float"]] if "v_float" in raw else [])
        out["electrodes"] = np.hstack(parts)
    return out


def iid_targets(u):
    ipc, ipc_names = ipc2_targets(u)
    return memory_targets(u), ipc, ipc_names, narma10_target(u)


def score_iid(F: np.ndarray, u: np.ndarray, floor: float = 0.0, min_std: float = MEAS_RES) -> dict:
    """Memory function (lags 0..MAX_LAG), degree-2 capacities, NARMA-10 NMSE; test scores plus validation
    summaries (used only to pick baseline hyperparameters). Capacity sums skip lag 0 for MC and count only
    values above `floor`."""
    mem, ipc, ipc_names, narma = iid_targets(u)
    ema = ema_targets(u)
    Y = np.hstack([mem, ipc, ema, narma[:, None]])
    train, val, test = split_indices(len(u))
    pred, alphas, val_pred = fit_predict(F, Y, train, val, test, min_std=min_std)
    n_mem, n_ipc = mem.shape[1], ipc.shape[1]
    c_test = capacity(pred[:, :-1], Y[test, :-1])
    c_val = capacity(val_pred[:, :-1], Y[val, :-1])
    mem_c, ipc_c, ema_c = c_test[:n_mem], c_test[n_mem:n_mem + n_ipc], c_test[n_mem + n_ipc:]
    return {
        "memory_function": mem_c.tolist(),
        "lag0": float(mem_c[0]),
        "MC": thresholded_sum(mem_c[1:], floor),
        "ipc2": ipc_c.tolist(),
        "ipc2_names": ipc_names,
        "IPC2": thresholded_sum(ipc_c, floor),
        "ema_capacity": ema_c.tolist(),
        "ema_lambdas": list(EMA_LAMBDAS),
        "narma10_nmse": nmse(pred[:, -1], Y[test, -1]),
        "narma10_pred": pred[:, -1].tolist(),
        "val_MC": float(np.sum(c_val[1:n_mem])),
        "val_IPC2": float(np.sum(c_val[n_mem:n_mem + n_ipc])),
        "val_narma10_nmse": nmse(val_pred[:, -1], Y[val, -1]),
    }


def score_sinesquare(F: np.ndarray, labels: np.ndarray, min_std: float = MEAS_RES) -> dict:
    y = 2.0 * np.asarray(labels) - 1                     # sine -1, square +1
    train, val, test = split_indices(len(y))
    pred, _, val_pred = fit_predict(F, y, train, val, test, score="accuracy", min_std=min_std)
    return {
        "accuracy": accuracy(np.sign(pred.ravel()), y[test]),
        "val_accuracy": accuracy(np.sign(val_pred.ravel()), y[val]),
    }


def noise_floor(u: np.ndarray, seeds=(0, 1, 2), dims=(32, 1000)) -> float:
    """Largest held-out capacity the readout finds in Gaussian-noise features (over all targets, seeds, dims)."""
    mem, ipc, _, _ = iid_targets(u)
    Y = np.hstack([mem, ipc])
    train, val, test = split_indices(len(u))
    worst = 0.0
    for d in dims:
        for s in seeds:
            pred, _, _ = fit_predict(B.noise_features(len(u), d, s), Y, train, val, test)
            worst = max(worst, float(capacity(pred, Y[test]).max()))
    return worst


# ------------------------------------------------------------------ baselines
def _esn_grid_scores(u, n_units, seed, score_fn):
    """Scores of the ESN at every grid point (spectral radius, input scaling)."""
    return [{**score_fn(B.esn_states(u, n_units, rho, sigma, seed)), "rho": rho, "sigma": sigma}
            for rho, sigma in B.ESN_GRID]


def _pick(scores, key, maximize):
    vals = [s[key] for s in scores]
    return scores[int(np.argmax(vals) if maximize else np.argmin(vals))]


def iid_baselines(u, floor, esn_sizes=(32, 200), esn_seeds=(0, 1, 2)) -> dict:
    """Baseline scores on the i.i.d. stream. ESN hyperparameters are chosen per task on validation; ESN scores
    are means over esn_seeds."""
    out = {
        "input_only": score_iid(B.input_only(u), u, floor),
        "delay_line_32": score_iid(B.delay_line(u, 32), u, floor),
        "quadratic_delay_8": score_iid(B.quadratic_delay_line(u, 8), u, floor),
    }
    select = {"MC": ("val_MC", True), "IPC2": ("val_IPC2", True), "narma10_nmse": ("val_narma10_nmse", False)}
    for n in esn_sizes:
        grids = [_esn_grid_scores(u, n, seed, lambda F: score_iid(F, u, floor)) for seed in esn_seeds]
        agg = {m: float(np.mean([_pick(g, key, mx)[m] for g in grids])) for m, (key, mx) in select.items()}
        best = _pick(grids[0], "val_MC", True)        # memory function of the MC-selected ESN, for figures
        agg.update(memory_function=best["memory_function"], lag0=best["lag0"],
                   ema_capacity=best["ema_capacity"], rho=best["rho"], sigma=best["sigma"])
        out[f"esn_{n}"] = agg
    return out


def sinesquare_baselines(u, labels, esn_sizes=(32, 200), esn_seeds=(0, 1, 2)) -> dict:
    out = {
        "input_only": score_sinesquare(B.input_only(u), labels),
        "delay_line_32": score_sinesquare(B.delay_line(u, 32), labels),
        "quadratic_delay_8": score_sinesquare(B.quadratic_delay_line(u, 8), labels),
    }
    for n in esn_sizes:
        accs = [_pick(_esn_grid_scores(u, n, seed, lambda F: score_sinesquare(F, labels)),
                      "val_accuracy", True)["accuracy"] for seed in esn_seeds]
        out[f"esn_{n}"] = {"accuracy": float(np.mean(accs))}
    return out


# ------------------------------------------------------------------ reservoir runs
def evaluate_iid_run(raw: dict, floor: float) -> dict:
    u = raw["u"]
    return {name: score_iid(F, u, floor) for name, F in observables(raw).items()}


def evaluate_sinesquare_run(raw: dict) -> dict:
    labels = raw["labels"]
    return {name: score_sinesquare(F, labels) for name, F in observables(raw).items()}


def state_observables(raw: dict) -> dict:
    """Observables with the trivial instantaneous echo of the input removed, so that they measure the network STATE.

    Two electrodes: conductance I/V_in and probe transfer ratios V_probe/V_in, which are independent of V_in for
    fixed junction states (the circuit is linear). Several driven electrodes (N5): the voltages depend on the
    whole vector of electrode voltages, so the part of each feature that is a linear function of the current
    electrode voltages is regressed out and the residual is kept. Junction states carry no instantaneous echo."""
    F = observables(raw)
    if "v_electrodes" in raw:
        X = np.hstack([raw["v_electrodes"], np.ones((len(raw["v_electrodes"]), 1))])

        def residual(A):
            coef, *_ = np.linalg.lstsq(X, A, rcond=None)
            return A - X @ coef
        out = {"current": residual(F["current"]), "probes": residual(F["probes"])}
        if "junctions" in F:
            out["junctions"] = F["junctions"]
        if "electrodes" in F:
            out["electrodes"] = residual(F["electrodes"])
        return out
    v = raw["v_in"][:, None]
    out = {"current": F["current"] / v, "probes": F["probes"] / v}
    if "junctions" in F:
        out["junctions"] = F["junctions"]
    return out


def evaluate_consistency(raw_a: dict, raw_b: dict) -> dict:
    """Consistency between two replicates driven by the same input, of each observable ("current", ...), of its
    state part ("current_state", ...; see state_observables), and of the sample-to-sample INCREMENTS of the state
    part ("current_state_inc", ...).

    The whole-series versions also reward a slow trend shared by both replicates (e.g. both still relaxing from
    their pre-drives), which is not a response to the input; the increment version removes such trends and is
    the one to read as the echo-state property (review 2026-09-27, scripts/review_checks.py)."""
    if not np.array_equal(raw_a["u"], raw_b["u"]):
        raise ValueError("replicates were not driven by the same input")
    out = {}
    for suffix, fn in (("", observables), ("_state", state_observables)):
        Fa, Fb = fn(raw_a), fn(raw_b)
        for name in [n for n in (*OBSERVABLES, "electrodes") if n in Fa]:
            a, b = Fa[name][MAX_LAG:], Fb[name][MAX_LAG:]
            out[name + suffix] = consistency(a, b, min_std=MEAS_RES)
            if suffix == "_state":
                out[name + "_state_inc"] = consistency(np.diff(a, axis=0), np.diff(b, axis=0), min_std=MEAS_RES)
    return out
