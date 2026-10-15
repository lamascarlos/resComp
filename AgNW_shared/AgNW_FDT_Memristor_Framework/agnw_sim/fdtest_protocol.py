"""FDTest protocol helpers (pure functions): chained-cycle schedule and stage extraction, operating-point
selection from a fixed-V pilot, junction properties for the hub analogue, and acceptance-gate checks.
No I/O, no plotting. See docs/superpowers/specs/2026-09-20-fdtest-measurement-design.md."""
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------- schedule and stages
@dataclass(frozen=True)
class Schedule:
    """Chained FDTest cycles. Steps per cycle: stage I (perturbation on, transient) T_relax, II (on, sampled)
    T_samp, III (off, transient) T_relax, IV (off, sampled) T_samp. Recorded samples are taken every
    `record_every` steps; `n_lags` counts recorded samples of the stage III relaxation."""
    V0: float
    eps: float
    T_relax: int
    T_samp: int
    n_cycles: int
    record_every: int = 10
    n_lags: int = 300

    def __post_init__(self):
        r = self.record_every
        if r < 1 or self.n_cycles < 1 or self.n_lags < 2:
            raise ValueError("record_every >= 1, n_cycles >= 1 and n_lags >= 2 are required")
        if self.T_relax % r or self.T_samp % r:
            raise ValueError("T_relax and T_samp must be multiples of record_every")
        if self.n_lags * r > self.T_relax:
            raise ValueError("n_lags * record_every must not exceed T_relax (the stage III length)")
        if self.n_lags > self.T_samp // r:
            raise ValueError("n_lags must not exceed T_samp / record_every (the autocorrelation lags come from stage IV)")

    @property
    def T_cycle(self):
        return 2 * self.T_relax + 2 * self.T_samp

    @property
    def n_steps(self):
        return self.n_cycles * self.T_cycle

    def voltage(self, step):
        """V_in applied at solver step `step`: V0 + eps during stages I and II, V0 otherwise."""
        return self.V0 + self.eps if (step % self.T_cycle) < self.T_relax + self.T_samp else self.V0

    def waveform_fn(self, dt):
        """Waveform for run_simulation (which passes the time t = step * dt)."""
        return lambda t: self.voltage(int(round(t / dt)))

    def snapshot_steps(self):
        """Steps at the end of every stage II and stage IV (full junction states are stored there)."""
        out = []
        for c in range(self.n_cycles):
            base = c * self.T_cycle
            out += [base + self.T_relax + self.T_samp - 1, base + self.T_cycle - 1]
        return out


def extract_stage_arrays(series, schedule):
    """Cut a recorded series (n_steps // record_every, ...) into per-cycle stage arrays (cycle axis first):
    x2 (n_cycles, T_samp / r, ...) stage II; x3 (n_cycles, n_lags, ...) from the removal instant on (lag 0 is the
    state at the removal, distributed as P_eps); x4 (n_cycles, T_samp / r, ...) stage IV."""
    s, r = schedule, schedule.record_every
    series = np.asarray(series)
    if series.shape[0] != s.n_steps // r:
        raise ValueError(f"series has {series.shape[0]} records, expected {s.n_steps // r}")
    n2 = s.T_samp // r
    x2, x3, x4 = [], [], []
    for c in range(s.n_cycles):
        i2 = c * s.T_cycle // r + s.T_relax // r
        i3 = i2 + n2
        i4 = i3 + s.T_relax // r
        x2.append(series[i2:i2 + n2])
        x3.append(series[i3:i3 + s.n_lags])
        x4.append(series[i4:i4 + n2])
    return np.stack(x2), np.stack(x3), np.stack(x4)


def extract_switch_on_arrays(series, schedule):
    """First n_lags records of stage I (perturbation just switched on) of every cycle: (n_cycles, n_lags, ...), lag 0 =
    the record at the cycle start c * T_cycle / record_every. That state is the end of the previous stage IV, i.e. the
    unperturbed steady state P_0, EXCEPT in cycle 0 of a run, which starts from the all-OFF initial condition: analyses
    of the switch-on transient must drop cycle 0 of every run (the drivers do)."""
    s, r = schedule, schedule.record_every
    series = np.asarray(series)
    if series.shape[0] != s.n_steps // r:
        raise ValueError(f"series has {series.shape[0]} records, expected {s.n_steps // r}")
    return np.stack([series[c * s.T_cycle // r:c * s.T_cycle // r + s.n_lags] for c in range(s.n_cycles)])


# ----------------------------------------------------------------------------- operating point
def pilot_summary(I_total, V_in, G_sample, block=5000, tail=30000, on_threshold=5e-4):
    """Stationarity and saturation summary of one fixed-V pilot run (uses the last `tail` steps).
    drift = |first block mean - last block mean| / |mean| of g_eff = I / V_in; p_on = per-junction ON fraction
    (a sample counts as ON if it exceeds on_threshold: conductances or 0/1 flags both work)."""
    g_eff = np.asarray(I_total, dtype=float) / np.asarray(V_in, dtype=float)
    tail_g = g_eff[-tail:]
    n_blocks = len(tail_g) // block
    if n_blocks < 2:
        raise ValueError("need at least two blocks in the tail to measure the drift")
    means = np.array([tail_g[i * block:(i + 1) * block].mean() for i in range(n_blocks)])
    drift = float(abs(means[-1] - means[0]) / abs(means.mean())) if means.mean() != 0 else float("nan")
    samples = np.asarray(G_sample)
    per_step = max(len(g_eff) // len(samples), 1)                       # recording interval of G_sample
    p_on = (samples[-max(tail // per_step, 1):] > on_threshold).mean(axis=0)
    return {"mean_geff": float(tail_g.mean()), "sigma_geff": float(tail_g.std()), "drift": drift,
            "frac_intermediate": float(np.mean((p_on > 0.1) & (p_on < 0.9))), "p_on": p_on}


def choose_v0(summaries, drift_tol=0.02):
    """summaries: {V: pilot_summary}. Among the stationary voltages (drift <= drift_tol) the one with the
    largest fraction of junctions of intermediate ON probability (least saturated). None if none is stationary."""
    ok = {v: s for v, s in summaries.items() if np.isfinite(s["drift"]) and s["drift"] <= drift_tol}
    if not ok:
        return None
    return max(ok, key=lambda v: ok[v]["frac_intermediate"])


def choose_epsilon(volts, mean_geff, sigma_geff, V0, target_sigma=0.5):
    """Perturbation eps whose mean g_eff response at V0 equals target_sigma * sigma_geff, from the piecewise-linear
    slope of the pilot curve mean_geff(V) around V0 (revision criterion: response of 0.3-1 sigma)."""
    v = np.asarray(volts, dtype=float)
    m = np.asarray(mean_geff, dtype=float)
    order = np.argsort(v)
    v, m = v[order], m[order]
    i = int(np.searchsorted(v, V0))
    lo, hi = max(i - 1, 0), min(i + 1, len(v) - 1)
    slope = (m[hi] - m[lo]) / (v[hi] - v[lo])
    if not np.isfinite(slope) or slope == 0:
        raise ValueError("g_eff does not respond to V around V0: cannot choose eps")
    return float(target_sigma * sigma_geff / abs(slope))


# ----------------------------------------------------------------------------- junction properties
def junction_strength(network):
    """strength_j = deg(a) + deg(b) - 2 for junction j between wires a and b: the number of OTHER junctions on its
    two wires (the analogue of node strength)."""
    deg = np.bincount(network.junctions.ravel(), minlength=network.n_wires)
    return deg[network.junctions[:, 0]] + deg[network.junctions[:, 1]] - 2


def current_share(circuit, snapshots, V0, g_on=1e-3, g_off=1e-9, threshold=1e-3):
    """Per junction, over the given ON/OFF snapshots (n_snap, n_junctions): the mean of |I_mem| / I_total, and
    the fraction of snapshots in which the junction is ON and carries more than threshold * I_total."""
    snapshots = np.asarray(snapshots)
    share = np.zeros(snapshots.shape[1])
    backbone = np.zeros_like(share)
    for state in snapshots:
        G = np.where(state > 0, g_on, g_off)
        V = circuit.solve(G, V0)
        i_mem = np.abs(G * circuit.junction_voltage_drops(V))
        i_total = abs(circuit.total_current(V))
        share += i_mem / i_total
        backbone += (state > 0) & (i_mem > threshold * i_total)
    return share / len(snapshots), backbone / len(snapshots)


def junction_classes(strength, backbone):
    """Boolean class masks (5, K) and names: strength terciles (low / mid / high) and backbone (majority of
    snapshots) versus non-backbone."""
    t1, t2 = np.quantile(strength, [1 / 3, 2 / 3])
    on = np.asarray(backbone) >= 0.5
    masks = np.stack([strength <= t1, (strength > t1) & (strength <= t2), strength > t2, on, ~on])
    return masks, ["strength low", "strength mid", "strength high", "backbone", "non-backbone"]


# ----------------------------------------------------------------------------- achieved perturbation
def achieved_response(x2, x4):
    """A-posteriori strength of the perturbation on the network observable: achieved_response_sigma =
    (mean(x2) - mean(x4)) / std(x4) (signed, pooled over cycles and samples; NaN if x4 has no spread) and
    overlap_fraction = share of stage II samples inside the central 95 % of the stage IV distribution."""
    a, b = np.asarray(x2, dtype=float).ravel(), np.asarray(x4, dtype=float).ravel()
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"achieved_response_sigma": float("nan"), "overlap_fraction": float("nan")}
    sd = b.std()
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"achieved_response_sigma": float((a.mean() - b.mean()) / sd) if sd > 0 else float("nan"),
            "overlap_fraction": float(np.mean((a >= lo) & (a <= hi)))}


# ----------------------------------------------------------------------------- acceptance gate
def gate_drift(x4, n_blocks=3, tol=0.02):
    """(a) drift of the cycle-averaged stage IV g_eff over n_blocks consecutive time blocks: (max - min) / |mean|."""
    mean_curve = np.asarray(x4, dtype=float).mean(axis=0)
    means = np.array([b.mean() for b in np.array_split(mean_curve, n_blocks)])
    value = float((means.max() - means.min()) / abs(means.mean())) if means.mean() != 0 else float("nan")
    return {"passed": bool(np.isfinite(value) and value <= tol), "value": value, "tol": tol}


def gate_halves(x4, t_max=3.0, min_fraction=0.95):
    """(b) P_0 consistency: per series (g_eff, or every junction), the first and second half of stage IV give
    the same mean. Paired t statistic of the per-CYCLE half means (cycles are independent; the time samples
    inside a cycle are not, which is why a KS test on pooled samples would over-reject). passed if at least
    min_fraction of the series have |t| < t_max."""
    x = np.asarray(x4, dtype=float)
    h = x.shape[1] // 2
    d = x[:, :h].mean(axis=1) - x[:, h:2 * h].mean(axis=1)
    se = d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d.mean(axis=0) / se
    ok = np.where(se > 0, np.abs(t) < t_max, d.mean(axis=0) == 0)
    fraction = float(np.mean(ok))
    return {"passed": bool(fraction >= min_fraction), "value": fraction, "t_max": t_max}


def gate_odd_even(d2_odd, ci_odd, d2_even, ci_even):
    """(c) independence of the chained cycles: each of the odd-cycle and even-cycle estimates lies within the
    other's bootstrap interval."""
    ok = bool(ci_even[0] <= d2_odd <= ci_even[1] and ci_odd[0] <= d2_even <= ci_odd[1])
    return {"passed": ok, "odd": float(d2_odd), "even": float(d2_even)}


def gate_scaling(d2_by_cycles, floor=0.0):
    """(d) finite-M behaviour: delta2 computed from increasing numbers of cycles ends below where it started, or already
    at (twice) the noise floor. This is an endpoint sanity check (delta2 must not grow with more cycles, or must sit
    at the floor), NOT a test of 1/M scaling."""
    v = np.asarray(d2_by_cycles, dtype=float)
    ok = bool(v[-1] < v[0] or v[-1] <= 2 * floor)
    return {"passed": ok, "first": float(v[0]), "last": float(v[-1])}


def gate_power(z0_bootstrap, min_snr=3.0):
    """(e) power: signal-to-noise of <Z(0)> (bootstrap mean over bootstrap standard deviation). Fails closed: fewer
    than 2 samples, any non-finite sample, or zero mean with zero spread (no signal) gives passed False. A nonzero
    mean with exactly zero spread is a legitimate pass (snr = inf)."""
    z = np.asarray(z0_bootstrap, dtype=float).ravel()
    if z.size < 2 or not np.isfinite(z).all():
        return {"passed": False, "value": float("nan"), "min_snr": min_snr}
    mean, std = float(z.mean()), float(z.std(ddof=1))
    if std > 0:
        snr = abs(mean) / std
    else:
        snr = float("inf") if mean != 0 else float("nan")
    return {"passed": bool(snr >= min_snr), "value": snr, "min_snr": min_snr}
