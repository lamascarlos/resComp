"""Analysis functions for AgNW simulation trajectories: autocorrelation,
hysteresis loop area, entropy production (framework §6.2, §8)."""
import numpy as np


def normalize_conductance(G, G_off, G_on):
    """Map conductance values to the dimensionless order parameter
    g = (G - G_off) / (G_on - G_off), per framework §4.2."""
    return (G - G_off) / (G_on - G_off)


def autocorrelation(traces, max_lag):
    """Ensemble-averaged autocorrelation C(tau) = <dg(0) dg(tau)> via FFT
    (Wiener-Khinchin), unbiased per-lag normalization (framework §6.2).

    Args:
        traces: (n_steps, n_series) array -- one column per series
            (e.g. one per sampled junction).
        max_lag: maximum lag in samples; must be < n_steps.

    Returns:
        (max_lag + 1,) array: C[0]..C[max_lag].
    """
    traces = np.asarray(traces, dtype=float)
    n_steps, n_series = traces.shape
    if max_lag >= n_steps:
        raise ValueError(f"max_lag ({max_lag}) must be < n_steps ({n_steps})")
    if n_series == 0:
        return np.full(max_lag + 1, np.nan)

    demeaned = traces - traces.mean(axis=0, keepdims=True)

    n_fft = 1
    while n_fft < 2 * n_steps:
        n_fft *= 2

    counts = n_steps - np.arange(max_lag + 1)
    C_per_series = np.empty((max_lag + 1, n_series))
    for j in range(n_series):
        x = demeaned[:, j]
        f = np.fft.rfft(x, n=n_fft)
        acf_unnorm = np.fft.irfft(f * np.conj(f), n=n_fft)[:max_lag + 1]
        C_per_series[:, j] = acf_unnorm / counts

    return C_per_series.mean(axis=1)


def hysteresis_loop_area(I_total, V_in):
    """Enclosed area of a closed I-V trajectory, oint I dV (units A*V).

    NOTE: this is a hysteresis/memory-window metric, NOT dissipated energy
    (framework Eq. 16 is wrong; see docs/theory_audit/THEORY_AUDIT.md, F8).
    It vanishes for a linear resistor, which dissipates. Energy dissipated
    per cycle is oint I*V dt. Caller must pass exactly one closed cycle
    (the trajectory should start and end at the same V_in)."""
    return float(np.trapezoid(I_total, V_in))


def entropy_production(I_total, V_in, dt, T_eff=300.0):
    """Network-level entropy production rate and cycle total (framework
    Eq. 17-18). Simplifies to I*V/T via g_eff = I/V (see spec §2.3) --
    exact given that network-level substitution, with no division-by-zero
    risk since g_eff never appears explicitly.

    Returns:
        (sigma_dot, sigma_total): per-step rate array and the trapezoidal
        integral of that rate over the given trajectory.
    """
    I_total = np.asarray(I_total, dtype=float)
    V_in = np.asarray(V_in, dtype=float)
    sigma_dot = I_total * V_in / T_eff
    sigma_total = float(np.trapezoid(sigma_dot, dx=dt))
    return sigma_dot, sigma_total
