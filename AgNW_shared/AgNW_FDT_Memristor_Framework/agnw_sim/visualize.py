"""Sanity-check plots: network layout, I-V hysteresis, conductance traces."""
import matplotlib.pyplot as plt


def plot_network(network, G=None, G_on_threshold=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    for (x1, y1), (x2, y2) in network.wire_endpoints:
        ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.5, zorder=1)
    if G is not None and network.junctions.size and G_on_threshold is not None:
        on = G >= G_on_threshold
        pts = network.junction_pos[on]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], color="red", s=4, zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def plot_iv_hysteresis(V_in, I_total, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(V_in, I_total)
    ax.set_xlabel("V_in (V)")
    ax.set_ylabel("I_total (A)")
    return ax


def plot_conductance_trace(t, G_sample, ax=None, max_traces=20):
    if ax is None:
        _, ax = plt.subplots()
    n_traces = min(max_traces, G_sample.shape[1])
    for j in range(n_traces):
        ax.plot(t, G_sample[:, j], linewidth=0.5, alpha=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("G (S)")
    return ax


def plot_autocorrelation(lags, C_tau, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(lags, C_tau, color="#1f77b4", linewidth=1.5)
    ax.axhline(0, color="lightgray", linewidth=0.5, zorder=0)
    ax.set_xlabel("lag (steps)")
    ax.set_ylabel("C(τ)")
    return ax
