import numpy as np
from agnw_sim.geometry import build_network
from agnw_sim.visualize import plot_network, plot_iv_hysteresis, plot_conductance_trace, plot_autocorrelation


def test_plot_network_runs_and_draws_all_wires():
    rng = np.random.default_rng(9)
    net = build_network(n_wires=20, area=1e-6, length=70e-6, rng=rng)
    ax = plot_network(net)
    assert len(ax.lines) == net.n_wires


def test_plot_network_marks_on_junctions():
    rng = np.random.default_rng(10)
    net = build_network(n_wires=200, area=1e-6, length=70e-6, rng=rng)
    n_junctions = net.junctions.shape[0]
    assert n_junctions > 0, "test network must have junctions"
    G = np.full(n_junctions, 1e-9)
    G[0] = 1e-3
    ax = plot_network(net, G=G, G_on_threshold=1e-3)
    assert len(ax.collections) == 1


def test_plot_iv_hysteresis_runs():
    V = np.linspace(-1, 1, 50)
    I = np.sin(V)
    ax = plot_iv_hysteresis(V, I)
    assert len(ax.lines) == 1


def test_plot_conductance_trace_runs():
    t = np.linspace(0, 1, 30)
    G_sample = np.random.default_rng(0).uniform(1e-9, 1e-3, size=(30, 5))
    ax = plot_conductance_trace(t, G_sample)
    assert len(ax.lines) == 5


def test_plot_autocorrelation_runs():
    lags = np.arange(21)
    C_tau = np.exp(-lags / 5.0)
    ax = plot_autocorrelation(lags, C_tau)
    # 2 lines: the C(tau) data line plus the axhline(0) zero-reference line
    # (matplotlib's axhline adds a Line2D that is also counted in ax.lines).
    assert len(ax.lines) == 2
