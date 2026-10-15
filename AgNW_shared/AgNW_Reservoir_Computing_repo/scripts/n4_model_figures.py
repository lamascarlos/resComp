#!/usr/bin/env python3
"""Descriptive figures of the N4 continuous-filament junction model (docs/N4_DESIGN.md).

  figN4_1_junction_model: (a) junction schematic, (b) equivalent circuit, (c) rate law dLambda/dt(V),
                          (d) conductance G(Lambda) and gap s(Lambda).
  figN4_2_single_junction: response of ONE junction to a voltage pulse train, continuous filament (noise-free and
                          noisy) vs the framework's binary stochastic junction.
  figN4_3_architecture:  where the junction model sits in the simulation pipeline.

All curves are computed from agnw_rc.filament and agnw_sim.dynamics; the single-junction panel uses a fixed voltage
across one junction (no network). Seeds are fixed below.

    python scripts/n4_model_figures.py [--mode paper|presentation]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_sim.dynamics import update_dynamics  # noqa: E402

from agnw_rc import config as C  # noqa: E402
from agnw_rc.filament import FilamentParams, conductance, gap, lambda_rate, step_lambda  # noqa: E402

FIG = ROOT / "figures"
STYLE = {"paper": FIG / "styles" / "paper_style.mplstyle",
         "presentation": FIG / "styles" / "presentation_style.mplstyle"}
WIDE = 7.0
SEED = 7
# Illustrative network-scale thresholds (the starting point of the calibration, docs/N4_DESIGN.md section 5): the
# source's V_set = 10 mV is for 10 um wires; the framework's junction threshold is V_th = 0.1 V.
P_SRC = FilamentParams()
P_ILL = FilamentParams(V_set=0.1, V_reset=0.05)


def save(fig, name, what):
    meta = {"pdf": {"Creator": "scripts/n4_model_figures.py", "Subject": what},
            "png": {"Software": "scripts/n4_model_figures.py", "Description": what}}
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None, metadata=meta[ext])
    plt.close(fig)
    print(f"figures/{name}.pdf|png")


def label(ax, letter):
    ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=8)


def resistor(ax, x0, x1, y, n=6, amp=0.06, **kw):
    """Horizontal zigzag resistor from x0 to x1 at height y."""
    xs = np.linspace(x0, x1, 2 * n + 1)
    ys = y + amp * np.array([0] + [(-1) ** k for k in range(2 * n - 1)] + [0])
    ax.plot(xs, ys, **kw)


def fig_junction_model():
    fig = plt.figure(figsize=(WIDE, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # (a) schematic cross-section of a junction
    ax = fig.add_subplot(gs[0, 0])
    ax.add_patch(Rectangle((0, 0), 10, 1.4, color="#B8B8B8"))              # bottom wire
    ax.add_patch(Rectangle((0, 2.6), 10, 1.4, color="#B8B8B8"))            # top wire
    ax.add_patch(Rectangle((0, 1.4), 10, 1.2, color="#F3E3B0"))            # PVP layer
    ax.add_patch(Polygon([[3.9, 1.4], [6.1, 1.4], [5.25, 2.15], [4.75, 2.15]], color="#6E6E6E"))  # Ag filament
    ax.annotate("", xy=(5.0, 2.15), xytext=(5.0, 2.6), arrowprops=dict(arrowstyle="<->", lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(5.25, 2.37, "gap $s(\\Lambda)$", va="center", fontsize=6.5)
    ax.annotate("", xy=(8.9, 1.4), xytext=(8.9, 2.6), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(9.05, 2.0, "$s_{max}$", va="center", fontsize=6.5)
    ax.text(0.3, 3.3, "Ag wire (top)", fontsize=6.5)
    ax.text(0.3, 0.5, "Ag wire (bottom)", fontsize=6.5)
    ax.text(0.3, 1.85, "PVP coating", fontsize=6.5)
    ax.text(5.0, 1.0, "Ag filament\n(grows with $|\\Lambda|$)", ha="center", va="top", fontsize=6, color="0.15")
    ax.set(xlim=(0, 10.8), ylim=(-0.2, 4.2), aspect="equal")
    ax.axis("off")
    ax.set_title("junction: filament across the insulating layer", fontsize=7, loc="right")
    label(ax, "a")

    # (b) equivalent circuit: (R_t(Lambda) + R_f) in parallel with R_off
    ax = fig.add_subplot(gs[0, 1])
    k = dict(color="k", lw=1)
    ax.plot([0, 1], [1, 1], **k); ax.plot([9, 10], [1, 1], **k)
    ax.plot([1, 1], [0.3, 1.7], **k); ax.plot([9, 9], [0.3, 1.7], **k)
    ax.plot([1, 2], [1.7, 1.7], **k); resistor(ax, 2, 4.5, 1.7, **k); ax.plot([4.5, 5.5], [1.7, 1.7], **k)
    resistor(ax, 5.5, 8, 1.7, **k); ax.plot([8, 9], [1.7, 1.7], **k)
    ax.annotate("", xy=(4.4, 2.05), xytext=(2.1, 1.35), arrowprops=dict(arrowstyle="->", lw=0.7))   # variable R
    ax.plot([1, 3.8], [0.3, 0.3], **k); resistor(ax, 3.8, 6.2, 0.3, **k); ax.plot([6.2, 9], [0.3, 0.3], **k)
    ax.text(3.25, 2.1, "$R_t(\\Lambda)$ tunnelling", ha="center", fontsize=6.5)
    ax.text(6.75, 2.1, "$R_f = 1/G_{on}$", ha="center", fontsize=6.5)
    ax.text(5.0, -0.2, "$R_{off} = 1/G_{off}$ (leakage)", ha="center", va="top", fontsize=6.5)
    ax.text(0.0, 0.8, "node $2j$", fontsize=6, va="top"); ax.text(9.1, 0.8, "node $2j{+}1$", fontsize=6, va="top")
    ax.text(5, 2.7, "$G(\\Lambda) = [R_t(\\Lambda) + R_f]^{-1} + G_{off}$", ha="center", fontsize=7.5)
    ax.set(xlim=(-0.3, 10.5), ylim=(-0.8, 3.1))
    ax.axis("off")
    ax.set_title("equivalent circuit of one junction (dual-node edge)", fontsize=7, loc="right")
    label(ax, "b")

    # (c) rate law
    ax = fig.add_subplot(gs[1, 0])
    V = np.linspace(-0.25, 0.25, 2001)
    p = P_ILL
    ax.axvspan(-p.V_reset, p.V_reset, color="#0072B2", alpha=0.08, lw=0)
    ax.axvspan(p.V_reset, p.V_set, color="0.85", alpha=0.6, lw=0)
    ax.axvspan(-p.V_set, -p.V_reset, color="0.85", alpha=0.6, lw=0)
    ax.plot(V, lambda_rate(V, np.full_like(V, +0.005), p), color="#D55E00", label="$\\Lambda > 0$")
    ax.plot(V, lambda_rate(V, np.full_like(V, -0.005), p), color="#0072B2", ls="--", label="$\\Lambda < 0$")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.text(0.0, -0.62, "dissolution: $b(|V|-V_{reset})\\,\\mathrm{sgn}\\Lambda$", ha="center", fontsize=5.5)
    ax.text(0.19, -0.1, "growth\n$(|V|-V_{set})\\,\\mathrm{sgn}V$", ha="center", fontsize=5.5)
    ax.text(0.075, -0.2, "hold", ha="center", fontsize=5.5)
    ax.set(xlabel="junction voltage $V$ (V)", ylabel="$d\\Lambda/dt$ (V)", ylim=(-0.7, 0.6))
    ax.set_title(f"rate law, $V_{{set}}$ = {p.V_set} V, $V_{{reset}}$ = {p.V_reset} V, $b$ = {p.b:g} (illustrative)",
                 fontsize=6.5, loc="right")
    ax.legend(fontsize=6, loc="upper left")
    label(ax, "c")

    # (d) conductance and gap
    ax = fig.add_subplot(gs[1, 1])
    lam = np.linspace(0, 0.015, 3001)
    ax.semilogy(lam * 1e3, conductance(lam, P_SRC), color="k", label="$G(\\Lambda)$ (this project: $G_{on}$ = 1 mS)")
    Pq = P_SRC.with_(G_on=7.748e-5, G_off=7.748e-8)
    ax.semilogy(lam * 1e3, conductance(lam, Pq), color="0.55", ls="--",
                label="source values: $R_f = 1/G_0$, $R_{off} = 10^3/G_0$")
    ax.axvline(P_SRC.lambda_crit * 1e3, color="0.5", ls=":", lw=0.8)
    ax.text(P_SRC.lambda_crit * 1e3 + 0.15, 3e-8, "$\\Lambda_{crit}$", fontsize=6)
    ax.set(xlabel="filament state $|\\Lambda|$ (mV s)", ylabel="junction conductance $G$ (S)", ylim=(3e-11, 3e-3))
    ax2 = ax.twinx()
    ax2.plot(lam * 1e3, gap(lam, P_SRC) * 1e9, color="#009E73", lw=0.9)
    ax2.set_ylabel("gap $s$ (nm)", color="#009E73")
    ax2.tick_params(axis="y", colors="#009E73")
    ax2.spines["right"].set_visible(True)
    ax.legend(fontsize=5.5, loc="upper left")
    ax.set_title("tunnelling (Simmons) then filament contact", fontsize=6.5, loc="right")
    label(ax, "d")
    save(fig, "figN4_1_junction_model", "scripts/n4_model_figures.py:fig_junction_model | agnw_rc.filament")


def fig_single_junction():
    """One junction under a pulse train of 0.2 V: 0.4 s ON / 0.6 s OFF, 3 cycles (dt = 1 ms)."""
    dt = 1e-3
    t = np.arange(0, 3.0, dt)
    V = np.where((t % 1.0) < 0.4, 0.2, 0.0)
    runs = {}
    for sigma in (0.0, 0.002, 0.006):
        p = P_ILL.with_(sigma=sigma)
        rng = np.random.default_rng(SEED)
        lam = np.zeros(1)
        L = np.empty(len(t))
        for i, v in enumerate(V):
            lam = step_lambda(lam, np.array([v]), p, rng if sigma else None)
            L[i] = lam[0]
        runs[sigma] = (L, conductance(L, p))
    params, _ = C.network_params()                       # framework binary junction, same voltage across it
    rng = np.random.default_rng(SEED)
    G = np.array([params.G_off])
    Gb = np.empty(len(t))
    for i, v in enumerate(V):
        G = update_dynamics(G, np.array([abs(v)]), G * v, params, rng)
        Gb[i] = G[0]

    fig, axes = plt.subplots(3, 1, figsize=(WIDE * 0.75, 4.2), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [0.6, 1, 1]})
    axes[0].plot(t, V, color="k", lw=0.9)
    axes[0].set_ylabel("$V$ (V)")
    axes[0].set_title("voltage across one junction (0.2 V pulses)", fontsize=7, loc="right")
    cols = {0.0: "#D55E00", 0.002: "#E69F00", 0.006: "#CC79A7"}
    for sigma, (L, Gc) in runs.items():
        axes[1].plot(t, L * 1e3, color=cols[sigma], lw=0.9,
                     label="noise-free" if sigma == 0 else f"$\\sigma$ = {sigma} V s$^{{1/2}}$")
    axes[1].axhline(P_ILL.lambda_crit * 1e3, color="0.5", ls=":", lw=0.8)
    axes[1].set_ylabel("$\\Lambda$ (mV s)")
    axes[1].legend(fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(0.15, 0.95))
    axes[1].set_title("continuous filament (N4): grows while driven, dissolves in the pauses", fontsize=7,
                      loc="right")
    for sigma, (L, Gc) in runs.items():
        axes[2].semilogy(t, Gc, color=cols[sigma], lw=0.9)
    axes[2].semilogy(t, Gb, color="#0072B2", lw=1.2, ls="--", label="framework binary junction (p_relax = 1e-4)")
    axes[2].set(ylabel="$G$ (S)", xlabel="time $t$ (s)", ylim=(3e-10, 3e-3))
    axes[2].legend(fontsize=6, loc="center right")
    axes[2].set_title("conductance: graded and forgetting (N4) vs one random switch that persists (binary)",
                      fontsize=7, loc="right")
    for ax, letter in zip(axes, "abc"):
        label(ax, letter)
    save(fig, "figN4_2_single_junction",
         f"scripts/n4_model_figures.py:fig_single_junction | agnw_rc.filament, agnw_sim.dynamics | seed {SEED}")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(WIDE, 2.6), constrained_layout=True)
    ax.axis("off")
    ax.set(xlim=(0, 10), ylim=(0, 4))

    def box(x, y, w, h, text, fc, fs=6.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="0.3", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="0.3"))

    box(0.1, 2.6, 2.2, 1.1, "framework (imported)\nwire geometry, junctions,\ndual-node graph, $G_{seg}$", "#E8E8E8")
    box(0.1, 0.4, 2.2, 1.1, "electrodes (this project)\ntwo bands | split bands\n| pad array (N5)", "#FBE3D6")
    box(3.0, 1.5, 2.0, 1.2, "Kirchhoff solve\n$\\mathbf{Y}(G)\\,\\mathbf{V} = \\mathbf{B}\\,V_{el}(t)$\n"
        "(two- or multi-terminal)", "#E8E8E8")
    box(5.7, 2.9, 2.3, 1.0, "binary stochastic junction\n(framework: SET / RESET / relax)", "#DCE9F5", fs=6)
    box(5.7, 1.65, 2.3, 1.0, "mean-field junction\n(N1 ablation: $P_j$)", "#DCE9F5", fs=6)
    box(5.7, 0.2, 2.3, 1.2, "continuous filament (N4)\n$\\dot\\Lambda = f(V,\\Lambda) + \\sigma\\xi$\n"
        "$G = G(\\Lambda)$", "#D9F0E3", fs=6)
    box(8.5, 1.5, 1.4, 1.2, "readout\ncurrents, probes,\nstates -> ridge", "#E8E8E8", fs=6)
    arrow(2.3, 3.15, 3.0, 2.4); arrow(2.3, 0.95, 3.0, 1.8)
    for y in (3.4, 2.15, 0.8):
        arrow(5.0, 2.1, 5.7, y)
        arrow(8.0, y, 8.5, 2.1)
    ax.text(5.35, 3.75, "junction model: one of", fontsize=6, style="italic")
    ax.text(3.0, 0.9, "each step: $V$ -> junction drops $V_j$\n-> update states -> new $G$", fontsize=5.5,
            style="italic")
    save(fig, "figN4_3_architecture", "scripts/n4_model_figures.py:fig_architecture | schematic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(STYLE), default="paper")
    a = ap.parse_args()
    plt.style.use(STYLE[a.mode])
    fig_junction_model()
    fig_single_junction()
    fig_architecture()


if __name__ == "__main__":
    main()
