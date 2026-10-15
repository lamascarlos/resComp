#!/usr/bin/env python3
"""Schematic figures of docs/MANUAL.md, drawn from the project's own model code where there is physics to show.

    python scripts/manual_figures.py [--mode paper|presentation] [--only fig1 fig3 ...]

Writes figures/manual/manual_fig{1..5}_*.{png,svg}:
numbered in the order they appear in the manual:
  fig1 pipeline     what is fixed (agnw_sim), designed here (interface, junction variants) and trained (readout)
  fig2 taxonomy     experiments by junction model, in the order they were run, coloured by outcome
  fig3 dualnode     physical crossings -> dual-node circuit graph (agnw_sim.dualnode convention)
  fig4 junctions    switching laws of the binary, N4 and N10 junctions (agnw_rc.filament / volatile, config params)
  fig5 layouts      electrode layouts at true scale on network seed 1 (agnw_rc.electrodes)
Deterministic: the only random input is network seed 1 (fig5).
"""
import argparse
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "figures" / "manual"

# Wong colour-blind-safe palette
BLUE, VERM, GREEN, PINK, SKY, ORANGE, BLACK = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#000000"
GREY, LIGHT = "#6b6b6b", "#f3f3f3"
ROLE_COLOR = {"in": VERM, "gnd": BLUE, "float": GREEN}

FS = 1.0          # font scale, set by --mode


def setup_style(mode: str):
    global FS
    FS = 1.0 if mode == "paper" else 1.35
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 9 * FS, "axes.titlesize": 10 * FS, "axes.labelsize": 9 * FS,
        "xtick.labelsize": 8 * FS, "ytick.labelsize": 8 * FS, "legend.fontsize": 8 * FS,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "legend.frameon": False, "savefig.dpi": 200, "figure.dpi": 100,
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def box(ax, x, y, w, h, text, fc="white", ec=BLACK, lw=1.2, fs=9, ls="-", weight="normal", radius=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec, lw=lw,
                                ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs * FS, weight=weight, linespacing=1.35)


def arrow(ax, p, q, color=BLACK, lw=1.3, style="-|>", rad=0.0, ls="-", ms=12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def label_panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=11 * FS, weight="bold", ha="right", va="bottom")


# ----------------------------------------------------------------------------------------------------------------
# fig1: pipeline
# ----------------------------------------------------------------------------------------------------------------
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(12.6, 3.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    w, h, y = 1.65, 1.25, 1.2
    xs = [0.15, 2.15, 4.4, 6.6, 8.75, 10.85]
    groups = [  # (first box, last box, label, colour)
        (0, 1, "interface: designed here\n(agnw_rc.drive / multidrive / electrodes)", VERM),
        (2, 3, "substrate: fixed network (agnw_sim)\n+ junction model variant", BLUE),
        (4, 4, "measured\n(agnw_rc.evaluate)", GREEN),
        (5, 5, "trained\n(agnw_rc.readout)", PINK),
    ]
    tint = {VERM: "#fbe9df", BLUE: "#e1eef7", GREEN: "#e0f3ec", PINK: "#f7e8f1"}
    for a, b, lab, c in groups:
        x0, x1 = xs[a] - 0.08, xs[b] + w + 0.08
        ax.add_patch(FancyBboxPatch((x0, y - 0.18), x1 - x0, h + 0.36, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=tint[c], ec=c, lw=1.0, ls="--"))
        ax.text((x0 + x1) / 2, y + h + 0.3, lab, ha="center", va="bottom", fontsize=8 * FS, color=c, weight="bold")
    texts = [
        "input\n$u_k \\sim U(0,1)$\niid / sine-square",
        "encode\n$V_e(t)=V_c+\\frac{V_w}{2}\\,x\\,w_e$\nheld $H$ steps",
        "AgNW network\n2000 wires, 1 mm$^2$\nKirchhoff $\\mathbf{A}\\mathbf{V}=\\mathbf{b}$",
        "junctions\n$G_j(t)$: binary / N1 /\nN4 / N10",
        "observables\n$I_e$, $V_p$, $V_\\mathrm{float}$\n× virtual nodes",
        "ridge readout\n$\\hat y = X\\mathbf{w}$\n$\\alpha$ on validation",
    ]
    for x, t in zip(xs, texts):
        box(ax, x, y, w, h, t, fs=8.5)
    for i in range(5):
        if i == 2:
            continue
        arrow(ax, (xs[i] + w, y + h / 2), (xs[i + 1], y + h / 2))
    # network <-> junction feedback loop
    xa, xb = xs[2] + w, xs[3]
    arrow(ax, (xa, y + h * 0.7), (xb, y + h * 0.7), color=BLUE)
    arrow(ax, (xb, y + h * 0.3), (xa, y + h * 0.3), color=BLUE)
    ax.text((xa + xb) / 2, y + h * 0.85, "$dV_j$", ha="center", fontsize=8 * FS, color=BLUE)
    ax.text((xa + xb) / 2, y + h * 0.05, "$G_j$", ha="center", fontsize=8 * FS, color=BLUE)
    ax.text(6.3, 0.35, "every step $\\Delta t$ = 1-2 ms: solve circuit $\\to$ junction drops $dV_j$ $\\to$ update "
            "junction states $\\to$ new $G_j$;   one sample $u_k$ is held for $H$ steps (10 ms - 3 s)",
            ha="center", fontsize=8 * FS, color=GREY)
    save(fig, "manual_fig1_pipeline")


# ----------------------------------------------------------------------------------------------------------------
# fig2: junction models
# ----------------------------------------------------------------------------------------------------------------
def fig4_junctions():
    from agnw_rc import config as C
    from agnw_rc import filament as F
    from agnw_rc import volatile as Vol

    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)

    # (a) binary: per-step probabilities (Lamas 2026 preset, framework values)
    ax = axs[0]
    dv = np.linspace(0, 3, 400)
    p_set = 1e-3 * 400 * np.maximum(0, 1 - np.exp(-1.0 * (dv - 0.1)))
    p_reset = np.minimum(1, 1e-3 * 10 * (1e-3 * dv / 1e-3) ** 2)
    ax.semilogy(dv, np.where(p_set > 0, p_set, np.nan), color=VERM, label="$p_\\mathrm{set}$ (OFF$\\to$ON)")
    ax.semilogy(dv, np.where(p_reset > 0, p_reset, np.nan), color=BLUE, label="$p_\\mathrm{reset}$ (ON$\\to$OFF)")
    ax.axhline(1e-4, color=GREY, ls="--", lw=1)
    ax.text(3, 1.3e-4, "$p_\\mathrm{relax}$ = $10^{-4}$ (10 s)", ha="right", va="bottom", fontsize=7.5 * FS, color=GREY)
    ax.axvline(0.1, color=GREY, ls=":", lw=1)
    ax.text(0.13, 2e-5, "$V_\\mathrm{th}$", fontsize=7.5 * FS, color=GREY)
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel("junction drop $|dV|$ (V)")
    ax.set_ylabel("probability per 1 ms step")
    ax.set_title("binary stochastic (framework)")
    ax.legend(loc="lower right")
    label_panel(ax, "a")

    # (b) N4 rate law: hold band where nothing changes
    fp = C.filament_params()
    ax = axs[1]
    v = np.linspace(0, 0.1, 500)
    for ratio, ls, lab in ((0.5, "-", "calibrated $r=0.5$"), (0.9, "--", "narrow band $r=0.9$")):
        p = fp.__class__(**{**fp.__dict__, "V_reset": ratio * fp.V_set})
        rate = F.lambda_rate(v, np.full_like(v, 0.5 * fp.lambda_crit), p)
        ax.plot(v * 1e3, rate * 1e3, color=BLACK, ls=ls, label=lab)
    ax.axvspan(0.5 * fp.V_set * 1e3, fp.V_set * 1e3, color=ORANGE, alpha=0.25, lw=0)
    ax.axvspan(0.9 * fp.V_set * 1e3, fp.V_set * 1e3, fc="none", ec="#8a5a00", hatch="///", lw=0)
    ax.text(0.7 * fp.V_set * 1e3, 22, "hold band\n$d\\Lambda/dt=0$\n($r$ = 0.5; hatched:\n$r$ = 0.9)", ha="center",
            fontsize=7.5 * FS, color="#8a5a00")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xlabel("junction drop $|V|$ (mV)")
    ax.set_ylabel("growth rate $d\\Lambda/dt$ (mV)")
    ax.set_title("N4 filament: rate law ($\\Lambda>0$)")
    ax.legend(loc="upper left")
    label_panel(ax, "b")

    # (c) N4 conductance vs state
    ax = axs[2]
    lam = np.linspace(0, fp.lambda_max, 400)
    ax.semilogy(lam / fp.lambda_crit, F.conductance(lam, fp), color=BLUE)
    ax.axvline(1, color=GREY, ls=":", lw=1)
    ax.text(1.03, 1e-8, "gap closed\n$\\Lambda=\\Lambda_\\mathrm{crit}$", fontsize=7.5 * FS, color=GREY)
    ax.axhline(fp.G_on, color=GREY, ls="--", lw=0.8)
    ax.axhline(fp.G_off, color=GREY, ls="--", lw=0.8)
    ax.text(0.02, fp.G_on * 1.6, "$G_\\mathrm{on}$", fontsize=7.5 * FS, color=GREY)
    ax.text(0.02, fp.G_off * 1.6, "$G_\\mathrm{off}$", fontsize=7.5 * FS, color=GREY)
    ax.set_xlabel("filament state $|\\Lambda|/\\Lambda_\\mathrm{crit}$")
    ax.set_ylabel("conductance $G$ (S)")
    ax.set_title("N4 filament: Simmons tunnelling")
    label_panel(ax, "c")

    # (d) N10 volatile rates: growth vs decay, no hold band
    vp = C.volatile_params().with_(tau_s=1.0)
    ax = axs[3]
    v = np.linspace(0, 0.2, 500)
    k_p, k_d, _ = Vol.rates(v, np.full_like(v, vp.G_off), vp, np.full_like(v, 1 / vp.tau_s))
    ax.semilogy(v * 1e3, k_p, color=VERM, label="growth $k_p$")
    ax.semilogy(v * 1e3, k_d, color=BLUE, label="decay $k_d$ ($\\tau_s$ = 1 s)")
    i = np.argmin(np.abs(np.log(k_p) - np.log(k_d)))
    ax.plot(v[i] * 1e3, k_p[i], "o", color=BLACK, ms=4)
    ax.text(v[i] * 1e3 + 8, k_p[i] * 3, f"$k_p=k_d$ at {v[i] * 1e3:.0f} mV", fontsize=7.5 * FS)
    ax.axhline(vp.k_max, color=GREY, ls="--", lw=0.8)
    ax.text(2, vp.k_max * 1.3, "$k_\\max$ = 100 s$^{-1}$", fontsize=7.5 * FS, color=GREY)
    ax.set_ylim(1e-6, 1e3)
    ax.set_xlabel("junction drop $|V|$ (mV)")
    ax.set_ylabel("rate (s$^{-1}$)")
    ax.set_title("N10 volatile ($v_\\mathrm{scale}$ = 5)")
    ax.legend(loc="lower right")
    label_panel(ax, "d")
    save(fig, "manual_fig4_junctions")


# ----------------------------------------------------------------------------------------------------------------
# fig3: electrode layouts at true scale
# ----------------------------------------------------------------------------------------------------------------
def fig5_layouts():
    from agnw_rc import config as C
    from agnw_rc import electrodes as El

    net = C.make_network(1)
    S = np.sqrt(net.area) * 1e6  # um
    ends = net.wire_endpoints * 1e6
    layouts = [("two_bands", "two bands\nE1-E4, N4, N10-1"), ("split_bands", "split bands 8 + 8\nN5a"),
               ("pads", "4 × 4 pads, 50 µm\nN5a, N10-E/M/S/R"), ("pads_float", "pads_float 4 + 4 + 8 floating\nN10-E")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.0), constrained_layout=True)
    for ax, (name, title) in zip(axs, layouts):
        els = El.make_layout(name, net)
        ax.add_collection(LineCollection(ends, colors="#9a9a9a", linewidths=0.35))
        for e in els:
            c = ROLE_COLOR[e.role]
            seg = ends[e.wires]
            ax.add_collection(LineCollection(seg, colors=c, linewidths=0.7, alpha=0.9))
        if name in ("two_bands", "split_bands"):
            bw = 0.05 * S
            if name == "two_bands":
                spans = [(0, 0, S, "in"), (S - bw, 0, S, "gnd")]
            else:
                spans = [(0, k * S / 8, S / 8, "in") for k in range(8)] + [(S - bw, k * S / 8, S / 8, "gnd")
                                                                          for k in range(8)]
            for x0, y0, hh, role in spans:
                ax.add_patch(Rectangle((x0, y0), bw, hh, fc=ROLE_COLOR[role], ec="none", alpha=0.55))
                # dark grey outline on top of the wires, so adjacent electrodes are clearly separated
                ax.add_patch(Rectangle((x0, y0), bw, hh, fc="none", ec="#333333", lw=1.3, zorder=4))
        else:
            ps = 50.0
            for i in range(4):
                for j in range(4):
                    e = els[4 * i + j]
                    cx, cy = (i + 0.5) * S / 4, (j + 0.5) * S / 4
                    ax.add_patch(Rectangle((cx - ps / 2, cy - ps / 2), ps, ps, fc=ROLE_COLOR[e.role], ec=BLACK,
                                           lw=0.8, zorder=3))
        n_contacts = np.mean([len(e.wires) for e in els])
        ax.text(0.0, -0.025, f"{len(els)} electrodes, {n_contacts:.0f} wires each (mean)", transform=ax.transAxes,
                ha="left", va="top", fontsize=8 * FS, color=GREY)
        ax.plot([S - 200, S], [-40, -40], color=BLACK, lw=2, clip_on=False)
        ax.text(S - 100, -55, "200 µm", ha="center", va="top", fontsize=7.5 * FS)
        ax.set_xlim(0, S)
        ax.set_ylim(0, S)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GREY)
        ax.set_title(title)
    handles = [Patch(fc=VERM, label="driven (input)"), Patch(fc=BLUE, label="grounded (current read)"),
               Patch(fc=GREEN, label="floating (voltage read)")]
    handles.append(Patch(fc="none", ec="none", label="true scale, network seed 1 (1 mm$^2$, 2000 wires of 70 µm); "
                                                      "coloured wires touch an electrode"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1))
    save(fig, "manual_fig5_layouts")


# ----------------------------------------------------------------------------------------------------------------
# fig4: dual-node graph
# ----------------------------------------------------------------------------------------------------------------
def zigzag(ax, p, q, color=BLACK, n=6, amp=0.09, lw=1.4, frac=0.5):
    """Resistor symbol between p and q (zigzag over the central `frac` of the segment)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.hypot(*d)
    u, nrm = d / L, np.array([-d[1], d[0]]) / L
    a, b = (1 - frac) / 2, (1 + frac) / 2
    t = np.linspace(a, b, 2 * n + 1)
    off = np.zeros_like(t)
    off[1:-1:2], off[2:-1:2] = amp, -amp
    pts = p + np.outer(t * L, u) + np.outer(off, nrm)
    pts = np.vstack([p, pts, q])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_joinstyle="miter")


def memristor(ax, p, q, color=VERM, lw=1.6):
    """Memristor symbol: line with a filled box, thick band at one end."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, zorder=1)
    vertical = abs(q[1] - p[1]) > abs(q[0] - p[0])
    w, h = (0.22, 0.42) if vertical else (0.42, 0.22)
    ax.add_patch(Rectangle((m[0] - w / 2, m[1] - h / 2), w, h, fc="white", ec=color, lw=lw, zorder=2))
    if vertical:
        ax.add_patch(Rectangle((m[0] - w / 2, m[1] + h / 2 - 0.09), w, 0.09, fc=color, ec=color, zorder=3))
    else:
        ax.add_patch(Rectangle((m[0] + w / 2 - 0.09, m[1] - h / 2), 0.09, h, fc=color, ec=color, zorder=3))


def node(ax, xy, label=None, dx=0.0, dy=0.22, color=BLACK, fc="white"):
    ax.plot(*xy, "o", ms=7, mfc=fc, mec=color, mew=1.5, zorder=4)
    if label:
        ax.text(xy[0] + dx, xy[1] + dy, label, ha="center", va="bottom" if dy > 0 else "top", fontsize=8.5 * FS,
                color=color)


def fig3_dualnode():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.35]})
    # ---- (a) physical picture: three sticks, two electrode bands
    ax = a1
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-0.6, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-0.3, -0.4), 1.1, 5.8, fc=VERM, alpha=0.25, lw=0))
    ax.add_patch(Rectangle((8.9, -0.4), 1.4, 5.8, fc=BLUE, alpha=0.25, lw=0))
    ax.text(0.25, -0.3, "driven\nband", ha="center", va="bottom", fontsize=8 * FS, color=VERM)
    ax.text(9.6, -0.3, "grounded\nband", ha="center", va="bottom", fontsize=8 * FS, color=BLUE)
    A = np.array([[0.2, 3.0], [8.0, 3.0]])
    B = np.array([[2.0, 5.0], [6.0, 0.0]])
    Cw = np.array([[4.0, 0.0], [9.3, 4.6]])
    for (p, q), name, off in ((A, "wire A", (-0.1, 0.25)), (B, "wire B", (0.15, 0.1)), (Cw, "wire C", (0.2, -0.3))):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=BLACK, lw=3, solid_capstyle="round")
    ax.text(1.4, 3.2, "A", fontsize=10 * FS, weight="bold")
    ax.text(1.6, 4.7, "B", fontsize=10 * FS, weight="bold")
    ax.text(4.5, 0.15, "C", fontsize=10 * FS, weight="bold")

    def cross(P, Q):
        (p1, p2), (q1, q2) = P, Q
        d1, d2 = p2 - p1, q2 - q1
        c2 = lambda a, b: a[0] * b[1] - a[1] * b[0]
        t = c2(q1 - p1, d2) / c2(d1, d2)
        return p1 + t * d1
    J = [cross(A, B), cross(B, Cw), cross(A, Cw)]
    for k, j in enumerate(J):
        ax.plot(*j, "o", ms=11, mfc=VERM, mec="white", mew=1.5, zorder=5)
        ax.text(j[0] - 0.75, j[1] + 0.2, f"$j_{k}$", fontsize=10 * FS, color=VERM, weight="bold")
    ax.plot(*A[0], "s", ms=8, color=VERM, zorder=5)
    ax.plot(*Cw[1], "s", ms=8, color=BLUE, zorder=5)
    ax.text(5.0, -0.55, "junction = wire crossing (memristive contact)\nsquares: wire ends inside a band = contacts",
            ha="center", va="top", fontsize=8 * FS, color=GREY)
    ax.set_title("physical network", fontsize=10 * FS)
    label_panel(ax, "a")

    # ---- (b) dual-node graph
    ax = a2
    ax.set_xlim(-1.6, 11.4)
    ax.set_ylim(-1.9, 5.3)
    ax.axis("off")
    yA, yB, yC = 4.0, 2.2, 0.4
    # node x positions follow the order along each wire in (a)
    xA = {"c": 0.6, 0: 3.6, 4: 7.4}
    xB = {1: 3.6, 2: 5.4}
    xC = {3: 5.4, 5: 7.4, "c": 9.4}
    for y, name in ((yA, "wire A"), (yB, "wire B"), (yC, "wire C")):
        ax.text(-1.5, y, name, va="center", fontsize=9 * FS, weight="bold")
    # wire segments (resistors G_seg)
    zigzag(ax, (xA["c"], yA), (xA[0], yA))
    zigzag(ax, (xA[0], yA), (xA[4], yA))
    zigzag(ax, (xB[1], yB), (xB[2], yB))
    zigzag(ax, (xC[3], yC), (xC[5], yC))
    zigzag(ax, (xC[5], yC), (xC["c"], yC))
    ax.text((xA[0] + xA[4]) / 2, yA + 0.25, "$G_\\mathrm{seg}=\\min(A_\\mathrm{cs}/\\rho d,\\,G_\\mathrm{seg,max})$",
            ha="center", fontsize=8 * FS)
    # memristor edges between the two nodes of each junction
    memristor(ax, (xA[0], yA), (xB[1], yB))
    memristor(ax, (xB[2], yB), (xC[3], yC))
    memristor(ax, (xA[4], yA), (xC[5], yC))
    for (x, y), lab in (((xA[0], (yA + yB) / 2), "$G_0$"), ((xB[2], (yB + yC) / 2), "$G_1$"),
                        ((xA[4], (yA + yC) / 2 + 0.4), "$G_2$")):
        ax.text(x + 0.3, y, lab, va="center", fontsize=9.5 * FS, color=VERM, weight="bold")
    # nodes
    node(ax, (xA[0], yA), "0")
    node(ax, (xA[4], yA), "4")
    node(ax, (xB[1], yB), "1", dx=-0.3, dy=0.12)
    node(ax, (xB[2], yB), "2", dx=-0.3, dy=0.12)
    node(ax, (xC[3], yC), "3", dy=-0.22)
    node(ax, (xC[5], yC), "5", dy=-0.22)
    node(ax, (xA["c"], yA), "contact", color=VERM, fc="#fbe9df")
    node(ax, (xC["c"], yC), "contact", dy=0.22, color=BLUE, fc="#e1eef7")
    # super-nodes
    box(ax, -1.5, yA - 1.65, 1.3, 0.7, "$V_\\mathrm{in}$\nsuper-node", fc="#fbe9df", ec=VERM, fs=8)
    box(ax, 9.95, yC - 1.75, 1.35, 0.7, "GND\nsuper-node", fc="#e1eef7", ec=BLUE, fs=8)
    zigzag(ax, (-0.85, yA - 0.95), (xA["c"], yA), color=VERM, frac=0.4, n=4)
    zigzag(ax, (xC["c"], yC), (10.6, yC - 1.05), color=BLUE, frac=0.4, n=4)
    ax.text(0.15, yA - 0.75, "$G_\\mathrm{CONTACT}$ = 1 S", fontsize=7.5 * FS, color=VERM, ha="left")
    ax.text(10.4, yC + 0.15, "$G_\\mathrm{CONTACT}$", fontsize=7.5 * FS, color=BLUE, ha="left")
    ax.text(4.9, -1.15, "junction $j$ $\\to$ node pair $(2j, 2j{+}1)$, one on each wire, joined by memristor $G_j$;  "
            "$dV_j = V_{2j}-V_{2j+1}$\n"
            "unknowns: $n = 2n_j + n_\\mathrm{contacts} + 2$ (+ $n_e$ super-nodes in the multi-terminal circuit)",
            ha="center", va="top", fontsize=8 * FS, color=GREY)
    ax.set_title("dual-node circuit graph (agnw_sim.dualnode)", fontsize=10 * FS)
    label_panel(ax, "b")
    fig.tight_layout()
    save(fig, "manual_fig3_dualnode")


# ----------------------------------------------------------------------------------------------------------------
# fig5: experiment taxonomy
# ----------------------------------------------------------------------------------------------------------------
OUTCOME = {"fail": ("#f6d5d0", VERM, "-", "fails: not a reservoir"),
           "diag": ("#e8e8e8", GREY, "-", "diagnostic / mechanism found"),
           "pass": ("#e3f5ee", GREEN, "-", "passes G1"),
           "g2": ("#a8e0c9", "#00664f", "-", "passes G1 and G2 loose (G2 strict: none)"),
           "open": ("white", GREY, "--", "open / deferred")}


def fig2_taxonomy():
    lanes = [("binary\n(framework)", 3), ("mean-field\n(N1 ablation)", 2), ("N4 filament\n(Fig. 4 fit)", 1),
             ("N10 volatile\n(measured data)", 0)]
    L = {"b": 3, "m": 2, "f": 1, "v": 0}
    # (column, lane, name, outcome, class); columns follow the order the experiments were run
    items = [
        (0, "b", "E1–E3", "scores at\ninput-only\nlevel", "fail"),
        (1, "m", "N1/M1", "no noise:\nnonlinear,\nMC 1.8", "diag"),
        (2, "m", "N2", "spread\nforgetting:\n~2× memory", "diag"),
        (3, "b", "N7a/b", "averaging,\nhigher V:\nno rescue", "fail"),
        (4, "b", "N5a", "16\nelectrodes:\nno gain", "fail"),
        (5, "f", "N4-1", "frozen:\ncapacity 0", "fail"),
        (7, "f", "N8 short", "hold band\nfreezes N4", "diag"),
        (8, "f", "N4-nb", "narrow\nhold band,\nE4 protocol", "open"),
        (8, "v", "N10-1", "32 probes,\ntwo bands", "pass"),
        (9, "v", "N10-E", "16 pads,\n3 networks", "g2"),
        (10, "v", "N10-M", "memory ≈ 4;\nσ ≥ 0.1 fails", "g2"),
        (11, "v", "N10-S", "settings\nrobust", "g2"),
        (12, "v", "N10-R", "ruptures\n≥ 0.01/s\nbreak G1", "fail"),
        (13, "v", "N10-C", "calibrate\non Milano\n2025 data", "open"),
    ]
    dx, dy, w, h = 1.42, 2.05, 1.3, 1.7
    ncol = 14
    fig, ax = plt.subplots(figsize=(18, 9.2))
    ax.set_xlim(-1.75, ncol * dx)
    ax.set_ylim(-1.0, 4 * dy + 0.45)
    ax.axis("off")
    for k, (lab, li) in enumerate(lanes):
        y = li * dy
        ax.add_patch(Rectangle((-1.7, y - 0.1), ncol * dx + 1.7, dy - 0.15, fc=LIGHT if k % 2 == 0 else "white", lw=0))
        ax.text(-1.6, y + h / 2, lab, va="center", fontsize=8.5 * FS, weight="bold")

    def card(x, y, hh, name, out, cls):
        fc, ec, ls, _ = OUTCOME[cls]
        passed = cls in ("pass", "g2")
        ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec,
                                    lw=2.6 if passed else 1.2, ls=ls))
        top = y + hh / 2 + 0.5 if hh <= h else y + hh / 2 + 0.75
        boxes.append((x, y, w, hh))
        b = len(boxes) - 1
        texts.append((ax.text(x + w / 2, top, name, ha="center", va="top", fontsize=8.5 * FS, weight="bold"), b))
        texts.append((ax.text(x + w / 2, top - 0.3, out, ha="center", va="top", fontsize=7.5 * FS, linespacing=1.3), b))
        if passed:   # verdict badge at the bottom of the card
            badge = "G1 ✓\nG2 loose ✓" if cls == "g2" else "G1 ✓\nG2 ✗"
            texts.append((ax.text(x + w / 2, y + 0.16, badge, ha="center", va="bottom", fontsize=7.5 * FS,
                                  weight="bold", color="white", linespacing=1.2,
                                  bbox=dict(boxstyle="round,pad=0.25", fc="#00664f" if cls == "g2" else GREEN, ec="none")), b))

    texts, boxes = [], []
    pos = {}
    for c, lane, name, out, cls in items:
        x, y = c * dx, L[lane] * dy
        card(x, y, h, name, out, cls)
        pos[name] = (x, y)
    # E4 spans the binary and N4 lanes
    x, y = 6 * dx, L["f"] * dy
    hh = L["b"] * dy + h - y
    card(x, y, hh, "E4", "0.3–3 s\ninputs,\nbinary + N4:\nG1 fails\nboth\n\ngo/no-go\ncriteria\nG0–G2 fixed\nbefore\nrunning", "fail")
    pos["E4"] = (x, y + hh / 2 - h / 2)

    def link(a, b, rad=0.0):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        arrow(ax, (xa + w, ya + h / 2), (xb, yb + h / 2), color=GREY, lw=1.1, rad=rad, ms=10)
    for a, b, r in (("E1–E3", "N1/M1", 0), ("N1/M1", "N2", 0), ("N2", "N7a/b", 0), ("N7a/b", "N5a", 0),
                    ("N5a", "N4-1", 0), ("N4-1", "E4", 0), ("E4", "N8 short", 0), ("N8 short", "N4-nb", 0),
                    ("N8 short", "N10-1", 0), ("N10-1", "N10-E", 0), ("N10-E", "N10-M", 0), ("N10-M", "N10-S", 0),
                    ("N10-S", "N10-R", 0), ("N10-R", "N10-C", 0)):
        link(a, b, r)
    ax.annotate("", xy=(ncol * dx - 0.1, -0.4), xytext=(0, -0.4), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1))
    for c, t in ((0, "2026-09-25"), (6, "09-27 (review)"), (8, "09-28/29"), (12, "09-30")):
        ax.text(c * dx, -0.5, t, fontsize=7.5 * FS, color=GREY, va="top")
    ax.text(ncol * dx - 0.1, -0.5, "order run", fontsize=7.5 * FS, color=GREY, va="top", ha="right")
    handles = [Patch(fc=fc, ec=ec, ls=ls, label=lab) for fc, ec, ls, lab in OUTCOME.values()]
    ax.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=8 * FS)
    ax.text((ncol * dx - 1.75) / 2, 4 * dy + 0.3, "Experiments by junction model (same 2000-wire network throughout); "
            "arrows: what each result led to", ha="center", fontsize=10 * FS, weight="bold")
    # every label must lie inside its card (2 per card: name, outcome)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for t, b in texts:
        bb = t.get_bbox_patch().get_window_extent() if t.get_bbox_patch() else t.get_window_extent()
        (x0, y0), (x1, y1) = inv.transform(bb.get_points())
        bx, by, bw, bh = boxes[b]
        m = 0.1                                       # required clearance from the card edge (data units)
        if x0 < bx + m or x1 > bx + bw - m or y0 < by + m or y1 > by + bh - m:
            print(f"warning: text {t.get_text()!r} overflows its card")
    ext = [((t.get_bbox_patch() or t).get_window_extent(), t, b) for t, b in texts]
    for i, (ea, ta, ba) in enumerate(ext):
        for eb, tb, bb in ext[i + 1:]:
            if ba == bb and ea.overlaps(eb):
                print(f"warning: {ta.get_text()!r} overlaps {tb.get_text()!r}")
    save(fig, "manual_fig2_taxonomy")


FIGS = {"fig1": fig1_pipeline, "fig2": fig2_taxonomy, "fig3": fig3_dualnode, "fig4": fig4_junctions,
        "fig5": fig5_layouts}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["paper", "presentation"], default="paper")
    ap.add_argument("--only", nargs="+", choices=list(FIGS))
    a = ap.parse_args()
    setup_style(a.mode)
    for k in a.only or FIGS:
        FIGS[k]()
