#!/usr/bin/env python3
"""Figures from results/metrics.json, results/pilot.npz and a few raw runs.

    python scripts/make_figures.py [--mode paper|presentation] [--only fig1 fig3 ...]

Writes figures/figN_*.pdf (vector) and .png (for the Markdown docs). Deterministic: no random numbers are drawn
here (all stochasticity lives in the recorded simulations, whose seeds are in their file names).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.drive import load_raw, probe_nodes  # noqa: E402
from agnw_rc.evaluate import OBSERVABLES  # noqa: E402
from agnw_rc.tasks import MAX_LAG, narma10_target, split_indices  # noqa: E402

FIG = ROOT / "figures"
FRAMEWORK_TRACE = (Path(__import__("agnw_sim").__file__).resolve().parents[1]
                   / "validation_output/lamas_fig4/N2000_lamas_eps1e-4_seed1.npz")
RAW = ROOT / "results" / "raw"
STYLE = {"paper": FIG / "styles" / "paper_style.mplstyle",
         "presentation": FIG / "styles" / "presentation_style.mplstyle"}
OBS_LABEL = {"current": "current (2-terminal)", "probes": f"{C.N_PROBES} voltage probes",
             "junctions": "all junction states (oracle)"}
WINDOW_COLOR = {1.25: "#0072B2", 1.75: "#009E73", 2.25: "#D55E00"}
MF_WINDOW_COLOR = {3.5: "#0072B2", 4.5: "#009E73", 5.5: "#D55E00"}      # N1 mean-field windows
BASE_STYLE = {"delay_line_32": ("delay line (32 taps)", "0.35", ":"),
              "esn_32": ("ESN, 32 units", "#CC79A7", "--"),
              "esn_200": ("ESN, 200 units", "#E69F00", "-."),
              "input_only": ("input only", "0.6", (0, (1, 3)))}
WIDE = 7.0      # inches, two-column width


_SOURCE = {"function": "", "data": ""}       # set by main() for the figure being drawn


def save(fig, name):
    """Save PDF + PNG with provenance (script, function, data files, framework commit) in the file metadata."""
    prov = (f"scripts/make_figures.py:{_SOURCE['function']} | data: {_SOURCE['data']} | "
            f"agnw_sim commit {C.framework_commit()[:7]}")
    meta = {"pdf": {"Creator": "scripts/make_figures.py", "Subject": prov},
            "png": {"Software": "scripts/make_figures.py", "Description": prov}}
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None,
                    metadata=meta[ext])
    plt.close(fig)
    print(f"figures/{name}.pdf|png")


def panel_labels(axes):
    """(a), (b), ... as left-aligned titles; centred titles, if any, stay in place."""
    for ax, letter in zip(np.ravel(axes), "abcdefghijklmnop"):
        ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=8)


# ------------------------------------------------------------------ data access
def load_metrics():
    return json.loads((ROOT / "results" / "metrics.json").read_text())


def stochastic(r):
    """Framework two-electrode stochastic runs (multi-electrode N5 runs excluded)."""
    return r.get("model", "stochastic") == "stochastic" and not r.get("layout", "")


def e1(r):
    return (stochastic(r) and r["p_relax"] == C.P_RELAX and r["v_width"] == 1.0 and not r.get("n_samples", 0)
            and r["v_center"] in C.E1_CENTERS and r.get("replicate", 0) in C.E1_REPLICATES)


def n7b(r):
    """Stochastic framework model at the mean-field (M1) windows."""
    return (stochastic(r) and r["p_relax"] == C.P_RELAX and r["v_center"] in C.M1_CENTERS
            and r.get("replicate", 0) in C.E1_REPLICATES)


def e3(r):
    return stochastic(r) and r["p_relax"] == C.P_RELAX and bool(r.get("n_samples", 0))


def m1(r):
    return (r.get("model") == "meanfield" and r["p_relax"] == C.P_RELAX and not r.get("p_relax_max", 0)
            and not r.get("layout", ""))


E3_LABEL = "1.25-2.25 V, long hold (800 samples)"


def group(runs, stream, metric, obs, keyf):
    """{key: [values]} of one metric of one observable."""
    g = defaultdict(list)
    for r in runs:
        if r["stream"] == stream:
            g[keyf(r)].append(r["scores"][obs][metric])
    return g


def baseline_mean(M, stream, name, metric):
    vals = [M["baselines"][s][stream][name][metric] for s in M["baselines"]]
    return float(np.mean(vals)), float(np.std(vals))


# ------------------------------------------------------------------ figure 1: setup
def fig_setup(M):
    P = np.load(ROOT / "results" / "pilot.npz")
    net = C.make_network(1)
    from agnw_sim.dualnode import DualNodeCircuit
    circuit = DualNodeCircuit(net, C.network_params()[1])
    probes = probe_nodes(net, circuit, C.N_PROBES, 1)

    fig = plt.figure(figsize=(WIDE, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    side = np.sqrt(C.AREA) * 1e6
    seg = net.wire_endpoints * 1e6
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(seg, colors="0.55", linewidths=0.15))
    band = 0.05 * side
    ax.axvspan(-5, band, color="#D55E00", alpha=0.2, lw=0)
    ax.axvspan(side - band, side + 5, color="#0072B2", alpha=0.2, lw=0)
    pp = net.junction_pos[probes // 2] * 1e6
    ax.plot(pp[:, 0], pp[:, 1], "o", ms=3, color="k", mfc="#F0E442", mew=0.5)
    ax.text(band / 2, side / 2, "$V_{in}$ electrode", rotation=90, ha="center", va="center", fontsize=6)
    ax.text(side - band / 2, side / 2, "GND electrode", rotation=90, ha="center", va="center", fontsize=6)
    ax.set(xlim=(-5, side + 5), ylim=(-5, side + 5), aspect="equal", xlabel="$x$ (µm)", ylabel="$y$ (µm)")
    ax.set_title(f"{net.n_wires} wires, {len(net.junctions)} junctions\n(yellow: {C.N_PROBES} probes)",
                 fontsize=7, loc="right")

    ax = fig.add_subplot(gs[0, 1])
    v, g, on = P["stair_v"], P["g_level"] * 1e3, P["on_level"]
    n_up = int(np.argmax(v)) + 1
    ax.plot(v[:n_up], g[:n_up], "o-", color="k", label="increasing $V$")
    ax.plot(v[n_up - 1:], g[n_up - 1:], "s--", color="0.5", mfc="white", label="decreasing $V$")
    for vc, col in WINDOW_COLOR.items():
        ax.axvspan(vc - 0.5, vc + 0.5, color=col, alpha=0.12, lw=0)
    ax.set(xlabel="electrode voltage $V$ (V)", ylabel="conductance $I/V$ (mS)")
    ax.set_title(f"quasi-static $I/V$\n({int(P['per'])} steps/level)", fontsize=7, loc="right")
    ax.legend(fontsize=6)

    ax = fig.add_subplot(gs[0, 2])
    t = P["step_snaps"] * C.DT
    ax.plot(t, P["step_on"], color="k")
    ax.set(xlabel="time $t$ (s)", ylabel="ON fraction of junctions")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(P["step_v"])) * C.DT, P["step_v"], color="#D55E00", lw=0.8)
    ax2.set_ylabel("$V_{in}$ (V)", color="#D55E00")
    ax2.spines["right"].set_visible(True)
    ax2.tick_params(axis="y", colors="#D55E00")
    ax.set_title(f"step response\n($1/p_{{relax}}$ = {1 / C.P_RELAX * C.DT:.0f} s)", fontsize=7, loc="right")

    spec = C.RunSpec("iid", 30, 1.75, 1, 0)
    raw = load_raw(spec, RAW)
    n_show = 40
    steps = np.arange(n_show * spec.hold)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(steps * C.DT, np.repeat(raw["v_in"][:n_show], spec.hold), color="#D55E00", lw=0.9)
    ax.set(xlabel="time since end of washout (s)", ylabel="$V_{in}$ (V)")
    ax.yaxis.label.set_color("#D55E00")
    ax2 = ax.twinx()
    ax2.plot(steps * C.DT, raw["i_total_scored"][steps] * 1e3, color="k", lw=0.8)
    ends = np.arange(n_show) * spec.hold + spec.hold - 1
    ax2.plot(ends * C.DT, raw["current"][:n_show, -1] * 1e3, "o", color="k", ms=2.5, mfc="white")
    ax2.set_ylabel("total current $I$ (mA)")
    ax2.spines["right"].set_visible(True)
    ax.set_title(f"reservoir drive: i.i.d. input, hold = {spec.hold} steps ({spec.hold * C.DT * 1e3:.0f} ms), "
                 f"window {spec.v_lo:.2f}-{spec.v_hi:.2f} V; open circles: end-of-block samples", fontsize=7,
                 loc="right")
    panel_labels(fig.axes[:3] + [fig.axes[4]])
    save(fig, "fig1_setup")


# ------------------------------------------------------------------ figure 2: consistency
def fig_consistency(M):
    fig, axes = plt.subplots(1, 4, figsize=(WIDE, 2.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    spec = C.RunSpec("iid", 30, 1.75, 1, 0)
    a, b = load_raw(spec, RAW), load_raw(C.RunSpec("iid", 30, 1.75, 1, 1), RAW)
    idx = np.arange(MAX_LAG, MAX_LAG + 120)
    ax = axes[0]
    for raw, col, ls, lab in ((a, "k", "-", "replicate 1"), (b, "#D55E00", "--", "replicate 2")):
        ax.plot(idx, raw["current"][idx, -1] / raw["v_in"][idx] * 1e3, color=col, ls=ls, lw=0.9, label=lab)
    ax.set(xlabel="input sample $t$", ylabel="conductance $I/V_{in}$ (mS)")
    ax.set_title("same input, other noise +\ninitial state (hold 30)", fontsize=7, loc="right")
    ax.legend(fontsize=6, loc="lower right")
    cons = [c for c in M["consistency"] if e1(c)]
    c3 = [c for c in M["consistency"] if e3(c)]
    for ax, obs in zip(axes[1:], OBSERVABLES):
        key_inc, key_whole = obs + "_state_inc", obs + "_state"
        for vc, col in WINDOW_COLOR.items():
            holds = sorted({c["hold"] for c in cons})
            for k, ls, alpha, lab in ((key_inc, "-", 1.0, f"{vc - 0.5:.2f}-{vc + 0.5:.2f} V"),
                                      (key_whole, ":", 0.45, None)):
                vals = [[c["consistency"][k] for c in cons if c["hold"] == h and c["v_center"] == vc]
                        for h in holds]
                ax.errorbar(holds, [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals], marker="o",
                            ms=2.5, color=col, ls=ls, alpha=alpha, label=lab)
        if c3:
            h3 = sorted({c["hold"] for c in c3})
            v3 = [[c["consistency"][key_inc] for c in c3 if c["hold"] == h] for h in h3]
            ax.errorbar(h3, [np.mean(v) for v in v3], yerr=[np.std(v) for v in v3], marker="o", mfc="white",
                        ls="--", color=WINDOW_COLOR[1.75], label=E3_LABEL)
        ttl = {"current": "$I/V_{in}$",
               "probes": "$V_p/V_{in}$",
               "junctions": "junction states"}[obs]
        ax.set(xscale="log", xlabel="hold (steps)", ylim=(-0.05, 1.02))
        ax.set_title(ttl, fontsize=6.5, loc="right")
    axes[1].set_ylabel("state consistency (mean corr.)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=6)
    fig.suptitle("consistency across replicates: solid = sample-to-sample increments (input-driven part), "
                 "dotted = whole series (includes shared slow trends)", fontsize=7)
    panel_labels(axes)
    save(fig, "fig2_consistency")


# ------------------------------------------------------------------ figure 3: sweep grid
ROWS = [("iid", "MC", "linear memory\ncapacity MC", False),
        ("iid", "IPC2", "degree-2\ncapacity", False),
        ("iid", "narma10_nmse", "NARMA-10\ntest NMSE", True),
        ("sinesquare", "accuracy", "sine/square\ntest accuracy", False)]


def fig_sweep(M, select=e1, colors=None, fig_name="fig3_sweep", overlay_e3=True, consistency_row=False,
              obs_label=None):
    """Metric x observable grid vs hold for the runs picked by `select`, coloured by voltage window."""
    colors = colors or WINDOW_COLOR
    runs = [r for r in M["runs"] if select(r)]
    rows = ROWS + ([("iid", "consistency", "state consistency\n(increments)", False)] if consistency_row else [])
    fig, axes = plt.subplots(len(rows), 3, figsize=(WIDE, 1.8 * len(rows)), constrained_layout=True,
                             sharex=True, sharey="row")
    for i, (stream, metric, label, logy) in enumerate(rows):
        for j, obs in enumerate(OBSERVABLES):
            ax = axes[i, j]
            if metric == "consistency":
                key = obs + "_state_inc"
                g = defaultdict(list)
                for c in M["consistency"]:
                    if select(c):
                        g[(c["v_center"], c["hold"])].append(c["consistency"][key])
                for vc, col in colors.items():
                    holds = sorted(h for (v, h) in g if v == vc)
                    ax.errorbar(holds, [np.mean(g[(vc, h)]) for h in holds],
                                yerr=[np.std(g[(vc, h)]) for h in holds], marker="o", color=col)
                ax.set_ylim(-0.05, 1.05)
                if j == 0:
                    ax.set_ylabel(label)
                if i == len(rows) - 1:
                    ax.set_xlabel("hold (simulation steps of 1 ms)")
                continue
            g = group(runs, stream, metric, obs, lambda r: (r["v_center"], r["hold"]))
            for vc, col in colors.items():
                holds = sorted(h for (v, h) in g if v == vc)
                ax.errorbar(holds, [np.mean(g[(vc, h)]) for h in holds],
                            yerr=[np.std(g[(vc, h)]) for h in holds], marker="o", color=col,
                            label=f"{vc - 0.5:.2f}-{vc + 0.5:.2f} V")
            if stream == "iid" and overlay_e3:
                g3 = group([r for r in M["runs"] if e3(r)], stream, metric, obs, lambda r: r["hold"])
                if g3:
                    h3 = sorted(g3)
                    ax.errorbar(h3, [np.mean(g3[h]) for h in h3], yerr=[np.std(g3[h]) for h in h3], marker="o",
                                mfc="white", ls="--", color=WINDOW_COLOR[1.75], label=E3_LABEL)
            for name, (blabel, col, ls) in BASE_STYLE.items():
                m, _ = baseline_mean(M, stream, name, metric)
                if name == "delay_line_32" and metric == "MC":
                    continue            # 30 by construction; off the scale
                ax.axhline(m, color=col, ls=ls, lw=1, label=blabel)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            elif metric in ("MC", "IPC2"):
                ax.set_yscale("symlog", linthresh=1, linscale=1.5)
                ax.set_ylim(-0.05, 40)
            if i == 0:
                ax.set_title((obs_label or OBS_LABEL)[obs], fontsize=8)
            if j == 0:
                ax.set_ylabel(label)
            if i == len(rows) - 1:
                ax.set_xlabel("hold (simulation steps of 1 ms)")
    h, l = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[2, 0].get_legend_handles_labels()
    extra = [(hh, ll) for hh, ll in zip(h2, l2) if ll not in l]
    fig.legend(h + [e[0] for e in extra], l + [e[1] for e in extra], loc="outside lower center", ncol=4,
               fontsize=6)
    panel_labels(axes)
    save(fig, fig_name)


# ------------------------------------------------------------------ figure 4: memory function + NARMA trace
def best_config(M, obs, metric="MC", maximize=True):
    runs = [r for r in M["runs"] if e1(r) and r["stream"] == "iid"]
    g = group(runs, "iid", metric, obs, lambda r: (r["hold"], r["v_center"]))
    pick = max if maximize else min
    return pick(g, key=lambda k: np.mean(g[k]))


def fig_memory(M):
    runs = [r for r in M["runs"] if e1(r) and r["stream"] == "iid"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.4), constrained_layout=True)
    ax = axes[0]
    lags = np.arange(MAX_LAG + 1)
    floor = np.mean(list(M["noise_floor"].values()))
    for obs, col in zip(OBSERVABLES, ("#0072B2", "#009E73", "#D55E00")):
        hold, vc = best_config(M, obs)
        mf = np.mean([r["scores"][obs]["memory_function"] for r in runs
                      if r["hold"] == hold and r["v_center"] == vc], axis=0)
        ax.plot(lags, mf, "o-", ms=2, color=col, label=f"{obs}, hold {hold}")
    r3 = [r for r in M["runs"] if e3(r) and r["stream"] == "iid" and r["hold"] == max(C.E3_HOLDS)]
    if r3:
        mf = np.mean([r["scores"]["current"]["memory_function"] for r in r3], axis=0)
        ax.plot(lags, mf, "o--", ms=2, mfc="white", color="#0072B2", label=f"current, hold {max(C.E3_HOLDS)}")
    for name in ("esn_32", "esn_200"):
        mf = np.mean([M["baselines"][s]["iid"][name]["memory_function"] for s in M["baselines"]], axis=0)
        lab, col, ls = BASE_STYLE[name]
        ax.plot(lags, mf, color=col, ls=ls, label=lab)
    ax.axhline(floor, color="0.5", lw=0.8, ls=":", label="noise floor")
    ax.set(xlabel="lag $k$ (input samples)", ylabel="capacity $C_k$ for $u_{t-k}$", ylim=(-0.02, 1.02))
    ax.set_title("memory function\n(best-MC setting)", fontsize=7, loc="right")
    ax.legend(fontsize=5.5, loc="center right")

    # integration: capacity for exponential moving averages of the input vs their memory time
    ax = axes[1]
    lam = np.array(runs[0]["scores"]["probes"]["ema_lambdas"])
    tau = 1 / (1 - lam)
    for obs, col in zip(OBSERVABLES, ("#0072B2", "#009E73", "#D55E00")):
        hold, vc = best_config(M, obs)
        c = [r["scores"][obs]["ema_capacity"] for r in runs if r["hold"] == hold and r["v_center"] == vc]
        ax.errorbar(tau, np.mean(c, axis=0), yerr=np.std(c, axis=0), marker="o", ms=2.5, color=col,
                    label=obs)
        r3 = [r for r in M["runs"] if e3(r) and r["stream"] == "iid" and r["hold"] == max(C.E3_HOLDS)]
        if r3:
            c3 = [r["scores"][obs]["ema_capacity"] for r in r3]
            ax.plot(tau, np.mean(c3, axis=0), marker="o", ms=2.5, mfc="white", ls="--", color=col)
    for name in ("esn_200",):
        c = np.mean([M["baselines"][s]["iid"][name]["ema_capacity"] for s in M["baselines"]], axis=0)
        lab, col, ls = BASE_STYLE[name]
        ax.plot(tau, c, color=col, ls=ls, label=lab)
    ax.set(xscale="log", xlabel="memory time $1/(1-\\lambda)$ (input samples)",
           ylabel="capacity for EMA target", ylim=(-0.02, 1.02))
    ax.set_title("integration target\n$z_t=(1-\\lambda)\\sum_k \\lambda^k u_{t-k}$", fontsize=7, loc="right")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", marker="o", ms=2.5), Line2D([], [], color="0.3", marker="o", ms=2.5,
                                                                     mfc="white", ls="--")]
    labels += ["best E1 hold (100)", f"hold {max(C.E3_HOLDS)} (E3)"]
    ax.legend(handles, labels, fontsize=5, loc="upper right")

    # NARMA-10 test trace
    ax = axes[2]
    obs = "junctions"
    hold, vc = best_config(M, obs, "narma10_nmse", maximize=False)
    r = next(r for r in runs if r["hold"] == hold and r["v_center"] == vc and r["network_seed"] == 1
             and r["replicate"] == 0)
    raw = load_raw(C.RunSpec("iid", hold, vc, 1, 0), RAW)
    y = narma10_target(raw["u"])
    _, _, test = split_indices(len(y))
    t = test[:100]
    ax.plot(t, y[t], color="k", label="target")
    ax.plot(t, np.array(r["scores"][obs]["narma10_pred"])[:100], color="#D55E00", ls="--",
            label=f"{obs} readout (NMSE {r['scores'][obs]['narma10_nmse']:.2f})")
    probe_pred = np.array(r["scores"]["probes"]["narma10_pred"])[:100]
    ax.plot(t, probe_pred, color="#009E73", lw=0.8,
            label=f"probes readout (NMSE {r['scores']['probes']['narma10_nmse']:.2f})")
    ax.set(xlabel="input sample $t$ (test segment)", ylabel="NARMA-10 $y_{t+1}$")
    ax.set_title(f"NARMA-10 test segment\n(hold {hold}, {vc - 0.5:.2f}-{vc + 0.5:.2f} V)", fontsize=7, loc="right")
    ax.legend(fontsize=5.5)
    panel_labels(axes)
    save(fig, "fig4_memory_nonlinear")


# ------------------------------------------------------------------ figure 5: relaxation sensitivity (E2)
def fig_prelax(M):
    runs = [r for r in M["runs"] if stochastic(r) and r["v_width"] == 1.0 and not r.get("n_samples", 0)]
    prs = sorted({r["p_relax"] for r in runs})
    if len(prs) < 2:
        print("no E2 runs; skipping fig5")
        return
    e2_hold, e2_vc = next((r["hold"], r["v_center"]) for r in runs if r["p_relax"] != C.P_RELAX)
    runs = [r for r in runs if r["hold"] == e2_hold and r["v_center"] == e2_vc]
    cons = [c for c in M["consistency"] if c["hold"] == e2_hold and c["v_center"] == e2_vc
            and not c.get("n_samples", 0) and stochastic(c)]
    panels = [("iid", "MC", "MC"), ("iid", "IPC2", "degree-2\ncapacity"),
              ("iid", "narma10_nmse", "NARMA-10\nNMSE"), ("sinesquare", "accuracy", "sine/square\naccuracy"),
              (None, "consistency", "state\nconsistency")]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDE, 2.1), constrained_layout=True)
    xpos = np.arange(len(prs))
    for ax, (stream, metric, label) in zip(axes, panels):
        for obs, col in zip(OBSERVABLES, ("#0072B2", "#009E73", "#D55E00")):
            if stream is None:
                k = obs + "_state_inc"
                vals = [[c["consistency"][k] for c in cons if c["p_relax"] == p] for p in prs]
            else:
                vals = [[r["scores"][obs][metric] for r in runs if r["stream"] == stream and r["p_relax"] == p]
                        for p in prs]
            ax.errorbar(xpos, [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals], marker="o",
                        color=col, label=obs)
        ax.set_xticks(xpos, ["0" if p == 0 else f"$10^{{{int(np.log10(p))}}}$" for p in prs], fontsize=6)
        ax.axvspan(prs.index(C.P_RELAX) - 0.3, prs.index(C.P_RELAX) + 0.3, color="0.85", lw=0, zorder=0)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set(xlabel="$p_{relax}$ (per step)")
        ax.set_title(label, fontsize=7, loc="right")
        if metric == "narma10_nmse":
            ax.set_yscale("log")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"relaxation sensitivity at hold {e2_hold}, window {e2_vc - 0.5:.2f}-{e2_vc + 0.5:.2f} V "
                 f"(0 = published model; shaded = framework working value)", fontsize=8)
    panel_labels(axes)
    save(fig, "fig5_prelax")


# ------------------------------------------------------------------ figure 7: heterogeneous forgetting (N2)
BANK = {}      # ideal leaky-integrator bank reference values, filled by fig_n2


def n2_condition(r):
    """Label of the forgetting condition of a mean-field run at the N2 window, or None."""
    if r.get("model") != "meanfield" or r["v_center"] != C.N2_CENTER or r["hold"] not in C.N2_HOLDS:
        return None
    if r.get("p_relax_max", 0):
        return f"log-unif.\n$10^{{-4}}$-$10^{{{int(np.log10(r['p_relax_max']))}}}$"
    return f"homog.\n$10^{{{int(np.log10(r['p_relax']))}}}$"


def fig_n2(M):
    global BANK
    bank_path = ROOT / "results" / "leaky_bank_reference.json"
    if bank_path.exists():                  # scripts/leaky_bank_reference.py: K = 200, tau = 1..100 samples
        J = json.loads(bank_path.read_text())
        b = J["banks"]
        BANK = {"MC": [(J["noise"]["noise-free, wide grid (alpha >= 1e-14)"]["MC"], ":", "ideal leaky bank K=200, "
                        "noise-free"), (J["noise"]["SNR 1e4"]["MC"], "-.", "same bank, feature SNR 1e4")],
                "IPC2": [(b["K=200, tau=1..100, +x^2"]["IPC2"], ":", "ideal leaky bank K=200 (+x^2), noise-free")],
                "narma10_nmse": [(b["K=200, tau=1..100"]["narma10_nmse"], ":", "ideal leaky bank K=200, "
                                  "noise-free")]}
    runs = [r for r in M["runs"] if n2_condition(r)]
    cons = [c for c in M["consistency"] if n2_condition(c)]
    if not any(r.get("p_relax_max", 0) for r in runs):
        print("no N2 runs; skipping fig7")
        return
    order = ["homog.\n$10^{-4}$", "log-unif.\n$10^{-4}$-$10^{-3}$", "log-unif.\n$10^{-4}$-$10^{-2}$",
             "log-unif.\n$10^{-4}$-$10^{-1}$", "homog.\n$10^{-3}$", "homog.\n$10^{-2}$"]
    order = [o for o in order if any(n2_condition(r) == o for r in runs)]
    panels = [("iid", "MC", "MC"), ("iid", "IPC2", "degree-2@capacity"),
              ("iid", "narma10_nmse", "NARMA-10@NMSE"), ("sinesquare", "accuracy", "sine/square@accuracy"),
              (None, "consistency", "state consistency@(increments)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDE, 2.5), constrained_layout=True)
    x = np.arange(len(order))
    for ax, (stream, metric, label) in zip(axes, panels):
        for obs, col in zip(OBSERVABLES, ("#0072B2", "#009E73", "#D55E00")):
            for hold, ls, mfc in zip(C.N2_HOLDS, ("-", "--"), (col, "white")):
                if stream is None:
                    k = obs + "_state_inc"
                    vals = [[c["consistency"][k] for c in cons if n2_condition(c) == o and c["hold"] == hold]
                            for o in order]
                else:
                    vals = [[r["scores"][obs][metric] for r in runs if r["stream"] == stream
                             and n2_condition(r) == o and r["hold"] == hold] for o in order]
                ax.errorbar(x + (0.1 if hold == C.N2_HOLDS[1] else -0.1), [np.mean(v) for v in vals],
                            yerr=[np.std(v) for v in vals], marker="o", ms=3, color=col, ls="none", mfc=mfc,
                            elinewidth=0.8, label=f"{obs}, hold {hold}")
        for ref, ls_ref, lab_ref in BANK.get(metric, []):
            ax.axhline(ref, color="0.4", ls=ls_ref, lw=1, label=lab_ref)
        ax.axvspan(0.5, 3.5, color="0.92", lw=0, zorder=0)
        ax.set_xticks(x, order, fontsize=5, rotation=90)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_title(label.replace("@", "\n"), fontsize=7, loc="right")
        if metric == "narma10_nmse":
            ax.set_yscale("log")
    axes[0].set_ylabel("score")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=6)
    fig.suptitle(f"N2 heterogeneous forgetting (mean-field, window {C.N2_CENTER - 0.5:.1f}-{C.N2_CENTER + 0.5:.1f} V);"
                 " shaded: per-junction $p_{relax}$ spread", fontsize=8)
    panel_labels(axes)
    save(fig, "fig7_n2_heterogeneous")


# ------------------------------------------------------------------ figure S1: mean-field vs stochastic onset
def fig_onset(M):
    path = ROOT / "results" / "meanfield_pilot.npz"
    if not path.exists():
        print("no results/meanfield_pilot.npz (run scripts/meanfield_pilot.py); skipping figS1")
        return
    S, F = np.load(ROOT / "results" / "pilot.npz"), np.load(path)
    fig, axes = plt.subplots(1, 2, figsize=(WIDE * 0.75, 2.4), constrained_layout=True)
    ax = axes[0]
    for d, v, g, col, lab in ((S, S["stair_v"], S["g_level"], "k", "stochastic (framework)"),
                              (F, F["v"], F["g"], "#D55E00", "mean-field (N1)")):
        n_up = int(np.argmax(v)) + 1
        ax.plot(v[:n_up], g[:n_up] * 1e3, "o-", color=col, ms=3, label=f"{lab}, up")
        ax.plot(v[n_up - 1:], g[n_up - 1:] * 1e3, "s--", color=col, ms=3, mfc="white", label=f"{lab}, down")
    ax.set(xlabel="electrode voltage $V$ (V)", ylabel="conductance $I/V$ (mS)")
    ax.set_title("quasi-static staircase", fontsize=7, loc="right")
    ax.legend(fontsize=5.5)
    ax = axes[1]
    v = F["v"]
    n_up = int(np.argmax(v)) + 1
    ax.plot(v[:n_up], F["max_dv"][:n_up], "o-", color="#D55E00", ms=3, label="largest junction drop, up")
    ax.axhline(C.network_params()[0].V_th, color="0.5", ls=":", label="$V_{th}$")
    ax.plot(v[:n_up], F["mean_p"][:n_up], "^-", color="#0072B2", ms=3, label="mean $P_j$, up")
    ax.plot(v[:n_up], F["frac_on"][:n_up], "v--", color="#0072B2", ms=3, mfc="white",
            label="fraction $P_j > 0.5$, up")
    ax.set(xlabel="electrode voltage $V$ (V)", ylabel="drop (V) or probability (-)")
    ax.set_title("mean-field junction voltages\nand ON probabilities", fontsize=7, loc="right")
    ax.legend(fontsize=5.5)
    panel_labels(axes)
    save(fig, "figS1_meanfield_onset")


# ------------------------------------------------------------------ figure 8: noise-averaging curve (N7a)
def fig_noise_avg(M):
    path = ROOT / "results" / "noise_averaging.json"
    if not path.exists():
        print("no results/noise_averaging.json (run scripts/noise_averaging.py); skipping fig8")
        return
    D = json.loads(path.read_text())
    Rs, seeds = D["Rs"], list(D["per_seed"])
    panels = [("MC", "MC"), ("IPC2", "degree-2@capacity"), ("narma10_nmse", "NARMA-10@NMSE"),
              ("consistency", "state consistency, increments@(two groups of R)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDE, 2.2), constrained_layout=True)
    for ax, (metric, label) in zip(axes, panels):
        for obs, col in zip(OBSERVABLES, ("#0072B2", "#009E73", "#D55E00")):
            if metric == "consistency":
                ck = "consistency_inc" if "consistency_inc" in D["per_seed"][seeds[0]] else "consistency"
                rr = [R for R in Rs if str(R) in D["per_seed"][seeds[0]][ck]]
                vals = [[D["per_seed"][s][ck][str(R)][obs] for s in seeds] for R in rr]
            else:
                rr = Rs
                vals = [[D["per_seed"][s]["scores"][str(R)][obs][metric] for s in seeds] for R in rr]
            ax.errorbar(rr, [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals], marker="o", ms=3,
                        color=col, label=obs)
        if metric in ("MC", "IPC2", "narma10_nmse"):
            m, _ = baseline_mean(M, "iid", "input_only", metric)
            ax.axhline(m, color="0.6", ls=(0, (1, 3)), label="input only")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Rs, [str(R) for R in Rs])
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel("replicates averaged $R$")
        ax.set_title(label.replace("@", "\n"), fontsize=7, loc="right")
        if metric == "narma10_nmse":
            ax.set_yscale("log")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"N7a noise averaging (stochastic framework model, hold {D['hold']}, "
                 f"{D['v_center'] - 0.5:.2f}-{D['v_center'] + 0.5:.2f} V; mean ± sd over 3 networks)", fontsize=8)
    panel_labels(axes)
    save(fig, "fig8_noise_averaging")


# ------------------------------------------------------------------ figure S2: multi-electrode pilot (N5)
def fig_multi_pilot(M):
    path = ROOT / "results" / "multi_pilot.npz"
    if not path.exists():
        print("no results/multi_pilot.npz (run scripts/multi_pilot.py); skipping figS2")
        return
    D = np.load(path)
    v = D["v"]
    n_up = int(np.argmax(v)) + 1
    fig, axes = plt.subplots(1, 2, figsize=(WIDE * 0.75, 2.4), constrained_layout=True)
    styles = {"two_bands": ("k", "two bands (framework layout)"), "split_bands": ("#0072B2", "split bands, 8 in / 8 out"),
              "pads": ("#D55E00", "pads 4x4, 8 in / 8 out")}
    for key, (col, lab) in styles.items():
        for ax, q, scale in ((axes[0], "g", 1e3), (axes[1], "on", 1.0)):
            y = D[f"{q}_{key}"] * scale
            ls = ":" if key == "two_bands" else "-"
            ax.plot(v[:n_up], y[:n_up], marker="o", ms=2.5, ls=ls, color=col, label=f"{lab}, up")
            ax.plot(v[n_up - 1:], y[n_up - 1:], marker="s", ms=2.5, ls="--", mfc="white", color=col,
                    label=f"{lab}, down")
    axes[0].set(yscale="log", xlabel="driven-electrode voltage $V$ (V)", ylabel="conductance $I_{in}/V$ (mS)")
    axes[0].set_ylim(1e-2, 50)
    axes[1].set(xlabel="driven-electrode voltage $V$ (V)", ylabel="ON fraction of junctions")
    axes[0].set_title("quasi-static staircase", fontsize=7, loc="right")
    axes[1].legend(fontsize=5)
    panel_labels(axes)
    save(fig, "figS2_multi_pilot")


# ------------------------------------------------------------------ figure S3: network + electrode layouts
LAYOUT_TITLES = {"two_bands": "two bands + 32 probes (E1-E4, N4, N10-1)",
                 "split_bands": "split bands, 8 + 8 segments (N5a, N10-E0)",
                 "pads": "4x4 pads, 50 µm, 8 in + 8 out (N5a, N10-E, N10-M)"}


def fig_layouts(M):
    """The three electrode configurations on network seed 1: wires (grey), electrodes (driven orange, grounded blue),
    the wires contacting each electrode (coloured) with their contact points, and the 32 voltage probes (yellow).
    Saved as the full figure S3 and as two parts: the layout in use (setup_current) and the ones planned next
    (setup_next)."""
    _layouts_figure(("two_bands", "split_bands", "pads"), "figS3_electrode_layouts")
    _layouts_figure(("two_bands",), "fig_setup_current")
    _layouts_figure(("split_bands", "pads"), "fig_setup_next")


def _layouts_figure(layouts, name):
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from agnw_rc.electrodes import make_layout
    from agnw_rc.multicircuit import MultiTerminalCircuit
    from agnw_rc.multidrive import _probe_nodes

    net = C.make_network(1)
    wd = C.network_params()[1]
    side = np.sqrt(C.AREA) * 1e6
    band = 0.05 * side
    role_col = {"in": "#D55E00", "gnd": "#0072B2", "float": "0.3"}
    n = len(layouts)
    fig, axes = plt.subplots(1, n, figsize=(WIDE * n / 3 + (0.6 if n == 1 else 0), 2.9 + (0.25 if n == 1 else 0)),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, layout in zip(axes, layouts):
        els = make_layout(layout, net)
        mc = MultiTerminalCircuit(net, wd, els)
        seg = net.wire_endpoints * 1e6
        ax.add_collection(LineCollection(seg, colors="0.72", linewidths=0.12))
        # electrode areas
        if layout == "pads":
            n_side = 4
            for k, e in enumerate(els):
                i, j = divmod(k, n_side)
                cx, cy = (i + 0.5) * side / n_side, (j + 0.5) * side / n_side
                ax.add_patch(Rectangle((cx - 25, cy - 25), 50, 50, color=role_col[e.role], alpha=0.35, lw=0))
                ax.text(cx, cy + 40, "in" if e.role == "in" else "out", ha="center", va="bottom", fontsize=5,
                        color=role_col[e.role])
        else:
            n_seg = {"in": sum(e.role == "in" for e in els), "gnd": sum(e.role == "gnd" for e in els)}
            count = {"in": 0, "gnd": 0}
            for e in els:
                x0 = 0 if e.role == "in" else side - band
                h = side / n_seg[e.role]
                y0 = count[e.role] * h
                count[e.role] += 1
                ax.add_patch(Rectangle((x0, y0), band, h, color=role_col[e.role], alpha=0.3, lw=0))
                if layout == "split_bands" and y0 > 0:             # segment boundaries, drawn above the wires
                    ax.plot([x0 - 20, x0 + band + 20], [y0, y0], color="k", lw=1.0, zorder=5)
        # contacting wires and contact points
        for e in els:
            if len(e.wires):
                ax.add_collection(LineCollection(seg[e.wires], colors=role_col[e.role], linewidths=0.6))
                p1, p2 = net.wire_endpoints[e.wires, 0], net.wire_endpoints[e.wires, 1]
                pts = (p1 + (p2 - p1) * (e.t / net.wire_length)[:, None]) * 1e6
                ax.plot(pts[:, 0], pts[:, 1], "o", ms=1.2, color="k")
        probes = _probe_nodes(mc, C.N_PROBES, 1)
        pp = net.junction_pos[probes // 2] * 1e6
        ax.plot(pp[:, 0], pp[:, 1], "o", ms=2.5, color="k", mfc="#F0E442", mew=0.4)
        n_in = sum(len(e.wires) for e in els if e.role == "in")
        n_out = sum(len(e.wires) for e in els if e.role == "gnd")
        ax.set(xlim=(-10, side + 10), ylim=(-10, side + 10), aspect="equal", xlabel="$x$ (µm)")
        ax.set_title(f"{LAYOUT_TITLES[layout]}\nwires: {n_in} driven, {n_out} grounded", fontsize=6.5, loc="right")
    axes[0].set_ylabel("$y$ (µm)")
    handles = [Line2D([], [], color=role_col["in"], lw=3, alpha=0.6, label="driven electrode + its wires"),
               Line2D([], [], color=role_col["gnd"], lw=3, alpha=0.6, label="grounded electrode (current readout)"),
               Line2D([], [], color="k", marker="o", ms=2, ls="none", label="contact point"),
               Line2D([], [], color="k", marker="o", ms=4, mfc="#F0E442", ls="none", label="voltage probe (32)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2 if n == 1 else 4, fontsize=6)
    fig.suptitle(f"network seed 1: {net.n_wires} wires, {len(net.junctions)} junctions, 1 mm x 1 mm", fontsize=8)
    if n > 1:
        panel_labels(axes)
    save(fig, name)


MULTI_OBS_LABEL = {"current": "16 electrode currents", "probes": f"{C.N_PROBES} voltage probes",
                   "junctions": "all junction states (oracle)"}


# ------------------------------------------------------------------ figure N4-4: calibration to Lamas Fig. 4
def fig_n4_calibration(M):
    """(a) loss map V_set x b (bipolar, N = 2000, seed 1; minimum over Lambda_crit), (b) calibration metrics:
    paper vs N4 at the calibrated point (mean +- sd over 4 network seeds) for N = 2000 (fitted) and 3000
    (not fitted), (c) I-V loop of the calibrated point vs the framework binary model."""
    path = ROOT / "results" / "n4_calibration.json"
    if not path.exists():
        print("no results/n4_calibration.json (run scripts/n4_calibrate.py); skipping figN4_4")
        return
    from agnw_rc.fig4 import paper_targets
    R = json.loads(path.read_text())
    pc = C.N4_PARAMS
    grid = [r for r in R.values() if r["n_wires"] == 2000 and r["network_seed"] == 1 and not r.get("unipolar")]
    vs = sorted({round(r["V_set"], 6) for r in grid})
    bs = sorted({round(r["b"], 6) for r in grid})
    L = np.full((len(vs), len(bs)), np.nan)
    for r in grid:
        i, j = vs.index(round(r["V_set"], 6)), bs.index(round(r["b"], 6))
        L[i, j] = np.nanmin([L[i, j], np.log10(r["loss"])])
    T = paper_targets()
    fig = plt.figure(figsize=(WIDE, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(L, origin="lower", cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(bs)), [f"{v:g}" for v in bs], fontsize=5.5, rotation=45)
    ax.set_yticks(range(len(vs)), [f"{v:g}" for v in vs], fontsize=5.5)
    ax.tick_params(which="minor", left=False, bottom=False)
    ax.plot(bs.index(round(pc["b"], 6)), vs.index(round(pc["V_set"], 6)), "r*", ms=9)
    ax.set(xlabel="dissolution factor $b$", ylabel="$V_{set}$ (V)")
    ax.set_title("log$_{10}$ loss (min over $\\Lambda_{crit}$)", fontsize=7, loc="right")
    fig.colorbar(im, ax=ax, shrink=0.9)
    ax = fig.add_subplot(gs[0, 1])
    keys = ["Imax", "onset_V", "area_pos", "area_neg", "ratio"]
    labels = ["$I_{max}$", "onset", "$A_+$", "$A_-$", "$A_-/A_+$"]
    x = np.arange(len(keys))
    for off, N, col in ((-0.2, 2000, "#D55E00"), (0.2, 3000, "#0072B2")):
        rs = [r for r in R.values() if r["n_wires"] == N and not r.get("unipolar")
              and all(abs(r[k] - pc[k]) < 1e-12 for k in ("V_set", "b", "lambda_crit"))]
        rel = np.array([[r["metrics"][k] / T[N][k] for k in keys] for r in rs])
        ax.bar(x + off, rel.mean(0), 0.38, yerr=rel.std(0), color=col, capsize=2,
               label=f"N = {N} ({'fitted' if N == 2000 else 'not fitted'}, {len(rs)} seeds)")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(x, labels, fontsize=6.5)
    ax.tick_params(which="minor", bottom=False)
    ax.set(ylabel="N4 / paper", ylim=(0, 1.5))
    ax.legend(fontsize=6)
    ax.set_title("calibrated point vs paper", fontsize=7, loc="right")
    ax = fig.add_subplot(gs[1, :])
    k = f"N2000_s1_Vset{pc['V_set']:g}_b{pc['b']:g}_lc{pc['lambda_crit']:g}"
    z = np.load(ROOT / "results" / "n4_calib" / f"{k}.npz")
    ax.plot(z["V_in"], z["I_total"] * 1e3, color="#D55E00", lw=0.9,
            label=f"N4 calibrated: $V_{{set}}$ = {pc['V_set']:g} V, $b$ = {pc['b']:g}, "
                  f"$\\Lambda_{{crit}}$ = {pc['lambda_crit']:g} V s")
    fwz = np.load(FRAMEWORK_TRACE)
    ax.plot(fwz["V_in"], fwz["I_total"] * 1e3, color="0.45", lw=0.7, ls="--",
            label="framework binary model ($p_{relax}$ = 1e-4), same protocol")
    ax.axvline(T[2000]["onset_V"], color="0.6", ls=":", lw=0.8)
    ax.text(T[2000]["onset_V"] + 0.05, 0.85 * T[2000]["Imax"], "paper onset", fontsize=6)
    ax.axhline(T[2000]["Imax"], color="0.6", ls=":", lw=0.8)
    ax.text(-3.5, T[2000]["Imax"] + 0.1, "paper $I_{max}$", fontsize=6)
    ax.set(xlabel="$V_{in}$ (V)", ylabel="$I$ (mA)")
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("I-V loop, one 50 s cycle of a 3.6 V sine, N = 2000, network seed 1", fontsize=7, loc="right")
    panel_labels([fig.axes[0], fig.axes[2], fig.axes[3]])
    save(fig, "figN4_4_calibration")


def n4_1_select(r):
    return (r.get("model") == "filament" and r.get("layout") == "two_bands" and not r.get("sigma", 0)
            and r["v_center"] in C.N4_1_CENTERS)


def n5a_select(layout):
    return lambda r: (r.get("layout", "") == layout and r.get("encoding") == "weights"
                      and r.get("model", "stochastic") == "stochastic")


def _layout_colors(layout):
    return dict(zip(C.N5A_CENTERS[layout], ("#0072B2", "#009E73", "#D55E00")))


FIGURES = {   # name: (function, data files it reads)
    "fig1": (fig_setup, "results/pilot.npz, results/raw/iid_h30_v1.75_*_n1_r0.npz, network seed 1"),
    "fig2": (fig_consistency, "results/metrics.json, results/raw/iid_h30_v1.75_*_n1_r{0,1}.npz"),
    "fig3": (fig_sweep, "results/metrics.json (E1, E3, baselines)"),
    "fig4": (fig_memory, "results/metrics.json, results/raw/iid_h100_*_n1_r0.npz"),
    "fig5": (fig_prelax, "results/metrics.json (E2)"),
    "fig6": (lambda M: fig_sweep(M, select=m1, colors=MF_WINDOW_COLOR, fig_name="fig6_meanfield_sweep",
                                 overlay_e3=False, consistency_row=True) if any(m1(r) for r in M["runs"])
             else print("no M1 runs; skipping fig6"), "results/metrics.json (M1)"),
    "fig7": (fig_n2, "results/metrics.json (N2, M1), results/leaky_bank_reference.json"),
    "figS1": (fig_onset, "results/pilot.npz, results/meanfield_pilot.npz"),
    "fig8": (fig_noise_avg, "results/noise_averaging.json, results/metrics.json (baselines)"),
    "fig10a": (lambda M: fig_sweep(M, select=n5a_select("split_bands"), colors=_layout_colors("split_bands"),
                                   fig_name="fig10a_n5a_split_bands", overlay_e3=False, consistency_row=True,
                                   obs_label=MULTI_OBS_LABEL)
               if any(n5a_select("split_bands")(r) for r in M["runs"]) else print("no N5a runs; skipping fig10a"),
               "results/metrics.json (N5a split bands)"),
    "fig10b": (lambda M: fig_sweep(M, select=n5a_select("pads"), colors=_layout_colors("pads"),
                                   fig_name="fig10b_n5a_pads", overlay_e3=False, consistency_row=True,
                                   obs_label=MULTI_OBS_LABEL)
               if any(n5a_select("pads")(r) for r in M["runs"]) else print("no N5a runs; skipping fig10b"),
               "results/metrics.json (N5a pads)"),
    "fig11": (lambda M: fig_sweep(M, select=n4_1_select, colors=dict(zip(C.N4_1_CENTERS,
                                                                     ("#0072B2", "#009E73", "#D55E00"))),
                                  fig_name="fig11_n4_1", overlay_e3=False, consistency_row=True)
              if any(n4_1_select(r) for r in M["runs"]) else print("no N4-1 runs; skipping fig11"),
              "results/metrics.json (N4-1)"),
    "figS2": (fig_multi_pilot, "results/multi_pilot.npz"),
    "figS3": (fig_layouts, "network seed 1 (agnw_rc.config.make_network), agnw_rc.electrodes layouts"),
    "figN4_4": (fig_n4_calibration, "results/n4_calibration.json, results/n4_calib/*.npz, framework Fig. 4 trace"),
    "fig9": (lambda M: fig_sweep(M, select=n7b, colors=MF_WINDOW_COLOR, fig_name="fig9_stochastic_high_v",
                                 overlay_e3=False, consistency_row=True) if any(n7b(r) for r in M["runs"])
             else print("no N7b runs; skipping fig9"), "results/metrics.json (N7b)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(STYLE), default="paper")
    ap.add_argument("--only", nargs="*", choices=list(FIGURES), help="figures to draw (default: all)")
    a = ap.parse_args()
    plt.style.use(STYLE[a.mode])
    M = load_metrics()
    for key in a.only or FIGURES:
        fn, data = FIGURES[key]
        _SOURCE["function"] = {"fig6": "fig_sweep(select=m1)", "fig9": "fig_sweep(select=n7b)",
                               "fig10a": "fig_sweep(select=n5a split_bands)",
                               "fig10b": "fig_sweep(select=n5a pads)",
                               "fig11": "fig_sweep(select=n4_1)"}.get(
            key, getattr(fn, "__name__", key))
        _SOURCE["data"] = data
        fn(M)


if __name__ == "__main__":
    main()
