#!/usr/bin/env python3
"""N10 phenomenology (docs/N10_DESIGN.md sections 1 and 4): does the volatile junction reproduce the measured
AgNW network behaviour? Network: framework geometry (N = 2000, seed 1), two bands, 1 ms step, noise-free.

Protocols (voltages on the driven band, the other grounded), for tau_s in TAUS:
  sweep   one sine cycle, amplitude V_SWEEP, period 20 s: onset and hysteresis (and the Fig. 4 shape, qualitatively)
  pulse   V_ACT for 3 s (activation, M3), then read at 0.1 V for 12 s (spontaneous relaxation, M1)
  ppf     two 0.1 s pulses at V_ACT separated by gaps GAPS, read 0.1 V (paired-pulse facilitation, M6)
  consol  10 s pulse at V_LOW or V_ACT with consolidation on, then 30 s read at 0.1 V (M2)

    python scripts/n10_phenomenology.py vscale     # first: network onset vs v_scale -> config.VOLATILE_V_SCALE
    python scripts/n10_phenomenology.py sweep      # onset at the chosen v_scale
    python scripts/n10_phenomenology.py all [--procs 12]
    python scripts/n10_phenomenology.py figure

Traces: results/n10_phen/*.npz; figure figures/fig14_n10_phenomenology.png.
"""
import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.electrodes import two_bands  # noqa: E402
from agnw_rc.multicircuit import MultiTerminalCircuit  # noqa: E402
from agnw_rc.volatile import conductance, junction_kd0, step_state  # noqa: E402

OUT = ROOT / "results" / "n10_phen"
FIG = ROOT / "figures" / "fig14_n10_phenomenology.png"
TAUS = (0.015, 0.3, 3.0)
V_SWEEP = 5.0
V_ACT, V_LOW, V_READ = 3.5, 2.0, 0.1       # above / just below the ~2.2 V onset at v_scale 5 (vscale stage)
V_SCALES = (3.0, 5.0, 8.0)
GAPS = (0.05, 0.2, 1.0, 5.0)
RECORD = 10                                # record every RECORD steps


def run_waveform(v, tau_s, consolidation=False, network_seed=1, v_scale=None):
    """Drive the network with electrode voltages v (one per 1 ms step); returns t, V, I (A), <x>, <y>, ON fraction
    every RECORD steps."""
    vp = C.volatile_params(None, tau_s=tau_s, k_c=C.VOLATILE_K_C if consolidation else 0.0, dt=C.DT,
                           **({"v_scale": v_scale} if v_scale else {}))
    _, wd = C.network_params()
    net = C.make_network(network_seed)
    mc = MultiTerminalCircuit(net, wd, two_bands(net))
    kd0 = junction_kd0(vp, len(net.junctions), network_seed)
    state = np.zeros((2, len(net.junctions)))
    rec = {k: [] for k in ("t", "V", "I", "x", "y", "on")}
    for i, vi in enumerate(v):
        Vn = mc.solve(conductance(state, vp), [vi, 0.0], method="pcg")
        if i % RECORD == 0:
            rec["t"].append(i * C.DT)
            rec["V"].append(vi)
            rec["I"].append(mc.electrode_currents(Vn)[0])
            rec["x"].append(state[0].mean())
            rec["y"].append(state[1].mean())
            rec["on"].append(np.mean(np.minimum(1, state.sum(axis=0)) >= 0.5))
        state = step_state(state, mc.junction_voltage_drops(Vn), vp, kd0)
    return {k: np.array(val) for k, val in rec.items()}


def waveform(kind, gap=None):
    n = lambda s: int(round(s / C.DT))  # noqa: E731
    if kind == "sweep":
        t = np.arange(n(20.0)) * C.DT
        return V_SWEEP * np.sin(2 * np.pi * t / 20.0)
    if kind == "pulse":
        return np.concatenate([np.full(n(3.0), V_ACT), np.full(n(12.0), V_READ)])
    if kind == "ppf":
        p = np.full(n(0.1), V_ACT)
        return np.concatenate([np.full(n(0.5), V_READ), p, np.full(n(gap), V_READ), p, np.full(n(2.0), V_READ)])
    if kind.startswith("consol"):
        amp = V_LOW if kind == "consol_low" else V_ACT
        return np.concatenate([np.full(n(10.0), amp), np.full(n(30.0), V_READ)])
    raise ValueError(kind)


def task(args):
    kind, tau_s, gap = args
    name = f"{kind}_tau{tau_s:g}" + (f"_gap{gap:g}" if gap is not None else "")
    path = OUT / f"{name}.npz"
    if not path.exists():
        rec = run_waveform(waveform(kind, gap), tau_s, consolidation=kind.startswith("consol"))
        np.savez_compressed(path, **rec)
    return name


def tasks(stage):
    if stage == "sweep":
        return [("sweep", t, None) for t in TAUS]
    out = [("sweep", t, None) for t in TAUS] + [("pulse", t, None) for t in TAUS]
    out += [("ppf", t, g) for t in TAUS for g in GAPS]
    out += [(k, t, None) for k in ("consol_low", "consol_high") for t in TAUS]
    return out


def load(name):
    return dict(np.load(OUT / f"{name}.npz"))


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for tau, c in zip(TAUS, ("C0", "C1", "C2")):
        lab = f"τ_s = {tau:g} s"
        r = load(f"sweep_tau{tau:g}")
        ax[0, 0].plot(r["V"], r["I"] * 1e3, c, lw=0.8, label=lab)
        r = load(f"pulse_tau{tau:g}")
        ax[0, 1].semilogy(r["t"], np.abs(r["I"] / r["V"]), c, label=lab)
        ratio = []
        for g in GAPS:
            r = load(f"ppf_tau{tau:g}_gap{g:g}")
            G = np.abs(r["I"] / r["V"])
            t1 = 0.5 + 0.1 - 0.02
            t2 = 0.5 + 0.1 + g + 0.1 - 0.02
            ratio.append(G[np.searchsorted(r["t"], t2)] / G[np.searchsorted(r["t"], t1)])
        ax[0, 2].semilogx(GAPS, ratio, c, marker="o", label=lab)
        for kind, ls in (("consol_low", "--"), ("consol_high", "-")):
            r = load(f"{kind}_tau{tau:g}")
            ax[1, 0].semilogy(r["t"], np.abs(r["I"] / r["V"]), c, ls=ls,
                              label=f"{lab}, {V_LOW if kind == 'consol_low' else V_ACT:g} V")
            ax[1, 1].plot(r["t"], r["y"], c, ls=ls)
        r = load(f"sweep_tau{tau:g}")
        ax[1, 2].plot(r["t"], r["on"], c, label=lab)
    ax[0, 0].set(xlabel="V (V)", ylabel="I (mA)", title=f"sine sweep ±{V_SWEEP:g} V, 20 s")
    ax[0, 1].set(xlabel="t (s)", ylabel="G = I / V (S)", title=f"{V_ACT:g} V for 3 s, then read {V_READ} V")
    ax[0, 2].axhline(1, color="0.5", lw=0.8)
    ax[0, 2].set(xlabel="gap between pulses (s)", ylabel="G after pulse 2 / G after pulse 1",
                 title=f"paired pulses (0.1 s, {V_ACT:g} V)")
    ax[1, 0].set(xlabel="t (s)", ylabel="G (S)", title="10 s pulse, consolidation on (dashed: low amplitude)")
    ax[1, 1].set(xlabel="t (s)", ylabel="mean consolidated filament <y>", title="consolidation")
    ax[1, 2].set(xlabel="t (s)", ylabel="fraction of junctions ON", title="sweep: ON fraction")
    for a in ax.flat:
        a.legend(fontsize=7)
    fig.suptitle("N10 volatile junction: network phenomenology (N = 2000, two bands, noise-free)")
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


def vscale_task(args):
    vs, tau = args
    t = np.arange(10_000) * C.DT
    r = run_waveform(5.0 * np.sin(2 * np.pi * t / 20.0), tau, v_scale=vs)        # quarter cycle up to 5 V
    G = np.abs(r["I"] / np.where(r["V"] == 0, np.nan, r["V"]))
    G0 = np.nanmin(G[1:50])
    onset = float(r["V"][np.argmax(G > 10 * G0)]) if np.any(G > 10 * G0) else None
    return {"v_scale": vs, "tau_s": tau, "onset_V": onset, "I_max_mA": float(np.abs(r["I"]).max() * 1e3),
            "on_max": float(r["on"].max())}


def vscale(procs):
    """Network onset (conductance x 10 above its start) on a 0 -> 5 V quarter sine (5 s) vs v_scale and tau_s."""
    import json
    with ProcessPoolExecutor(procs) as pool:
        rows = list(pool.map(vscale_task, [(vs, t) for vs in V_SCALES for t in TAUS]))
    (ROOT / "results" / "n10_vscale.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["vscale", "sweep", "all", "figure"])
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    if a.stage == "figure":
        return figure()
    if a.stage == "vscale":
        return vscale(a.procs)
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.procs) as pool:
        for name in pool.map(task, tasks(a.stage)):
            print(name, flush=True)
    if a.stage == "sweep":
        for t in TAUS:
            r = load(f"sweep_tau{t:g}")
            up = (r["V"] > 0) & (np.gradient(r["V"]) > 0)
            G = np.abs(r["I"] / np.where(r["V"] == 0, np.nan, r["V"]))
            on = r["V"][up][np.argmax(G[up] > 10 * np.nanmin(G[up]))] if np.any(G[up] > 10 * np.nanmin(G[up])) else None
            print(f"tau_s {t:g}: onset (10 x conductance) {on} V, max I {np.abs(r['I']).max() * 1e3:.2f} mA, "
                  f"max ON fraction {r['on'].max():.3f}")


if __name__ == "__main__":
    main()
