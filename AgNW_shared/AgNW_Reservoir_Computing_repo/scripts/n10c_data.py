#!/usr/bin/env python3
"""N10-C stage A: the measured network-level statistics of Milano et al. 2025 (Nat. Commun., DOI
10.1038/s41467-025-58741-2), from their open dataset (Zenodo 10.5281/zenodo.15050217, CC-BY) in data/milano2025/.

Measured device: Ag NWs 20-50 um long, ~115 nm diameter, 20 uL drop at 0.09 mg/mL on ~1 x 1 cm, two electrodes
~7 mm apart, sampled every 0.6 s. Extracted here (all from the data files, not from the paper text):
  * steady state G~(V) (Fig. 2b) and the onset with our definition (driven conductance 10x its low-voltage value,
    on their model curve because the measured low-voltage conductance is at the instrument floor);
  * the 3.6 V stationary state (Figs. 3, 5a): mean G, Gaussian noise of dG/dt (Fig. 3d = jumps removed), the jump
    threshold (largest |dG/dt| kept as Gaussian), jumps per second (Fig. 3g), up/down balance, amplitude power law
    (continuous MLE above the threshold, as in their Fig. 3i), all also as relative changes per sample, Delta G / G;
  * relaxation at 10 mV (Fig. 1e) and the fitted reversion speed theta(V) (Fig. 2e);
  * density n L^2 of their network (from the deposited mass) vs ours.

    python scripts/n10c_data.py   -> results/n10c_data.json, figures/fig21_n10c_data.png
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agnw_rc import config as C  # noqa: E402

DATA = ROOT / "data" / "milano2025"
OUT = ROOT / "results" / "n10c_data.json"
FIG = ROOT / "figures" / "fig21_n10c_data.png"
RHO_AG = 10490.0          # kg/m^3


def load(name, cols=(0, 1)):
    return np.loadtxt(DATA / f"Figure {name}.txt", skiprows=2, usecols=cols)


def powerlaw_mle(x, xmin):
    """Continuous power-law exponent alpha (p ~ x^-alpha) for x >= xmin (Clauset et al. 2009) and its standard error."""
    x = x[x >= xmin]
    a = 1.0 + len(x) / np.sum(np.log(x / xmin))
    return float(a), float((a - 1.0) / np.sqrt(len(x))), len(x)


def main():
    out = {}
    # ---- steady state vs voltage (Fig. 2b) and fitted rates (Figs. 2d-g)
    ss = np.loadtxt(DATA / "Figure 2b.txt", skiprows=2)
    v, g_exp, g_mod = ss.T
    onset_model = float(v[np.argmax(g_mod > 10 * g_mod[0])])
    onset_exp = float(v[np.argmax(g_exp > 1e-9)])
    rates = np.loadtxt(DATA / "Figure 2d,e,f,g.txt", skiprows=2)
    out["steady_state"] = {"onset_V_model_10x": onset_model, "first_V_measured_above_1nS": onset_exp,
                           "G_sat_S": float(np.median(g_exp[v >= 5.0])), "gap_m": 7e-3,
                           "theta_low_V_per_s": float(rates[0, 2]),
                           "theta_3p6V_per_s": float(np.interp(3.6, rates[:, 0], rates[:, 2]))}
    # ---- 3.6 V stationary state
    t5, G5 = load("5a").T
    G36 = float(G5[t5 > 18500].mean())
    raw = load("3a,b,c")
    gauss = load("3d,e,f")
    dt = float(np.median(np.diff(raw[:, 0])))
    thr = float(np.abs(gauss[:, 1]).max())
    jumps = raw[np.abs(raw[:, 1]) > thr, 1]
    ev = load("3g")
    slope = float(np.polyfit(ev[:, 0], ev[:, 1], 1)[0])
    alpha, alpha_se, n_fit = powerlaw_mle(np.abs(jumps), thr)
    sig = float(gauss[:, 1].std())
    out["stationary_3p6V"] = {
        "G_mean_S": G36, "sample_dt_s": dt, "duration_s": float(raw[-1, 0] - raw[0, 0]),
        "gauss_sd_dGdt_S_per_s": sig, "gauss_sd_rel_per_sample": sig * dt / G36,
        "threshold_dGdt_S_per_s": thr, "threshold_over_gauss_sd": thr / sig, "threshold_rel_per_sample": thr * dt / G36,
        "jump_samples": int(len(jumps)), "jump_samples_up": int((jumps > 0).sum()),
        "jump_samples_down": int((jumps < 0).sum()),
        "jump_rate_per_s_fig3g": slope, "jump_rate_per_s_samples": len(jumps) / float(raw[-1, 0] - raw[0, 0]),
        "jump_rel_median": float(np.median(np.abs(jumps)) * dt / G36),
        "jump_rel_max": float(np.abs(jumps).max() * dt / G36),
        "powerlaw_alpha": alpha, "powerlaw_alpha_se": alpha_se, "powerlaw_n": n_fit,
        "paper": {"jump_rate_per_s": 0.082, "powerlaw_alpha": 2.78, "powerlaw_alpha_se": 0.07}}
    # ---- relaxation at 10 mV (Fig. 1e)
    t1, G1 = load("1e").T
    out["relaxation_10mV"] = {"duration_s": float(t1[-1]), "G_start_S": float(G1[0]), "G_end_S": float(G1[-1]),
                              "fractional_drop": float(1 - G1[-1] / G1[0]),
                              "fractional_drop_first_10s": float(1 - G1[t1 <= 10][-1] / G1[0])}
    # ---- density: wires from the deposited mass, n L^2 vs ours (percolation threshold n_c L^2 ~ 5.64)
    mass = 20e-9 * 0.09                                     # 20 uL x 0.09 mg/mL = 1.8 ug -> kg: 1.8e-9
    L, d = 35e-6, 115e-9
    n_w = mass / (RHO_AG * np.pi * (d / 2) ** 2 * L)
    out["density"] = {"their_wires_est": float(n_w), "their_area_m2": 1e-4, "their_L_m": L,
                      "their_nL2": float(n_w / 1e-4 * L ** 2), "our_wires": C.N_WIRES, "our_area_m2": C.AREA,
                      "our_L_m": C.LENGTH, "our_nL2": C.N_WIRES / C.AREA * C.LENGTH ** 2, "percolation_nL2": 5.64,
                      "note": "mass-based estimate assumes all wires land on the 1 cm^2 area and L = 35 um (20-50 um)"}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    ax[0].semilogy(v, np.where(g_exp > 0, g_exp, np.nan), "o", ms=3, label="measured")
    ax[0].semilogy(v, g_mod, "-", label="their model")
    ax[0].axvline(onset_model, color="r", ls="--", lw=0.8, label=f"onset (10x) {onset_model:.1f} V")
    ax[0].set(xlabel="applied voltage (V)", ylabel="steady-state G (S)", title="steady state, 7 mm gap (Fig. 2b)")
    ax[0].legend(fontsize=8)
    ax[1].plot(raw[:, 0], raw[:, 1] * dt / G36 * 1e3, lw=0.3, color="0.4")
    ax[1].axhline(thr * dt / G36 * 1e3, color="r", ls="--", lw=0.8)
    ax[1].axhline(-thr * dt / G36 * 1e3, color="r", ls="--", lw=0.8)
    ax[1].set(xlabel="time (s)", ylabel=r"$\Delta G/G$ per 0.6 s sample ($10^{-3}$)", title="3.6 V stationary (Fig. 3a)")
    ax[2].plot(ev[:, 0], ev[:, 1], ".", ms=1)
    ax[2].set(xlabel="time (s)", ylabel="jumps", title=f"jump count: {slope:.3f} /s (Fig. 3g)")
    a = np.sort(np.abs(jumps))[::-1] * dt / G36
    ax[3].loglog(a, np.arange(1, len(a) + 1) / len(a), ".", ms=2)
    ax[3].set(xlabel=r"jump size $\Delta G/G$", ylabel="P(size > x)",
              title=rf"jump sizes: $\alpha$ = {alpha:.2f} $\pm$ {alpha_se:.2f}")
    ax[3].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.suptitle("N10-C: measured network-level statistics, Milano et al. 2025 (open dataset)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
