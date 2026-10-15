#!/usr/bin/env python3
"""N11 stage 0: summary statistics of the measured records of Milano et al. 2025 (Nat. Commun., DOI
10.1038/s41467-025-58741-2; open dataset Zenodo 10.5281/zenodo.15050217, CC-BY, in data/milano2025/), with the
estimators of agnw_rc/series_stats.py (docs/PLAN_N11_CALIBRATION.md section 5). Measured files only: Fig. 1e, 2a, 2b
(points), 2b inset, 5a (stationary part 18 500-35 600 s); their fitted theta (Fig. 2e) is a cross-check.

    python scripts/n11_data.py   -> results/n11_data.json, figures/fig33_n11_data.png
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agnw_rc import series_stats as S  # noqa: E402

DATA = ROOT / "data" / "milano2025"
OUT = ROOT / "results" / "n11_data.json"
FIG = ROOT / "figures" / "fig33_n11_data.png"
N10C = ROOT / "results" / "n10c_data.json"
V_SAT = 5.0                 # V   saturated conductance: median measured G at V >= V_SAT
MIN_EVENTS = 20             #     stage-2 voltages need at least this many jump events in 500 s
N_BOOT = 200
T5_END = 35600.0            # s   Fig. 5a: G at the instrument floor from 35653 s on (bias off)


def load_pairs(name):
    """[(V, t, G)] for the 66 (time, conductance) column pairs of a multi-voltage file, cleaned."""
    path = DATA / f"Figure {name}.txt"
    with open(path) as f:
        head = [next(f) for _ in range(3)]
    volts = [float(s.split("=")[1].replace("V", "")) for s in head[2].rstrip("\n").split("\t") if "=" in s]
    d = np.genfromtxt(path, delimiter="\t", skip_header=3)
    assert d.shape[1] == 2 * len(volts), (path, d.shape, len(volts))
    return [(v, *S.clean(d[:, 2 * j], d[:, 2 * j + 1])) for j, v in enumerate(volts)]


def load2(name):
    return np.loadtxt(DATA / f"Figure {name}.txt", skiprows=2)


def validation(rng):
    sd = 4.8e-4
    t = np.arange(0, 20_000.0, S.GRID_DT)
    flat = np.full(len(t), 3e-5)
    sd_r = S.jump_stats(t, S.synthetic_series(rng, t, flat, sd, 0.08, 3.0, 2.6 * sd))["sd_rel"] / sd
    rate_r = S.jump_stats(t, S.synthetic_series(rng, t, flat, sd, 0.08, 3.0, 5 * sd))["rate_per_s"] / 0.08
    eff = {}
    for m in (2.0, 2.6, 3.0, 4.0, 5.0, 8.0):              # (nearly) fixed sizes m sd: Pareto exponent 60
        G = S.synthetic_series(rng, t, flat, sd, 0.08, 60.0, m * sd)
        eff[f"{m:g}"] = S.jump_stats(t, G)["rate_per_s"] / 0.08
    return {"sd_ratio": sd_r, "rate_ratio_5sd": rate_r, "detection_efficiency": eff}


def main():
    rng = np.random.default_rng(2026_10_13)
    out = {}
    # ---- steady state (Fig. 2b points): onset and shape
    v, g_exp, _ = load2("2b").T
    g_sat = float(np.median(g_exp[v >= V_SAT]))
    v_on = S.onset(v, g_exp, g_sat)
    out.update(onset_V=v_on, onset_V_resolution=float(np.median(np.diff(v)) / 2), G_sat_S=g_sat,
               onset_definition=f"G reaches {S.ONSET_FRAC:g} x G_sat (median at V >= {V_SAT} V), log-interpolated")
    # ---- inset: 500 s stationary records per voltage
    inset, shape, pooled = [], [], []
    for V, t, G in load_pairs("2b_inset"):
        row = {"V": V, "native_dt_s": float(np.median(np.diff(t))), "interp_index": S.interpolation_index(G)}
        tg, Gg = S.to_grid(t, G)
        b = S.block_bootstrap(Gg, np.mean, round(S.BLOCK_S / S.GRID_DT), N_BOOT, rng)
        row.update(G_mean_S=float(Gg.mean()), G_mean_ci=[b["lo"], b["hi"]])
        i2b = int(np.argmin(np.abs(v - V)))
        row["ratio_2b_over_inset"] = float(g_exp[i2b] / Gg.mean()) if Gg.mean() > 0 else None
        row["stationary"] = S.is_stationary(g_exp[i2b], Gg.mean())
        if v_on is not None and V > v_on and np.all(Gg > 0) and row["stationary"]:
            js = S.jump_stats(tg, Gg)
            sizes = js.pop("sizes")
            pooled.append(sizes)
            row["sizes"] = sizes.tolist()
            rel = np.diff(Gg) / Gg[:-1]
            sdb = S.block_bootstrap(rel, S.clipped_sd, round(S.BLOCK_S / S.GRID_DT), N_BOOT, rng)
            row.update(jumps=js, sd_ci=[sdb["lo"], sdb["hi"]], sd_log_delta=sdb["log_sd"])
            shape.append({"V": V, "ratio": float(g_exp[i2b] / g_sat), "log_delta": b["log_sd"]})   # Fig. 2b point
        inset.append(row)
    out["inset"], out["shape"] = inset, shape
    jr = [r for r in inset if "jumps" in r]
    out["stage2_voltages"] = sorted(r["V"] for r in jr if r["jumps"]["events"] >= MIN_EVENTS)
    out["excluded_not_stationary"] = [r["V"] for r in inset if v_on is not None and r["V"] > v_on and not r["stationary"]]
    sl, se = S.loglog_fit([r["G_mean_S"] / g_sat for r in jr], [r["jumps"]["rate_per_s"] for r in jr])
    out["lambda_vs_conducting_slope"] = {"slope": sl, "se": se, "n_rates_not_positive":
                                         int(sum(r["jumps"]["rate_per_s"] <= 0 for r in jr))}
    pooled = np.concatenate(pooled) if pooled else np.zeros(0)
    out["pooled_sizes"] = {"n": int(len(pooled)),
                           "abs_quantiles_50_90_99": np.quantile(np.abs(pooled), [0.5, 0.9, 0.99]).tolist()}
    # ---- transients (Fig. 2a): theta per voltage above onset; their fitted theta (Fig. 2e) for comparison
    theirs = load2("2d,e,f,g")
    th = []
    for V, t, G in load_pairs("2a"):
        if v_on is None or V <= v_on or len(t) < 20:
            continue
        f = S.fit_theta(t, G, N_BOOT, rng)
        f.update(V=V, theirs_per_s=float(np.interp(V, theirs[:, 0], theirs[:, 2])))
        th.append(f)
    out["theta"] = th
    # ---- relaxation at 10 mV (Fig. 1e)
    t1, G1 = load2("1e").T
    out["relaxation"] = S.fit_relaxation(t1, G1)
    # ---- 3.6 V anchor (Fig. 5a, stationary part) with the same estimator; N10-C values for reference
    t5, G5 = load2("5a").T
    m = (t5 > 18500) & (t5 < T5_END)       # stationary part; the record falls to the instrument floor at 35653 s
    a = S.jump_stats(*S.to_grid(*S.clean(t5[m], G5[m])))
    a.pop("sizes")
    out["anchor_3p6V"] = {"fig5a": a, "n10c": json.loads(N10C.read_text())["stationary_3p6V"] if N10C.exists() else None}
    out["validation"] = validation(rng)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"onset {v_on:.3f} V, G_sat {g_sat:.3g} S, stage-2 voltages {out['stage2_voltages']}")
    print(f"excluded (not stationary): {out['excluded_not_stationary']}")
    print(f"lambda vs G/G_sat log-log slope {out['lambda_vs_conducting_slope']}; validation {out['validation']}")
    print(f"relaxation: {out['relaxation']}")
    print(f"anchor 3.6 V (Fig. 5a): rate {a['rate_per_s']:.4f} /s, sd {a['sd_rel']:.2e}, events {a['events']}")
    figure(out, v, g_exp, pooled)


def figure(out, v, g_exp, pooled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    ax[0].semilogy(v, np.where(g_exp > 0, g_exp, np.nan), "o", ms=3, label="measured (Fig. 2b)")
    ax[0].axvline(out["onset_V"], color="r", ls="--", lw=0.8, label=f"onset {out['onset_V']:.2f} V")
    ax[0].set(xlabel="applied voltage (V)", ylabel="steady-state G (S)", title="steady state, 7 mm gap")
    ax[0].legend(fontsize=8)
    th = out["theta"]
    ok = [r for r in th if not r["unresolved"]]
    ax[1].errorbar([r["V"] for r in ok], [r["theta_per_s"] for r in ok],
                   yerr=[[r["theta_per_s"] - r["lo"] for r in ok], [r["hi"] - r["theta_per_s"] for r in ok]],
                   fmt="o", ms=3, label="this fit (Fig. 2a)")
    ax[1].plot([r["V"] for r in th], [r["theirs_per_s"] for r in th], "-", label="their model fit (Fig. 2e)")
    ax[1].set(yscale="log", xlabel="applied voltage (V)", ylabel=r"$\theta$ (1/s)", title="potentiation rate")
    ax[1].legend(fontsize=8)
    jr = [r for r in out["inset"] if "jumps" in r]
    rate = np.array([r["jumps"]["rate_per_s"] for r in jr])
    ci = np.array([r["jumps"]["rate_ci_per_s"] for r in jr])
    ax[2].errorbar([r["V"] for r in jr], rate, yerr=[np.maximum(rate - ci[:, 0], 0), np.maximum(ci[:, 1] - rate, 0)],
                   fmt="o", ms=3, label=r"$\lambda$")
    ax2 = ax[2].twinx()
    ax2.plot([r["V"] for r in jr], [r["jumps"]["sd_rel"] * 1e3 for r in jr], "s", ms=3, color="C3")
    ax2.set_ylabel(r"Gaussian sd of $\Delta G/G$ ($10^{-3}$)", color="C3")
    ax[2].set(xlabel="applied voltage (V)", ylabel="jumps per s", title="jumps and noise, 500 s per voltage")
    a = np.sort(np.abs(pooled))[::-1]
    ax[3].loglog(a, np.arange(1, len(a) + 1) / len(a), ".", ms=2)
    ax[3].set(xlabel=r"jump size $|\Delta G/G|$", ylabel="P(size > x)", title=f"pooled jump sizes ({len(a)} events)")
    fig.suptitle("N11 stage 0: measured statistics, Milano et al. 2025 (open dataset)")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
