"""FDTest of the measured AgNW network conductance (Clarius+ export, one sheet per run, two cycles per sheet).

  python -m scripts.fdtest_experiment INPUT --out DIR [--csv PATH] [--tail 200] [--n-lags 150] [--n-boot 200] [--seed 12345]

INPUT is the .xls export (converted to the long-format CSV first; written next to it as <stem>_long.csv unless --csv is
given) or an existing long CSV (agnw_sim.experiment.COLUMNS). Observable, drift removal, readout-jump correction and the
stage windows: see agnw_sim.experiment. Writes DIR/fdtest_experiment.json (strict JSON), DIR/experiment_overview.png and
DIR/experiment_z.png (200 dpi).

Analyses (all with the noise cutoff chosen once on the full sample and the 5 % rule, bootstrap over cycles, mean of 30
split-half floors, reverse 95 % interval; as scripts.fdtest_geff):
  cycle2      cycle 2 of every sheet (M = n_sheets): OFF, ON (switch-on with the roles swapped) and BOTH, via
              scripts.fdtest_geff.analyze_geff. Cycle 1 has no valid switch-on (it starts after the unrecorded gap).
  off_all     OFF on both cycles of every sheet (M = 2 n_sheets); the perturbed tail of cycle 1 follows the post-gap transient.
  off_early / off_late   off_all on the first / second half of the sheets (stationarity of the result).
Each is run on the jump-corrected observable ("corrected") and on the raw one ("raw", no readout correction).
--max-step PPM drops the sheets whose largest one-sample step inside a segment (experiment.max_inner_step, first 20 samples of
each segment skipped) exceeds PPM parts per million (sudden resistance jumps); default: keep all sheets."""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agnw_sim import experiment as ex
from agnw_sim import fdtest as ft
from agnw_sim import fdtest_protocol as fp
from scripts.fdtest_analyze import sanitize, fmt, fmt_ci, N_FLOOR_SPLITS
from scripts.fdtest_geff import analyze_geff, delta2_at, RULES


def load(path, csv_path=None):
    """Long table from the .xls (writing the CSV) or from a long CSV."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    df = ex.to_long(ex.read_xls(path))
    csv_path = csv_path or os.path.splitext(path)[0] + "_long.csv"
    df.to_csv(csv_path, index=False, float_format="%.10g")
    print(f"wrote {csv_path} ({len(df)} rows, {df.sheet.nunique()} sheets)")
    return df


def analyze_off(x2, x3, x4, n_boot, seed=12345, n_floor=N_FLOOR_SPLITS):
    """Switch-off FDTest alone (same rules, bootstrap and floor as analyze_geff's OFF variant). Returns {c5, noise} -> block and
    the curves."""
    free = ft.continuous_fdtest(x2, x3, x4)
    cut, hit_end = ft.noise_cutoff(free["z3"])
    rng = np.random.default_rng([seed, 0])
    n = x2.shape[0]
    boot = {r: np.empty(n_boot) for r in RULES}
    z0 = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r = ft.continuous_fdtest(x2[idx], x3[idx], x4[idx])
        boot["c5"][b], boot["noise"][b], z0[b] = r["delta2"], delta2_at(r, cut), r["z_mean"][0]
    resp = fp.achieved_response(x2, x4)
    out = {}
    for rule, est, cutoff, he in (("c5", free["delta2"], None, free["tau_c"] == x3.shape[1] - 1),
                                  ("noise", delta2_at(free, cut), cut, hit_end)):
        fl = ft.continuous_split_half_floor(x2, x3, x4, np.random.default_rng([seed, 1]), n_splits=n_floor, cutoff=cutoff)
        vs = ft.violation_summary(est, boot[rule], fl)
        out[rule] = {"delta2": vs["delta2"], "ci95_reverse": vs["ci95_reverse"], "noise_floor": fl, "excess": vs["excess"],
                     "excludes_floor": vs["excludes_floor"], "cutoff": int(free["tau_c"] if cutoff is None else cut),
                     "hit_end": bool(he), **resp, "z0_snr": fp.gate_power(z0)["value"], "M": int(n)}
    sigma, _, zs = ft.noise_level(free["z3"])
    curves = {"z_mean": free["z_mean"], "c_zz": free["c_zz"], "z_smooth": zs, "sigma_noise": sigma, "noise_cutoff": cut,
              "noise_hit_end": bool(hit_end), "c5_cutoff": int(free["tau_c"])}
    return out, curves


def run_all(segs, jump, tail, n_lags, n_boot, seed):
    """All analyses for one readout correction `jump`. Returns (results, curves)."""
    res, curves = {}, {}
    x2, x3, x4, xI = ex.stage_arrays(segs, tail, n_lags, cycles=(2,), jump=jump)
    g = analyze_geff(x2, x3, x4, xI, n_boot, seed)
    res["cycle2"] = {v: {r: {**g["variants"][v][r], "M": int(x2.shape[0])} for r in RULES} for v in g["variants"]}
    curves.update({f"cycle2_{k}": c for k, c in g["curves"].items()})
    h = len(segs) // 2
    for name, part in (("off_all", segs), ("off_early", segs[:h]), ("off_late", segs[h:])):
        x2, x3, x4, _ = ex.stage_arrays(part, tail, n_lags, cycles=(1, 2), jump=jump)
        res[name], c = analyze_off(x2, x3, x4, n_boot, seed)
        if name == "off_all":
            curves["off_all"] = c
    return res, curves


def print_table(label, res):
    print(f"\n[{label}]")
    print(f"{'analysis':<16} {'rule':<6} {'M':>4} {'delta2':>8} {'ci95 reverse':>18} {'floor':>8} {'viol':>5} {'cut':>5} {'end':>4} "
          f"{'resp/sd':>8} {'z0 snr':>7}")
    rows = [(f"cycle2 {v}", res["cycle2"][v]) for v in ("OFF", "ON", "BOTH")] + [(k, res[k]) for k in ("off_all", "off_early", "off_late")]
    for name, blk in rows:
        for rule in RULES:
            b = blk[rule]
            print(f"{name:<16} {rule:<6} {b['M']:>4} {fmt(b['delta2'])} {fmt_ci(b['ci95_reverse']):>18} {fmt(b['noise_floor'])} "
                  f"{'yes' if b['excludes_floor'] else 'no':>5} {b['cutoff']:>5} {'y' if b['hit_end'] else 'n':>4} "
                  f"{fmt(b['achieved_response_sigma'], 2):>8} {fmt(b['z0_snr'], 1):>7}")


def plot_overview(df, segs, jump, tail, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    R = df["R"].to_numpy()
    ax[0].plot(np.arange(len(R)), R, lw=0.3, color="k")
    ax[0].set_xlabel("sample (sheets concatenated, gaps not shown)"); ax[0].set_ylabel(r"R = V/I ($\Omega$)")
    n = min(len(s) for sg in segs for s in sg)
    dt = 0.0288
    for k, (lab, col) in enumerate(zip(("A on (cycle 1)", "B off", "C on (cycle 2)", "D off"), ("tab:red", "tab:blue", "tab:orange", "tab:cyan"))):
        m = np.array([sg[k][:n] for sg in segs])
        start = sum(min(len(sg[j]) for sg in segs) for j in range(k))
        tt = (start + np.arange(n)) * dt
        mu, se = 1e6 * m.mean(0), 1e6 * m.std(0) / np.sqrt(len(m))
        ax[1].fill_between(tt, mu - se, mu + se, color=col, alpha=0.3, lw=0); ax[1].plot(tt, mu, color=col, lw=1, label=lab)
    ax[1].axhline(0, color="0.5", lw=0.5); ax[1].axhline(1e6 * jump, color="0.5", lw=0.5, ls="--", label="readout jump")
    ax[1].set_xlabel("time in sheet (s)"); ax[1].set_ylabel(r"$x = G/G_{trend}-1$ (ppm), mean $\pm$ SE")
    ax[1].set_title("sheet-averaged detrended conductance"); ax[1].legend(fontsize=7)
    x2, _, x4, _ = ex.stage_arrays(segs, tail, 150, cycles=(1, 2), jump=jump)
    ax[0].set_title(f"raw resistance, {len(segs)} sheets")
    bins = np.linspace(*np.percentile(np.r_[x2.ravel(), x4.ravel()], [0.5, 99.5]), 60)
    ax[2].hist(1e6 * x4.ravel(), 1e6 * bins, density=True, alpha=0.5, label=r"$P_0$ (unperturbed tails)")
    ax[2].hist(1e6 * x2.ravel(), 1e6 * bins, density=True, alpha=0.5, label=r"$P_\epsilon$ (perturbed tails, jump-corrected)")
    ax[2].set_xlabel("x (ppm)"); ax[2].set_ylabel("density"); ax[2].legend(fontsize=7); ax[2].set_title("stationary distributions")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_z(curves, path, dt=0.0288):
    keys = [("off_all", "OFF, both cycles"), ("cycle2_OFF", "OFF, cycle 2"), ("cycle2_ON", "ON, cycle 2 (roles swapped)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (k, title) in zip(axes, keys):
        c = curves[k]
        lag = np.arange(len(c["z_mean"])) * dt
        ax.axhspan(-c["sigma_noise"], c["sigma_noise"], color="0.85", label=r"$\pm\sigma_{noise}$")
        ax.plot(lag, c["z_mean"], lw=1.2, label=r"$\langle z(\tau)\rangle$ (transient)")
        ax.plot(lag, c["c_zz"], lw=1.2, label=r"$C_{zz}(\tau)$ (stationary)")
        ax.axvline(c["noise_cutoff"] * dt, color="k", ls="--", lw=1, label="noise cutoff" + (" (end)" if c["noise_hit_end"] else ""))
        ax.axvline(c["c5_cutoff"] * dt, color="tab:red", ls=":", lw=1, label="5 % cutoff")
        ax.set_xlabel(r"lag $\tau$ (s)"); ax.set_title(title); ax.legend(fontsize=7)
    axes[0].set_ylabel("z")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main(argv=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _main(argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("input"); p.add_argument("--out", required=True); p.add_argument("--csv")
    p.add_argument("--tail", type=int, default=200); p.add_argument("--n-lags", type=int, default=150)
    p.add_argument("--n-boot", type=int, default=200); p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-step", type=float, help="drop sheets with an in-segment step above this (ppm)")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    df = load(a.input, a.csv)
    segs = [ex.detrend(s, a.tail) for s in ex.sheet_segments(df)]
    n_all = len(segs)
    if a.max_step is not None:
        keep = ex.max_inner_step(segs) * 1e6 <= a.max_step
        segs = [s for s, k in zip(segs, keep) if k]
        df = df[df.sheet.isin(np.flatnonzero(keep) + 1)]
        print(f"--max-step {a.max_step:g} ppm: kept {len(segs)} of {n_all} sheets")
    steps = ex.switch_steps(segs)
    jump = float(steps.mean())
    print(f"{len(segs)} sheets; readout jump {1e6 * jump:.1f} ppm (A->B {1e6 * steps[:, 0].mean():.1f}, "
          f"B->C {1e6 * steps[:, 1].mean():.1f}, C->D {1e6 * steps[:, 2].mean():.1f}; SE {1e6 * steps.std() / np.sqrt(steps.size):.1f})")
    out = {"input": os.path.basename(a.input), "n_sheets": len(segs), "n_sheets_total": n_all, "max_step_ppm": a.max_step, "tail": a.tail, "n_lags": a.n_lags, "n_boot": a.n_boot,
           "seed": a.seed, "readout_jump": jump, "switch_steps_mean": steps.mean(0), "R_first_last_sheet": [
               float(df[df.sheet == 1].R.mean()), float(df[df.sheet == df.sheet.max()].R.mean())], "results": {}}
    for label, j in (("corrected", jump), ("raw", 0.0)):
        res, curves = run_all(segs, j, a.tail, a.n_lags, a.n_boot, a.seed)
        out["results"][label] = res
        print_table(label, res)
        if label == "corrected":
            out["curves"] = curves
            plot_z(curves, os.path.join(a.out, "experiment_z.png"))
    plot_overview(df, segs, jump, a.tail, os.path.join(a.out, "experiment_overview.png"))
    with open(os.path.join(a.out, "fdtest_experiment.json"), "w") as f:
        json.dump(sanitize(out), f, indent=1, allow_nan=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
