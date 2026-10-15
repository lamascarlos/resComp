"""Delta^2(g_eff) as a function of M (number of cycles), to see where it saturates (report section 5.8's "weak
endpoint test" made finer): reuses the already-recorded run files, no re-simulation. Read-only analysis + figure.

  python -m scripts.geff_delta2_vs_m --dir validation_output/fdtest_eps3.0 --N 2000 [--n-grid 15] [--n-boot 60] [--seed 12345]

M is defined exactly as in scripts.fdtest_analyze: cycles pooled across all seed files, cycle 0 included, in
file-sort order (so smaller-M points are a deterministic prefix of the pooled cycle axis, the same convention as
the existing quarter/half/all scaling check in fdtest_analyze.py). The noise floor is computed once at full M via
the split-half method and scaled as floor * M_full / m (1/M scaling), matching fdtest_analyze.py's `scaling` block;
this is an approximation at small m, not a fresh split-half floor at each m.

Writes geff_delta2_vs_m_N{N}.json and .png to --out (default: --dir)."""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agnw_sim import fdtest as ft
from agnw_sim import fdtest_protocol as fp
from scripts.fdtest_analyze import load_runs, sanitize

N_FLOOR_SPLITS = 30


def m_grid(m_full, n_grid):
    """Geometric grid of cycle counts from 8 to m_full, n_grid points, unique, ascending, m_full always last."""
    lo = min(8, m_full)
    pts = np.unique(np.round(np.geomspace(lo, m_full, n_grid)).astype(int))
    pts = pts[pts <= m_full]
    if pts[-1] != m_full:
        pts = np.append(pts, m_full)
    return pts.tolist()


def delta2_curve(x2, x3, x4, ms, n_boot, rng):
    """{"m": [...], "delta2": [...], "ci95": [[lo, hi], ...]} for each m in ms (bootstrap over that m's cycles)."""
    out = {"m": [], "delta2": [], "ci95": []}
    for m in ms:
        d2 = ft.continuous_fdtest(x2[:m], x3[:m], x4[:m])["delta2"]
        ci = [float("nan"), float("nan")]
        if n_boot and m >= 8:
            boot = ft.bootstrap_continuous_delta2(x2[:m], x3[:m], x4[:m], n_boot, rng)
            ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        out["m"].append(int(m)); out["delta2"].append(float(d2)); out["ci95"].append(ci)
    return out


def plot(curve, floor_full, m_full, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    m, d2 = np.array(curve["m"]), np.array(curve["delta2"])
    ci = np.array(curve["ci95"])
    have_ci = np.isfinite(ci).all(axis=1)
    ax.plot(m, d2, "o-", color="C0", label=r"$\Delta^2(g_{eff})$")
    if have_ci.any():
        ax.fill_between(m[have_ci], ci[have_ci, 0], ci[have_ci, 1], color="C0", alpha=0.2, label="95% CI")
    if np.isfinite(floor_full):
        ax.plot(m, floor_full * m_full / m, "--", color="k", lw=1, label=r"noise floor $\propto 1/M$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("M (cycles)"); ax.set_ylabel(r"$\Delta^2$")
    ax.set_title(r"$g_{eff}$: $\Delta^2$ vs. number of cycles $M$")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True); p.add_argument("--N", type=int, required=True); p.add_argument("--out")
    p.add_argument("--n-grid", type=int, default=15); p.add_argument("--n-boot", type=int, default=60)
    p.add_argument("--n-lags", type=int); p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args(argv)
    runs, meta, files = load_runs(a.dir, a.N)
    n_lags = meta["n_lags"] if a.n_lags is None else a.n_lags
    sch = fp.Schedule(V0=meta["V0"], eps=meta["eps"], T_relax=meta["T_relax"], T_samp=meta["T_samp"],
                       n_cycles=meta["n_cycles"], record_every=meta["record_every"], n_lags=n_lags)
    steps = np.arange(sch.n_steps)
    V_in = np.where(steps % sch.T_cycle < sch.T_relax + sch.T_samp, sch.V0 + sch.eps, sch.V0)
    geff = [fp.extract_stage_arrays((z["I_total"] / V_in)[::sch.record_every], sch) for z in runs]
    x2, x3, x4 = (np.concatenate([g[i] for g in geff]) for i in range(3))
    m_full = x2.shape[0]
    if m_full < 8:
        sys.exit(f"need at least 8 pooled cycles (have {m_full})")
    rng = np.random.default_rng(a.seed)
    floor = ft.continuous_split_half_floor(x2, x3, x4, rng, n_splits=N_FLOOR_SPLITS)
    ms = m_grid(m_full, a.n_grid)
    print(f"N={a.N}: {len(runs)} runs, M_full={m_full} cycles, V0={sch.V0} V, eps={sch.eps:.4f} V, "
          f"floor(M_full)={floor:.4g}, grid={ms}")
    curve = delta2_curve(x2, x3, x4, ms, a.n_boot, rng)
    for m, d2, ci in zip(curve["m"], curve["delta2"], curve["ci95"]):
        print(f"  M={m:>5}  delta2={d2:.5f}  ci95=[{ci[0]:.5f}, {ci[1]:.5f}]  floor_scaled={floor * m_full / m:.5f}")
    out = a.out or a.dir
    os.makedirs(out, exist_ok=True)
    result = {"N": a.N, "n_runs": len(runs), "m_full": m_full, "floor_full": float(floor),
              "config": {"n_grid": a.n_grid, "n_boot": a.n_boot, "seed": a.seed, "n_lags": n_lags}, "curve": curve}
    with open(os.path.join(out, f"geff_delta2_vs_m_N{a.N}.json"), "w") as fh:
        json.dump(sanitize(result), fh, indent=1, allow_nan=False)
    plot(curve, floor, m_full, os.path.join(out, f"geff_delta2_vs_m_N{a.N}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
