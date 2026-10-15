"""FDTest of alternative observables (docs/alt_observables_design.md): the network conductance g_eff (reference, as in the
main analysis), the ON fraction m_K of the K recorded junctions, and the cross-fitted optimal linear observable s = w . g.

  python -m scripts.fdtest_observables N [--out validation_output/fdtest] [--n-boot 200] [--n-lags INT] [--seed 12345]
                                       [--max-snapshots 40] [--ridge-frac 0.1] [--stride 30]

Reads the run_N{N}_seed*.npz of one schedule exactly as scripts.fdtest_analyze does (same checks; --n-lags as there) and
writes fdtest_observables_N{N}.json (strict JSON: non-finite values are null), fdtest_weights_N{N}.png and
fdtest_observables_N{N}.png (200 dpi). For every observable it reports, unbinned and lag-binned (default scheme of
docs/lag_binning_design.md): delta2 with percentile and reverse 95 % bootstrap intervals over cycles, the split-half noise floor (mean
of 30 random splits), excess, excludes_floor, tau_c with the cutoff-at-end flag, the achieved response in sigma, the
signal-to-noise of <Z(0)> and the noise correlation length across lags.

Optimal observable: the weights are a regularised linear discriminant of stage II against stage IV fitted on one random
half of the cycles and applied to the other half and vice versa (cross-fitting). The bootstrap over cycles does NOT refit
the weights (fold assignment and weights are fixed), so its intervals ignore the uncertainty of w. The reported weight map
w_j is the mean of the two folds' unit-norm weight vectors, renormalised to unit norm (the sign convention makes both folds'
mean response positive, so no sign alignment is needed); fold_cosine is their cosine (stability of the map): below 0.3 the averaged
map is mostly noise and the Spearman associations / backbone share are blanked (null) with a note. The scores of each evaluation fold
are standardised by that fold's own stage-IV mean and std (see fdtest_observables.crossfit_scores).

Assumption of the cross-fitting: the folds are random subsets of the cycles of the chained runs, so adjacent cycles can fall in
different folds; the cycles are treated as independent because stage I (T_relax steps) separates the stage IV of one cycle from
the stage II of the next (this is the same independence the odd/even gate of fdtest_analyze checks).

Memory: the three stage arrays of the junction states (uint8) are filled run by run, about M * (2 * T_samp + n_lags) / record_every * K bytes
(2 GB for M = 480, K = 1000, 1400 lags); the discriminant uses strided samples only. Exit status 0, or 1 for inconsistent inputs."""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agnw_sim import fdtest as ft
from agnw_sim import fdtest_protocol as fp
from agnw_sim import fdtest_observables as fo
from agnw_sim.dualnode import DualNodeCircuit
from scripts.fdtest_analyze import load_runs, sanitize, fmt, fmt_ci, N_FLOOR_SPLITS, SCHEDULE_KEYS
from scripts.fdtest_common import build_setup

NOTE = ("the bootstrap over cycles does not refit the weights of the optimal observable (fold assignment and weights are "
        "fixed), so its intervals ignore the uncertainty of w; cross-fitting removes the in-sample bias of the response")
OBS_LABEL = {"geff": "g_eff", "m_K": "m_K", "optimal": "optimal"}


def analyze_observable(x2, x3, x4, n_boot, rng, edges):
    """Unbinned and lag-binned FDTest of one continuous observable (x2, x3, x4 as in ft.continuous_fdtest): delta2 with
    bootstrap intervals over the cycles and split-half floor, plus the a-posteriori response, the SNR of <Z(0)> and the
    noise correlation length across lags. Returns a dict (numpy values; sanitize before json)."""
    L = x3.shape[1]
    res = ft.continuous_fdtest(x2, x3, x4)
    boot, boot_z0 = ft.bootstrap_continuous_delta2(x2, x3, x4, n_boot, rng, return_z0=True)
    floor, z_a, z_b = ft.continuous_split_half_floor(x2, x3, x4, rng, n_splits=N_FLOOR_SPLITS, return_halves=True)
    vs = ft.violation_summary(res["delta2"], boot, floor)
    unbinned = {"delta2": vs["delta2"], "ci95": vs["ci95_percentile"], "ci95_reverse": vs["ci95_reverse"], "noise_floor": floor,
                "excess": vs["excess"], "excludes_floor": vs["excludes_floor"], "tau_c": int(res["tau_c"]),
                "cutoff_at_end": bool(res["tau_c"] == L - 1)}
    res_b = ft.continuous_fdtest(x2, x3, x4, edges=edges)
    boot_b = ft.bootstrap_continuous_delta2(x2, x3, x4, n_boot, rng, edges=edges)
    floor_b = ft.continuous_split_half_floor(x2, x3, x4, rng, n_splits=N_FLOOR_SPLITS, edges=edges)
    vb = ft.violation_summary(res_b["delta2"], boot_b, floor_b)
    binned = {"delta2": vb["delta2"], "ci95": vb["ci95_percentile"], "ci95_reverse": vb["ci95_reverse"], "noise_floor": floor_b,
              "excess": vb["excess"], "excludes_floor": vb["excludes_floor"], "tau_c": int(res_b["tau_c"]),
              "cutoff_at_end": bool(res_b["cutoff_at_end"])}
    _, corr = ft.lag_noise_correlation(z_a, z_b)
    return {"unbinned": unbinned, "binned": binned, **fp.achieved_response(x2, x4),
            "z0_snr": fp.gate_power(boot_z0)["value"], "noise_corr_length": corr,
            "excluded_fraction": res["excluded_fraction"], "z_mean": res["z_mean"], "c_zz": res["c_zz"]}


MIN_FOLD_COSINE = 0.3


def weight_map(info, strength, share, backbone, min_fold_cosine=MIN_FOLD_COSINE):
    """Weight map of the optimal observable (see the module docstring) and its association with junction properties.
    If the two folds' weight vectors disagree (cosine < min_fold_cosine, or undefined), the averaged map is mostly noise:
    the Spearman associations and the backbone share of |w| are then NaN (null in the JSON) and `associations_blanked` and
    `associations_note` say so; the map itself is still returned."""
    unit = lambda v: v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v
    w0, w1 = unit(info["w"][0]), unit(info["w"][1])
    w = unit(w0 + w1)
    fold_cos = float(w0 @ w1) if np.linalg.norm(w0) > 0 and np.linalg.norm(w1) > 0 else float("nan")
    blank = not (np.isfinite(fold_cos) and fold_cos >= min_fold_cosine)

    def rho(x, y):
        return float(spearmanr(x, y)[0]) if not blank and len(x) > 2 and np.ptp(x) > 0 and np.ptp(y) > 0 else float("nan")

    on_bb = np.asarray(backbone) >= 0.5
    absw = np.abs(w)
    note = (f"fold cosine {fold_cos:.3f} < {min_fold_cosine}: the weight vectors of the two folds disagree, the averaged map is "
            f"mostly noise and its associations are not reported" if blank else
            f"fold cosine {fold_cos:.3f} >= {min_fold_cosine}: associations of the averaged map")
    return {"w": w, "w_fold": info["w"], "fold_cosine": fold_cos, "associations_blanked": bool(blank), "associations_note": note,
            "spearman_strength": rho(w, strength), "spearman_current_share": rho(w, share), "spearman_backbone": rho(w, backbone),
            "spearman_abs_strength": rho(absw, strength), "spearman_abs_current_share": rho(absw, share),
            "backbone_fraction_weighted": float(absw[on_bb].sum() / absw.sum()) if not blank and absw.sum() > 0 else float("nan"),
            "backbone_fraction_recorded": float(on_bb.mean()),
            "response_in_sample": info["response_in_sample"], "response_crossfit": info["response_crossfit"]}


def main(argv=None):
    """Run the analysis; RuntimeWarnings from degenerate data (all-NaN slices, empty means, 0/0) are silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _main(argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("N", type=int); p.add_argument("--out", default="validation_output/fdtest")
    p.add_argument("--n-boot", type=int, default=200); p.add_argument("--max-snapshots", type=int, default=40)
    p.add_argument("--n-lags", type=int, default=None, help="lags of the relaxation / autocorrelation window (default: as stored in the runs)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--ridge-frac", type=float, default=0.1, help="ridge of the discriminant as a fraction of trace(Sigma)/K")
    p.add_argument("--stride", type=int, default=30, help="every stride-th record enters the discriminant covariance")
    a = p.parse_args(argv)
    if a.stride < 1 or not (np.isfinite(a.ridge_frac) and a.ridge_frac >= 0):
        sys.exit(f"--stride must be >= 1 and --ridge-frac finite and >= 0 (got {a.stride} and {a.ridge_frac})")
    rng = np.random.default_rng(a.seed)

    runs, meta, files = load_runs(a.out, a.N)
    n_lags = meta["n_lags"] if a.n_lags is None else a.n_lags
    try:
        sch = fp.Schedule(V0=meta["V0"], eps=meta["eps"], T_relax=meta["T_relax"], T_samp=meta["T_samp"],
                          n_cycles=meta["n_cycles"], record_every=meta["record_every"], n_lags=n_lags)
    except ValueError as err:
        r = meta["record_every"]
        sys.exit(f"--n-lags {n_lags} is not allowed for these runs ({err}); the allowed range is 2 to "
                 f"{min(meta['T_samp'] // r, meta['T_relax'] // r)}")
    net, ids, params, wd = build_setup(a.N, meta["network_seed"], meta["k"], meta["p_relax"])
    for f, z in zip(files, runs):
        if not np.array_equal(ids, z["sampled_junction_ids"]):
            sys.exit(f"{f}: sampled_junction_ids differ from the junctions selected by build_setup(N={a.N}, network_seed="
                     f"{meta['network_seed']}, k={meta['k']}); the run was made for another network or junction selection")
        if z["G_sampled"].shape[1] != len(ids) or z["G_snapshots"].shape[1] != net.junctions.shape[0]:
            sys.exit(f"{f}: G_sampled / G_snapshots widths ({z['G_sampled'].shape[1]}, {z['G_snapshots'].shape[1]}) do not match "
                     f"the rebuilt network ({len(ids)} recorded, {net.junctions.shape[0]} junctions)")
    circuit = DualNodeCircuit(net, wd)
    steps = np.arange(sch.n_steps)
    V_in = np.where(steps % sch.T_cycle < sch.T_relax + sch.T_samp, sch.V0 + sch.eps, sch.V0)

    # ---- stage arrays: g_eff per run (small), junction states written into preallocated uint8 arrays (no concatenation copy)
    K, L, nc, r = len(ids), sch.n_lags, sch.n_cycles, sch.record_every
    M = nc * len(runs)
    xg = [np.empty((M, n)) for n in (sch.T_samp // r, L, sch.T_samp // r)]
    gs = [np.empty((M, n, K), np.uint8) for n in (sch.T_samp // r, L, sch.T_samp // r)]
    for i, z in enumerate(runs):
        sl = slice(i * nc, (i + 1) * nc)
        for dst, part in zip(xg, fp.extract_stage_arrays((z["I_total"] / V_in)[::r], sch)):
            dst[sl] = part
        gz = z["G_sampled"]
        for dst, part in zip(gs, fp.extract_stage_arrays(gz, sch)):
            dst[sl] = part
        del gz, dst, part
    print(f"N={a.N}: {len(runs)} runs, M = {M} cycles, K = {K} recorded junctions, {L} lags "
          f"(lag step {r}), V0 = {sch.V0} V, eps = {sch.eps:.4f} V")

    obs = {"geff": tuple(xg), "m_K": tuple(fo.on_fraction(g) for g in gs)}
    obs["optimal"] = fo.crossfit_scores(*gs, rng, ridge_frac=a.ridge_frac, stride=a.stride)
    info = obs["optimal"][3]
    obs["optimal"] = obs["optimal"][:3]
    del gs, xg                                                                            # the junction states are not needed any more

    # ---- FDTest of the three observables
    edges = ft.lag_bin_edges(L)
    results = {name: analyze_observable(*obs[name], a.n_boot, rng, edges) for name in ("geff", "m_K", "optimal")}

    # ---- weight map and junction properties (snapshots as in fdtest_analyze)
    strength = fp.junction_strength(net)
    snaps = np.concatenate([z["G_snapshots"][1::2] for z in runs])                        # ends of stage IV
    if len(snaps) > a.max_snapshots:
        snaps = snaps[np.linspace(0, len(snaps) - 1, a.max_snapshots).astype(int)]
    share, backbone = fp.current_share(circuit, snaps, sch.V0)
    wm = weight_map(info, strength[ids], share[ids], backbone[ids])
    pos = net.junction_pos[ids]

    out = {"schedule": {**{k: meta[k] for k in SCHEDULE_KEYS}, "n_lags": int(L), "n_lags_stored": int(meta["n_lags"])},
           "M": int(M), "K": int(K), "n_boot": a.n_boot, "seed": a.seed, "note": NOTE,
           "bin_scheme": {"n_first": 8, "ratio": 1.15, "n_bins": len(edges) - 1, "edges": edges.tolist()},
           "optimal_fit": {"ridge_frac": a.ridge_frac, "stride": a.stride},
           "observables": results, "weights": {**wm, "pos": pos, "strength": strength[ids], "current_share": share[ids],
                                                "backbone": backbone[ids]},
           "git_commit": meta.get("git_commit")}
    base = os.path.join(a.out, f"fdtest_observables_N{a.N}.json")
    with open(base, "w") as fh:
        json.dump(sanitize(out), fh, indent=1, allow_nan=False)

    # ---- figures (200 dpi)
    lags = np.arange(L) * r
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, name in zip(axes, results):
        o = results[name]
        ax.plot(lags, o["z_mean"], label="<Z(tau)>"); ax.plot(lags, o["c_zz"], "--", label="C_ZZ(tau)")
        ax.axvline(lags[o["unbinned"]["tau_c"]], color="gray", lw=0.6)
        ax.set_xlabel("lag (steps)"); ax.legend(fontsize=8)
        ax.set_title(f"{OBS_LABEL[name]}: delta2 = {fmt(o['unbinned']['delta2'], 3)} (floor {fmt(o['unbinned']['noise_floor'], 3)}), "
                     f"binned {fmt(o['binned']['delta2'], 3)}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"fdtest_observables_N{a.N}.png"), dpi=200); plt.close(fig)
    w = wm["w"]
    lim = max(float(np.nanmax(np.abs(w))), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    sc = axes[0].scatter(pos[:, 0] * 1e6, pos[:, 1] * 1e6, c=w, s=10, cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[0].set_xlabel("x (um)"); axes[0].set_ylabel("y (um)"); axes[0].set_title("weight of each recorded junction", fontsize=9)
    fig.colorbar(sc, ax=axes[0], label="w_j")
    axes[1].scatter(pos[:, 0] * 1e6, w, c=w, s=10, cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[1].axhline(0, color="gray", lw=0.6); axes[1].set_xlabel("x position (um)"); axes[1].set_ylabel("w_j")
    axes[2].hist(w[np.isfinite(w)], bins=40, color="tab:green", alpha=0.8); axes[2].set_xlabel("w_j"); axes[2].set_ylabel("junctions")
    axes[2].set_title(f"fold cosine {fmt(wm['fold_cosine'], 2)}, rho(w, strength) {fmt(wm['spearman_strength'], 2)}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"fdtest_weights_N{a.N}.png"), dpi=200); plt.close(fig)

    # ---- comparison table
    print("comparison (decision: reverse-interval lower bound above the floor; tau_c in lags, * = cutoff at the last lag):")
    print(f"  {'observable':9s} {'variant':9s} {'delta2':>8s} {'reverse 95% CI':>20s} {'floor':>8s} {'excess':>8s} {'excludes':>8s} {'tau_c':>6s}"
          f" {'resp(sigma)':>11s} {'SNR<Z0>':>8s} {'noise corr':>10s}")
    for name, o in results.items():
        for variant in ("unbinned", "binned"):
            v = o[variant]
            tc = f"{v['tau_c']}{'*' if v['cutoff_at_end'] else ''}"
            print(f"  {OBS_LABEL[name]:9s} {variant:9s} {fmt(v['delta2']):>8s} {fmt_ci(v['ci95_reverse']):>20s} {fmt(v['noise_floor']):>8s} "
                  f"{fmt(v['excess']):>8s} {str(v['excludes_floor']):>8s} {tc:>6s} {fmt(o['achieved_response_sigma'], 3):>11s} "
                  f"{fmt(o['z0_snr'], 1):>8s} {fmt(o['noise_corr_length'], 1):>10s}")
    print(f"optimal weights: FOLD COSINE {fmt(wm['fold_cosine'], 3)} (stability of the map; associations are reported only if >= {MIN_FOLD_COSINE}); "
          f"response in-sample per fold {[fmt(v, 3) for v in wm['response_in_sample']]} sigma vs cross-fitted {fmt(wm['response_crossfit'], 3)} sigma")
    print(f"  Spearman(w, strength) {fmt(wm['spearman_strength'], 3)}, (w, current share) {fmt(wm['spearman_current_share'], 3)}, "
          f"(w, backbone) {fmt(wm['spearman_backbone'], 3)}; share of |w| on backbone junctions {fmt(wm['backbone_fraction_weighted'], 3)} "
          f"(backbone fraction of the recorded junctions {fmt(wm['backbone_fraction_recorded'], 3)})")
    if wm["associations_blanked"]:
        print("  WARNING:", wm["associations_note"])
    print("note:", NOTE)
    print("->", base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
