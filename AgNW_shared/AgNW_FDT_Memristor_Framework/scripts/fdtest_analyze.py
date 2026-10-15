"""Analyze the FDTest runs of one network (spec sections 5-6): network-level g_eff, per-junction and class-averaged
delta2 with cycle-block bootstrap intervals and split-half noise floors, junction-property associations, acceptance
gate, figures.

  python -m scripts.fdtest_analyze N [--out validation_output/fdtest] [--n-boot 200] [--max-snapshots 40] [--n-lags INT]
                                  [--no-binned] [--bin-first 8] [--bin-ratio 1.15] [--fixed-window W | --no-fixed]
                                  [--interval-method reverse|percentile|sqrt_reverse]

--n-lags overrides the number of lags stored in the run files (longer or shorter relaxation window; no re-simulation is
needed because the runs keep the whole stage III and IV). Allowed: 2 <= n_lags <= min(T_samp, T_relax) / record_every,
otherwise the analyzer exits with status 1 before any work. The lags used are printed and stored in the JSON schedule.

Reads run_N{N}_seed*.npz (all with the same schedule; the recorded junction ids and the snapshot width are checked against
the network rebuilt from network_seed), writes fdtest_results_N{N}.json (strict JSON: non-finite values are null) and
fdtest_*_N{N}.png (200 dpi). Exit status 0 if the acceptance gate passes, 2 if it fails (1 is a crash or inconsistent inputs).

Lag-binned violation (docs/lag_binning_design.md; on by default, --no-binned skips it): the lags are grouped in bins (first
--bin-first single lags, then each edge max(previous + 1, previous * --bin-ratio)), delta2_bin = sum_b n_b (C_b - Z_b)^2 /
sum_b n_b Z_b^2 up to the cutoff bin chosen on the binned Z curve, with the same reverse-interval decision rule and a floor computed
with the same binning. It is stored under `binned` (g_eff and the five classes; tau_c is the last lag of the cutoff bin), with
`bin_scheme` and `lag_noise_corr` (correlation length across lags of the half-difference of <Z>: about 1 for white scatter, so
binning n_b lags reduces the noise variance by about n_b / corr_length). The acceptance gate stays on the unbinned quantities.

Fixed window (on by default, --no-fixed skips it): delta2, its bootstrap intervals and the noise floor computed over the FIXED lags
0..W (--fixed-window W, default the whole window n_lags - 1) instead of the data-dependent cutoff tau_c, which jumps between
bootstrap resamples and inflates the variance. Stored under `fixed_window` (`window`, `geff`, the five classes; same fields as
`binned`) and printed in its own section; drawn after the binned block, so no other result changes. --interval-method picks
the interval that decides `excludes_floor` in every block (default reverse = the historical rule; percentile, or reverse
on sqrt(delta2)); the chosen name is stored as `interval_method` and the interval used as `ci95_decision`.

Outputs for g_eff and for each of the five junction classes: delta2, its percentile interval ('ci95', kept) and the
REVERSE interval 2 d - q (the decision rule: `excludes_floor` is True only if the reverse lower bound exceeds the split-half
noise floor, a mean over N_FLOOR_SPLITS random splits), the excess delta2 - floor, the cutoff lag `tau_c` with the flag
`cutoff_at_end` (relaxation never fell below 5 %: delta2 is truncated), and the a-posteriori perturbation (g_eff:
`achieved_response_sigma`, `overlap_fraction`, excluded fraction; classes: `achieved_shift_sigma`). Required gates: drift,
P_0 halves (g_eff, junctions), odd/even (g_eff; classes: at least 80 % of the classes), power (g_eff, classes). The two
scaling checks are ADVISORY (stored under gate with advisory = True, not part of gate_passed) and the quarter / half / all
values with their excess over the (1/M scaled) floor are stored under `scaling`. Associations are descriptive only.

Memory: the stage arrays hold about M * (2 * T_samp/record_every + n_lags) * K bytes (float32/uint8), and the peak while
stacking the seeds is about 4 GB for M = 480 cycles, K = 1000. Standalone driver (no unit tests; see tests/test_fdtest_smoke.py)."""
import argparse
import glob
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
from agnw_sim.dualnode import DualNodeCircuit
from scripts.fdtest_common import build_setup

CHUNK = 100
N_FLOOR_SPLITS = 30           # random half assignments averaged in every noise floor
CLASS_ODD_EVEN_MIN = 0.8      # fraction of classes whose odd/even estimates must mutually contain each other
ASSOC_NOTE = ("descriptive only: single-junction delta2 has a noise floor that depends on p_on, relaxation time and |beta|, "
              "the properties being correlated")
SCHEDULE_KEYS = ("V0", "eps", "T_relax", "T_samp", "n_cycles", "record_every", "n_lags", "network_seed", "k", "p_relax", "N")


def load_runs(out, n_wires):
    files = sorted(glob.glob(os.path.join(out, f"run_N{n_wires}_seed*.npz")))
    if not files:
        sys.exit(f"no run files for N={n_wires} in {out}")
    runs, first = [], None
    for f in files:
        z = np.load(f)
        meta = json.loads(str(z["schedule_json"]))
        if first is None:
            first = meta
        elif any(meta[k] != first[k] for k in SCHEDULE_KEYS):
            sys.exit(f"{f}: schedule differs from {files[0]}; analyze runs of one schedule only")
        runs.append(z)
    return runs, first, files


def weight_rows(n, n_boot, rng):
    """All cycle resamplings needed by the analysis, evaluated in ONE pass over the junction chunks:
    row 0 uniform; n_boot bootstraps; odd-cycles and even-cycles subsets and their bootstraps; first quarter and
    first half of the cycles (finite-M scaling)."""
    idx = np.arange(n)
    odd, even = idx[1::2], idx[0::2]
    rows, tags = [np.full(n, 1.0 / n)], ["uniform"]

    def counts(pool):
        c = np.bincount(rng.choice(pool, size=len(pool)), minlength=n).astype(float)
        return c / c.sum()

    def uniform_on(pool):
        w = np.zeros(n); w[pool] = 1.0 / len(pool); return w

    rows += [counts(idx) for _ in range(n_boot)]; tags += ["boot"] * n_boot
    if len(odd) >= 2 and len(even) >= 2:
        rows += [uniform_on(odd), uniform_on(even)]; tags += ["odd", "even"]
        rows += [counts(odd) for _ in range(n_boot)]; tags += ["boot_odd"] * n_boot
        rows += [counts(even) for _ in range(n_boot)]; tags += ["boot_even"] * n_boot
    rows += [uniform_on(idx[: max(n // 4, 1)]), uniform_on(idx[: max(n // 2, 1)])]; tags += ["quarter", "half"]
    return np.stack(rows), np.array(tags)


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [2.5, 97.5])] if x.size else [float("nan"), float("nan")]


def fmt(v, nd=4):
    return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def fmt_ci(c):
    return "[" + ", ".join(fmt(v, 3) for v in c) + "]"


def sanitize(o):
    """Recursively turn a results structure into strict JSON: NaN/Inf -> None, numpy scalars/arrays -> python types."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    if isinstance(o, np.generic):
        return sanitize(o.item())
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    return o


def main(argv=None):
    """Run the analysis; RuntimeWarnings from degenerate data (all-NaN slices, empty means, 0/0) are silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _main(argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("N", type=int); p.add_argument("--out", default="validation_output/fdtest")
    p.add_argument("--n-boot", type=int, default=200); p.add_argument("--max-snapshots", type=int, default=40)
    p.add_argument("--no-binned", action="store_true", help="skip the lag-binned block")
    p.add_argument("--bin-first", type=int, default=8, help="number of single-lag bins at the start (default 8)")
    p.add_argument("--bin-ratio", type=float, default=1.15, help="growth ratio of the bin edges (default 1.15)")
    p.add_argument("--fixed-window", type=int, default=None, help="last lag W of the fixed window 0..W (default: the whole window, n_lags - 1)")
    p.add_argument("--no-fixed", action="store_true", help="skip the fixed-window block")
    p.add_argument("--interval-method", choices=ft.INTERVAL_METHODS, default="reverse",
                   help="interval that decides excludes_floor (default reverse)")
    p.add_argument("--n-lags", type=int, default=None, help="lags of the relaxation / autocorrelation window (default: as stored in the runs)")
    a = p.parse_args(argv)
    if a.bin_first < 1 or not (np.isfinite(a.bin_ratio) and a.bin_ratio > 1):
        sys.exit(f"--bin-first must be >= 1 and --bin-ratio finite and > 1 (got {a.bin_first} and {a.bin_ratio})")
    if a.no_fixed and a.fixed_window is not None:
        sys.exit("--fixed-window and --no-fixed are mutually exclusive")
    rng = np.random.default_rng(12345)

    runs, meta, files = load_runs(a.out, a.N)
    n_lags = meta["n_lags"] if a.n_lags is None else a.n_lags
    try:
        sch = fp.Schedule(V0=meta["V0"], eps=meta["eps"], T_relax=meta["T_relax"], T_samp=meta["T_samp"],
                          n_cycles=meta["n_cycles"], record_every=meta["record_every"], n_lags=n_lags)
    except ValueError as err:
        r = meta["record_every"]
        sys.exit(f"--n-lags {n_lags} is not allowed for these runs ({err}); the allowed range is 2 to "
                 f"{min(meta['T_samp'] // r, meta['T_relax'] // r)} (min of T_samp // record_every = {meta['T_samp'] // r} and "
                 f"T_relax // record_every = {meta['T_relax'] // r})")
    fixed_w = sch.n_lags - 1 if a.fixed_window is None else a.fixed_window
    if not a.no_fixed and not 0 <= fixed_w <= sch.n_lags - 1:
        sys.exit(f"--fixed-window {fixed_w} is outside the analysis window: allowed 0 to {sch.n_lags - 1} (n_lags = {sch.n_lags})")
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

    # ---- stage arrays, all seeds stacked along the cycle axis
    geff = [fp.extract_stage_arrays((z["I_total"] / V_in)[::sch.record_every], sch) for z in runs]
    gbin = [fp.extract_stage_arrays(z["G_sampled"], sch) for z in runs]
    x2, x3, x4 = (np.concatenate([g[i] for g in geff]) for i in range(3))
    g2, g3, g4 = (np.concatenate([g[i] for g in gbin]) for i in range(3))
    del geff, gbin
    M, K, L = x2.shape[0], g2.shape[2], sch.n_lags
    print(f"N={a.N}: {len(runs)} runs, M = {M} cycles, K = {K} recorded junctions, {L} lags "
          f"(lag step {sch.record_every}), V0 = {sch.V0} V, eps = {sch.eps:.4f} V")

    # ---- network level (g_eff)
    res = ft.continuous_fdtest(x2, x3, x4)
    boot_d2, boot_z0 = ft.bootstrap_continuous_delta2(x2, x3, x4, a.n_boot, rng, return_z0=True)
    floor = ft.continuous_split_half_floor(x2, x3, x4, rng, n_splits=N_FLOOR_SPLITS)
    variants = {f"hist{nb}": ft.continuous_fdtest(x2, x3, x4, n_bins=nb)["delta2"] for nb in (40, 100)}
    variants["kde"] = ft.continuous_fdtest(x2, x3, x4, estimator="kde")["delta2"]
    odd_ev = None
    if M >= 8:
        bo = ft.bootstrap_continuous_delta2(x2[1::2], x3[1::2], x4[1::2], a.n_boot, rng)
        be = ft.bootstrap_continuous_delta2(x2[0::2], x3[0::2], x4[0::2], a.n_boot, rng)
        d_o = ft.continuous_fdtest(x2[1::2], x3[1::2], x4[1::2])["delta2"]
        d_e = ft.continuous_fdtest(x2[0::2], x3[0::2], x4[0::2])["delta2"]
        odd_ev = fp.gate_odd_even(d_o, ci(bo), d_e, ci(be))
    geff_vs = ft.violation_summary(res["delta2"], boot_d2, floor, method=a.interval_method)
    geff_out = {"delta2": geff_vs["delta2"], "ci95": geff_vs["ci95_percentile"], "ci95_reverse": geff_vs["ci95_reverse"],
                "ci95_decision": geff_vs["ci95"], "noise_floor": floor, "excess": geff_vs["excess"], "excludes_floor": geff_vs["excludes_floor"],
                "tau_c": int(res["tau_c"]), "cutoff_at_end": bool(res["tau_c"] == L - 1),
                "excluded_fraction": res["excluded_fraction"], **fp.achieved_response(x2, x4),
                "estimator_variants": variants}

    # ---- junction properties and classes
    strength = fp.junction_strength(net)
    snaps = np.concatenate([z["G_snapshots"][1::2] for z in runs])            # ends of stage IV
    if len(snaps) > a.max_snapshots:
        snaps = snaps[np.linspace(0, len(snaps) - 1, a.max_snapshots).astype(int)]
    share, backbone = fp.current_share(circuit, snaps, sch.V0)
    masks, names = fp.junction_classes(strength[ids], backbone[ids])

    # ---- junction level, chunks of junctions, one pass for every resampling
    W, tags = weight_rows(M, a.n_boot, rng)
    splits = [rng.permutation(M) for _ in range(N_FLOOR_SPLITS)]               # fixed random half assignments
    C = len(names)
    acc_z, acc_c, acc_n = np.zeros((len(W), C, L)), np.zeros((len(W), C, L)), np.zeros((len(W), C))
    sf, sn = np.zeros((C, L)), np.zeros(C)
    sa, sb = np.zeros((N_FLOOR_SPLITS, C, L)), np.zeros((N_FLOOR_SPLITS, C, L))   # per-split sums: floors need complete sums
    shift_sum, shift_n = np.zeros(C), np.zeros(C)
    d2_j, valid_j, p0_j = np.full(K, np.nan), np.zeros(K, bool), np.zeros(K)
    halves_ok, halves_n = 0.0, 0
    for lo in range(0, K, CHUNK):
        sl = slice(lo, min(lo + CHUNK, K))
        stats = ft.binary_stage_stats(g2[:, :, sl], g3[:, :, sl], g4[:, :, sl], L - 1)
        z_, c_, n_ = ft.bootstrap_class_sums(stats, W, masks[:, sl])
        acc_z += z_; acc_c += c_; acc_n += n_
        for i, perm in enumerate(splits):
            parts = ft.split_half_class_sums(stats, masks[:, sl], perm[: M // 2], perm[M // 2:])
            if i == 0:
                sf += parts[0]; sn += parts[3]                                    # identical for every split
            sa[i] += parts[1]; sb[i] += parts[2]
        pj = ft.binary_fdtest(stats)
        shift = ft.binary_shift_sigma(stats)
        okj = pj["valid"] & np.isfinite(shift)
        shift_sum += (masks[:, sl] & okj).astype(float) @ np.where(okj, shift, 0.0); shift_n += (masks[:, sl] & okj).sum(axis=1)
        d2_j[sl], valid_j[sl], p0_j[sl] = pj["delta2"], pj["valid"], stats["m4"].mean(axis=0)
        if pj["valid"].any():
            g = fp.gate_halves(g4[:, :, sl][:, :, pj["valid"]])
            halves_ok += g["value"] * pj["valid"].sum(); halves_n += int(pj["valid"].sum())

    rows = lambda tag: np.nonzero(tags == tag)[0]
    d2_tau = [ft.delta2_from_sums(acc_z[r], acc_c[r], acc_n[r], return_tau=True) for r in range(len(W))]
    d2_rows, tau_rows = np.stack([t[0] for t in d2_tau]), np.stack([t[1] for t in d2_tau])            # (rows, C)
    z0_rows = acc_z[:, :, 0] / np.where(acc_n > 0, acc_n, np.nan)
    floors = np.mean([ft.floor_from_sums(sf, sa[i], sb[i], sn) for i in range(N_FLOOR_SPLITS)], axis=0)
    classes_out = {}
    for c, name in enumerate(names):
        nv = int(acc_n[0, c])
        vs = ft.violation_summary(d2_rows[0, c], d2_rows[rows("boot"), c], floors[c], method=a.interval_method)
        classes_out[name] = {"n_valid": nv, "delta2": vs["delta2"], "ci95": vs["ci95_percentile"], "ci95_reverse": vs["ci95_reverse"],
                             "ci95_decision": vs["ci95"], "noise_floor": float(floors[c]), "excess": vs["excess"], "excludes_floor": vs["excludes_floor"],
                             "tau_c": int(tau_rows[0, c]) if nv else None, "cutoff_at_end": bool(nv and tau_rows[0, c] == L - 1),
                             "achieved_shift_sigma": float(shift_sum[c] / shift_n[c]) if shift_n[c] else float("nan"),
                             "z0_snr": fp.gate_power(z0_rows[rows("boot"), c])["value"]}

    # ---- junction-property associations (Spearman, bootstrap over junctions)
    props = {"strength": strength[ids], "p_on": p0_j, "current_share": share[ids], "backbone": backbone[ids]}
    good = valid_j & np.isfinite(d2_j)
    assoc = {}
    for name, x in props.items():
        if good.sum() > 5 and np.ptp(x[good]) > 0:
            rho = spearmanr(x[good], d2_j[good])[0]
            idx = np.nonzero(good)[0]
            bs = [spearmanr(x[i], d2_j[i])[0] for i in (rng.choice(idx, len(idx)) for _ in range(200))]
            assoc[name] = {"rho": float(rho), "ci95": ci(bs)}

    # ---- acceptance gate
    gate = {"drift": fp.gate_drift(x4),
            "p0_halves_geff": fp.gate_halves(x4, min_fraction=1.0),
            "p0_halves_junctions": {"passed": bool(halves_n > 0 and halves_ok / halves_n >= 0.95),
                                    "value": halves_ok / halves_n if halves_n else float("nan")}}
    if len(rows("odd")):
        o, e = rows("odd")[0], rows("even")[0]
        bo, be = rows("boot_odd"), rows("boot_even")
        per = {n: (bool(fp.gate_odd_even(d2_rows[o, c], ci(d2_rows[bo, c]), d2_rows[e, c], ci(d2_rows[be, c]))["passed"])
                   if np.isfinite(d2_rows[o, c]) and np.isfinite(d2_rows[e, c]) else None) for c, n in enumerate(names)}
        n_ok = sum(v is True for v in per.values())
        gate["odd_even_junction_classes"] = {"passed": bool(n_ok >= int(np.ceil(CLASS_ODD_EVEN_MIN * C))),
                                             "n_classes": sum(v is not None for v in per.values()), "fraction": n_ok / C,
                                             "min_fraction": CLASS_ODD_EVEN_MIN, "per_class": per}
    else:
        gate["odd_even_junction_classes"] = {"passed": False, "skipped": "M<4"}
    gate["odd_even_geff"] = odd_ev if odd_ev is not None else {"passed": False, "skipped": "M<8"}
    # finite-M scaling: delta2 from the first quarter / half / all of the cycles. ADVISORY: an endpoint test is unreliable
    # when the true delta2 is large, so it is stored and printed but does not enter gate_passed. The floor scales as 1/M.
    n_q, n_h = max(M // 4, 1), max(M // 2, 1)
    r_q, r_h, r_all = rows("quarter")[0], rows("half")[0], 0
    ms = [n_q, n_h, M]
    scaling = {"n_cycles": ms, "classes": {}}
    if M < 8:
        gate["scaling_geff"] = {"passed": False, "skipped": "M<8", "advisory": True}
        scaling["geff"] = None
    else:
        d_q = ft.continuous_fdtest(x2[: M // 4], x3[: M // 4], x4[: M // 4])["delta2"]
        d_h = ft.continuous_fdtest(x2[: M // 2], x3[: M // 2], x4[: M // 2])["delta2"]
        gate["scaling_geff"] = {**fp.gate_scaling([d_q, d_h, res["delta2"]], floor=floor if np.isfinite(floor) else 0.0), "advisory": True}
        scaling["geff"] = {"delta2": [d_q, d_h, res["delta2"]], "floor_scaled": [floor * M / m for m in ms],
                           "excess": [d - floor * M / m for d, m in zip((d_q, d_h, res["delta2"]), ms)]}
    for c, name in enumerate(names):
        ds = [d2_rows[r_q, c], d2_rows[r_h, c], d2_rows[r_all, c]]
        scaling["classes"][name] = {"delta2": ds, "floor_scaled": [floors[c] * M / m for m in ms],
                                    "excess": [d - floors[c] * M / m for d, m in zip(ds, ms)]}
    gate["scaling_classes"] = {**fp.gate_scaling([np.nanmean(d2_rows[r_q]), np.nanmean(d2_rows[r_h]), np.nanmean(d2_rows[r_all])],
                                                 floor=float(np.nanmean(floors))), "advisory": True}
    gate["power_geff"] = fp.gate_power(boot_z0)
    failing = [n for n, v in classes_out.items() if v["n_valid"] == 0 or not np.isfinite(v["z0_snr"]) or v["z0_snr"] < 3]
    snrs = [v["z0_snr"] for v in classes_out.values() if np.isfinite(v["z0_snr"])]
    gate["power_classes"] = {"passed": not failing, "min_snr": float(min(snrs)) if snrs else float("nan"),
                             "failing_classes": failing}       # an empty class cannot support an SNR claim
    failed = [k for k, g in gate.items() if not g.get("advisory") and not g["passed"]]
    gate_ok = not failed

    # ---- lag-binned violation (after every unbinned random draw, so the unbinned results do not depend on it)
    binned = bin_scheme = noise_corr = None
    if not a.no_binned:
        edges = ft.lag_bin_edges(L, a.bin_first, a.bin_ratio)
        bin_scheme = {"n_first": a.bin_first, "ratio": a.bin_ratio, "n_bins": len(edges) - 1, "edges": edges.tolist()}
        res_b = ft.continuous_fdtest(x2, x3, x4, edges=edges)
        boot_b = ft.bootstrap_continuous_delta2(x2, x3, x4, a.n_boot, rng, edges=edges)
        floor_b, z_half_a, z_half_b = ft.continuous_split_half_floor(x2, x3, x4, rng, n_splits=N_FLOOR_SPLITS, edges=edges, return_halves=True)
        vs = ft.violation_summary(res_b["delta2"], boot_b, floor_b, method=a.interval_method)
        geff_b = {"delta2": vs["delta2"], "ci95": vs["ci95_percentile"], "ci95_reverse": vs["ci95_reverse"], "ci95_decision": vs["ci95"], "noise_floor": floor_b,
                  "excess": vs["excess"], "excludes_floor": vs["excludes_floor"], "tau_c": int(res_b["tau_c"]),
                  "cutoff_at_end": bool(res_b["cutoff_at_end"])}
        use = np.concatenate([[0], rows("boot")])                                # uniform row and the bootstrap rows
        d2b, taub = zip(*[ft.delta2_from_sums(acc_z[r], acc_c[r], acc_n[r], return_tau=True, edges=edges) for r in use])
        d2b, taub = np.stack(d2b), np.stack(taub)
        floors_b = np.mean([ft.floor_from_sums(sf, sa[i], sb[i], sn, edges=edges) for i in range(N_FLOOR_SPLITS)], axis=0)
        classes_b = {}
        for c, name in enumerate(names):
            nv = int(acc_n[0, c])
            v = ft.violation_summary(d2b[0, c], d2b[1:, c], floors_b[c], method=a.interval_method)
            classes_b[name] = {"n_valid": nv, "delta2": v["delta2"], "ci95": v["ci95_percentile"], "ci95_reverse": v["ci95_reverse"],
                               "ci95_decision": v["ci95"], "noise_floor": float(floors_b[c]), "excess": v["excess"], "excludes_floor": v["excludes_floor"],
                               "tau_c": int(taub[0, c]) if nv else None, "cutoff_at_end": bool(nv and taub[0, c] == L - 1)}
        binned = {"geff": geff_b, "classes": classes_b}
        # correlation length of the lag-to-lag scatter of <Z>: half-difference of the first random split (g_eff: the split of the floor)
        rho, corr = ft.lag_noise_correlation(z_half_a, z_half_b)
        noise_corr = {"max_lag": ft.LAG_NOISE_MAX_LAG, "geff": {"corr_length": corr, "rho": rho}, "classes": {}}
        for c, name in enumerate(names):
            rho, corr = ft.lag_noise_correlation(sa[0, c] / sn[c], sb[0, c] / sn[c]) if sn[c] > 0 else (np.array([np.nan]), float("nan"))
            noise_corr["classes"][name] = {"corr_length": corr, "rho": rho}

    # ---- fixed-window violation (after every other random draw, so no other result depends on it)
    fixed = None
    if not a.no_fixed:
        res_f = ft.continuous_fdtest(x2, x3, x4, cutoff=fixed_w)
        boot_f = ft.bootstrap_continuous_delta2(x2, x3, x4, a.n_boot, rng, cutoff=fixed_w)
        floor_f = ft.continuous_split_half_floor(x2, x3, x4, rng, n_splits=N_FLOOR_SPLITS, cutoff=fixed_w)
        vs = ft.violation_summary(res_f["delta2"], boot_f, floor_f, method=a.interval_method)
        geff_f = {"delta2": vs["delta2"], "ci95": vs["ci95_percentile"], "ci95_reverse": vs["ci95_reverse"], "ci95_decision": vs["ci95"],
                  "noise_floor": floor_f, "excess": vs["excess"], "excludes_floor": vs["excludes_floor"], "tau_c": int(res_f["tau_c"]),
                  "cutoff_at_end": bool(res_f["tau_c"] == L - 1)}
        use = np.concatenate([[0], rows("boot")])                                # uniform row and the bootstrap rows
        d2f = np.stack([ft.delta2_from_sums(acc_z[r], acc_c[r], acc_n[r], cutoff=fixed_w) for r in use])
        floors_f = np.mean([ft.floor_from_sums(sf, sa[i], sb[i], sn, cutoff=fixed_w) for i in range(N_FLOOR_SPLITS)], axis=0)
        classes_f = {}
        for c, name in enumerate(names):
            nv = int(acc_n[0, c])
            v = ft.violation_summary(d2f[0, c], d2f[1:, c], floors_f[c], method=a.interval_method)
            classes_f[name] = {"n_valid": nv, "delta2": v["delta2"], "ci95": v["ci95_percentile"], "ci95_reverse": v["ci95_reverse"],
                               "ci95_decision": v["ci95"], "noise_floor": float(floors_f[c]), "excess": v["excess"],
                               "excludes_floor": v["excludes_floor"], "tau_c": fixed_w if nv else None,
                               "cutoff_at_end": bool(nv and fixed_w == L - 1)}
        fixed = {"window": int(fixed_w), "geff": geff_f, "classes": classes_f}

    result = {"schedule": {**{k: meta[k] for k in SCHEDULE_KEYS}, "n_lags": int(sch.n_lags), "n_lags_stored": int(meta["n_lags"])}, "M": int(M), "K": int(K), "n_valid_junctions": int(valid_j.sum()),
              "geff": geff_out, "classes": classes_out, "junction_delta2": d2_j,
              "associations": assoc, "associations_note": ASSOC_NOTE, "scaling": scaling, "gate": gate, "gate_passed": bool(gate_ok), "git_commit": meta.get("git_commit"),
              "interval_method": a.interval_method}
    if binned is not None:
        result.update(binned=binned, bin_scheme=bin_scheme, lag_noise_corr=noise_corr)
    if fixed is not None:
        result["fixed_window"] = fixed
    base = os.path.join(a.out, f"fdtest_results_N{a.N}.json")
    with open(base, "w") as fh:
        json.dump(sanitize(result), fh, indent=1, allow_nan=False)

    # ---- figures (200 dpi)
    lags = np.arange(L) * sch.record_every
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lags, res["z_mean"], label="<Z(tau)> (relaxation after removal)"); ax.plot(lags, res["c_zz"], "--", label="C_ZZ(tau) (stationary)")
    ax.axvline(lags[res["tau_c"]], color="gray", lw=0.6); ax.set_xlabel("lag (steps)"); ax.legend(fontsize=8)
    ax.set_title(f"g_eff: delta2 = {fmt(res['delta2'], 3)} rev {fmt_ci(geff_out['ci95_reverse'])}, floor {fmt(floor, 3)}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"fdtest_geff_N{a.N}.png"), dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(C)
    d = np.array([classes_out[n]["delta2"] for n in names]); lo = np.array([classes_out[n]["ci95_decision"][0] for n in names])
    hi = np.array([classes_out[n]["ci95_decision"][1] for n in names]); fl = np.array([classes_out[n]["noise_floor"] for n in names])
    ax.bar(xs, d, yerr=[np.maximum(d - lo, 0), np.maximum(hi - d, 0)], capsize=3, color="tab:purple", alpha=0.7)
    ax.plot(xs, fl, "kx", label="noise floor"); ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\nn={classes_out[n]['n_valid']}" for n in names], fontsize=7)
    ax.set_ylabel(f"class-averaged delta2\n(bars: {a.interval_method} 95% interval)"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"fdtest_classes_N{a.N}.png"), dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(d2_j[good], bins=40, color="tab:green", alpha=0.8); axes[0].set_xlabel("single-junction delta2 (descriptive)")
    sc = axes[1].scatter(strength[ids][good], d2_j[good], c=backbone[ids][good], s=8, cmap="viridis")
    axes[1].set_xlabel("junction strength"); axes[1].set_ylabel("delta2_j"); fig.colorbar(sc, ax=axes[1], label="backbone fraction")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"fdtest_junctions_N{a.N}.png"), dpi=200); plt.close(fig)

    print(f"g_eff: delta2 = {fmt(res['delta2'])}, reverse 95% CI {fmt_ci(geff_out['ci95_reverse'])} (percentile {fmt_ci(geff_out['ci95'])}), "
          f"floor {fmt(floor)}, excess {fmt(geff_out['excess'])}, excludes_floor = {geff_out['excludes_floor']}")
    print(f"       tau_c = {geff_out['tau_c']}{' (CUTOFF AT END: delta2 truncated)' if geff_out['cutoff_at_end'] else ''}, "
          f"achieved response = {fmt(geff_out['achieved_response_sigma'], 3)} sigma, stage-II samples inside the central 95% of stage IV: "
          f"{fmt(geff_out['overlap_fraction'], 3)}, excluded fraction {fmt(geff_out['excluded_fraction'], 3)}; "
          f"estimator variants {{{', '.join(f'{k}: {fmt(v)}' for k, v in variants.items())}}}")
    print("classes (decision: reverse-interval lower bound above the floor):")
    for n in names:
        v = classes_out[n]
        tc = "-" if v["tau_c"] is None else f"{v['tau_c']}{'*' if v['cutoff_at_end'] else ''}"
        print(f"  {n:13s} n={v['n_valid']:4d} delta2 {fmt(v['delta2'])} rev {fmt_ci(v['ci95_reverse'])} floor {fmt(v['noise_floor'])} "
              f"excess {fmt(v['excess'])} excludes_floor {str(v['excludes_floor']):5s} tau_c {tc:>4s} "
              f"shift {fmt(v['achieved_shift_sigma'], 3)} sigma  SNR(<Z(0)>) {fmt(v['z0_snr'], 1)}")
    print("  (tau_c marked * = cutoff at the last lag: the relaxation never fell below 5%, delta2 is truncated)")
    if binned is not None:
        gb = binned["geff"]
        print(f"lag-binned violation ({bin_scheme['n_bins']} bins over {L} lags: {a.bin_first} single lags, then x{a.bin_ratio}; tau_c in lags, * = cutoff at end):")
        cl = lambda ncorr: fmt(ncorr, 1)
        print(f"  {'g_eff':13s}        delta2_bin {fmt(gb['delta2'])} rev {fmt_ci(gb['ci95_reverse'])} floor {fmt(gb['noise_floor'])} "
              f"excess {fmt(gb['excess'])} excludes_floor {str(gb['excludes_floor']):5s} tau_c {str(gb['tau_c']) + ('*' if gb['cutoff_at_end'] else ''):>4s} "
              f"noise corr length {cl(noise_corr['geff']['corr_length'])} lags")
        for n in names:
            v, nc_ = binned["classes"][n], noise_corr["classes"][n]
            tc = "-" if v["tau_c"] is None else f"{v['tau_c']}{'*' if v['cutoff_at_end'] else ''}"
            print(f"  {n:13s} n={v['n_valid']:4d} delta2_bin {fmt(v['delta2'])} rev {fmt_ci(v['ci95_reverse'])} floor {fmt(v['noise_floor'])} "
                  f"excess {fmt(v['excess'])} excludes_floor {str(v['excludes_floor']):5s} tau_c {tc:>4s} noise corr length {cl(nc_['corr_length'])} lags")
    if fixed is not None:
        gf = fixed["geff"]
        print(f"fixed window (lags 0..{fixed_w} of {L}, no data-dependent cutoff; {a.interval_method} interval decides):")
        print(f"  {'g_eff':13s}        delta2_fix {fmt(gf['delta2'])} rev {fmt_ci(gf['ci95_reverse'])} floor {fmt(gf['noise_floor'])} "
              f"excess {fmt(gf['excess'])} excludes_floor {gf['excludes_floor']}")
        for n in names:
            v = fixed["classes"][n]
            print(f"  {n:13s} n={v['n_valid']:4d} delta2_fix {fmt(v['delta2'])} rev {fmt_ci(v['ci95_reverse'])} floor {fmt(v['noise_floor'])} "
                  f"excess {fmt(v['excess'])} excludes_floor {str(v['excludes_floor']):5s}")
    print("associations:", {k: round(v["rho"], 3) for k, v in assoc.items()}, "--", ASSOC_NOTE)
    show = lambda g: {kk: (round(float(vv), 4) if isinstance(vv, (float, np.floating)) else vv)
                      for kk, vv in g.items() if kk not in ("passed", "advisory")}
    for k, g in gate.items():
        if not g.get("advisory"):
            print(f"  gate {k:28s} {'PASS' if g['passed'] else 'FAIL'}  {show(g)}")
    print("advisory checks (not part of the gate):")
    for k, g in gate.items():
        if g.get("advisory"):
            print(f"  {k:28s} {'ok' if g['passed'] else 'flag'}  {show(g)}")
    if scaling["geff"]:
        print("  scaling of g_eff delta2 (quarter, half, all cycles):", [round(v, 4) for v in scaling["geff"]["delta2"]],
              "excess over the 1/M-scaled floor:", [round(v, 4) for v in scaling["geff"]["excess"]])
    for n, v in scaling["classes"].items():
        print(f"  scaling of {n:13s}:", [fmt(d) for d in v["delta2"]], "excess:", [fmt(d) for d in v["excess"]])
    print("acceptance gate:", "PASSED" if gate_ok else f"FAILED ({', '.join(failed)}; increase T_relax / cycles)", "->", base)
    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
