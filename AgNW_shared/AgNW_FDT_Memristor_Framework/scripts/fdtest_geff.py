"""FDTest of the network conductance g_eff with the switch-ON doubling and a noise-based cutoff.

  python -m scripts.fdtest_geff N --out DIR [--n-boot 200] [--n-lags 1400] [--seed 12345]

Reads only the `I_total` array of the run_N{N}_seed*.npz files of one schedule (same schedule-consistency checks as
scripts.fdtest_analyze; junction arrays are not read) with g_eff = I_total / V_in recorded every `record_every` steps, and writes
fdtest_geff_N{N}.json (strict JSON, non-finite values are null) and fdtest_geff_switch_N{N}.png (200 dpi). Exit status 0, or 1
with a message for inconsistent inputs (e.g. --n-lags outside 2 .. min(T_samp, T_relax) / record_every).

Variants (Engbring et al., PRX 13, 021034: the statistics can be doubled with the switch-on transient):
  OFF   the usual analysis: x2 = stage II (P_eps), x3 = stage III from the removal instant, x4 = stage IV (P_0).
  ON    mirror analysis with the roles swapped: continuous_fdtest(x2 := stage IV, x3 := stage I lags, x4 := stage II). The mirror
        relation: the system starts (stage I) from P_0 and relaxes under the perturbed dynamics towards P_eps, whose steady state
        is stage II. With z'(x) = P_0(x) / P_eps(x) - 1 (z_map(x_eps := stage IV, x_0 := stage II)), <z'(tau)> over the stage I
        lags equals C_z'z'(tau) evaluated in the perturbed steady state (stage II), for ANY Markov observable, exactly as
        <z(tau)> = C_zz(tau) does for the switch-off (both follow from E_start[z(x_tau)] - E_stationary[z] and the stationarity
        of the reference ensemble). The lag-0 state of stage I is the last stage IV record plus one record, hence P_0.
  BOTH  the mean of the OFF and ON delta2 (and of their floors): the two directions use the same cycles.
Cycle 0 of EVERY run file is dropped from all three variants: it starts from the all-OFF initial condition, so its switch-on
transient does not start from P_0 (and its stage II has not equilibrated). Hence M = n_runs * (n_cycles - 1) cycles, and OFF here
differs slightly from scripts.fdtest_analyze (which keeps cycle 0).

Cutoff rules, per variant: `c5` the existing rule (first lag where |<z>| stays below 5 % of |<z>(0)|; chosen from the data of each
resample, as in the analyzer) and `noise` (fdtest.noise_cutoff, computed ONCE per direction on the full sample from the per-cycle
z curves and then held fixed: the estimate, every bootstrap resample and the noise floor use that window). For BOTH with the
noise rule the common window is the LARGER of the two directional cutoffs (max): a shared window keeps the average of the two
delta2 a plain average over one lag range, and the larger one avoids truncating the slower direction; the OFF and ON entries
use their own cutoff, so BOTH-noise is NOT the mean of the OFF-noise and ON-noise rows. For BOTH with the c5 rule each direction keeps its own data-driven cutoff (reported as cutoff_off / cutoff_on;
`cutoff` is their maximum). `hit_end` is True if the cutoff is the last lag (never reached / signal below the noise from lag 0).

Bootstrap: n_boot resamples of the cycles; ONE index vector per resample is drawn and applied to the OFF and to the ON data (the
same cycles in both directions, so BOTH is the average of two estimates on the same resample; z map re-estimated in each
direction). The `noise` delta2 of a resample uses the same z map and curves as its `c5` delta2 (only the window differs), so both
rules cost one continuous_fdtest per direction and resample. Floor: mean of 30 random split-half floors; OFF and ON use the SAME
random splits (identically seeded generators) and the BOTH floor is the average of the two per-direction floors. Intervals:
reverse 95 % interval (violation_summary); excludes_floor as elsewhere. z0_snr: gate_power on the bootstrap <z(0)> (BOTH: mean of
the two directions' values per resample). achieved_response_sigma / overlap_fraction: the physical perturbation, from stage II
against stage IV. noise_corr_length: correlation length across lags of the split-half scatter of <z> (BOTH: mean of the two).

Runtime/memory: only I_total is read (1-2 M doubles per run), the stage arrays hold about M * (2 * T_samp / record_every + 2 * n_lags)
doubles; the bootstrap (2 continuous_fdtest per resample) dominates the time."""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agnw_sim import fdtest as ft
from agnw_sim import fdtest_protocol as fp
from scripts.fdtest_analyze import load_runs, sanitize, fmt, fmt_ci, N_FLOOR_SPLITS

RULES = ("c5", "noise")
VARIANTS = ("OFF", "ON", "BOTH")
NOTE = ("OFF/ON: switch-off / switch-on transients (ON with the roles of the stages swapped); BOTH averages them on the same cycles; "
        "cycle 0 of every run is excluded; the noise cutoff is chosen once on the full sample and held fixed. BOTH with the noise rule "
        "is NOT the mean of the OFF-noise and ON-noise rows: it uses the max of the two directional cutoffs as a common window "
        "(estimate, bootstrap and floor on that window)")


def stage_arrays(runs, sch):
    """(x2, x3, x4, xI) of the g_eff series of all runs stacked along the cycle axis, cycle 0 of every run dropped (see the
    module docstring). g_eff = I_total / V_in with V_in rebuilt from the schedule, sampled every record_every steps."""
    steps = np.arange(sch.n_steps)
    V_in = np.where(steps % sch.T_cycle < sch.T_relax + sch.T_samp, sch.V0 + sch.eps, sch.V0)
    parts = []
    for z in runs:
        series = (z["I_total"] / V_in)[::sch.record_every]
        parts.append([a[1:] for a in (*fp.extract_stage_arrays(series, sch), fp.extract_switch_on_arrays(series, sch))])
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def delta2_at(res, cutoff):
    """delta2 of a free estimate `res` (continuous_fdtest output) over the fixed window 0..cutoff (clipped to the last lag):
    identical to continuous_fdtest(..., cutoff=cutoff) on the same data, without repeating the z map."""
    return ft.delta_squared(res["c_zz"], res["z_mean"], ft._fixed_end(cutoff, len(res["z_mean"])))


def bootstrap_switch(x2, x3, x4, xI, n_boot, rng, cut_off, cut_on):
    """Bootstrap over cycles with the SAME indices for both directions. Returns {"c5": {off, on, both}, "noise": {off, on, both}
    (n_boot,) delta2 arrays, "z0": {off, on, both}}. c5: each direction with its own data-driven cutoff; noise: fixed windows
    cut_off / cut_on, and for `both` the common window max(cut_off, cut_on)."""
    n = x2.shape[0]
    w = max(cut_off, cut_on)
    out = {r: {k: np.empty(n_boot) for k in ("off", "on", "both")} for r in RULES}
    z0 = {k: np.empty(n_boot) for k in ("off", "on", "both")}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        r_off = ft.continuous_fdtest(x2[idx], x3[idx], x4[idx])
        r_on = ft.continuous_fdtest(x4[idx], xI[idx], x2[idx])
        d = {"c5": (r_off["delta2"], r_on["delta2"]), "noise": (delta2_at(r_off, cut_off), delta2_at(r_on, cut_on))}
        for rule in RULES:
            out[rule]["off"][b], out[rule]["on"][b] = d[rule]
        out["c5"]["both"][b] = 0.5 * (d["c5"][0] + d["c5"][1])
        out["noise"]["both"][b] = 0.5 * (delta2_at(r_off, w) + delta2_at(r_on, w))
        z0["off"][b], z0["on"][b] = r_off["z_mean"][0], r_on["z_mean"][0]
        z0["both"][b] = 0.5 * (z0["off"][b] + z0["on"][b])
    return {**out, "z0": z0}


def analyze_geff(x2, x3, x4, xI, n_boot, seed=12345, n_floor=N_FLOOR_SPLITS):
    """Everything of the driver on ready stage arrays. Returns a dict with `variants` {OFF, ON, BOTH} -> {c5, noise} -> block, the
    directional curves and the noise-level information (numpy values; sanitize before json)."""
    dirs = {"off": (x2, x3, x4), "on": (x4, xI, x2)}
    L = x3.shape[1]
    free = {k: ft.continuous_fdtest(*v) for k, v in dirs.items()}
    nc = {k: ft.noise_cutoff(free[k]["z3"]) for k in dirs}
    cut = {k: nc[k][0] for k in dirs}
    w = max(cut.values())
    boot = bootstrap_switch(x2, x3, x4, xI, n_boot, np.random.default_rng([seed, 0]), cut["off"], cut["on"])
    z0_snr = {k: fp.gate_power(v)["value"] for k, v in boot["z0"].items()}

    floors, halves = {}, {}
    def floor(k, cutoff):
        key = (k, cutoff)
        if key not in floors:                                          # identically seeded generators: the same splits in both directions
            floors[key], za, zb = ft.continuous_split_half_floor(*dirs[k], np.random.default_rng([seed, 1]), n_splits=n_floor,
                                                                  return_halves=True, cutoff=cutoff)
            halves[k] = (za, zb)
        return floors[key]

    est = {"c5": {k: free[k]["delta2"] for k in dirs}, "noise": {k: delta2_at(free[k], cut[k]) for k in dirs}}
    fl = {"c5": {k: floor(k, None) for k in dirs}, "noise": {k: floor(k, cut[k]) for k in dirs}}
    fl["noise_both"] = {k: floor(k, w) for k in dirs}
    corr = {k: ft.lag_noise_correlation(*halves[k])[1] for k in dirs}
    resp = fp.achieved_response(x2, x4)
    tau = {"c5": {k: int(free[k]["tau_c"]) for k in dirs}, "noise": cut}
    hit = {"c5": {k: bool(free[k]["tau_c"] == L - 1) for k in dirs}, "noise": {k: bool(nc[k][1]) for k in dirs}}
    est_both = {"c5": 0.5 * (est["c5"]["off"] + est["c5"]["on"]),
                "noise": 0.5 * (delta2_at(free["off"], w) + delta2_at(free["on"], w))}
    floor_both = {"c5": 0.5 * (fl["c5"]["off"] + fl["c5"]["on"]), "noise": 0.5 * (fl["noise_both"]["off"] + fl["noise_both"]["on"])}

    variants = {}
    for name, key in (("OFF", "off"), ("ON", "on"), ("BOTH", "both")):
        variants[name] = {}
        for rule in RULES:
            d_hat = est[rule][key] if key != "both" else est_both[rule]
            fl_v = fl[rule][key] if key != "both" else floor_both[rule]
            vs = ft.violation_summary(d_hat, boot[rule][key], fl_v)
            if key == "both":
                cut_v = w if rule == "noise" else max(tau["c5"].values())
                extra = {"cutoff_off": tau[rule]["off"] if rule == "c5" else w, "cutoff_on": tau[rule]["on"] if rule == "c5" else w}
                he = any(hit[rule].values()) if rule == "c5" else bool(w == L - 1 or any(hit["noise"].values()))
                cl = 0.5 * (corr["off"] + corr["on"])
            else:
                cut_v, extra, he, cl = tau[rule][key], {}, hit[rule][key], corr[key]
            variants[name][rule] = {"delta2": vs["delta2"], "ci95_reverse": vs["ci95_reverse"], "ci95_percentile": vs["ci95_percentile"],
                                    "noise_floor": fl_v, "excess": vs["excess"], "excludes_floor": vs["excludes_floor"],
                                    "cutoff": int(cut_v), "hit_end": he, **extra, **resp, "z0_snr": z0_snr[key], "noise_corr_length": cl}
    curves = {}
    for k in dirs:
        sigma, zm, zs = ft.noise_level(free[k]["z3"])
        curves[k.upper()] = {"z_mean": free[k]["z_mean"], "c_zz": free[k]["c_zz"], "z_smooth": zs, "sigma_noise": sigma,
                             "noise_cutoff": cut[k], "noise_hit_end": bool(nc[k][1]), "c5_cutoff": tau["c5"][k]}
    return {"variants": variants, "curves": curves, "excluded_fraction": {k: free[k]["excluded_fraction"] for k in dirs}}


def print_table(res):
    print(f"{'variant':<6} {'rule':<6} {'delta2':>8} {'ci95 reverse':>18} {'floor':>8} {'excess':>8} {'viol':>5} {'cut':>5} {'end':>4} "
          f"{'z0 snr':>7} {'corrL':>6}")
    for name in VARIANTS:
        for rule in RULES:
            b = res["variants"][name][rule]
            print(f"{name:<6} {rule:<6} {fmt(b['delta2'])} {fmt_ci(b['ci95_reverse']):>18} {fmt(b['noise_floor'])} {fmt(b['excess'])} "
                  f"{'yes' if b['excludes_floor'] else 'no':>5} {b['cutoff']:>5} {'y' if b['hit_end'] else 'n':>4} {fmt(b['z0_snr'], 1):>7} "
                  f"{fmt(b['noise_corr_length'], 1):>6}")


def plot(res, sch, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    lag = np.arange(sch.n_lags)
    for ax, name in zip(axes, ("OFF", "ON")):
        c = res["curves"][name]
        ax.axhspan(-c["sigma_noise"], c["sigma_noise"], color="0.85", label=r"$\pm\sigma_{noise}$ (median SE)")
        ax.plot(lag, c["z_mean"], lw=1.2, label=r"$\langle z(\tau)\rangle$")
        ax.plot(lag, c["c_zz"], lw=1.2, label=r"$C_{zz}(\tau)$")
        ax.axvline(c["noise_cutoff"], color="k", ls="--", lw=1, label=f"noise cutoff {c['noise_cutoff']}" + (" (end)" if c["noise_hit_end"] else ""))
        ax.axvline(c["c5_cutoff"], color="tab:red", ls=":", lw=1, label=f"5 % cutoff {c['c5_cutoff']}")
        ax.set_xlabel(f"lag (units of {sch.record_every} steps)"); ax.set_title(f"{name}: " + ("switch-off (stage III)" if name == "OFF" else "switch-on (stage I, roles swapped)"))
        ax.legend(fontsize=7)
    axes[0].set_ylabel("z")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main(argv=None):
    """Run the analysis; RuntimeWarnings from degenerate data (all-NaN slices, empty means, 0/0) are silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _main(argv)


def _main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("N", type=int); p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=200); p.add_argument("--n-lags", type=int, default=1400)
    p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args(argv)
    runs, meta, files = load_runs(a.out, a.N)
    try:
        sch = fp.Schedule(V0=meta["V0"], eps=meta["eps"], T_relax=meta["T_relax"], T_samp=meta["T_samp"],
                          n_cycles=meta["n_cycles"], record_every=meta["record_every"], n_lags=a.n_lags)
    except ValueError as err:
        r = meta["record_every"]
        sys.exit(f"--n-lags {a.n_lags} is not allowed for these runs ({err}); the allowed range is 2 to "
                 f"{min(meta['T_samp'] // r, meta['T_relax'] // r)}")
    if sch.n_cycles < 2 or len(runs) * (sch.n_cycles - 1) < 4:
        sys.exit(f"need at least 4 cycles after dropping cycle 0 of every run (have {len(runs) * (sch.n_cycles - 1)})")
    x2, x3, x4, xI = stage_arrays(runs, sch)
    M = x2.shape[0]
    print(f"N={a.N}: {len(runs)} runs, M = {M} cycles (cycle 0 of every run excluded), {sch.n_lags} lags (lag step {sch.record_every}), "
          f"V0 = {sch.V0} V, eps = {sch.eps:.4f} V")
    res = analyze_geff(x2, x3, x4, xI, a.n_boot, seed=a.seed)
    print_table(res)
    out = {"N": a.N, "schedule": {**{k: meta[k] for k in ("V0", "eps", "T_relax", "T_samp", "n_cycles", "record_every")}, "n_lags": sch.n_lags},
           "n_runs": len(runs), "n_cycles_used": M, "cycle0_excluded": True, "n_boot": a.n_boot, "n_floor_splits": N_FLOOR_SPLITS,
           "seed": a.seed, "note": NOTE, **res}
    with open(os.path.join(a.out, f"fdtest_geff_N{a.N}.json"), "w") as fh:
        json.dump(sanitize(out), fh, indent=1, allow_nan=False)
    plot(res, sch, os.path.join(a.out, f"fdtest_geff_switch_N{a.N}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
