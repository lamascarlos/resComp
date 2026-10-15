#!/usr/bin/env python3
"""N10-RN: does the N10-E reservoir survive conductance fluctuations of the size MEASURED on a real AgNW network?

No new simulations. The noise-free N10-E runs (4x4 pads, random weights, 16 electrode currents, tau_s 0.3 s, spread 1,
100 ms holds, 3 networks x 2 replicates + sine/square, both windows) are re-scored after multiplying every recorded
electrode current by (1 + z_e(t)), with z the network-level noise measured by Milano et al. 2025 at 3.6 V
(scripts/n10c_data.py -> results/n10c_data.json):
  "ou":    their model, dz = -theta z dt + s dW + Gamma dq: Ornstein-Uhlenbeck with theta(3.6 V) = 0.116 /s, Gaussian
           increments of sd 4.8e-4 per 0.6 s, Poisson jumps at 0.080 /s, sign +/- with equal odds, |Gamma| power law
           p ~ |Gamma|^-alpha above the detection threshold 1.2e-3 (alpha 2.98), capped at the largest measured jump (2 %);
           jumps therefore persist ~ 1/theta = 9 s;
  "white": the Gaussian part as independent measurement noise per reading (sd 4.8e-4 / sqrt 2), jumps as above.
  "indep": an independent z per electrode; "common": one z shared by all electrodes (a common gain fluctuation, as
           a single network-level process would give).
Amplitudes x1 (measured), x3, x10, x30. Each replicate gets its own noise, so G0 (consistency) is tested too.
Caveat: the noise acts only on the readout, not on the junction states (that is N10-R / N10-K).

    python scripts/n10e_readout_noise.py [--procs N]   -> results/n10e_readout_noise.json, figures/fig23_n10e_readout_noise.png
"""
import argparse
import dataclasses
import json
import os
import sys
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agnw_rc import config as C  # noqa: E402
from agnw_rc import criteria as K  # noqa: E402
from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.drive import load_raw, raw_path  # noqa: E402
import n10r_noise as R  # noqa: E402

OUT = ROOT / "results" / "n10e_readout_noise.json"
FIG = ROOT / "figures" / "fig23_n10e_readout_noise.png"
OBS = "electrodes"
AMPS = (1, 3, 10, 30)
KINDS = (("ou", "indep"), ("ou", "common"), ("white", "indep"))


def measured():
    d = json.loads((ROOT / "results" / "n10c_data.json").read_text())
    s = d["stationary_3p6V"]
    return {"theta": d["steady_state"]["theta_3p6V_per_s"], "dt": s["sample_dt_s"],
            "sd_inc": s["gauss_sd_rel_per_sample"], "rate": s["jump_rate_per_s_fig3g"],
            "jmin": s["threshold_rel_per_sample"], "jmax": s["jump_rel_max"], "alpha": s["powerlaw_alpha"]}


def noise(t, n_ch, m, amp, kind, rng):
    """z (len(t), n_ch) at the (increasing) times t (s)."""
    T = len(t)
    jumps = np.zeros((T, n_ch))
    for c in range(n_ch):
        n_j = rng.poisson(m["rate"] * (t[-1] - t[0]))
        if n_j:
            at = np.searchsorted(t, rng.uniform(t[0], t[-1], n_j))
            size = np.minimum(m["jmin"] * rng.random(n_j) ** (-1.0 / (m["alpha"] - 1.0)), m["jmax"])
            np.add.at(jumps[:, c], np.minimum(at, T - 1), size * rng.choice((-1.0, 1.0), n_j))
    s = m["sd_inc"] / np.sqrt(m["dt"])                       # per sqrt(s)
    z = np.zeros((T, n_ch))
    if kind == "ou":
        dts = np.diff(t, prepend=t[0])
        for k in range(1, T):
            a = np.exp(-m["theta"] * dts[k])
            sd = s * np.sqrt((1 - a ** 2) / (2 * m["theta"]))
            z[k] = a * z[k - 1] + sd * rng.standard_normal(n_ch) + jumps[k]
    else:                                                    # white reading noise + decaying jumps
        dts = np.diff(t, prepend=t[0])
        jz = np.zeros(n_ch)
        for k in range(T):
            jz = jz * np.exp(-m["theta"] * dts[k]) + jumps[k]
            z[k] = jz
        z += m["sd_inc"] / np.sqrt(2) * rng.standard_normal((T, n_ch))
    return amp * z


def noisy_raw(spec, amp, kind, share):
    raw = dict(load_raw(spec, R.RAW))
    m = measured()
    n_e = int(raw["n_electrodes"])
    cur = raw["current"].reshape(len(raw["u"]), -1, n_e)
    offs = np.asarray(raw["virtual_offsets"], dtype=float) * C.DT
    t = (np.arange(cur.shape[0])[:, None] * spec.hold * C.DT + offs[None, :]).ravel()
    rng = np.random.default_rng(zlib.crc32(f"{spec.name}|{amp}|{kind}|{share}".encode()))
    z = noise(t, 1 if share == "common" else n_e, m, amp, kind, rng)
    z = np.broadcast_to(z, (len(t), n_e)).reshape(cur.shape)
    raw["current"] = (cur * (1.0 + z)).reshape(len(raw["u"]), -1)
    return raw


def _score(job):
    spec, amp, kind, share = job
    raw = noisy_raw(spec, amp, kind, share) if amp else load_raw(spec, R.RAW)
    if spec.stream == "sinesquare":
        return job, E.evaluate_sinesquare_run(raw), None
    return job, E.evaluate_iid_run(raw, E.noise_floor(raw["u"])), E.observables(raw)[OBS].shape[1]


def _cons(job):
    (a, b), amp, kind, share = job
    ra = noisy_raw(a, amp, kind, share) if amp else load_raw(a, R.RAW)
    rb = noisy_raw(b, amp, kind, share) if amp else load_raw(b, R.RAW)
    return job, E.evaluate_consistency(ra, rb)


def main(procs):
    base = [dataclasses.replace(s, rupture=0.0) for s in R.specs(("rupture_low",)) if s.rupture == 0.01]
    base = [s for s in base if raw_path(s, R.RAW).exists()]
    conds = [(0, "none", "none")] + [(a, k, sh) for k, sh in KINDS for a in AMPS]
    jobs = [(s, a, k, sh) for s in base for a, k, sh in conds]
    pairs = [((s, dataclasses.replace(s, replicate=1)), a, k, sh) for s in base if s.stream == "iid" and s.replicate == 0
             for a, k, sh in conds]
    with ProcessPoolExecutor(procs) as pool:
        scored = list(pool.map(_score, jobs, chunksize=4))
        cs = list(pool.map(_cons, pairs, chunksize=4))
    d = json.loads((ROOT / "results" / "metrics.json").read_text())
    floor = max(d["noise_floor_short"][f"{n}_{C.E4_N_SAMPLES}"] for n in C.E1_SEEDS)
    e16 = json.loads((ROOT / "results" / "esn16_baseline.json").read_text())["mean"]
    bl = [d["baselines_short"][f"{n}_{C.E4_N_SAMPLES}"]["iid"] for n in C.E1_SEEDS]
    esn = {16: e16["esn_16"]["MC"] + e16["esn_16"]["IPC2"], 32: e16["esn_32"]["MC"] + e16["esn_32"]["IPC2"],
           200: float(np.mean([b["esn_200"]["MC"] + b["esn_200"]["IPC2"] for b in bl]))}
    rows = []
    for a, k, sh in conds:
        runs = [dict(stream=s.stream, hold=s.hold, v_center=s.v_center, network_seed=s.network_seed,
                     replicate=s.replicate, scores=sc) for (s, a2, k2, sh2), sc, _ in scored if (a2, k2, sh2) == (a, k, sh)]
        cons = [dict(hold=p[0].hold, v_center=p[0].v_center, consistency=c) for (p, a2, k2, sh2), c in cs
                if (a2, k2, sh2) == (a, k, sh)]
        nf = next(n for (_, a2, k2, sh2), _, n in scored if (a2, k2, sh2) == (a, k, sh) and n)
        strict = K.evaluate(runs, cons, OBS, nf, esn, floor)
        loose = K.evaluate(runs, cons, OBS, 16, esn, floor)
        rows.append({"amp": a, "kind": k, "share": sh, **strict, "G2_loose": loose["G2"]})
        print(f"{k:5s} {sh:6s} x{a:<3} MC {strict['MC']['mean']:.2f}±{strict['MC']['sd']:.2f} IPC2 "
              f"{strict['IPC2']['mean']:.2f}±{strict['IPC2']['sd']:.2f} NARMA {strict['narma10_nmse']['mean']:.3f} "
              f"acc {strict['accuracy']['mean'] if strict['accuracy'] else float('nan'):.2f} cons "
              f"{strict['consistency_inc']['mean'] if strict['consistency_inc'] else float('nan'):.2f} "
              f"-> G0 {strict['G0']} G1 {strict['G1']} G2 loose {loose['G2']}", flush=True)
    OUT.write_text(json.dumps({"measured_noise": measured(), "esn_total": esn, "noise_floor": floor,
                               "base_runs": [s.name for s in base], "rows": rows}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = rows[0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, key, title, thr in zip(axes, ("MC", "IPC2", "narma10_nmse", "consistency_inc"),
                                   ("memory capacity", "degree-2 capacity", "NARMA-10 NMSE", "increment consistency"),
                                   (1, 1, 0.7, 0.8)):
        for k, sh in KINDS:
            rs = [r for r in rows if (r["kind"], r["share"]) == (k, sh)]
            ax.errorbar([r["amp"] for r in rs], [r[key]["mean"] for r in rs], [r[key]["sd"] for r in rs],
                        marker="o", capsize=3, label=f"{k}, {sh}")
        ax.axhline(ref[key]["mean"], color="0.5", lw=1, label="no noise (N10-E)")
        ax.axhline(thr, color="r", ls="--", lw=0.8)
        ax.set(xscale="log", xlabel="noise amplitude (x measured)", title=title)
    axes[0].legend(fontsize=8)
    fig.suptitle("N10-RN: measured network-level fluctuations (Milano 2025) added to the 16 electrode currents of N10-E")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    main(ap.parse_args().procs)
