"""Direct memory diagnostic on the g_eff(t) series (Markov test 4): does the future of g_eff depend on its past beyond g(t)?

  python -m scripts.markov_geff_test --dir validation_output/fdtest_eps0 [--N 2000] [--out DIR] [--lags 1,5,20,50] [--k 3]
                                     [--n-boot 200] [--n-surr 100] [--n-max 50000] [--control-trace PATH] [--skip-controls] [--seed 12345]

Reads the `I_total` arrays of the run files of one schedule (the null control: no perturbation), g_eff = I_total / V_in at full step
resolution, cycle 0 of every run dropped (all-OFF start). For every lag h it reports two statistics (agnw_sim.markov_tests):
  gain   cross-validated gain of adding g(t-jh), g(t-jh)^2 (j = 1..k) to a degree-7 polynomial predictor of g(t+h) from g(t); folds are runs;
         detected when the 95 % bootstrap interval excludes 0 AND the gain exceeds MIN_GAIN (1e-3, a practical-significance floor)
  cmi    conditional mutual information I(g(t+h); g(t-h) | g(t)) (kNN) against 100 Markov surrogates; detected above their 97.5th percentile
Controls run through the same code: an isolated exponential junction and a nonlinear Markov AR(1) (must show at most one detection over
all lags and statistics), and, if --control-trace is given, the trace sampled from the exact tiny circuit of scripts.tiny_circuit_exact (must be
detected when its exact delta2 exceeds 0.01). If a required control fails the verdict on the real series is "no verdict".
Writes markov_geff_test_N{N}.json (strict JSON) and .png to --out (default: --dir). Exit status 0, or 1 with a message for invalid input."""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agnw_sim import markov_tests as mt
from agnw_sim.dynamics import SimParams, update_dynamics
from scripts.fdtest_analyze import load_runs, sanitize

MIN_GAIN = 1e-3
PRESET = SimParams.from_step_probs(0.4, 0.01, dt=1e-3, alpha=1.0, V_th=0.1, I_unit=1e-3)
JUNCTION_V, JUNCTION_I = 0.30, 2.5e-3       # isolated junction: per-step SET 0.0725, RESET 0.0625 (the values of tests/test_benchmarks.py)


def geff_series(runs, meta):
    """g_eff = I_total / V_in of every run (V_in rebuilt from the schedule), the first cycle dropped."""
    T_cycle = 2 * (meta["T_relax"] + meta["T_samp"])
    n_steps = meta["n_cycles"] * T_cycle
    steps = np.arange(n_steps)
    V_in = np.where(steps % T_cycle < meta["T_relax"] + meta["T_samp"], meta["V0"] + meta["eps"], meta["V0"])
    out = []
    for z in runs:
        if len(z["I_total"]) != n_steps:
            raise ValueError(f"I_total has {len(z['I_total'])} samples, the schedule says {n_steps}")
        out.append((z["I_total"] / V_in)[T_cycle:])
    return out


def standardize(runs):
    """Runs scaled by the pooled mean and standard deviation."""
    flat = np.concatenate(runs)
    m, s = flat.mean(), flat.std()
    return [(np.asarray(r, float) - m) / s for r in runs]


def analyze(runs, lags, rng, k=3, degree=7, thin=5, n_boot=200, n_surr=100, n_max=50_000, cmi_k=5, stride_factor=3):
    """Both statistics for every lag on a list of series. Returns dict(corr_time, stride, lags=[per-lag dict], n_detections)."""
    runs = standardize(runs)
    tau = mt.correlation_time(runs, max_lag=min(2000, min(len(r) for r in runs) // 4))
    stride = max(int(stride_factor * tau), 1)
    rows = []
    for h in lags:
        gain = mt.regression_memory_gain(runs, h, k=k, degree=degree, thin=thin, n_boot=n_boot, rng=rng)
        x, y, z = mt.triples(runs, h, stride, n_max, rng)
        cmi = mt.cmi_test(x, y, z, rng, k=cmi_k, n_surr=n_surr)
        rows.append({"h": int(h), "gain": gain, "gain_detected": bool(gain["ci95"][0] > 0 and gain["gain"] > MIN_GAIN),
                     "cmi": {**cmi, "n_triples": int(len(x))}, "cmi_detected": cmi["detected"]})
    return {"corr_time": int(tau), "stride": int(stride), "lags": rows,
            "n_detections": int(sum(r["gain_detected"] + r["cmi_detected"] for r in rows))}


# ------------------------------------------------------------------------------------------ controls
def isolated_junction_runs(rng, n_runs=8, n_steps=200_000, burn_in=1000):
    """ON/OFF series of independent isolated exponential-lifetime-free junctions (two-state Markov chains, package dynamics)."""
    G = np.full(n_runs, PRESET.G_off)
    V, I = np.full(n_runs, JUNCTION_V), np.full(n_runs, JUNCTION_I)
    on = np.empty((n_steps + burn_in, n_runs))
    for t in range(n_steps + burn_in):
        G = update_dynamics(G, V, I, PRESET, rng)
        on[t] = G > PRESET.G_off
    return [on[burn_in:, i].copy() for i in range(n_runs)]


def nonlinear_markov_runs(rng, n_runs=8, n_steps=200_000):
    """x_{t+1} = tanh(2 x_t) + 0.5 noise: Markov, but E[x_{t+1} | x_t] is nonlinear."""
    out = []
    for _ in range(n_runs):
        e = 0.5 * rng.normal(size=n_steps)
        x = np.zeros(n_steps)
        for t in range(1, n_steps):
            x[t] = math.tanh(2.0 * x[t - 1]) + e[t]
        out.append(x)
    return out


def run_controls(rng, lags, trace_path, size, **kw):
    """{name: dict(analysis, expected, passed, required)}."""
    ctl = {}
    for name, runs in (("isolated_junction", isolated_junction_runs(rng, n_steps=size)),
                       ("nonlinear_markov", nonlinear_markov_runs(rng, n_steps=size))):
        a = analyze(runs, lags, rng, **kw)
        ctl[name] = {"analysis": a, "expected": "no memory (at most 1 detection)", "required": True, "passed": a["n_detections"] <= 1}
    if trace_path:
        tr = np.load(trace_path)
        key, exact = str(tr["control_key"]), float(tr["control_exact_delta2"])
        a = analyze([r.astype(float) for r in tr[key]], lags, rng, **kw)
        required = bool(exact > 0.01)
        ctl["tiny_circuit_trace"] = {"analysis": a, "expected": f"memory (exact delta2 of {key[:-4]} = {exact:.3e})", "exact_delta2": exact,
                                     "key": key, "required": required, "passed": a["n_detections"] >= 1 if required else True}
    return ctl


def verdict(real, controls):
    failed = [k for k, c in controls.items() if c["required"] and not c["passed"]]
    if failed:
        return {"text": f"no verdict: control(s) failed: {', '.join(failed)}", "failed_controls": failed}
    if real["n_detections"]:
        det = [f"h={r['h']}" for r in real["lags"] if r["gain_detected"] or r["cmi_detected"]]
        return {"text": f"memory detected in g_eff(t) at {', '.join(det)}", "failed_controls": []}
    return {"text": "no memory detected in g_eff(t) at any lag (controls passed)", "failed_controls": []}


# ------------------------------------------------------------------------------------------ output
def plot(real, controls, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sets = [("real g_eff", real)] + [(k, c["analysis"]) for k, c in controls.items()]
    for label, a in sets:
        h = [r["h"] for r in a["lags"]]
        g = np.array([r["gain"]["gain"] for r in a["lags"]])
        lo = np.array([r["gain"]["ci95"][0] for r in a["lags"]])
        hi = np.array([r["gain"]["ci95"][1] for r in a["lags"]])
        axes[0].errorbar(h, g, yerr=[g - lo, hi - g], marker="o", capsize=3, label=label)
        axes[1].plot(h, [r["cmi"]["excess"] for r in a["lags"]], marker="o", label=label)
    axes[0].axhline(MIN_GAIN, color="k", ls=":", lw=0.8)
    axes[0].set_xscale("log"); axes[0].set_yscale("symlog", linthresh=1e-4); axes[0].set_xlabel("lag h (steps)"); axes[0].set_ylabel("regression memory gain")
    axes[1].set_xscale("log"); axes[1].set_yscale("symlog", linthresh=1e-3); axes[1].set_xlabel("lag h (steps)")
    axes[1].set_ylabel("CMI excess over the Markov surrogate (nats)")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", required=True); p.add_argument("--N", type=int, default=2000); p.add_argument("--out")
    p.add_argument("--lags", default="1,5,20,50"); p.add_argument("--k", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=200); p.add_argument("--n-surr", type=int, default=100)
    p.add_argument("--n-max", type=int, default=50_000); p.add_argument("--control-trace")
    p.add_argument("--control-size", type=int, default=200_000, help="steps per run of the two synthetic controls")
    p.add_argument("--skip-controls", action="store_true"); p.add_argument("--seed", type=int, default=12345)
    a = p.parse_args(argv)
    lags = [int(v) for v in a.lags.split(",")]
    runs, meta, files = load_runs(a.dir, a.N)
    try:
        series = geff_series(runs, meta)
    except ValueError as err:
        sys.exit(str(err))
    if len(series) < 4:
        sys.exit(f"need at least 4 runs (have {len(series)})")
    if max(lags) > min(len(s) for s in series) // 10:
        sys.exit(f"--lags up to {max(lags)} exceed a tenth of the series length {min(len(s) for s in series)}")
    if a.control_trace and not os.path.exists(a.control_trace):
        sys.exit(f"control trace {a.control_trace} not found (run scripts.tiny_circuit_exact first)")
    rng = np.random.default_rng(a.seed)
    kw = dict(k=a.k, n_boot=a.n_boot, n_surr=a.n_surr, n_max=a.n_max)
    print(f"N={a.N}: {len(series)} runs, {len(series[0])} steps each after dropping cycle 0; lags {lags}")
    real = analyze(series, lags, rng, **kw)
    controls = {} if a.skip_controls else run_controls(rng, lags, a.control_trace, a.control_size, **kw)
    v = verdict(real, controls)
    for r in real["lags"]:
        print(f"  h={r['h']:>3}  gain {r['gain']['gain']:+.5f} [{r['gain']['ci95'][0]:+.5f}, {r['gain']['ci95'][1]:+.5f}]"
              f"  cmi {r['cmi']['cmi']:+.4f} (surrogate q97.5 {r['cmi']['surrogate_q975']:+.4f})"
              f"  {'DETECTED' if r['gain_detected'] or r['cmi_detected'] else 'no memory'}")
    for k, c in controls.items():
        print(f"  control {k}: {'passed' if c['passed'] else 'FAILED'}{'' if c['required'] else ' (informative)'}; {c['analysis']['n_detections']} detections; expected {c['expected']}")
    print("VERDICT:", v["text"])
    out = a.out or a.dir
    os.makedirs(out, exist_ok=True)
    result = {"N": a.N, "n_runs": len(series), "files": [os.path.basename(f) for f in files], "config": {**kw, "lags": lags, "seed": a.seed, "min_gain": MIN_GAIN},
              "real": real, "controls": controls, "verdict": v}
    with open(os.path.join(out, f"markov_geff_test_N{a.N}.json"), "w") as fh:
        json.dump(sanitize(result), fh, indent=1, allow_nan=False)
    plot(real, controls, os.path.join(out, f"markov_geff_test_N{a.N}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
