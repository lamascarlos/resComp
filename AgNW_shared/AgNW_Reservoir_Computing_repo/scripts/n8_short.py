#!/usr/bin/env python3
"""N8 (short version, 2026-09-27): can a single-timescale N4 junction forget fast enough to compute, and what does
that cost in the Lamas Fig. 4 fit? Fig. 4 is a qualitative simulation, not ground truth (memory
fig4-not-ground-truth), so this is a quick map, not a calibration.

Scan: V_set 0.06 V and Lambda_crit 0.01 V s (calibrated values); dissolution time
tau_d = Lambda_max / (b V_reset) in TAU_D (b set accordingly); hold-band ratio r = V_reset / V_set in RATIOS.
Per point:
  * Fig. 4 protocol (N = 2000, network seed 1, 2 ms step) -> metrics and calibration loss vs the paper's N = 2000;
  * reservoir probe: E4-like i.i.d. run (network 1, 400 samples, hold 300 ms, 50 virtual nodes, washout
    max(3 tau_d, 3 s)), windows WINDOWS; scored with the E4 pipeline. Per point the window is chosen on
    validation (val_MC + val_IPC2) and test MC, degree-2 capacity and NARMA-10 of the current readout are reported,
    plus how much the state moves (conductance variation, junctions changing per sample).

    python scripts/n8_short.py run [--procs 12]      # simulations -> results/n8_short/*.json (resumable)
    python scripts/n8_short.py report                # -> results/n8_short.json, figures/fig13_n8_short.png
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.fig4 import LOSS_KEYS, loss, metrics_fn, paper_targets, sine_protocol  # noqa: E402
from agnw_rc.multidrive import simulate_multi  # noqa: E402

OUT_DIR = ROOT / "results" / "n8_short"
OUT = ROOT / "results" / "n8_short.json"
FIG = ROOT / "figures" / "fig13_n8_short.png"
TAU_D = (0.3, 1.0, 3.0, 10.0, 25.0)       # s; 25 s ~ the calibrated point (b = 0.02, r = 0.5)
RATIOS = (0.5, 0.9)
WINDOWS = (1.75, 2.25, 2.75)
HOLD, N_SAMPLES, STRIDE = 300, 400, 2


def b_for(tau_d, r):
    d = C.N4_PARAMS
    return 1.5 * d["lambda_crit"] / (tau_d * r * d["V_set"])


def params(tau_d, r):
    return C.filament_params(b=b_for(tau_d, r), reset_ratio=r, stride=STRIDE)


def key(tau_d, r, what):
    return f"tau{tau_d:g}_r{r:g}_{what}"


def fig4_task(tau_d, r):
    v, i, _ = sine_protocol(params(tau_d, r))
    m = metrics_fn()(v, i * 1e3)
    return {"metrics": {k: float(m[k]) for k in LOSS_KEYS}, "loss": loss(m, paper_targets()[2000])}


def reservoir_task(tau_d, r, vc):
    washout = max(int(round(3 * tau_d / C.DT)), 3000)
    spec = C.RunSpec("iid", HOLD, vc, 1, 0, n_samples=N_SAMPLES, layout="two_bands", encoding="direct",
                     model="filament", stride=STRIDE, washout=-(-washout // HOLD) * HOLD, n_virtual=50)
    raw = simulate_multi(spec, fp=params(tau_d, r))
    floor = E.noise_floor(raw["u"])
    s = E.evaluate_iid_run(raw, floor)["current"]
    lam = raw["junction_lambda"].astype(float)
    n_e = int(raw["n_electrodes"])
    cur = raw["current"].reshape(len(lam), -1, n_e)[:, -1, raw["electrode_roles"] == "in"].sum(axis=1)
    g = cur / raw["v_electrodes"][:, 0]
    keep = ("MC", "IPC2", "narma10_nmse", "lag0", "val_MC", "val_IPC2", "memory_function")
    return {**{k: s[k] for k in keep}, "floor": floor, "G_rel_std": float(g.std() / g.mean()),
            "on_fraction": float(np.mean(np.abs(lam) >= 1.0)),
            "changing_per_sample": float((np.abs(np.diff(lam, axis=0)) > 1e-3).sum(axis=1).mean()),
            "corr_G_u": float(np.corrcoef(g, raw["u"])[0, 1]) if g.std() > 0 else 0.0,
            "seconds": float(raw["seconds"])}


def run_task(args):
    what, tau_d, r, vc = args
    path = OUT_DIR / (key(tau_d, r, what if what == "fig4" else f"v{vc:g}") + ".json")
    if path.exists():
        return path.name
    res = fig4_task(tau_d, r) if what == "fig4" else reservoir_task(tau_d, r, vc)
    res.update(tau_d=tau_d, reset_ratio=r, b=b_for(tau_d, r), v_center=vc)
    path.write_text(json.dumps(res))
    return path.name


def run(procs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [("res", t, r, vc) for t in TAU_D for r in RATIOS for vc in WINDOWS]
    tasks += [("fig4", t, r, None) for t in TAU_D for r in RATIOS]
    tasks.sort(key=lambda a: -a[1])                     # long washouts first
    with ProcessPoolExecutor(procs) as pool:
        futs = [pool.submit(run_task, a) for a in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(tasks)}] {f.result()}", flush=True)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for t in TAU_D:
        for r in RATIOS:
            f4 = json.loads((OUT_DIR / f"{key(t, r, 'fig4')}.json").read_text())
            res = [json.loads((OUT_DIR / f"{key(t, r, f'v{vc:g}')}.json").read_text()) for vc in WINDOWS]
            best = max(res, key=lambda x: x["val_MC"] + x["val_IPC2"])
            rows.append({"tau_d": t, "reset_ratio": r, "b": b_for(t, r), "fig4_loss": f4["loss"],
                         "fig4_metrics": f4["metrics"], "selected_window": best["v_center"],
                         **{k: best[k] for k in ("MC", "IPC2", "narma10_nmse", "lag0", "G_rel_std", "on_fraction",
                                                 "changing_per_sample", "corr_G_u", "floor")},
                         "all_windows": res})
    from agnw_rc import baselines as B
    from agnw_rc.drive import input_stream
    u = input_stream(C.RunSpec("iid", HOLD, 2.25, 1, 0, n_samples=N_SAMPLES))[0]     # same stream as the probes
    floor = E.noise_floor(u)
    base = {name: {k: E.score_iid(F, u, floor)[k] for k in ("MC", "IPC2", "narma10_nmse")}
            for name, F in (("input_only", B.input_only(u)), ("delay_line_32", B.delay_line(u, 32)))}
    print("baselines (400 samples):", base, "floor", round(floor, 3))
    OUT.write_text(json.dumps({"hold_ms": HOLD, "n_samples": N_SAMPLES, "baselines": base, "noise_floor": floor,
                               "rows": rows}, indent=1))
    for x in rows:
        print(f"tau_d {x['tau_d']:5g} s r {x['reset_ratio']:.1f} b {x['b']:.3g}: Fig.4 loss {x['fig4_loss']:.3f} "
              f"(A-/A+ {x['fig4_metrics']['ratio']:.2f}) | {x['selected_window']} V: MC {x['MC']:.2f} "
              f"IPC2 {x['IPC2']:.2f} NARMA {x['narma10_nmse']:.2f} G var {x['G_rel_std']:.3f} "
              f"changing {x['changing_per_sample']:.0f}")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for r, mk in zip(RATIOS, ("o", "s")):
        xs = [x for x in rows if x["reset_ratio"] == r]
        t = [x["tau_d"] for x in xs]
        axes[0].plot(t, [x["fig4_loss"] for x in xs], marker=mk, label=f"r = V_reset/V_set = {r}")
        axes[1].plot(t, [x["MC"] + x["IPC2"] for x in xs], marker=mk, label=f"r = {r}")
        axes[2].scatter([x["fig4_loss"] for x in xs], [x["MC"] + x["IPC2"] for x in xs], marker=mk,
                        label=f"r = {r}")
        for x in xs:
            axes[2].annotate(f"{x['tau_d']:g} s", (x["fig4_loss"], x["MC"] + x["IPC2"]), fontsize=7)
    axes[0].set(xscale="log", yscale="log", xlabel="dissolution time τ_d (s)", ylabel="Fig. 4 loss (N = 2000)",
                title="fit to Lamas Fig. 4 (qualitative target)")
    axes[1].axhline(2, color="r", ls=":", lw=1, label="G1 (MC ≥ 1 and degree-2 ≥ 1)")
    axes[1].set(xscale="log", xlabel="dissolution time τ_d (s)", ylabel="MC + degree-2 capacity",
                title=f"reservoir probe ({HOLD} ms holds, current readout)")
    axes[2].set(xscale="log", xlabel="Fig. 4 loss", ylabel="MC + degree-2 capacity", title="trade-off")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report"])
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    run(a.procs) if a.stage == "run" else report()
