#!/usr/bin/env python3
"""N10-PW: operating-point search for pads, two higher input windows on the N10-LS networks (decided by Juan 2026-10-14).

The N10-LS protocol offers two windows (1.05 and 1.25 x onset, width 0.45 x onset) and validation picks one. The N10-SB
window check (1 mm, networks 1-3, results/n10sb_win15.json) found pads better at 1.5 x onset, with ruptures validation
picking the highest window offered. N10-PW adds 1.5 and 1.75 x onset (same width) at every N10-LS size (1 / 2 / 2.83 /
4 mm), networks 1-10, ruptures 0 and 1e-2 /s per junction, and re-runs the N10-LS analysis with the operating point
chosen on validation data among the four windows (the same rule as before, on a wider grid). The two protocol windows
are the existing N10-LS runs; the 1 mm runs at 0.54 V, networks 1-3, are the N10-SB window-check runs (reused).
Split bands (b) got 1.5 x onset in the same check and lost there; any later layout gets the same grid.

    python scripts/n10pw_windows.py run [--procs N] [--only SIZE:RATE,...]   -> results/raw/*.npz
    python scripts/n10pw_windows.py report                                   -> results/n10pw_windows.json
"""
import argparse
import dataclasses
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import n10l_seeds as S  # noqa: E402
from agnw_rc.drive import raw_path  # noqa: E402

RAW = S.RAW
OUT = ROOT / "results" / "n10pw_windows.json"
FACTORS = (1.5, 1.75)                    # x onset, on top of the protocol 1.05 / 1.25
ONSET_1MM = 0.36                         # the N10-E pads onset behind C.N10M_WINDOWS (0.38 / 0.45) and the 0.54 V check
SIDES = (4.0, 2.83, 2.0, 1.0)            # queue order: longest runs first
RATES = (1e-2, 0.0)
SEEDS = tuple(range(1, 11))


def onset(side):
    if side == 1.0:
        return ONSET_1MM
    return json.loads((ROOT / "results" / "n10l_sweep.json").read_text())[f"{side:g}"]["onset_V"]


def new_specs(side, rate, seeds=SEEDS):
    """The N10-LS runs of (side, rate) moved to the new windows (width, model, layout unchanged)."""
    base = S.specs_for(side, rate, seeds)
    ref = base[0].v_center                                  # lower protocol window: one copy of every run
    return [dataclasses.replace(s, v_center=round(f * onset(side), 2)) for f in FACTORS for s in base if s.v_center == ref]


def run(procs, only=""):
    keys = [(s, r) for s in SIDES for r in RATES]
    if only:
        keep = [tuple(float(x) for x in k.split(":")) for k in only.split(",")]
        keys = [k for k in keys if k in keep]
    todo = [s for side, r in keys for s in new_specs(side, r)
            if not raw_path(s, RAW).exists() and not raw_path(s, RAW).with_suffix(".running").exists()]
    print(f"{len(todo)} runs to do", flush=True)
    with ProcessPoolExecutor(procs) as pool:
        for i, f in enumerate(as_completed([pool.submit(S.run_one, s) for s in todo]), 1):
            print(f"[{i}/{len(todo)}] {f.result()}", flush=True)


def report(procs=8):
    """Per size and rate: the N10-LS analysis with 2 windows (as published) and with 4 windows (operating point chosen on
    validation data among them). Groups are keyed with sigma = -2 / -4 to tell the two choices apart."""
    groups = {}
    for side in sorted(SIDES):
        for r in RATES:
            old = [s for s in S.specs_for(side, r, SEEDS) if raw_path(s, RAW).exists()]
            new = [s for s in new_specs(side, r) if raw_path(s, RAW).exists()]
            groups[(side, r, -2.0)] = old
            groups[(side, r, -4.0)] = old + new
    out = S.analyse({k: v for k, v in groups.items() if v}, SEEDS, procs)
    for g in out["groups"]:
        g["windows"] = int(-g.pop("sigma"))
        g["onset_V"] = onset(g["side_mm"])
    out["factors_new"] = list(FACTORS)
    OUT.write_text(json.dumps(out, indent=1))
    for g in out["groups"]:
        p = g["pooled"]
        print(f"{g['side_mm']:g} mm r {g['rupture']:g} {g['windows']} windows: op {p['operating_point']['v_center']:.2f} V "
              f"({p['operating_point']['v_center'] / g['onset_V']:.2f} x onset), MC {p['MC']['mean']:.2f} IPC2 "
              f"{p['IPC2']['mean']:.2f} NARMA {p['narma10_nmse']['mean']:.3f}, pooled G1 {p['G1']}, "
              f"{g['networks_pass_G1']}/{g['n_networks']} networks" + (f", rho {g['rho_median']:.2f}" if "rho_median" in g else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "report", "count"])
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.stage == "count":
        for side in SIDES:
            for r in RATES:
                sp = new_specs(side, r)
                print(f"{side:g} mm r {r:g}: windows {sorted({s.v_center for s in sp})} width {sp[0].v_width}, "
                      f"{len(sp)} runs, {sum(raw_path(s, RAW).exists() for s in sp)} exist")
    else:
        {"run": lambda: run(a.procs, a.only), "report": lambda: report(min(a.procs, 8))}[a.stage]()
