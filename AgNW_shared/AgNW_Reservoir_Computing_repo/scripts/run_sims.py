#!/usr/bin/env python3
"""Run the reservoir simulations of an experiment in parallel; existing results are skipped (resumable).

    python scripts/run_sims.py e1 [--procs 12]
    python scripts/run_sims.py e2 --hold 100 --v-center 1.75
    python scripts/run_sims.py e3 --v-center 1.75
    python scripts/run_sims.py m1                  # mean-field ablation (docs/NEXT_STEPS.md N1)
    python scripts/run_sims.py n2                  # heterogeneous forgetting, mean-field (N2)
    python scripts/run_sims.py n7a                 # 8 noise replicates of one E1 point (N7a)
    python scripts/run_sims.py n7b                 # stochastic model at the mean-field windows (N7b)
    python scripts/run_sims.py n5a                 # multi-electrode, random input weights (N5a)
    python scripts/run_sims.py n4_1                # calibrated continuous-filament model, E1 protocol (N4-1)
    python scripts/run_sims.py e4 [--model stochastic|filament]   # time-scale-matched inputs (E4, review plan)
    python scripts/run_sims.py n10_1               # N10 volatile junction, E4 protocol (docs/N10_DESIGN.md)
"""
import argparse
import os
import sys

# one BLAS thread per process: the scripts parallelize over runs with processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C                      # noqa: E402
from agnw_rc.drive import raw_path, save_raw, simulate  # noqa: E402
from agnw_rc.meanfield import simulate_meanfield  # noqa: E402
from agnw_rc.multidrive import simulate_multi  # noqa: E402

RAW = ROOT / "results" / "raw"


def run_one(spec):
    if raw_path(spec, RAW).exists():                  # done meanwhile (e.g. by another machine, results copied in)
        return spec.name, 0.0
    if spec.layout:
        raw = simulate_multi(spec)
    elif spec.model == "meanfield":
        raw = simulate_meanfield(spec)
    else:
        raw = simulate(spec)
    save_raw(spec, raw, RAW)
    return spec.name, raw["seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["e1", "e2", "e3", "m1", "n2", "n7a", "n7b", "n5a", "n4_1", "e4", "n10_1", "n10m"])
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--hold", type=int)
    ap.add_argument("--v-center", type=float)
    ap.add_argument("--model", choices=["stochastic", "filament"], help="e4: one model only")
    ap.add_argument("--holds", type=int, nargs="+", help="only runs with these hold times (split work across machines)")
    ap.add_argument("--reverse", action="store_true", help="work from the end of the queue (second machine)")
    ap.add_argument("--blocks", nargs="+", help="n10m: blocks to run (tau1 tau3 spread delays noise)")
    ap.add_argument("--limit", type=int, help="run only the first N runs of the queue (fill idle cores)")
    a = ap.parse_args()
    if a.experiment == "e1":
        specs = C.e1_specs()
    elif a.experiment == "m1":
        specs = C.m1_specs()
    elif a.experiment == "n2":
        specs = C.n2_specs()
    elif a.experiment == "n7a":
        specs = C.n7a_specs()
    elif a.experiment == "n7b":
        specs = C.n7b_specs()
    elif a.experiment == "n5a":
        specs = C.n5a_specs()
    elif a.experiment == "n4_1":
        specs = C.n4_1_specs()
    elif a.experiment == "n10m":
        specs = C.n10m_specs(tuple(a.blocks) if a.blocks else tuple(C.N10M_BLOCKS))
    elif a.experiment == "n10_1":
        specs = C.n10_1_specs()
    elif a.experiment == "e4":
        specs = C.e4_specs((a.model,) if a.model else ("stochastic", "filament"))
    elif a.experiment == "e2":
        if a.hold is None or a.v_center is None:
            ap.error("e2 needs --hold and --v-center")
        specs = C.e2_specs(a.hold, a.v_center)
    else:
        if a.v_center is None:
            ap.error("e3 needs --v-center")
        specs = C.e3_specs(a.v_center)
    if a.holds:
        specs = [s for s in specs if s.hold in a.holds]
    todo = [s for s in specs if not raw_path(s, RAW).exists()]
    # longest first, so the pool is not left waiting on one slow run at the end
    todo.sort(key=lambda s: -s.hold * s.length / s.stride)
    if a.reverse:
        todo.reverse()
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.experiment}: {len(specs)} runs, {len(todo)} to do, {a.procs} processes", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.procs) as pool:
        futures = [pool.submit(run_one, s) for s in todo]
        for i, f in enumerate(as_completed(futures), 1):
            name, sec = f.result()
            print(f"[{i}/{len(todo)}] {name} {sec:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
