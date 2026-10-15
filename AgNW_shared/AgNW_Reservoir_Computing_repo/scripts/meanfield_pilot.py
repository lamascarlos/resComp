#!/usr/bin/env python3
"""Quasi-static staircase of the MEAN-FIELD network (NEXT_STEPS N1), the counterpart of the stochastic staircase in
scripts/pilot.py: V = 1, 2, 3, 4, 5, 6, 8, 10 V up and back down to 1 V, 3000 steps per level, network seed 1.
Records at the end of each level the conductance I/V, the mean ON probability, the fraction of junctions with
P > 0.5 and the largest junction voltage drop. These are the numbers behind the "mean-field onset at ~3-4 V"
finding (docs/NEXT_STEPS.md, N1 notes).

    python scripts/meanfield_pilot.py   ->  results/meanfield_pilot.npz  (~3 min)
"""
import os
import sys

# one BLAS thread per process: the scripts parallelize over runs with processes
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_sim.dualnode import DualNodeCircuit  # noqa: E402

from agnw_rc import config as C  # noqa: E402
from agnw_rc.meanfield import MeanFieldSolver, meanfield_step  # noqa: E402

LEVELS = (1, 2, 3, 4, 5, 6, 8, 10, 6, 4, 2, 1)
PER = 3000


def main():
    params, wd = C.network_params()
    net = C.make_network(1)
    circuit = DualNodeCircuit(net, wd)
    solver = MeanFieldSolver(circuit)
    P = np.zeros(len(net.junctions))
    rows = []
    for V in LEVELS:
        for _ in range(PER):
            G = params.G_off + P * (params.G_on - params.G_off)
            Vn = solver.solve(G, V)
            dV = circuit.junction_voltage_drops(Vn)
            P = meanfield_step(P, dV, params)
        rows.append((V, circuit.total_current(Vn) / V, P.mean(), np.mean(P > 0.5), np.abs(dV).max()))
        print("V={:5.1f}  G={:.4f} mS  mean P={:.4f}  frac(P>0.5)={:.4f}  max|dV|={:.3f} V".format(
            rows[-1][0], rows[-1][1] * 1e3, *rows[-1][2:]), flush=True)
    r = np.array(rows)
    out = ROOT / "results" / "meanfield_pilot.npz"
    np.savez_compressed(out, v=r[:, 0], g=r[:, 1], mean_p=r[:, 2], frac_on=r[:, 3], max_dv=r[:, 4], per=PER,
                        framework_commit=C.framework_commit())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
