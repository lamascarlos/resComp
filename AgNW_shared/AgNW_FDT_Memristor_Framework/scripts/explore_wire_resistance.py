"""Sensitivity of the I-V response to intra-wire resistance (Lamas 2026
Sec. 2.2) using the packaged dual-node model (agnw_sim.dualnode).

Historical use: showed that a wire diameter of ~90-100 nm brings the peak
currents of our replication within ~8% of the paper's Fig. 4 (ideal wires give
~1.3x too much current), while leaving the onset voltage unchanged.

Usage: python -m scripts.explore_wire_resistance DIAMETER_NM [N] [seed] [quarter|full]
Runs the first quarter cycle (0 -> 3.6 V) by default, or the full 50000-step
cycle with 'full'; RESET off and V_th = 0.1 V. Standalone driver (no tests)."""
import os
import sys
import numpy as np

from agnw_sim.dynamics import SimParams
from agnw_sim.simulate import run_simulation

DT = 1e-3
STEPS_PER_CYCLE = 50000
V_AMP = 3.6


def main(d_wire_nm, n_wires=2000, seed=1, cycle="quarter",
         out_dir="validation_output/wire_resistance"):
    os.makedirs(out_dir, exist_ok=True)
    params = SimParams.from_step_probs(0.4, 0.0, dt=DT, alpha=1.0, V_th=0.1)
    n_steps = STEPS_PER_CYCLE if cycle == "full" else STEPS_PER_CYCLE // 4 + 1
    f = 1.0 / (STEPS_PER_CYCLE * DT)
    result = run_simulation(
        n_wires=n_wires, area=1e-6, length=70e-6,
        waveform_fn=lambda t: V_AMP * np.sin(2 * np.pi * f * t),
        n_steps=n_steps, dt=DT, params=params, rng=np.random.default_rng(seed),
        wire_diameter=d_wire_nm * 1e-9,
    )
    name = f"d{int(d_wire_nm)}nm_N{n_wires}" + ("_full" if cycle == "full" else "")
    np.savez_compressed(os.path.join(out_dir, name + ".npz"),
                        V_in=result.V_in, I_total=result.I_total)
    print("done", name)


if __name__ == "__main__":
    main(float(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 2000,
         int(sys.argv[3]) if len(sys.argv) > 3 else 1,
         sys.argv[4] if len(sys.argv) > 4 else "quarter")
