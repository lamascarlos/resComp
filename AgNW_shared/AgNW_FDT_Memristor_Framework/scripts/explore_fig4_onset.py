"""Sensitivity of the activation ("onset") voltage of the ascending I-V
branch to SET parameters, to diagnose the mismatch with Lamas 2026 Fig. 4
(paper: onset ~1.8 V at N=2000; our Table-1 replication: ~0.35 V).

Usage: python -m scripts.explore_fig4_onset NAME [N] [seed]
Runs the first quarter cycle (0 -> 3.6 V, 12500 steps of a 50000-step sine),
RESET off, and saves validation_output/onset_sweep/NAME_N{N}.npz.
Standalone exploration driver (no unit tests)."""
import os
import sys
import numpy as np

from agnw_sim.simulate import run_simulation
from agnw_sim.dynamics import SimParams

DT = 1e-3
STEPS_PER_CYCLE = 50000
V_AMP = 3.6

# name -> (p0_step [= dt*P0], alpha, V_th)
CONFIGS = {
    "base":        (0.4,    1.0,  0.01),
    "Vth0.05":     (0.4,    1.0,  0.05),
    "Vth0.1":      (0.4,    1.0,  0.1),
    "Vth0.2":      (0.4,    1.0,  0.2),
    "Vth0.5":      (0.4,    1.0,  0.5),
    "alpha0.1":    (0.4,    0.1,  0.01),
    "alpha10":     (0.4,    10.0, 0.01),
    "p0_0.04":     (0.04,   1.0,  0.01),
    "p0_0.004":    (0.004,  1.0,  0.01),
    "p0_0.0004":   (0.0004, 1.0,  0.01),
}


def main(name, n_wires=2000, seed=1, out_dir="validation_output/onset_sweep"):
    os.makedirs(out_dir, exist_ok=True)
    p0, alpha, vth = CONFIGS[name]
    params = SimParams.from_step_probs(p0, 0.0, dt=DT, alpha=alpha, V_th=vth)
    f = 1.0 / (STEPS_PER_CYCLE * DT)
    r = run_simulation(
        n_wires=n_wires, area=1e-6, length=70e-6,
        waveform_fn=lambda t: V_AMP * np.sin(2 * np.pi * f * t),
        n_steps=STEPS_PER_CYCLE // 4 + 1, dt=DT, params=params,
        rng=np.random.default_rng(seed), n_sampled_junctions=200,
    )
    np.savez_compressed(os.path.join(out_dir, f"{name}_N{n_wires}.npz"),
                        V_in=r.V_in, I_total=r.I_total,
                        on_frac=(r.G_sample >= params.G_on).mean(axis=1))
    print("done", name)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000,
         int(sys.argv[3]) if len(sys.argv) > 3 else 1)
