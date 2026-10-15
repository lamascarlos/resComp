"""Fixed-voltage (abrupt turn-on) dynamics over many lifetimes, for the model
variants that reproduce Lamas 2026 Fig. 4 (see scripts.replicate_lamas_fig4).

Usage: python -m scripts.fixed_v_stationarity N VARIANT V_IN [N_STEPS] [seed]
Saves validation_output/fixed_v/N{N}_{VARIANT}_V{V}_seed{seed}.npz with I_total(t),
G_eff(t)=I/V and the sampled-junction ON fraction. Standalone driver (no tests)."""
import os
import sys
import numpy as np

from agnw_sim.simulate import run_simulation
from scripts.replicate_lamas_fig4 import build_params, DT


def main(n_wires, variant, v_in, n_steps=60000, seed=1, out_dir="validation_output/fixed_v"):
    os.makedirs(out_dir, exist_ok=True)
    params, run_kw = build_params(variant)
    r = run_simulation(n_wires=n_wires, area=1e-6, length=70e-6, waveform_fn=lambda t: v_in,
                       n_steps=n_steps, dt=DT, params=params, rng=np.random.default_rng(seed),
                       n_sampled_junctions=200, **run_kw)
    on_frac = (r.G_sample >= params.G_on).mean(axis=1)
    path = os.path.join(out_dir, f"N{n_wires}_{variant}_V{v_in:g}_seed{seed}.npz")
    np.savez_compressed(path, I_total=r.I_total, V_in=r.V_in, on_frac=on_frac,
                        n_junctions=r.network.junctions.shape[0])
    print("saved", path)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], float(sys.argv[3]),
         int(sys.argv[4]) if len(sys.argv) > 4 else 60000,
         int(sys.argv[5]) if len(sys.argv) > 5 else 1)
