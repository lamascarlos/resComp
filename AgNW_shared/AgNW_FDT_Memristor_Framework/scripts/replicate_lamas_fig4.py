"""Replicate Lamas et al. 2026 Fig. 4: I-V hysteresis loops under a
sinusoidal drive (amplitude 3.6 V, 50000 steps/cycle), Table 1 parameters.

Usage: python -m scripts.replicate_lamas_fig4 N RESET_VARIANT [seed]
  RESET_VARIANT: 'lamas' -> packaged preset (agnw_sim.presets.lamas2026_fig4);
                 'lamas_xM' -> same with the RESET coefficient multiplied by M;
                 'lamas_epsE' -> same with spontaneous relaxation probability E per step
                 'lamas_lifeL' -> same with a fixed lifetime of L steps after SET
                 (tokens combine: 'lamas_x10_eps1e-4').
                 Others (see VARIANTS): 'mA'  -> paper-literal coefficient, I in mA (RESET ~inactive)
                 'fit' -> per-step coefficient ~3e7 A^-2 (matches Fig. 5A plateau)
Standalone validation driver (not part of the package; no unit tests)."""
import dataclasses
import os
import sys
import numpy as np

from agnw_sim.simulate import run_simulation
from agnw_sim.dynamics import SimParams
from agnw_sim.presets import lamas2026_fig4
from agnw_sim.waveforms import sine

DT = 1e-3
STEPS_PER_CYCLE = 50000
V_AMP = 3.6

VARIANTS = {
    "mA": dict(I_unit=1e-3),
    "fit": dict(I_unit=float(np.sqrt(0.01 / 3e7))),  # 0.01*(I/I_unit)^2 = 3e7*I_A^2
    # V_th sensitivity (diagnosed via scripts.explore_fig4_onset: Table-1
    # V_th=0.01 V gives onset ~0.35 V; V_th ~0.1 V matches the paper's ~1.8 V).
    "mA_Vth0.05": dict(I_unit=1e-3, V_th=0.05),
    "mA_Vth0.1": dict(I_unit=1e-3, V_th=0.1),
}


def build_params(variant, dt=DT):
    """(SimParams, run_simulation kwargs) for a variant name (see module docstring)."""
    run_kw = {}
    if variant.startswith("lamas"):   # packaged replication: V_th=0.1 V, D=90 nm dual-node
        params, run_kw = lamas2026_fig4(dt)
        for tok in variant.split("_")[1:]:   # 'lamas_x10_eps1e-4': RESET x10 and relaxation eps
            if tok.startswith("x"):          # RESET coefficient M times the literal Table 1 value
                params = dataclasses.replace(params, I_unit=params.I_unit / np.sqrt(float(tok[1:])))
            elif tok.startswith("life"):     # deterministic lifetime (steps) after SET
                params = dataclasses.replace(params, lifetime_steps=int(float(tok[4:])))
            elif tok.startswith("eps"):      # spontaneous relaxation probability per step
                params = dataclasses.replace(params, p_relax=float(tok[3:]))
    else:
        kw = {"alpha": 1.0, "V_th": 0.01, **VARIANTS[variant]}
        params = SimParams.from_step_probs(0.4, 0.01, dt=dt, **kw)
    return params, run_kw


def main(n_wires, variant, seed=1, out_dir="validation_output/lamas_fig4"):
    os.makedirs(out_dir, exist_ok=True)
    params, run_kw = build_params(variant)
    f = 1.0 / (STEPS_PER_CYCLE * DT)
    result = run_simulation(
        n_wires=n_wires, area=1e-6, length=70e-6,
        waveform_fn=lambda t: V_AMP * np.sin(2 * np.pi * f * t),
        n_steps=STEPS_PER_CYCLE, dt=DT, params=params,
        rng=np.random.default_rng(seed), n_sampled_junctions=200, **run_kw,
    )
    on_frac = (result.G_sample >= params.G_on).mean(axis=1)
    path = os.path.join(out_dir, f"N{n_wires}_{variant}_seed{seed}.npz")
    np.savez_compressed(path, V_in=result.V_in, I_total=result.I_total,
                        on_frac=on_frac, n_junctions=result.network.junctions.shape[0])
    print(f"saved {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
