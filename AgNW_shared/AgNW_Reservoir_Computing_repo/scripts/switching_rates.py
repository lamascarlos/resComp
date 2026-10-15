#!/usr/bin/env python3
"""How often each switching mechanism of the framework fires at the reservoir operating points
(docs/SWITCHING_MECHANISMS.md).

For recorded snapshots of E1 runs (network seed 1, replicate 0, hold 30), the junction voltages are recomputed from
the snapshot (framework circuit solver), and the per-step probabilities of the three mechanisms are evaluated with
the framework's own functions: SET p_set(|dV|) on OFF junctions; current-driven RESET p_reset(I_on = G_on |dV|)
and spontaneous relaxation p_relax on ON junctions. Reported: the expected number of events per step, network-wide,
per mechanism, and the distribution of the relevant voltages and currents.

    python scripts/switching_rates.py   ->  results/switching_rates.json
"""
import json
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
from agnw_sim.dynamics import p_reset, p_set  # noqa: E402

from agnw_rc import config as C  # noqa: E402
from agnw_rc.drive import load_raw, unpack_flags  # noqa: E402

RAW = ROOT / "results" / "raw"
N_SNAPSHOTS = 40


def main():
    params, wd = C.network_params()
    net = C.make_network(1)
    circuit = DualNodeCircuit(net, wd)
    out = {"params": {"dt": params.dt, "P0": params.P0, "alpha": params.alpha, "V_th": params.V_th,
                      "P_decay": params.P_decay, "I_unit": params.I_unit, "p_relax": params.p_relax,
                      "G_on": params.G_on, "G_off": params.G_off},
           "runs": {}}
    for vc in C.E1_CENTERS:
        spec = C.RunSpec("iid", 30, vc, 1, 0)
        raw = load_raw(spec, RAW)
        flags = unpack_flags(raw)
        idx = np.linspace(C.N_SAMPLES["iid"] // 10, len(flags) - 1, N_SNAPSHOTS).astype(int)
        ev = {"set": [], "reset": [], "relax": []}
        dv_off, dv_on, i_on = [], [], []
        for k in idx:
            on = flags[k]
            G = np.where(on, params.G_on, params.G_off)
            V = circuit.solve(G, raw["v_in"][k])
            dv = np.abs(circuit.junction_voltage_drops(V))
            ps = p_set(dv[~on], params)
            pr = p_reset(params.G_on * dv[on], params)
            ev["set"].append(ps.sum())
            ev["reset"].append(pr.sum())
            ev["relax"].append(params.p_relax * on.sum())
            dv_off.append(dv[~on]); dv_on.append(dv[on]); i_on.append(params.G_on * dv[on])
        dv_off, dv_on, i_on = map(np.concatenate, (dv_off, dv_on, i_on))
        q = lambda x: [float(v) for v in np.quantile(x, [0.5, 0.9, 0.99, 1.0])]
        out["runs"][spec.name] = {
            "window_V": [spec.v_lo, spec.v_hi],
            "on_fraction": float(flags[idx].mean()),
            "expected_events_per_step": {m: float(np.mean(v)) for m, v in ev.items()},
            "off_junction_dV_quantiles_50_90_99_max_V": q(dv_off),
            "fraction_off_above_V_th": float(np.mean(dv_off > params.V_th)),
            "on_junction_dV_quantiles_V": q(dv_on),
            "on_junction_current_quantiles_A": q(i_on),
        }
        e = out["runs"][spec.name]["expected_events_per_step"]
        print(f"{spec.v_lo:.2f}-{spec.v_hi:.2f} V: ON {flags[idx].mean():.3f}  events/step: SET {e['set']:.3g}  "
              f"RESET {e['reset']:.3g}  relax {e['relax']:.3g}  | OFF above V_th "
              f"{out['runs'][spec.name]['fraction_off_above_V_th']:.4f}  | max I_on "
              f"{i_on.max() * 1e3:.3f} mA", flush=True)
    path = ROOT / "results" / "switching_rates.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
