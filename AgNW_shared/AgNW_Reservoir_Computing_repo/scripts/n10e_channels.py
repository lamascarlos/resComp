#!/usr/bin/env python3
"""N10-E0: how much of the electrode readout's capacity comes from space (16 electrodes) and how much from time
(50 virtual nodes per hold)? Re-scores the pilot runs (scripts/n10e_pilot.py) with the electrode readout restricted
to: all 50 virtual nodes (as in the pilot), 5 evenly spaced virtual nodes, the end-of-hold value only (16 features),
and the end-of-hold value of a single electrode (1 feature). -> results/n10e_channels.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agnw_rc import evaluate as E  # noqa: E402
from agnw_rc.drive import load_raw  # noqa: E402
import n10e_pilot as P  # noqa: E402

OUT = ROOT / "results" / "n10e_channels.json"


def subsets(F, n_v):
    per = F.shape[1] // n_v                                  # features per virtual node (virtual-node-major)
    blocks = F.reshape(len(F), n_v, per)
    return {"all_50_nodes": F, "5_nodes": blocks[:, np.linspace(0, n_v - 1, 5).astype(int)].reshape(len(F), -1),
            "end_of_hold": blocks[:, -1], "one_electrode_end": blocks[:, -1, :1]}


def main():
    best = json.loads(P.OUT.read_text())["best"]
    rows = []
    for b in best:
        if b["readout"] != "electrodes":
            continue
        spec = next(s for s in P.specs() if s.layout == b["layout"] and s.encoding == b["encoding"]
                    and s.v_center == b["v_center"] and s.stream == "iid" and s.replicate == 0)
        raw = load_raw(spec, P.RAW)
        floor = E.noise_floor(raw["u"])
        n_v = len(raw["virtual_offsets"])
        for name, F in subsets(E.observables(raw)["electrodes"], n_v).items():
            s = E.score_iid(F, raw["u"], floor)
            rows.append({"layout": b["layout"], "encoding": b["encoding"], "subset": name, "n_features": F.shape[1],
                         **{m: float(s[m]) for m in ("MC", "IPC2", "narma10_nmse")}})
            print(f"{b['layout']:11s} {b['encoding']:8s} {name:18s} ({F.shape[1]:4d} f): MC {s['MC']:.2f} "
                  f"IPC2 {s['IPC2']:.2f} NARMA {s['narma10_nmse']:.2f}")
    OUT.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
