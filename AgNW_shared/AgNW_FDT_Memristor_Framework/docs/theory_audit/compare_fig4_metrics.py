"""Same envelope metrics for the digitized paper Fig. 4 and our simulated traces."""
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digitize_lamas_fig4 as D

IMG = sys.argv[1]
PANELS = {2000: ("p7-002.png", np.arange(-3, 4)), 3000: ("p7-004.png", np.arange(-10, 11, 5))}


metrics = D.metrics


rows = {}
for N, (fn, yt) in PANELS.items():
    V, I, _ = D.digitize(f"{IMG}/{fn}", np.arange(-3, 4), yt)
    rows[f"paper N={N}"] = metrics(V, I)
sims = {"dual-node D=90, no RESET": "validation_output/wire_resistance/d90nm_N{N}_full.npz",
        "wire-as-node V_th=.1, RESET mA": "validation_output/lamas_fig4/N{N}_mA_Vth0.1_seed1.npz",
        "PRESET dual-node D=90, RESET mA": "validation_output/lamas_fig4/N{N}_lamas_seed1.npz"}
for m in (3, 10, 30, 100, 300):
    sims[f"dual-node D=90, RESET x{m}"] = f"validation_output/lamas_fig4/N{{N}}_lamas_x{m}_seed1.npz"
for e in ("3e-5", "1e-4", "2e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2"):
    sims[f"relaxation eps={e}"] = f"validation_output/lamas_fig4/N{{N}}_lamas_eps{e}_seed1.npz"
for L in (5000, 10000, 15000, 17500, 20000, 25000, 30000):
    sims[f"fixed lifetime L={L}"] = f"validation_output/lamas_fig4/N{{N}}_lamas_life{L}_seed1.npz"
for name, pat in sims.items():
    for N in (2000, 3000):
        f = pat.format(N=N)
        if os.path.exists(f):
            z = np.load(f)
            rows[f"{name} N={N}"] = metrics(z["V_in"], z["I_total"] * 1e3)
hdr = f"{'':38s} {'Imax':>6s} {'onset':>6s} {'A+':>6s} {'A-':>6s} {'A-/A+':>6s} | |I|(-1V) small/large | |I|(-2V) small/large"
print(hdr)
for k, m in rows.items():
    print(f"{k:38s} {m['Imax']:6.2f} {m['onset_V']:6.2f} {m['area_pos']:6.2f} {m['area_neg']:6.2f} {m['ratio']:6.2f} |"
          f" {m['neg1V_absI_small']:5.2f} / {m['neg1V_absI_large']:5.2f}    | {m['neg2V_absI_small']:5.2f} / {m['neg2V_absI_large']:5.2f}")
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig4_metrics.json"), "w"), indent=1)
