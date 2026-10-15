"""Relaxation of the measured Milano et al. 2022 2 V pulse trace (data/milano2022/pulse_2V.txt).

Independent check for N11 stage 1 (not part of its loss): one- and two-exponential fits of the conductance read at
10 mV after the pulse, and the fraction of the pulse-induced excess conductance left at fixed delays.
Writes results/milano2022_relax.json.

    .venv/bin/python scripts/milano2022_relax.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DELAYS = (1, 5, 10, 30, 60, 120, 240)


def one_exp(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def two_exp(t, a, t1, b, t2, c):
    return a * np.exp(-t / t1) + b * np.exp(-t / t2) + c


def aic(resid, k):
    n = len(resid)
    return n * np.log(np.sum(resid**2) / n) + 2 * k


def main():
    d = np.loadtxt(ROOT / "data/milano2022/pulse_2V.txt")
    t, current, volt = d[:, 0] - d[0, 0], d[:, 1], d[:, 2]
    g = current / volt
    on = np.where(volt > 1.0)[0]
    after = slice(on[-1] + 1, None)
    tr, gr = t[after] - t[after][0], g[after]
    g0 = g[0]

    p1, _ = curve_fit(one_exp, tr, gr, p0=[gr[0] - gr[-1], 20, gr[-1]], maxfev=20000)
    p2, _ = curve_fit(two_exp, tr, gr, p0=[p1[0] / 2, 3, p1[0] / 2, 50, p1[2]], maxfev=50000)
    if p2[1] > p2[3]:  # order the two times
        p2 = np.array([p2[2], p2[3], p2[0], p2[1], p2[4]])

    left = {}
    for s in DELAYS:
        k = np.searchsorted(tr, s)
        if k < len(tr):
            left[str(s)] = float((gr[k] - g0) / (gr[0] - g0))

    out = {
        "source": "Milano et al. 2022 Nat. Mater. 21, 195; GitHub Fig_2/raw_data/pulse_2V.txt",
        "pulse": {"V": float(volt[on].mean()), "start_s": float(t[on[0]]), "end_s": float(t[on[-1]])},
        "read_V": float(volt[after].mean()),
        "dt_median_s": float(np.median(np.diff(t))),
        "relax_record_s": float(tr[-1]),
        "G_pristine_S": float(g0),
        "G_end_pulse_S": float(g[on[-1]]),
        "G_first_read_S": float(gr[0]),
        "G_last_read_S": float(gr[-1]),
        "one_exp": {"tau_s": float(p1[1]), "amp_S": float(p1[0]), "floor_S": float(p1[2]),
                    "aic": float(aic(gr - one_exp(tr, *p1), 3))},
        "two_exp": {"tau1_s": float(p2[1]), "amp1_S": float(p2[0]), "tau2_s": float(p2[3]),
                    "amp2_S": float(p2[2]), "floor_S": float(p2[4]),
                    "aic": float(aic(gr - two_exp(tr, *p2), 5))},
        "excess_fraction_left": left,
        "authors_fit": {"kp0": 8.422409820914783e-10, "kd0": 0.048697579017353006, "eta_p": 0.19999370301178968,
                        "eta_d": 158.02444821482402, "tau_d0_s": 1 / 0.048697579017353006,
                        "model": "grid graph 21 x 21, Fig_2/fit_testing.py"},
    }
    path = ROOT / "results/milano2022_relax.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
