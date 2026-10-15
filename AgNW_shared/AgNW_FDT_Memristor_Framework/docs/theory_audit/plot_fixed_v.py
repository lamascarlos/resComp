"""Overview + statistics of the fixed-voltage long runs (scripts/fixed_v_stationarity.py).
Usage: .venv/bin/python docs/theory_audit/plot_fixed_v.py [DPI]
Writes validation_output/fixed_v/fixed_v_overview.png and prints, per run: mean/std of
G_eff in consecutive 5000-step blocks, the dominant fluctuation period (FFT of the
detrended trailing 40000 steps) and the autocorrelation time of that trailing part."""
import glob
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "validation_output/fixed_v/"
DPI = int(sys.argv[1]) if len(sys.argv) > 1 else 200
files = sorted(glob.glob(D + "N*_seed1.npz"))
if not files:
    raise SystemExit("no runs found")


def label(f):
    m = re.match(r"N(\d+)_lamas(?:_(\S+?))?_V([\d.]+)_seed(\d+)\.npz", os.path.basename(f))
    N, var, V, sd = m.groups()
    return f"N={N}, V={V} V, " + {None: "literal RESET (no relaxation)", "life20000": "fixed lifetime L=20000",
                                   "eps1e-4": "relaxation eps=1e-4"}.get(var, var)


def stats(G, block=5000):
    out = []
    for k in range(0, len(G) - block + 1, block):
        b = G[k:k + block]
        out.append((b.mean(), b.std(), b.min(), b.max()))
    return out


def dominant_period(x):
    x = x - x.mean()
    f = np.fft.rfftfreq(len(x), d=1.0)
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    k = np.argmax(P[1:]) + 1
    return 1.0 / f[k], P[k] / P[1:].sum()


def acorr_time(x, max_lag=15000):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:max_lag] / (np.arange(n, n - max_lag, -1))
    ac /= ac[0]
    below = np.nonzero(ac < np.exp(-1))[0]
    return (below[0] if below.size else np.nan), ac


fig, axes = plt.subplots(len(files), 2, figsize=(16, 2.6 * len(files)), squeeze=False)
for r, f in enumerate(files):
    z = np.load(f)
    V = float(z["V_in"][0]); G = z["I_total"] / V * 1e3; on = z["on_frac"]
    t = np.arange(len(G))
    axes[r, 0].plot(t, G, lw=0.4, color="purple"); axes[r, 0].set_ylabel("G_eff (mS)")
    axes[r, 0].set_title(label(f), fontsize=9)
    axes[r, 1].plot(t, on, lw=0.5, color="tab:green"); axes[r, 1].set_ylabel("ON fraction (200 sampled junctions)")
    axes[r, 0].grid(alpha=.3); axes[r, 1].grid(alpha=.3)
    tail = G[-40000:] if len(G) >= 60000 else G[len(G) // 3:]
    per, frac = dominant_period(tail)
    tau, _ = acorr_time(tail)
    print(f"\n{label(f)}  ({len(G)} steps)")
    print("  5000-step blocks (mean+-std [min,max] mS): " + "  ".join(
        f"{m:.3f}±{s:.3f}[{lo:.2f},{hi:.2f}]" for m, s, lo, hi in stats(G)))
    print(f"  trailing part: dominant period {per:.0f} steps ({100*frac:.0f}% of spectral power), "
          f"1/e autocorrelation time {tau} steps, CV {tail.std()/tail.mean():.2f}")
for ax in axes[-1, :]:
    ax.set_xlabel("step (dt = 1 ms)")
fig.suptitle("Fixed V_in from t=0, dual-node D=90 nm, V_th=0.1 V, literal RESET (+ relaxation variants)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(D + "fixed_v_overview.png", dpi=DPI)
print("\nwrote", D + "fixed_v_overview.png")
