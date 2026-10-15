"""Slowest per-junction relaxation time of a pilot run (integrated autocorrelation time, lag units = steps)."""
import sys

import numpy as np

path, tail_steps = sys.argv[1], int(sys.argv[2])
z = np.load(path)
g = z["G_on"][-(tail_steps // 10):].astype(float)               # (samples, K), one sample per 10 steps
g = g[:, g.std(axis=0) > 0]
g -= g.mean(axis=0)
n = g.shape[0]
f = np.fft.rfft(g, 2 * n, axis=0)
acf = np.fft.irfft(f * np.conj(f), axis=0)[:n]
acf /= acf[0]
taus = []
for c in acf.T:
    neg = np.nonzero(c < 0)[0]
    stop = neg[0] if len(neg) else n // 2                     # sum to the first zero crossing
    taus.append(10 * (0.5 + c[1:stop].sum()))
taus = np.array(taus)
print(f"junctions {len(taus)}  tau_int (steps): median {np.median(taus):.0f}  90th pct {np.percentile(taus, 90):.0f}  max {taus.max():.0f}")
