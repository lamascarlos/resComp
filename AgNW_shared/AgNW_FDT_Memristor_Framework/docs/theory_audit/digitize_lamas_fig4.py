"""Digitize Lamas 2026 Fig. 4 (N=2000, N=3000 panels; low-res embedded JPEGs)
into upper/lower I(V) envelopes, to get quantitative calibration targets
(onset, half-cycle loop areas, conductance on each branch).
Usage: .venv/bin/python docs/theory_audit/digitize_lamas_fig4.py IMG_DIR
IMG_DIR holds p7-002.png (N=2000) and p7-004.png (N=3000) from
`pdfimages -png -f 7 -l 7 <pdf> IMG_DIR/p7`. Accuracy ~1 px = 0.025 V, 0.03 mA."""
import json
import sys
import numpy as np
from PIL import Image

PANELS = {2000: ("p7-002.png", np.arange(-3, 4), np.arange(-3, 4)),
          3000: ("p7-004.png", np.arange(-3, 4), np.arange(-10, 11, 5))}


def light_lines(gray, axis, lo=215, hi=250):
    """Positions (along `axis`) of rows/cols that are mostly light-gray grid pixels."""
    frac = ((gray > lo) & (gray < hi)).mean(axis=axis)
    idx = np.nonzero(frac > 0.45)[0]
    if idx.size == 0:
        return np.array([])
    groups = np.split(idx, np.nonzero(np.diff(idx) > 2)[0] + 1)
    return np.array([g.mean() for g in groups])


def runs_of(lines, k):
    return [lines[i:i + k] for i in range(len(lines) - k + 1)] if len(lines) >= k else [lines]


def digitize(path, xticks, yticks):
    im = np.asarray(Image.open(path).convert("RGB")).astype(int)
    gray = im.mean(axis=2)
    # The curve is the only dark thing inside the axes box (JPEG chroma blur makes a
    # colour criterion patchy on thin segments; an earlier colour mask lost ~10-70% of
    # the curve pixels and undercounted loop areas). Find the frame, keep dark pixels inside.
    dark = gray < 150
    cols = np.nonzero(dark.mean(axis=0) > 0.6)[0]
    rows = np.nonzero(dark.mean(axis=1) > 0.6)[0]
    x0, x1 = cols.min() + 3, cols.max() - 3
    y0, y1 = rows.min() + 3, rows.max() - 3
    interior = np.zeros_like(dark)
    interior[y0:y1 + 1, x0:x1 + 1] = True
    ys, xs = np.nonzero((gray < 205) & interior)
    # Grid lines: several light-gray candidates (frame/axis lines are also light). The drive is
    # +-3.6 V and the response nearly odd, so pick the run of candidates for which the digitized
    # curve is most symmetric about (0, 0). Calibration sanity: Imax ~ -Imin, Vmax ~ -Vmin.
    best = None
    for rx in runs_of(light_lines(gray, axis=0), len(xticks)):
        ax_, bx_ = np.polyfit(rx, xticks, 1)
        V = ax_ * xs + bx_
        cx = abs(V.max() + V.min())
        for ry in runs_of(light_lines(gray, axis=1), len(yticks)):
            ay_, by_ = np.polyfit(ry, yticks[::-1], 1)          # image rows grow downward
            I = ay_ * ys + by_
            score = cx + abs(I.max() + I.min())
            if best is None or score < best[0]:
                best = (score, V, I, (rx, ry))
    return best[1], best[2], best[3]


def envelopes(V, I, step=0.05):
    edges = np.arange(np.floor(V.min() / step) * step, V.max() + step, step)
    c, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (V >= a) & (V < b)
        if m.sum() >= 1:
            c.append((a + b) / 2); lo.append(I[m].min()); hi.append(I[m].max())
    return np.array(c), np.array(lo), np.array(hi)


def metrics(V, I):
    """Envelope metrics of an I(V) loop (same definitions for paper and simulations):
    onset of the ascending branch, half-cycle loop areas (upper-lower envelope),
    and |I| of the two branches at -1 V and -2 V."""
    c, lo, hi = envelopes(V, I)
    pos, neg = c > 0, c < 0
    imax = np.abs(I).max()
    a_pos = np.trapezoid((hi - lo)[pos], c[pos])
    a_neg = np.trapezoid((hi - lo)[neg], c[neg])
    onset = c[pos][np.argmax(lo[pos] > 0.03 * imax)]
    f = lambda v, arr: float(np.interp(v, c, arr))
    return dict(Imax=float(I.max()), Imin=float(I.min()), onset_V=float(onset),
                area_pos=float(abs(a_pos)), area_neg=float(abs(a_neg)),
                ratio=float(abs(a_neg / a_pos)),
                neg1V_absI_small=abs(f(-1.0, hi)), neg1V_absI_large=abs(f(-1.0, lo)),
                neg2V_absI_small=abs(f(-2.0, hi)), neg2V_absI_large=abs(f(-2.0, lo)),
                pos2V_low=f(2.0, lo), pos2V_high=f(2.0, hi))


if __name__ == "__main__":
    d = sys.argv[1]            # directory with p7-002.png / p7-004.png
    for N, (fn, xt, yt) in PANELS.items():
        V, I, _ = digitize(f"{d}/{fn}", xt, yt)
        print(json.dumps({"N": N, **{k: round(v, 3) for k, v in metrics(V, I).items()}}, indent=1))
