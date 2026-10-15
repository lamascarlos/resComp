"""Gallery of simulated Fig.-4 loops (one panel per run) with the digitized paper
curve overlaid in red, plus the paper's original panel, for visual comparison.
Usage: .venv/bin/python docs/theory_audit/plot_fig4_gallery.py N [DPI]   (N = 2000 or 3000)
       .venv/bin/python docs/theory_audit/plot_fig4_gallery.py seeds [DPI]
Writes validation_output/lamas_fig4/gallery_N{N}.png (all runs that exist, seed 1) and
gallery_seeds_N2000.png (seed-to-seed comparison for the two best candidates)."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import digitize_lamas_fig4 as D

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "paper_fig4")
PAPER = {2000: ("p7-002.png", np.arange(-3, 4)), 3000: ("p7-004.png", np.arange(-10, 11, 5))}
LIM = {2000: (4.6, 4.5), 3000: (15.0, 14.0)}      # (I limit mA, ...)
L4 = "validation_output/lamas_fig4/"
WR = "validation_output/wire_resistance/"


def runs(N):
    r = [("dual-node D=90 nm, NO RESET", f"{WR}d90nm_N{N}_full.npz"),
         ("PRESET: literal RESET (x1)", f"{L4}N{N}_lamas_seed1.npz")]
    if N == 2000:
        r += [(f"RESET x{m}", f"{L4}N{N}_lamas_x{m}_seed1.npz") for m in (3, 10, 30, 100, 300)]
        r += [(f"relaxation eps={e} (life ~{1/float(e):.0f} steps)", f"{L4}N{N}_lamas_eps{e}_seed1.npz")
              for e in ("3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2")]
        r += [(f"relaxation eps=2e-4 (life ~5000 steps)", f"{L4}N{N}_lamas_eps2e-4_seed1.npz")]
        r += [(f"fixed lifetime L={L} steps", f"{L4}N{N}_lamas_life{L}_seed1.npz")
              for L in (5000, 10000, 15000, 17500, 20000, 25000, 30000)]
    if N == 3000:
        r += [("relaxation eps=1e-4 (life ~10000 steps)", f"{L4}N{N}_lamas_eps1e-4_seed1.npz"),
              ("fixed lifetime L=20000 steps", f"{L4}N{N}_lamas_life20000_seed1.npz")]
    return [(name, f) for name, f in r if os.path.exists(f)]


def main(N, dpi=200):
    fn, yt = PAPER[N]
    pV, pI, _ = D.digitize(os.path.join(IMG, fn), np.arange(-3, 4), yt)
    pm = D.metrics(pV, pI)
    items = runs(N)
    ncol = 5
    nrow = int(np.ceil((len(items) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow), squeeze=False)
    ilim = 1.15 * max(abs(pI).max(), 1e-9)
    ax = axes[0, 0]
    ax.imshow(np.asarray(Image.open(os.path.join(IMG, fn)).convert("RGB")))
    ax.set_title(f"PAPER (Lamas 2026 Fig. 4, N={N})\nA-/A+={pm['ratio']:.2f}  Imax={pm['Imax']:.2f}  A+={pm['area_pos']:.2f}", fontsize=9)
    ax.axis("off")
    for k, (name, f) in enumerate(items, start=1):
        ax = axes.flat[k]
        z = np.load(f)
        V, I = z["V_in"], z["I_total"] * 1e3
        m = D.metrics(V, I)
        ax.plot(pV, pI, ".", color="tab:red", ms=1.6, alpha=0.55, label="paper (digitized)", zorder=1)
        ax.plot(V, I, "-", color="purple", lw=0.7, label="simulation", zorder=2)
        ax.set_xlim(-3.9, 3.9); ax.set_ylim(-ilim, ilim); ax.grid(alpha=0.3)
        ax.set_title(f"{name}\nA-/A+={m['ratio']:.2f}  Imax={m['Imax']:.2f}  A+={m['area_pos']:.2f}", fontsize=8.5)
        ax.tick_params(labelsize=7)
        if k == 1:
            ax.legend(fontsize=7, loc="upper left")
    for ax in axes.flat[len(items) + 1:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("V_in (V)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("I (mA)", fontsize=8)
    fig.suptitle(f"N={N}: simulated I-V loops (purple) vs. digitized paper curve (red). Titles: A-/A+ = negative/positive half-cycle "
                 f"loop-area ratio (paper {pm['ratio']:.2f}), Imax (mA), A+ = positive-half area (mA V)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = f"validation_output/lamas_fig4/gallery_N{N}.png"
    fig.savefig(out, dpi=dpi)
    print("wrote", out, "with", len(items), "runs")


def seeds_figure(dpi=200, N=2000):
    fn, yt = PAPER[N]
    pV, pI, _ = D.digitize(os.path.join(IMG, fn), np.arange(-3, 4), yt)
    pm = D.metrics(pV, pI)
    variants = [("fixed lifetime L=20000", "life20000"), ("relaxation eps=1e-4", "eps1e-4")]
    seeds = (1, 2, 3, 4)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.2), squeeze=False)
    ilim = 1.15 * abs(pI).max()
    table = {}
    for r, (label, tag) in enumerate(variants):
        axes[r, 0].imshow(np.asarray(Image.open(os.path.join(IMG, fn)).convert("RGB")))
        axes[r, 0].set_title(f"PAPER N={N}\nA-/A+={pm['ratio']:.2f}  Imax={pm['Imax']:.2f}  A+={pm['area_pos']:.2f}", fontsize=9)
        axes[r, 0].axis("off")
        ms = []
        for c, sd in enumerate(seeds, start=1):
            ax = axes[r, c]
            z = np.load(f"{L4}N{N}_lamas_{tag}_seed{sd}.npz")
            V, I = z["V_in"], z["I_total"] * 1e3
            m = D.metrics(V, I); ms.append(m)
            ax.plot(pV, pI, ".", color="tab:red", ms=1.6, alpha=0.55, zorder=1)
            ax.plot(V, I, "-", color="purple", lw=0.7, zorder=2)
            ax.set_xlim(-3.9, 3.9); ax.set_ylim(-ilim, ilim); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            ax.set_title(f"{label}, seed {sd}\nA-/A+={m['ratio']:.2f}  Imax={m['Imax']:.2f}  A+={m['area_pos']:.2f}", fontsize=9)
            ax.set_xlabel("V_in (V)", fontsize=8)
        axes[r, 1].set_ylabel("I (mA)", fontsize=8)
        table[label] = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms], ddof=1)))
                        for k in ("Imax", "onset_V", "area_pos", "area_neg", "ratio")}
    fig.suptitle(f"N={N}, dual-node D=90 nm, V_th=0.1 V, literal RESET: seed-to-seed comparison (purple) vs paper (red)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f"validation_output/lamas_fig4/gallery_seeds_N{N}.png"
    fig.savefig(out, dpi=dpi)
    print("wrote", out)
    for label, t in table.items():
        print(f"{label}: " + "  ".join(f"{k} {m:.2f}±{sd:.2f}" for k, (m, sd) in t.items()))
    print(f"paper: Imax {pm['Imax']:.2f}  onset {pm['onset_V']:.2f}  area_pos {pm['area_pos']:.2f}  area_neg {pm['area_neg']:.2f}  ratio {pm['ratio']:.2f}")


if __name__ == "__main__":
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    if sys.argv[1] == "seeds":
        seeds_figure(dpi)
    else:
        main(int(sys.argv[1]), dpi)
