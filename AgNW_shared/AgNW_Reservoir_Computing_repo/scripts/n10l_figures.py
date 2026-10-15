#!/usr/bin/env python3
"""Figures for the N10-L write-up and the report of 2026-10-07.

  fig29_n10l_layouts.png   the 1, 1.41 and 2 mm networks (seed 1) at true scale with their scaled 4x4 pads
                           (orange driven, blue grounded): a larger device is the 1 mm device enlarged geometrically
  fig30_n10l_tolerance.png tolerance vs network side. Tolerance r_75 = rupture rate (per junction) at which the median
                           run keeps 75 % of its rupture-free capacity (MC + degree-2 relative to the twin), interpolated
                           in log r between the tested rates; power-law fit r_75 ~ side^a and the side at which it
                           reaches the N10-C range for real junctions (results/n10c_sim.json)
                           -> also results/n10l_tolerance.json

    python scripts/n10l_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.electrodes import make_layout  # noqa: E402

FIGS = ROOT / "figures"
OUT = ROOT / "results" / "n10l_tolerance.json"
LEVEL = 0.75


def layouts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle
    sides = (1.0, 1.41, 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={"width_ratios": sides})
    for ax, side in zip(axes, sides):
        net = C.make_network(1, side=side)
        L = np.sqrt(net.area) * 1e3
        segs = net.wire_endpoints * 1e3
        ax.add_collection(LineCollection(segs, colors="0.55", linewidths=0.25))
        pad = 50e-3 * side
        for i, e in enumerate(make_layout("pads_scaled", net)):
            r, c = divmod(i, 4)
            cx, cy = (r + 0.5) * L / 4, (c + 0.5) * L / 4
            col = "#e8590c" if e.role == "in" else "#1c7ed6"
            ax.add_patch(Rectangle((cx - pad / 2, cy - pad / 2), pad, pad, color=col, zorder=3))
        ax.set(xlim=(0, L), ylim=(0, L), aspect="equal", xlabel="mm",
               title=f"{side:g} mm: {net.wire_endpoints.shape[0]} wires, {len(net.junctions)} junctions")
        ax.set_xticks([0, L])
        ax.set_yticks([0, L])
    fig.suptitle("N10-L: the 16-pad device enlarged geometrically (same wire density; pads 50 µm per mm; "
                 "orange driven, blue grounded)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig29_n10l_layouts.png", dpi=150)
    plt.close(fig)


def r_level(rates, ratios, level=LEVEL):
    """Rate where the median twin ratio crosses `level`, by linear interpolation in log10(rate)."""
    pts = sorted(zip(rates, ratios))
    for (r0, y0), (r1, y1) in zip(pts, pts[1:]):
        if y0 >= level > y1:
            return float(10 ** (np.log10(r0) + (y0 - level) / (y0 - y1) * (np.log10(r1) - np.log10(r0))))
    return None


def tolerance():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.loads((ROOT / "results" / "n10l_criteria.json").read_text())["rows"]
    real = json.loads((ROOT / "results" / "n10c_sim.json").read_text())["rupture_scaling"]
    r_real = sorted(v["r_needed_per_s"] for k, v in real.items() if k in ("0.0001", "0.001"))
    out = {"level": LEVEL, "sides": {}}
    for side in sorted({x["side_mm"] for x in rows}):
        pts = [(x["rupture"], x["twin_ratio_median"]) for x in rows
               if x["side_mm"] == side and x["rupture"] and x["twin_ratio_median"] is not None]
        if side == 1.0:                              # 1 mm: all N10-R rates (results/n10r_twins.json)
            tw = json.loads((ROOT / "results" / "n10r_twins.json").read_text())["summary"]
            pts = [(float(k), v["median_ratio"]) for k, v in tw.items()]
        rl = r_level(*zip(*pts))
        lo = min(pts)
        out["sides"][f"{side:g}"] = {"points": pts, "r_level": rl,
                                     # below LEVEL already at the lowest tested rate: only an upper bound
                                     "r_level_upper_bound": lo[0] if rl is None and lo[1] < LEVEL else None}
    xs = np.array([float(s) for s, v in out["sides"].items() if v["r_level"]])
    ys = np.array([v["r_level"] for v in out["sides"].values() if v["r_level"]])
    a, b = np.polyfit(np.log10(xs), np.log10(ys), 1)
    out["fit"] = {"exponent": float(a), "prefactor_at_1mm": float(10 ** b),
                  "real_range_per_s": r_real,
                  "side_to_reach_real_mm": [float(10 ** ((np.log10(r) - b) / a)) for r in r_real]}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["fit"], indent=1), {k: v["r_level"] for k, v in out["sides"].items()})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for s, v in out["sides"].items():
        p = sorted(v["points"])
        axes[0].semilogx([q[0] for q in p], [q[1] for q in p], "o-", label=f"{s} mm")
    axes[0].axhline(LEVEL, color="k", ls=":", lw=0.8)
    axes[0].set(xlabel="rupture rate per junction (1/s)", ylabel="median (MC + deg-2) / rupture-free twin",
                title="capacity kept vs rupture rate")
    axes[0].legend(fontsize=8)
    grid = np.logspace(0, np.log10(8), 50)
    axes[1].loglog(xs, ys, "o", ms=8, color="C3", label=rf"$r_{{75}}$ (measured)")
    axes[1].loglog(grid, 10 ** b * grid ** a, "--", color="C3", label=rf"fit $\propto$ side$^{{{a:.1f}}}$")
    ub = [(float(k), v["r_level_upper_bound"]) for k, v in out["sides"].items() if v.get("r_level_upper_bound")]
    if ub:
        axes[1].loglog([u[0] for u in ub], [u[1] for u in ub], "v", ms=9, color="C3", mfc="none",
                       label=r"upper bound ($\rho < 0.75$ at the lowest rate tested)")
    axes[1].axhspan(r_real[0], r_real[-1], color="0.85", label="real junctions (N10-C)")
    axes[1].axvspan(7, 8, color="#ffe8cc", label="measured device (7 mm gap)")
    axes[1].set(xlabel="network side (mm)", ylabel="tolerated rupture rate $r_{75}$ (1/s)",
                title="tolerance vs size (3 sizes within 2x + a bound: extrapolation indicative)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig30_n10l_tolerance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    layouts()
    tolerance()
