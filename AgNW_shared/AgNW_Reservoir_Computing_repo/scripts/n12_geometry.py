#!/usr/bin/env python3
"""N12: non-uniform wire placement (agnw_rc/substrate.py, docs/N12_GEOMETRY.md) - connectivity before any simulation.

  stats    every geometry in substrate.GEOMETRIES (or --only NAME,...) x network seeds 1-3: junctions, mean degree,
           isolated wires, largest connected wire cluster, percolation between the 4x4 pads and between the two bands
           -> results/n12_geometry_stats.json (+ a table on stdout)
  figure   the networks of seed 1 for each geometry, with the 4x4 pads (orange driven, blue grounded) and the cluster
           centres (2 sigma circles) -> figures/fig36_n12_geometries.png

    python scripts/n12_geometry.py stats [--only clusters4,aligned_x] [--seeds 1,2,3]
    python scripts/n12_geometry.py figure
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.electrodes import make_layout  # noqa: E402
from agnw_rc.substrate import GEOMETRIES, network_stats  # noqa: E402

OUT = ROOT / "results" / "n12_geometry_stats.json"
FIG = ROOT / "figures" / "fig36_n12_geometries.png"


def stats(names, seeds):
    rows = []
    for name in names:
        for seed in seeds:
            net = C.make_network(seed, geometry=name)
            bands = network_stats(net)
            pads = network_stats(net, make_layout("pads", net))
            rows.append({"geometry": name, "network_seed": seed, **{k: bands[k] for k in (
                "n_wires", "n_junctions", "mean_degree", "isolated_fraction", "largest_cluster_fraction")},
                "bands_percolate": bands["percolates"], "pads_percolate": pads["percolates"],
                "pads_in_linked": pads["in_linked"], "pads_gnd_linked": pads["gnd_linked"]})
            r = rows[-1]
            print(f"{name:16s} seed {seed}  junctions {r['n_junctions']:5d}  degree {r['mean_degree']:5.2f}  "
                  f"isolated {r['isolated_fraction']:5.1%}  largest {r['largest_cluster_fraction']:5.1%}  "
                  f"bands {'yes' if r['bands_percolate'] else 'NO '}  pads {'yes' if r['pads_percolate'] else 'NO '} "
                  f"(in {r['pads_in_linked']}, gnd {r['pads_gnd_linked']})")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"geometries": {n: repr(GEOMETRIES[n]) for n in names}, "rows": rows}, indent=1))
    print(f"-> {OUT.relative_to(ROOT)}")


def figure(names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Rectangle
    ncol = min(3, len(names))
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5.2 * nrow), squeeze=False)
    um = 1e6
    for ax, name in zip(axes.flat, names):
        net = C.make_network(1, geometry=name)
        side = np.sqrt(net.area) * um
        ax.add_collection(LineCollection(net.wire_endpoints * um, colors="0.35", linewidths=0.3))
        for i in range(4):
            for j in range(4):
                c = ((i + 0.5) * side / 4, (j + 0.5) * side / 4)
                color = "tab:orange" if (i + j) % 2 == 0 else "tab:blue"
                ax.add_patch(Rectangle((c[0] - 25, c[1] - 25), 50, 50, color=color, alpha=0.8, zorder=3))
        for cl in GEOMETRIES[name].clusters:
            ax.add_patch(Circle((cl.x * side, cl.y * side), 2 * cl.sigma * side, fill=False, ls="--", color="tab:red"))
        s = network_stats(net, make_layout("pads", net))
        ax.set_title(f"{name}\n{s['n_junctions']} junctions, largest cluster {s['largest_cluster_fraction']:.0%}, "
                     f"pads {'percolate' if s['percolates'] else 'do NOT percolate'}", fontsize=9)
        ax.set_xlim(0, side), ax.set_ylim(0, side), ax.set_aspect("equal")
        ax.set_xlabel("x (um)"), ax.set_ylabel("y (um)")
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"-> {FIG.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("stats", "figure"))
    ap.add_argument("--only", default="", help="comma-separated geometry names (default: all)")
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(GEOMETRIES)
    unknown = set(names) - set(GEOMETRIES)
    if unknown:
        ap.error(f"unknown geometries {sorted(unknown)}; known: {list(GEOMETRIES)}")
    if a.cmd == "stats":
        stats(names, [int(s) for s in a.seeds.split(",")])
    else:
        figure(names)


if __name__ == "__main__":
    main()
