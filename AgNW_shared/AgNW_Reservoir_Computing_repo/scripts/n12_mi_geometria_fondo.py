#!/usr/bin/env python3
"""N12: a cluster geometry of your own at a fixed number of wires, sweeping the fraction of uniformly placed wires.

The script version of the Colab notebook resComp_mi_geometria_fondo.ipynb, for a local machine: every network (each
background fraction, plus the uniform network with the same number of wires as reference) is built, its onset found
with the quasi-static ramp of n12_geometry.sweep_one, and the short N10-E reservoir run on it (16 pads, random input
weights, volatile junction tau_s 0.3 s, 100 ms holds, window centre 1.25 x onset, width 0.45 x onset, one network, one
replicate). Onsets and runs go in parallel, one process per core.

    python scripts/n12_mi_geometria_fondo.py                                # defaults below
    python scripts/n12_mi_geometria_fondo.py --n-wires 3000 --fondos 0.4,0.6,0.8,0.9 --procs 8
    python scripts/n12_mi_geometria_fondo.py --clusters "0.3,0.5,0.1;0.7,0.5,0.1" --fondos 0.2,0.5,0.8

--clusters: "x,y,sigma[,weight];..." in units of the substrate side (default: the 4 clusters of clusters4 + one at the
centre, sigma 0.08). Onsets are cached in results/n12_onsets_cache.json (keyed by layout, number of wires and the
geometry itself), so a rerun or another sweep with the same networks skips them. Output: <out>/resultados.json,
resultados.csv, redes.png, puntajes.png, vs_fondo.png, memoria.png (default out: results/n12_mi_geometria/<tag>).
"""
import argparse
import csv
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agnw_rc import config as C  # noqa: E402
from agnw_rc.electrodes import make_layout  # noqa: E402
from agnw_rc.substrate import GEOMETRIES, Cluster, GeometrySpec, cluster_counts, network_stats  # noqa: E402

CACHE = ROOT / "results" / "n12_onsets_cache.json"
LAYOUT = "pads"
DEFAULT_CLUSTERS = "0.25,0.25,0.08;0.75,0.25,0.08;0.25,0.75,0.08;0.75,0.75,0.08;0.5,0.5,0.08"


def parse_clusters(text):
    out = []
    for part in text.split(";"):
        v = [float(x) for x in part.split(",")]
        if len(v) not in (3, 4):
            raise ValueError(f"cluster '{part}': expected x,y,sigma[,weight]")
        out.append(Cluster(*v))
    return tuple(out)


def name_of(b):
    return f"mi_geometria_b{round(b * 100)}"


def register(clusters, fondos, mu, sigma):
    """Register one geometry per background fraction (also in every worker process: called at worker start)."""
    for b in fondos:
        GEOMETRIES[name_of(b)] = GeometrySpec(clusters=clusters, background=b, angle_mu=mu, angle_sigma=sigma)


def cache_key(name, n_wires):
    return f"{LAYOUT}|N{n_wires}|{GEOMETRIES[name]!r}"


def onset_job(args):
    name, n_wires, cfg = args
    register(*cfg)
    from n12_geometry import sweep_one
    t = time.time()
    r = sweep_one(name, LAYOUT, n_wires)[1]
    return name, r["onset_V"], time.time() - t


def run_job(args):
    name, n_wires, onset, seed, n_samples, cfg = args
    register(*cfg)
    from agnw_rc.evaluate import observables, noise_floor, score_iid
    from agnw_rc.multidrive import simulate_multi
    spec = C.RunSpec(stream="iid", model="volatile", layout=LAYOUT, network_seed=seed, replicate=0, hold=100,
                     v_center=round(1.25 * onset, 2), v_width=round(0.45 * onset, 2), n_samples=n_samples,
                     washout=7000, n_virtual=10, stride=2, tau_s=0.3, tau_spread=1.0, geometry=name, n_wires=n_wires)
    t = time.time()
    raw = simulate_multi(spec)
    s = score_iid(observables(raw)["electrodes"], raw["u"], floor=noise_floor(raw["u"]))
    keep = ("MC", "IPC2", "narma10_nmse", "memory_function")
    return name, {**{k: s[k] for k in keep}, "n_junctions": int(raw["n_junctions"]), "segundos": time.time() - t,
                  "u": raw["u"].tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-wires", type=int, default=3000, help="number of wires (default 3000)")
    ap.add_argument("--fondos", default="0.4,0.6,0.8,0.9", help="background fractions, comma-separated")
    ap.add_argument("--clusters", default=DEFAULT_CLUSTERS, help='"x,y,sigma[,weight];..." (units of the side)')
    ap.add_argument("--angle-mu", type=float, default=0.0)
    ap.add_argument("--angle-sigma", type=float, default=0.0, help="0 = uniform wire angles")
    ap.add_argument("--seed", type=int, default=1, help="network seed")
    ap.add_argument("--samples", type=int, default=300, help="scored input samples")
    ap.add_argument("--no-uniform", action="store_true", help="skip the uniform reference network")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="", help="output folder (default results/n12_mi_geometria/<tag>)")
    a = ap.parse_args()

    clusters, fondos = parse_clusters(a.clusters), [float(x) for x in a.fondos.split(",")]
    cfg = (clusters, fondos, a.angle_mu, a.angle_sigma)
    register(*cfg)
    names = ([] if a.no_uniform else ["uniform"]) + [name_of(b) for b in fondos]
    fondo_de = {name_of(b): b for b in fondos} | {"uniform": 1.0}
    tag = f"N{a.n_wires}_seed{a.seed}_{len(clusters)}clusters"
    out = Path(a.out) if a.out else ROOT / "results" / "n12_mi_geometria" / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"{a.n_wires} hilos, semilla {a.seed}, {a.samples} entradas, {a.procs} procesos -> {out}", flush=True)

    # 1. wire counts and connectivity (5 seeds), no simulation
    print(f"\n{'red':18s}{'fondo':>6s}  hilos fondo / por cluster   junturas   conduce (5 semillas)   pads entrada/tierra")
    conect = {}
    for g in names:
        n = cluster_counts(a.n_wires, GEOMETRIES[g])
        st = [network_stats(net, make_layout(LAYOUT, net))
              for net in (C.make_network(s, n_wires=a.n_wires, geometry=g) for s in range(1, 6))]
        if 1 <= a.seed <= 5:
            s1 = st[a.seed - 1]
        else:
            net = C.make_network(a.seed, n_wires=a.n_wires, geometry=g)
            s1 = network_stats(net, make_layout(LAYOUT, net))
        conect[g] = {"hilos_fondo": int(n[0]), "hilos_por_cluster": [int(x) for x in n[1:]],
                     "junturas": s1["n_junctions"], "conduce_5": int(sum(s["percolates"] for s in st)),
                     "pads_entrada": s1["in_linked"], "pads_tierra": s1["gnd_linked"]}
        c = conect[g]
        print(f"{g:18s}{fondo_de[g]:6.0%}  {c['hilos_fondo']:5d} / {'/'.join(map(str, c['hilos_por_cluster'])) or '-':14s}"
              f"{c['junturas']:9d}   {c['conduce_5']}/5                    {c['pads_entrada']} / {c['pads_tierra']}", flush=True)

    # 2. onsets (cached)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    onsets = {g: cache[cache_key(g, a.n_wires)] for g in names if cache_key(g, a.n_wires) in cache}
    todo = [g for g in names if g not in onsets]
    print(f"\nonsets en caché: {sorted(onsets)}; a calcular: {todo}", flush=True)
    if todo:
        with ProcessPoolExecutor(min(a.procs, len(todo))) as pool:
            for g, on, dt in pool.map(onset_job, [(g, a.n_wires, cfg) for g in todo]):
                onsets[g] = on
                cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
                cache[cache_key(g, a.n_wires)] = on
                CACHE.write_text(json.dumps(cache, indent=1))
                print(f"  onset {g}: {on} V ({dt:.0f} s)", flush=True)
    for g in [g for g in names if onsets[g] is None]:
        print(f"  {g}: no conduce por debajo de 3 V; se saltea")

    # 3. runs
    runs = [g for g in names if onsets[g] is not None]
    res = {}
    print(f"\nsimulando {len(runs)} redes con {min(a.procs, len(runs))} procesos ...", flush=True)
    with ProcessPoolExecutor(min(a.procs, len(runs))) as pool:
        jobs = [(g, a.n_wires, onsets[g], a.seed, a.samples, cfg) for g in runs]
        for g, r in pool.map(run_job, jobs):
            res[g] = r
            print(f"  {g:18s} {r['segundos']:6.0f} s  {r['n_junctions']:6d} junturas  MC {r['MC']:.2f}  "
                  f"IPC2 {r['IPC2']:.2f}  NARMA-10 {r['narma10_nmse']:.3f}", flush=True)

    # 4. table, files, figures
    from agnw_rc.evaluate import noise_floor, score_iid
    u0 = np.array(next(iter(res.values()))["u"])
    base = score_iid(u0[:, None], u0, floor=noise_floor(u0))
    rows = [{"red": g, "fondo": fondo_de[g], "onset_V": onsets[g], **conect[g],
             **{k: res[g][k] for k in ("MC", "IPC2", "narma10_nmse", "segundos")}} for g in runs]
    print(f"\n{'red':18s}{'fondo':>6s}{'junturas':>10s}{'onset (V)':>11s}{'MC':>7s}{'IPC2':>7s}{'NARMA-10':>10s}")
    for r in rows:
        print(f"{r['red']:18s}{r['fondo']:6.0%}{r['junturas']:10d}{r['onset_V']:11.3f}{r['MC']:7.2f}{r['IPC2']:7.2f}"
              f"{r['narma10_nmse']:10.3f}")
    print(f"{'entrada sola':18s}{'':16s}{'':11s}{base['MC']:7.2f}{base['IPC2']:7.2f}{base['narma10_nmse']:10.3f}")
    meta = {"n_wires": a.n_wires, "seed": a.seed, "samples": a.samples, "clusters": [c.__dict__ for c in clusters],
            "angle_mu": a.angle_mu, "angle_sigma": a.angle_sigma, "layout": LAYOUT,
            "entrada_sola": {k: base[k] for k in ("MC", "IPC2", "narma10_nmse")}}
    (out / "resultados.json").write_text(json.dumps(
        {"meta": meta, "filas": rows, "memoria": {g: res[g]["memory_function"] for g in runs}}, indent=1))
    with open(out / "resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "hilos_por_cluster"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(1, len(names), figsize=(4.7 * len(names), 5.2), squeeze=False)
    for ax, g in zip(axes[0], names):
        net = C.make_network(a.seed, n_wires=a.n_wires, geometry=g)
        ax.add_collection(LineCollection(net.wire_endpoints * 1e6, colors="0.35", linewidths=0.2))
        for e in make_layout(LAYOUT, net):
            i, j = int(e.name[1]), int(e.name[2])
            ax.add_patch(plt.Rectangle(((i + 0.5) * 250 - 25, (j + 0.5) * 250 - 25), 50, 50, zorder=3,
                                       color="tab:orange" if e.role == "in" else "tab:blue"))
        for cl in GEOMETRIES[g].clusters:
            ax.add_patch(plt.Circle((cl.x * 1000, cl.y * 1000), 2 * cl.sigma * 1000, fill=False, ls="--", color="tab:red"))
        ax.set(xlim=(0, 1000), ylim=(0, 1000), aspect="equal", xticks=[], yticks=[],
               title=f"{g} ({fondo_de[g]:.0%} fondo)\n{conect[g]['junturas']} junturas")
    fig.tight_layout()
    fig.savefig(out / "redes.png", dpi=130)

    labels = [r["red"] for r in rows] + ["entrada sola"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k, t, thr in zip(axes, ("MC", "IPC2", "narma10_nmse"),
                             ("capacidad de memoria", "capacidad de grado 2", "NARMA-10 (menor es mejor)"), (1, 1, 0.7)):
        vals = [r[k] for r in rows] + [base[k]]
        ax.bar(labels, vals, color=["0.5" if n == "uniform" else "0.85" if n == "entrada sola" else "tab:blue" for n in labels])
        ax.axhline(thr, color="r", ls="--", lw=0.8)
        ax.set_title(t)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"{a.n_wires} hilos, semilla {a.seed}, {a.samples} entradas")
    fig.tight_layout()
    fig.savefig(out / "puntajes.png", dpi=130)

    cl_rows = [r for r in rows if r["red"] != "uniform"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, k, t, thr in zip(axes, ("MC", "IPC2", "narma10_nmse"),
                             ("capacidad de memoria", "capacidad de grado 2", "NARMA-10 (menor es mejor)"), (1, 1, 0.7)):
        ax.plot([r["fondo"] * 100 for r in cl_rows], [r[k] for r in cl_rows], "o-", label="tu geometría")
        if "uniform" in res:
            ax.axhline(res["uniform"][k], color="0.5", lw=1.5, label="uniforme (mismo N)")
        ax.axhline(thr, color="r", ls="--", lw=0.8, label="umbral G1")
        ax.set(xlabel="hilos uniformes (fondo, %)", title=t)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{a.n_wires} hilos, semilla {a.seed}: desempeño vs fracción de fondo")
    fig.tight_layout()
    fig.savefig(out / "vs_fondo.png", dpi=130)

    fig = plt.figure(figsize=(8, 3.5))
    for g in runs:
        mf = res[g]["memory_function"]
        plt.plot(range(len(mf)), mf, "o-", ms=3, label=g, lw=2 if g == "uniform" else 1)
    plt.xlim(0, 15)
    plt.xlabel("retardo (muestras atrás)")
    plt.ylabel("capacidad C_k")
    plt.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "memoria.png", dpi=130)
    print(f"\n-> {out}/resultados.json, resultados.csv, redes.png, puntajes.png, vs_fondo.png, memoria.png")


if __name__ == "__main__":
    main()
