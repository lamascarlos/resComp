#!/bin/bash
# Red uniforme con distinto número de hilos (densidad), usando scripts/n12_mi_geometria_fondo.py sin modificarlo:
# clona o actualiza el repo, arma el entorno (una sola vez), corre el script una vez por cada N y al final junta la fila
# "uniform" de cada corrida en una tabla y una figura de MC, IPC2 y NARMA-10 vs número de hilos.
#
#   bash run_densidad_local.sh                                          # N = 2000 2500 3000 3500 4000
#   HILOS="2000 2250 2500 2750 3000 3250 3500 3750 4000" bash run_densidad_local.sh
#   SAMPLES=500 bash run_densidad_local.sh                              # más entradas evaluadas (más lento)
#   nohup bash run_densidad_local.sh > densidad.log 2>&1 &              # sigue aunque cierres la terminal
#
# El script siempre corre además al menos una red con clusters (--fondos); se usa un solo fondo (0.9) y se ignora. Con
# 2 o más núcleos corre en paralelo con la uniforme, así que casi no agrega tiempo.
# Si un N ya tiene resultados se saltea (borrá su carpeta para repetirlo).
# Resultados en ~/rescomp/AgNW_shared/AgNW_Reservoir_Computing_repo/results/n12_densidad_uniforme/
set -euo pipefail
BRANCH="${BRANCH:-feature/geometria-substrato}"
DIR="${DIR:-$HOME/rescomp}"
HILOS="${HILOS:-2000 2500 3000 3500 4000}"
SEED="${SEED:-1}"
SAMPLES="${SAMPLES:-300}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ -d "$DIR/.git" ]; then
    git -C "$DIR" fetch -q origin "$BRANCH" && git -C "$DIR" checkout -q "$BRANCH" && git -C "$DIR" pull -q --ff-only origin "$BRANCH"
else
    git clone -q -b "$BRANCH" https://github.com/lamascarlos/rescomp.git "$DIR"
fi
cd "$DIR/AgNW_shared/AgNW_Reservoir_Computing_repo"
if [ ! -x .venv/bin/python ]; then                      # necesita python3-venv: sudo apt install python3-venv
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -e ../AgNW_FDT_Memristor_Framework -e ".[dev]"
fi
OUT=results/n12_densidad_uniforme
mkdir -p "$OUT"
echo "== $(date)  $(hostname)  $(nproc) núcleos  N = $HILOS  semilla $SEED  $SAMPLES entradas"

for N in $HILOS; do
    if [ -f "$OUT/N$N/resultados.json" ]; then
        echo "== N = $N ya está hecho, se saltea"
        continue
    fi
    echo "== N = $N  ($(date +%H:%M))"
    .venv/bin/python scripts/n12_mi_geometria_fondo.py --n-wires "$N" --fondos 0.9 --seed "$SEED" \
        --samples "$SAMPLES" --out "$OUT/N$N"
done

echo "== resumen"
.venv/bin/python - "$OUT" <<'EOF'
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
rows = []
for f in out.glob("N*/resultados.json"):
    d = json.loads(f.read_text())
    r = next((r for r in d["filas"] if r["red"] == "uniform"), None)
    if r:
        rows.append({"N": d["meta"]["n_wires"], "semilla": d["meta"]["seed"], "junturas": r["junturas"],
                     "onset_V": r["onset_V"], "MC": r["MC"], "IPC2": r["IPC2"], "narma10_nmse": r["narma10_nmse"],
                     "G1": r["MC"] >= 1 and r["IPC2"] >= 1 and r["narma10_nmse"] <= 0.7})
        base = d["meta"]["entrada_sola"]
rows.sort(key=lambda r: r["N"])
if not rows:
    sys.exit("no hay resultados")
with open(out / "resumen.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{'N':>6s}{'junturas':>10s}{'onset (V)':>11s}{'MC':>7s}{'IPC2':>7s}{'NARMA-10':>10s}{'G1':>5s}")
for r in rows:
    print(f"{r['N']:6d}{r['junturas']:10d}{r['onset_V']:11.3f}{r['MC']:7.2f}{r['IPC2']:7.2f}{r['narma10_nmse']:10.3f}"
          f"{'sí' if r['G1'] else 'no':>5s}")
print(f"{'entrada sola':>27s}{base['MC']:7.2f}{base['IPC2']:7.2f}{base['narma10_nmse']:10.3f}")
for k, best in (("MC", max), ("IPC2", max), ("narma10_nmse", min)):
    b = best(rows, key=lambda r: r[k])
    print(f"  mejor {k}: N = {b['N']} ({b[k]:.3f})")

n = [r["N"] for r in rows]
fig, axes = plt.subplots(1, 4, figsize=(19, 4))
for ax, k, t, thr in zip(axes, ("MC", "IPC2", "narma10_nmse", "onset_V"),
                         ("capacidad de memoria", "capacidad de grado 2", "NARMA-10 NMSE (menor es mejor)", "onset (V)"),
                         (1, 1, 0.7, None)):
    ax.plot(n, [r[k] for r in rows], "o-")
    if thr is not None:
        ax.axhline(thr, color="r", ls="--", lw=0.8, label="umbral G1")
        ax.axhline(base[k], color="0.6", ls=":", label="entrada sola")
    ax.set(xlabel="número de hilos (por mm²)", title=t)
axes[0].legend(fontsize=8)
fig.suptitle(f"red uniforme vs número de hilos (semilla {rows[0]['semilla']})")
fig.tight_layout()
fig.savefig(out / "vs_N.png", dpi=130)
print(f"-> {out}/resumen.csv, {out}/vs_N.png")
EOF
echo "== listo $(date)"
