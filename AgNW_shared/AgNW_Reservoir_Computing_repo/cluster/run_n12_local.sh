#!/bin/bash
# N12 en una PC con Ubuntu (sin SLURM): clona o actualiza el repo, arma el entorno, corre las geometrías pedidas con
# todos los núcleos y, al terminar, hace el report y sube results/n12_criteria.json + la figura a la rama.
#
#   bash run_n12_local.sh                                   # las geometrías de clusters (por defecto)
#   bash run_n12_local.sh clusters4,clusters_line           # otras
#   PROCS=6 bash run_n12_local.sh                           # limitar los núcleos (por defecto: todos menos 1)
#
# Se puede cortar y volver a lanzar: las corridas ya guardadas en results/raw/ no se repiten.
# Para que siga aunque cierres la terminal o la sesión SSH:
#   nohup bash run_n12_local.sh > n12.log 2>&1 &        (y mirar el avance con: tail -f n12.log)
set -euo pipefail

GEOMS="${1:-clusters4,clusters4_dense,clusters_line}"
BRANCH="${BRANCH:-feature/geometria-substrato}"
DIR="${DIR:-$HOME/rescomp}"
PROCS="${PROCS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # una hebra por corrida: el paralelismo es entre corridas

echo "== $(date)  máquina $(hostname)  núcleos $(nproc), uso $PROCS  geometrías $GEOMS"

# 1. repo
if [ -d "$DIR/.git" ]; then
    git -C "$DIR" fetch -q origin "$BRANCH" && git -C "$DIR" checkout -q "$BRANCH" && git -C "$DIR" pull -q --ff-only origin "$BRANCH"
else
    git clone -q -b "$BRANCH" https://github.com/lamascarlos/rescomp.git "$DIR"
fi
cd "$DIR/AgNW_shared/AgNW_Reservoir_Computing_repo"

# 2. entorno (una sola vez; necesita python3-venv: sudo apt install python3-venv)
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -e ../AgNW_FDT_Memristor_Framework -e ".[dev]"
fi
PY=.venv/bin/python
$PY -m pytest -q tests/test_substrate.py

# 3. corridas (onsets ya calculados en results/n12_sweep.json; sweep solo calcula los que falten)
$PY scripts/n12_geometry.py sweep --only "$GEOMS" --procs "$PROCS"
$PY scripts/n12_geometry.py run --only "$GEOMS" --procs "$PROCS"

# 4. report y subida de los resultados (no de los archivos crudos, que quedan en results/raw/)
git pull -q --ff-only origin "$BRANCH" || true          # trae lo que otra máquina haya reportado
$PY scripts/n12_geometry.py report --only "$GEOMS" --procs "$PROCS"
git add results/n12_criteria.json results/n12_sweep.json figures/fig37_n12_scores.png
if git commit -q -m "N12: resultados de $GEOMS ($(hostname))"; then
    git push -q origin "$BRANCH" && echo "resultados subidos a $BRANCH" \
        || echo "no se pudo hacer push (¿credenciales de GitHub?): los resultados quedan en results/n12_criteria.json"
fi
echo "== listo $(date)"
