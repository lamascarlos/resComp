#!/bin/bash
# Tu geometría de clusters con barrido de fondo (scripts/n12_mi_geometria_fondo.py) en una PC con Ubuntu: clona o
# actualiza el repo, arma el entorno (una sola vez) y corre con todos los núcleos menos uno. Los argumentos se pasan
# tal cual al script de Python:
#
#   bash run_mi_geometria_local.sh                                         # 3000 hilos, fondos 0.4,0.6,0.8,0.9, 5 clusters
#   bash run_mi_geometria_local.sh --n-wires 2000 --fondos 0.2,0.5,0.8
#   bash run_mi_geometria_local.sh --clusters "0.3,0.5,0.1;0.7,0.5,0.1"
#   nohup bash run_mi_geometria_local.sh > mi_geometria.log 2>&1 &         # sigue aunque cierres la terminal
#
# Resultados en ~/rescomp/AgNW_shared/AgNW_Reservoir_Computing_repo/results/n12_mi_geometria/<N..._seed..._..clusters>/
set -euo pipefail
BRANCH="${BRANCH:-feature/geometria-substrato}"
DIR="${DIR:-$HOME/rescomp}"
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
echo "== $(date)  $(hostname)  $(nproc) núcleos"
.venv/bin/python scripts/n12_mi_geometria_fondo.py "$@"
echo "== listo $(date)"
