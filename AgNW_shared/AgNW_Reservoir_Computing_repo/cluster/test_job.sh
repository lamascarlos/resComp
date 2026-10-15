#!/bin/bash
#SBATCH --job-name=agnw_test
#SBATCH --output=logs/test_%j.log
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --account=colisionesNuevo
#SBATCH --partition=colisionesNuevo
# Cluster test (2026-10-08): node info, step benchmark (1 and 8 concurrent 2.83 mm networks), full test suite.
# Submit from ~/AgNW/AgNW_Reservoir_Computing:  sbatch cluster/test_job.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/miniforge3/envs/agnw/bin/python
echo "node $(hostname)  $(date)"; lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2
echo "== benchmark, 1 process"
$PY scripts/bench_step.py 1.0 0.45
$PY scripts/bench_step.py 2.83 1.40
echo "== benchmark, 8 concurrent processes (2.83 mm)"
for i in $(seq 8); do $PY scripts/bench_step.py 2.83 1.40 > /tmp/bench_$SLURM_JOB_ID_$i.txt 2>&1 & done; wait
cat /tmp/bench_$SLURM_JOB_ID_*.txt | grep side; rm -f /tmp/bench_$SLURM_JOB_ID_*.txt
echo "== tests"
$PY -m pytest -q -p no:cacheprovider 2>&1 | tail -4
echo "done $(date)"
