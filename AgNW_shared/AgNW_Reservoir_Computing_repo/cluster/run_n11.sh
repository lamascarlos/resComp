#!/bin/bash
#SBATCH --job-name=agnw_n11
#SBATCH --output=logs/n11_%j.log
#SBATCH --error=logs/n11_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
# One node, all its cores, on one phase of scripts/n11_fit.py (atomic .running markers on the shared /home, so several
# node jobs can work through the same phase). Submit from ~/AgNW/AgNW_Reservoir_Computing, e.g.
#   sbatch --account=colisionesNuevo --partition=colisionesNuevo --cpus-per-task=10 --mem=24G cluster/run_n11.sh stage1 grid
# vscale is sequential (each iteration needs the previous onset): run it on one node only.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "node $(hostname)  cores $SLURM_CPUS_PER_TASK  phase $1 $2 $3  $(date)"
~/miniforge3/envs/agnw/bin/python scripts/n11_fit.py "$1" "$2" $3 --procs "$SLURM_CPUS_PER_TASK"
echo "done $(date)"
