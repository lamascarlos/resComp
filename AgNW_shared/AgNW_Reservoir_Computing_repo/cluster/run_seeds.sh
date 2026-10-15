#!/bin/bash
#SBATCH --job-name=agnw_seeds
#SBATCH --output=logs/seeds_%j.log
#SBATCH --error=logs/seeds_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
# One node, all its cores, working through the shared N10-LS run list (scripts/n10l_seeds.py; atomic .running markers
# on the shared /home, so several node jobs never take the same run). Submit from ~/AgNW/AgNW_Reservoir_Computing, e.g.
#   sbatch --account=colisionesNuevo --partition=colisionesNuevo --cpus-per-task=10 --mem=24G cluster/run_seeds.sh "2.83:0.01,2.83:0"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "node $(hostname)  cores $SLURM_CPUS_PER_TASK  groups $1  $(date)"
~/miniforge3/envs/agnw/bin/python scripts/n10l_seeds.py run --procs "$SLURM_CPUS_PER_TASK" --only "$1"
echo "done $(date)"
