#!/bin/bash
#SBATCH --job-name=agnw_pw
#SBATCH --output=logs/pw_%j.log
#SBATCH --error=logs/pw_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
# One node, all its cores, working through the shared N10-PW run list (scripts/n10pw_windows.py: pads at 1.5 / 1.75 x
# onset; atomic .running markers on the shared /home, so several node jobs never take the same run). Submit from
# ~/AgNW/AgNW_Reservoir_Computing, e.g.
#   sbatch --account=colisiones --partition=colisiones --cpus-per-task=4 --mem=12G cluster/run_pw.sh ""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "node $(hostname)  cores $SLURM_CPUS_PER_TASK  groups ${1:-all}  $(date)"
~/miniforge3/envs/agnw/bin/python scripts/n10pw_windows.py run --procs "$SLURM_CPUS_PER_TASK" --only "$1"
echo "done $(date)"
