#!/bin/bash
#SBATCH --job-name=agnw_n10sb
#SBATCH --output=logs/n10sb_%j.log
#SBATCH --error=logs/n10sb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
# One node, all its cores, working through the shared N10-SB run list (scripts/n10sb_layout.py, split-band layout;
# atomic .running markers on the shared /home). Submit from ~/AgNW/AgNW_Reservoir_Computing, e.g.
#   sbatch --account=colisionesNuevo --partition=colisionesNuevo --cpus-per-task=10 --mem=24G cluster/run_n10sb.sh "2.83:0.01,2.83:0"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "node $(hostname)  cores $SLURM_CPUS_PER_TASK  groups $1  $(date)"
~/miniforge3/envs/agnw/bin/python scripts/n10sb_layout.py run --procs "$SLURM_CPUS_PER_TASK" --only "$1"
echo "done $(date)"
