#!/bin/bash
#SBATCH --job-name=agnw_n11time
#SBATCH --output=logs/n11time_%j.log
#SBATCH --error=logs/n11time_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
# N11 timing on one node: ms per 2 ms step of the 15.6 mm strip (47 280 junctions), one process alone, then
# $SLURM_CPUS_PER_TASK processes at once (the load of a vscale round). Nothing is written to results/. Submit e.g.
#   sbatch --account=colisiones --partition=colisiones --cpus-per-task=4 --mem=12G cluster/time_n11.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/miniforge3/envs/agnw/bin/python
echo "node $(hostname)  cores $SLURM_CPUS_PER_TASK  $(lscpu | grep 'Model name')  $(date)"
read -r -d '' CODE <<'PYEOF'
import importlib.util, sys, time
s = importlib.util.spec_from_file_location("f", "scripts/n11_fit.py"); F = importlib.util.module_from_spec(s)
s.loader.exec_module(F)
st, mc = F.strip_circuit(1, 15.6, 0.5)
p = F.point(1, 40.0, setup=F.FULL, **F.REF); vp = F.vparams(p)
kd0 = F.junction_kd0(vp, len(st.network.junctions), 1)
a = F.run_constant(mc, vp, kd0, 3.6, 0.2)
t0 = time.time(); F.run_constant(mc, vp, kd0, 3.6, 300 * vp.dt, state=a["state"])
print(f"tag {sys.argv[1]} ms_per_step {(time.time() - t0) / 300 * 1e3:.1f}", flush=True)
PYEOF
$PY -c "$CODE" alone
for i in $(seq 1 "$SLURM_CPUS_PER_TASK"); do $PY -c "$CODE" "loaded$i" & done
wait
echo "done $(date)"
