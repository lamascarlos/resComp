# Running jobs on the CCT-Rosario cluster "Capitan"

*Written 2026-10-08, from the N10-LS setup (jobs 2421365-2421385); updated 2026-10-13 for N11 (jobs 2425222,
2425224: section 7a). Access details and rules are also in memory/cct-cluster.md.*

This tutorial covers the whole cycle: log in, set up the code once, test a node, submit simulation jobs, monitor them,
recover from failures and pull the results back for analysis on the laptop. The rule of thumb: **heavy runs go to the
cluster, analysis and reports stay on the laptop.**

## 1. What the cluster is

* Login node: `login-capitan.rosario-conicet.gov.ar`, user `<user>`. Scheduler: **SLURM**.
* The login node is an old AMD Opteron and is shared: **no simulations there**, only editing, `rsync`, `sbatch`,
  `squeue`. Rules: https://cluster.rosario-conicet.gov.ar/cluster_utilizacion.php (also printed in the login banner).
* `/home` is shared by all nodes (that is how several jobs share one run list); node-local scratch is `/local`.
* Quota: 100 GB (`miQuota`). Raw runs without junction states are small, but check before big campaigns.

Partitions available to us:

| partition | account | nodes | per node | limits |
|---|---|---|---|---|
| `colisionesNuevo` | `colisionesNuevo` | 4 | 8-10 cores, 32 GB (compute-6-32: i9-9900K, 62 GB) | group only, no time limit |
| `colisiones` | `colisiones` | 19 | 4 cores, 16 GB (i7-3770) | group only, no time limit |
| `eth_hi` (default) | `internos` | many | | 240 h, at most 20 % of the cores per user |
| `eth_low` | `internos` | many | | 48 h, preemptible |
| `ib100` | | | | at least 2 nodes (MPI); not useful here |

The two group partitions are shared with the rest of the group: look at what is free (`sinfo`, section 5) before sizing
a campaign. At most **100 queued jobs per user**, so the pattern is **one job per node, running several simulations in
parallel** (one process per core), not one job per simulation.

Speed (2.83 mm network, 2 ms step): 59 ms per step alone on compute-6-32, ~127 ms with 8 concurrent processes; 79 ms
alone and ~110 ms with 4 concurrent on a `colisiones` node. Per core this equals the laptop, but 76 cores on 15 nodes
give ~10-20x the laptop's throughput under load.

## 2. Logging in

```bash
ssh -i ~/.ssh/id_ed25519 <user>@login-capitan.rosario-conicet.gov.ar
```

The key works without a password, so commands can also be run in one shot from the laptop:

```bash
ssh -i ~/.ssh/id_ed25519 <user>@login-capitan.rosario-conicet.gov.ar 'squeue -u $USER'
```

The module messages printed on login (`Loading compiler version 2022.0.2 ...`) are harmless. Optional shortcut in
`~/.ssh/config` on the laptop:

```
Host capitan
    HostName login-capitan.rosario-conicet.gov.ar
    User <user>
    IdentityFile ~/.ssh/id_ed25519
```

after which `ssh capitan` and `rsync ... capitan:...` work. The commands below use the long form.

## 3. One-time setup (already done on 2026-10-08)

Layout on the cluster (same as on the laptop, relative):

```
~/AgNW/AgNW_FDT_Memristor_Framework/   framework at commit f6b6951 (code + .git + 2 reference folders)
~/AgNW/AgNW_Reservoir_Computing/       this project, without .venv and results/raw
~/miniforge3/envs/agnw/                Python 3.14.7, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1
```

To redo it (for example after a cluster reinstall), from the laptop, in the parent folder of both projects:

```bash
H=<user>@login-capitan.rosario-conicet.gov.ar; S="ssh -i $HOME/.ssh/id_ed25519"
$S $H 'mkdir -p ~/AgNW'
# framework: code + .git + the reference files some tests need
rsync -a -e "$S" --exclude .venv --exclude '__pycache__' --exclude '*.egg-info' \
  AgNW_FDT_Memristor_Framework/ $H:AgNW/AgNW_FDT_Memristor_Framework/
# project: everything except the virtual env and the raw runs
rsync -a -e "$S" --exclude .venv --exclude 'results/raw/' --exclude '__pycache__' --exclude '*.egg-info' \
  AgNW_Reservoir_Computing/ $H:AgNW/AgNW_Reservoir_Computing/
```

(The framework copy used in 2026-10 is only its code, `.git`, `validation_output/lamas_fig4` and `docs/theory_audit`,
52 MB; exclude its large output folders if the full copy is too big.)

Then on the cluster (login node; installing is light work):

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda create -y -n agnw python=3.14.7
PY=~/miniforge3/envs/agnw/bin/python
cd ~/AgNW/AgNW_Reservoir_Computing
$PY -m pip install -e ../AgNW_FDT_Memristor_Framework
$PY -m pip install -e ".[dev]"
mkdir -p logs
```

Pin the package versions to the laptop's (section 3 of RESUME_FROM_HERE.md) if pip picks newer ones. Do **not** run the
test suite on the login node: run it inside a job (next section).

## 4. Test job: is the node working, how fast is it?

`cluster/test_job.sh` prints the node's CPU and memory, benchmarks one and eight concurrent 2.83 mm networks
(`scripts/bench_step.py`) and runs the whole test suite. Submit it from the project folder on the cluster:

```bash
cd ~/AgNW/AgNW_Reservoir_Computing
sbatch cluster/test_job.sh                 # -> "Submitted batch job 2421365"
cat logs/test_2421365.log                  # when finished (~3 min)
```

All tests must pass before trusting cluster results. To benchmark a `colisiones` node instead, override the header
options on the command line (they win over the `#SBATCH` lines):

```bash
sbatch --account=colisiones --partition=colisiones --cpus-per-task=4 cluster/test_job.sh
```

**N11 timing job** (2026-10-13): `cluster/time_n11.sh` times the 15.6 mm strip of N11 (47 280 junctions), first alone
and then with `$SLURM_CPUS_PER_TASK` copies at once, and writes nothing to `results/`:

```bash
sbatch --account=colisiones --partition=colisiones --cpus-per-task=4 --mem=12G cluster/time_n11.sh
```

Measured on colisiones compute-6-3 (Intel i7-3770, job 2425222): 36 ms per step alone, 52-55 ms with 4 at once. The
laptop needs 19.6 ms alone, so a colisiones core costs about 2.7 times the laptop budget under load.

## 5. Before submitting: what is free?

```bash
sinfo -p colisiones,colisionesNuevo                       # idle / mix / alloc nodes per partition
sinfo -p colisiones,colisionesNuevo -N -o "%N %T %C %m"   # per node: state, CPUs (alloc/idle/other/total), memory
squeue -p colisiones,colisionesNuevo                      # who else is running there
```

Size the campaign to the idle nodes. A job asking for more cores or memory than a node has stays pending forever
(`(Resources)` or `(ReqNodeNotAvail)` in `squeue`).

## 6. Anatomy of a job script

`cluster/run_seeds.sh` is the model for every campaign:

```bash
#!/bin/bash
#SBATCH --job-name=agnw_seeds
#SBATCH --output=logs/seeds_%j.log        # %j = job id; the folder logs/ must exist
#SBATCH --error=logs/seeds_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1                        # one Python process ...
#SBATCH --time=4-00:00:00
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # ... with one BLAS thread per worker
echo "node $(hostname)  cores $SLURM_CPUS_PER_TASK  groups $1  $(date)"
~/miniforge3/envs/agnw/bin/python scripts/n10l_seeds.py run --procs "$SLURM_CPUS_PER_TASK" --only "$1"
echo "done $(date)"
```

Points that matter:

* **One task, many cores**: `--cpus-per-task` is set at submission and the script passes `$SLURM_CPUS_PER_TASK` to the
  Python process pool, so the number of workers always equals the cores SLURM gave the job.
* **One BLAS thread per process** (project rule): without it each worker spawns a thread per core and the node is
  overloaded.
* **Account and partition are not in the header**, so the same script works on both group partitions; give them on the
  `sbatch` line.
* Use the full path of the env's Python (no `conda activate` needed in batch jobs).
* Jobs start in the folder `sbatch` was called from: always submit from `~/AgNW/AgNW_Reservoir_Computing`.
* `--mem` must fit the node (24G on `colisionesNuevo`, 12G on `colisiones` worked for 2.83 mm networks).

### Several jobs on one run list: atomic claims

Many node jobs work through the **same** list of runs. They must not compute the same run twice. `n10l_seeds.run_one`
handles this with a marker file on the shared `/home`:

1. if `results/raw/<run>.npz` exists, skip it;
2. otherwise create `results/raw/<run>.running` with `os.open(..., O_CREAT | O_EXCL)`; this is atomic, so exactly one
   job wins and the others skip the run ("running elsewhere");
3. the winner writes its host name and PID into the marker, runs, saves the `.npz`, deletes the marker.

`scripts/n11_fit.py` does the same (function `claim`) and adds two things. A marker older than 24 h is taken over (it
belongs to a killed job, since the longest N11 run is about 4 h). Each job also shuffles its run list, seeded by
`SLURM_JOB_ID`, so jobs that start together do not race for the same first runs.

**Only `scripts/n10l_seeds.py` and `scripts/n11_fit.py` do this so far.** The other runners (`scripts/run_sims.py`, `n10r_noise.py`, ...) only
skip runs whose `.npz` already exists, so two cluster jobs running the same list at the same time will duplicate work.
For those, either submit one job only, split the list by hand (`--limit` / `--reverse` in `run_sims.py`), or copy the
claim logic of `n10l_seeds.run_one` into the script first (preferred for any new campaign).

## 7. Submitting a campaign

Example: the N10-LS campaign of 2026-10-08 (348 runs, 15 jobs, 76 cores). `G` lists the groups `side:rate[:sigma]` in
priority order:

```bash
cd ~/AgNW/AgNW_Reservoir_Computing
G="2.83:0.01,2.83:0,2:0.01,2:0,1:0.01,1:0,2.83:0.01:0.03,4:0.01,4:0"
# the 3 big nodes
for c in 10 10 8; do
  sbatch --account=colisionesNuevo --partition=colisionesNuevo --cpus-per-task=$c --mem=24G cluster/run_seeds.sh "$G"
done
# 12 small nodes
for i in $(seq 12); do
  sbatch --account=colisiones --partition=colisiones --cpus-per-task=4 --mem=12G cluster/run_seeds.sh "$G"
done
```

Every job gets the full list and takes the next unclaimed run whenever a worker is free, so fast and slow nodes balance
themselves and the high-priority groups finish first. Write down the job ids (and the submission time) in
docs/NEXT_STEPS.md.

For a **new experiment**: write its runner with a `run --procs N` stage and atomic claims, copy `cluster/run_seeds.sh`
to `cluster/run_<name>.sh` changing the job name, log names and the Python line, sync the code (section 10), run one
small job first, then submit the rest.

## 8. Monitoring

From the laptop, in one command (this is the check in RESUME_FROM_HERE.md):

```bash
ssh -i ~/.ssh/id_ed25519 <user>@login-capitan.rosario-conicet.gov.ar \
  'cd ~/AgNW/AgNW_Reservoir_Computing; squeue -u $USER; echo done $(ls results/raw/*.npz | wc -l) of 348; \
   echo claimed $(ls results/raw/*.running 2>/dev/null | wc -l); tail -n 2 logs/seeds_*.log | grep -c done'
```

* `squeue -u $USER`: `R` running (with node and elapsed time), `PD` pending with the reason (`Resources` = waiting for
  a free node; `Priority` = others first). The `TIME` column counts from the job's start, not its submission.
* `done N of 348`: finished runs (`.npz`). `claimed`: runs in progress right now (should equal the number of busy cores).
* last number: jobs whose log ends with `done` (finished their part).
* `grep -c` returns exit code 1 when it counts 0; that is not an error.

More detail:

```bash
tail -f logs/seeds_2421373.log         # progress of one job: "[i/n] run name"
cat logs/seeds_2421373.err             # Python errors land here
scontrol show job 2421376              # full job info (reason, node, requested resources)
sacct -j 2421373 --format=JobID,State,Elapsed,MaxRSS,NodeList    # after the job ended: state and memory used
ssh compute-6-33 top -b -n1 | head -20 # load of a node running your job (untested: works only if SSH to nodes is allowed)
```

`MaxRSS` from `sacct` tells you whether `--mem` can be lowered (more jobs fit) or must be raised.

## 7a. The N11 campaign (junction calibration, 2026-10-13 on)

`cluster/run_n11.sh <stage> <phase> [--refine]` runs one phase of `scripts/n11_fit.py` on one node with all its cores.
Always pass `--cpus-per-task`, since the script hands it to `--procs`. The order is fixed (docs/NEXT_STEPS.md START
HERE), and each step needs Juan's go:

```bash
cd ~/AgNW/AgNW_Reservoir_Computing
P="--account=colisiones --partition=colisiones --cpus-per-task=4 --mem=14G"   # or colisionesNuevo, 8-10 cores, 24G
sbatch $P cluster/run_n11.sh stage1 vscale      # ONE node only: each round needs the previous onset (job 2425224)
# after vscale converged (results/n11_stage1_vscale.json): one job per free node, same command on each
sbatch $P cluster/run_n11.sh stage1 grid
sbatch $P cluster/run_n11.sh stage1 report       # quick; refine and checks refresh the report themselves
sbatch $P cluster/run_n11.sh stage1 refine       # several nodes; reuses the recorded refine grid (--recenter: new one)
sbatch $P cluster/run_n11.sh stage1 checks       # several nodes
sbatch $P cluster/run_n11.sh stage1 report
# stage 2, after reviewing stage 1 with Juan
sbatch $P cluster/run_n11.sh stage2 plan
sbatch $P cluster/run_n11.sh stage2 run          # several nodes; noise runs wait for the base states
sbatch $P cluster/run_n11.sh stage2 report
sbatch $P cluster/run_n11.sh stage2 refine
sbatch $P cluster/run_n11.sh stage2 run --refine
sbatch $P cluster/run_n11.sh stage2 report
```

What the phases write on the cluster:
* raw runs in `results/raw/n11/`;
* state files in `results/`: `n11_stage1_vscale.json`, `n11_stage1_refines.json`, `n11_fit_stage1.json`,
  `n11_stage2_plan.json`, `n11_fit_stage2.json`;
* figures 34 and 35.

These state files live on the cluster: later phases read them there. Check progress:

```bash
squeue -u $USER; tail logs/n11_<jobid>.log
cat results/n11_stage1_vscale.json      # vscale: one entry per round (onset, next v_scale)
ls results/raw/n11/*.npz | wc -l; ls results/raw/n11/*.running
```

## 9. When things go wrong

| symptom | cause | fix |
|---|---|---|
| job pending for hours with `(Resources)` | group nodes busy, or it asks more cores / memory than any free node | wait, or `scancel` and resubmit smaller |
| state `OUT_OF_MEMORY` in `sacct`, `oom-kill` in `.err` | `--mem` too small for that many workers | resubmit with more `--mem` or fewer `--cpus-per-task` |
| `.err` shows a Python traceback | code / env problem | fix, sync, test with one small job |
| `squeue` empty but runs missing, `.running` files left | jobs killed (node failure, `scancel`, time limit) | remove stale claims, resubmit (below) |
| node load far above its cores | missing `OMP_NUM_THREADS=1` | add the three `export`s |

**Stale claims.** A killed job leaves its `.running` markers behind and those runs are then skipped forever. Remove them
**only when no job of yours is running** (otherwise you delete live claims and runs get computed twice):

```bash
squeue -u $USER                                   # must be empty
rm ~/AgNW/AgNW_Reservoir_Computing/results/raw/*.running
```

then resubmit exactly as in section 7: finished runs are skipped, so resubmitting is always safe. For N11:
`~/miniforge3/envs/agnw/bin/python scripts/n11_fit.py stage2 clean-markers` removes the markers in
`results/raw/n11/`, again only when no job is running. Without it, markers older than 24 h are taken over
automatically. A `vscale` job that finds a run held by another job stops with a message instead of waiting: run vscale
on one node only.

Cancel jobs:

```bash
scancel 2421376                  # one job
scancel -u $USER -n agnw_seeds   # all your jobs with that name
```

A cancelled job stops mid-run; its `.tmp.npz` and `.running` files are left behind (see stale claims).

## 10. Moving code and results

**Code changes go laptop to cluster**, never the reverse (the laptop copy is the reference). Sync only when no job is
running the files you change (Python reads the modules at start, but workers start later):

```bash
rsync -a -e "ssh -i $HOME/.ssh/id_ed25519" \
  --exclude .venv --exclude 'results/raw/' --exclude 'results/metrics.json' --exclude logs/ \
  --exclude '__pycache__' --exclude '*.egg-info' \
  ./ <user>@login-capitan.rosario-conicet.gov.ar:AgNW/AgNW_Reservoir_Computing/
```

(run from the project folder; `logs/` is excluded so cluster job logs are not overwritten). The framework never changes
(commit f6b6951), so it does not need syncing.

**During N11, also add** `--exclude 'results/n11_stage*' --exclude 'results/n11_fit_stage*'` to that push. The cluster's
N11 state files (section 7a) are newer than any laptop copy, and a push must not overwrite them.

**Results go cluster to laptop**, finished runs only:

```bash
rsync -a --ignore-existing --exclude '*.running' --exclude '*.tmp.npz' -e "ssh -i $HOME/.ssh/id_ed25519" \
  <user>@login-capitan.rosario-conicet.gov.ar:AgNW/AgNW_Reservoir_Computing/results/raw/ results/raw/
```

then analyse locally (for N10-LS: `.venv/bin/python scripts/n10l_seeds.py report`). For N11, pull
`results/raw/n11/` as above and the state files with
`rsync -a -e "ssh -i $HOME/.ssh/id_ed25519" '<user>@login-capitan.rosario-conicet.gov.ar:AgNW/AgNW_Reservoir_Computing/results/n11_*' results/`.
The N11 reports can then be rerun locally (`scripts/n11_fit.py stage1 report`), giving the same json and the figures. Copy the job logs too if they will
be cited (`rsync ... :AgNW/AgNW_Reservoir_Computing/logs/seeds_* logs/cluster/`). Reports and figures are made on the
laptop, never on the cluster.

## 11. Checklist for a new campaign

1. Runner script has a `run --procs N` stage, skips existing `.npz` and claims runs atomically.
2. Code synced; laptop and cluster at the same state.
3. `sinfo`: how many nodes are free in each group partition.
4. One small job first (e.g. one group, a few runs); check its log, `.err` and `sacct` memory.
5. Submit the rest; record job ids and time in docs/NEXT_STEPS.md (and the check command in RESUME_FROM_HERE.md).
6. Monitor (section 8); on failure clean stale claims and resubmit (section 9).
7. Pull results, analyse on the laptop, update RESULTS / NEXT_STEPS / memory.
8. Publications using cluster results must carry the acknowledgment:
   "The results presented in this work have been obtained by using the facilities of the CCT-Rosario Computational
   Center, member of the High Performance Computing National System (SNCAD, MincyT- Argentina)."
