# Plan: more networks for the size result (N10-LS), weekend of 10-11 October 2026

*Written 2026-10-08, before the runs; the analysis below is fixed in advance and will be reported as planned.*

## Why

The headline result of N10-L (docs/RESULTS.md section 27) is that a 2.83 mm network meets G1 at $10^{-2}$ ruptures per
second per junction, the range inferred for real junctions, and that larger networks vary less from device to device.
Each point rests on **3 random networks** (network seeds 1-3, i.e. 3 independently drawn devices) $\times$ 2 replicates.
G1 requires mean $- 2$ sd $\ge$ threshold over the 6 runs at the validation-chosen window; with 3 devices the sd itself is
poorly known (a sample sd from 3 values can easily be off by a factor 2). The claim "size reduces the spread" is a
statement about these sds. A referee will ask for more devices.

## What

Seven more networks (seeds 4-10, so 10 per point) at the three sizes that carry the claim, at the noise-free and the
realistic rupture rate, with the unchanged N10-L protocol (16 pads with random input weights, 16 electrode currents,
$\tau_s = 0.3$ s with spread 1, 100 ms per input, 800 inputs, windows $1.05$ and $1.25 \times$ onset with width
$0.45 \times$ onset, 2 i.i.d. replicates + 1 sine/square run per window):

| side | layout | rupture rate (/s per junction) | new seeds | runs |
|---|---|---|---|---|
| 1 mm | `pads` (the N10-E / N10-R runs) | 0, $10^{-2}$ | 4-10 | 84 |
| 2 mm | `pads_scaled`, side 2 | 0, $10^{-2}$ | 4-10 | 84 |
| 2.83 mm | `pads_scaled`, side 2.83 | 0, $10^{-2}$ | 4-10 | 84 |
| | | | **total** | **252** |

Junction states are not stored (default since 2026-10-08): about 5 MB per run, 1.3 GB in all.

## Cost and schedule (this machine, 15 of 20 cores, nice 19)

Run times measured under the same load: 1 mm $\approx$ 0.2 h, 2 mm $\approx$ 1.5 h, 2.83 mm $\approx$ 4.9 h per run, i.e.
$84 \times (0.2 + 1.5 + 4.9) \approx 555$ core-hours, **about 37 h** on 15 cores. Memory: 2.83 mm processes need
$\approx 0.33$ GB each without stored junction states ($\approx 5$ GB for 15).

* Start: Friday 9 October, 18:00 (or when Juan says); expected end: Sunday 11 October, around 07:00.
* Queue order (so that a partial run is still useful): 2.83 mm at $10^{-2}$, 2.83 mm at 0, 2 mm at $10^{-2}$, 2 mm at 0,
  1 mm at both rates.
* Optional: ultron (12 cores) on the 1 mm and 2 mm share would cut the wall time to about 26 h.
* Resumable: completed runs are skipped on restart; `.running` markers prevent duplicates.

## Analysis, fixed in advance

For each size and rate, with all 10 networks (seeds 1-10; 20 i.i.d. runs + 10 sine/square at each window):

1. **Pooled criteria** (unchanged code, `agnw_rc.criteria.evaluate`): operating point chosen on validation data, G0, G1
   (mean $- 2$ sd over the 20 runs) and G2 loose / strict.
2. **Devices that pass.** A network passes G1 if, at the group's operating point, the mean of its 2 replicates has
   MC $\ge 1$, degree-2 $\ge 1$, NARMA-10 $\le 0.70$ and at least 2 memory lags above the noise floor. Reported as
   $k$ of 10, with a 95 % Clopper-Pearson interval.
3. **Spread between devices.** sd over the 10 networks of the per-network mean of MC + degree-2, with a 95 % bootstrap
   interval (10 000 resamples of networks); and the ratio of the 1 mm to the 2.83 mm sd with its bootstrap interval.
4. **Capacity kept.** Median over the 20 runs of $\rho = (\mathrm{MC} + C_2)/(\mathrm{MC} + C_2)_{r=0}$ (twin = same network,
   replicate and window without ruptures), with a bootstrap interval, and the number of collapses ($\rho < 0.5$).
5. The noise floor of the capacities is recomputed for the input streams of all 10 networks (the streams depend on the
   network seed); the floor used is the maximum over the networks, as before.

**Decision rules (stated before the data):**

* The headline "a 2.83 mm network computes beyond its input at $10^{-2}$ ruptures /s" stands if the pooled G1 passes at
  2.83 mm and $10^{-2}$ /s **and** at least 9 of 10 networks pass individually. If the pooled G1 fails but most networks
  pass, it is reported as "most devices" with the fraction; if fewer than 7 of 10 pass, the claim is withdrawn.
* "Size reduces the spread between devices" stands if the bootstrap interval of the 1 mm / 2.83 mm sd ratio excludes 1.
* "Realistic noise costs about half the capacity at every size" stands if the 95 % intervals of the median $\rho$ at
  $10^{-2}$ /s overlap across the three sizes.

## Weekend extension (Juan, 2026-10-08: both machines free Friday night to Monday night)

With about 72 h on this machine and ultron, two more groups are added, in priority order after the seeds:

| priority | group | runs | why |
|---|---|---|---|
| 1 (must) | networks 4-10 at 1, 2, 2.83 mm, 0 and $10^{-2}$ /s (above) | 252 | statistics of the headline |
| 2 (should) | 2.83 mm, ruptures $10^{-2}$ /s **plus** Gaussian junction noise $\sigma = 0.03$, networks 1-10; scored also with the measured readout noise (N10-RN, $\times 1$) | 60 | all three noise sources at once |
| 3 (nice) | 4 mm (onset swept beforehand), 0 and $10^{-2}$ /s, networks 1-3 | 36 | one more size |

Split (ultron about 1.5 $\times$ slower per run):

| machine | queue | estimate |
|---|---|---|
| this machine, 18 procs (12 for 4 mm, memory) | 2.83 mm seeds $\to$ combined noise $\to$ 4 mm at $10^{-2}$ | $\approx 23 + 17 + 20 = 60$ h |
| ultron, 12 procs | 1 mm and 2 mm seeds $\to$ 4 mm at 0 | $\approx 18 + 30 = 48$ h |

Start Friday 2026-10-09 21:00 $\to$ expected end Monday morning, about 12 h before the machines are needed again.
Launch: `bash logs/weekend_launch.sh "2026-10-09 21:00"` (waits until then; syncs the code to ultron, starts both queues,
pulls ultron's runs when it is done and writes the report). If time runs short, priority 3 is dropped first.

Additional analysis for priority 2: pooled G1 and networks passing, with and without the measured readout noise, and
$\rho$ against the rupture- and noise-free twin. The 4 mm groups are reported like the other sizes (3 networks, so the
spread there is indicative only).

## Update 2026-10-08: everything runs on the HPC cluster

The CCT-Rosario cluster (memory/cct-cluster.md) has 76 cores in Juan's group partitions, as fast per process as this
machine. The whole plan (348 runs, ~390 core-hours) was submitted on 2026-10-08 at 11:58 as 15 node jobs
(`cluster/run_seeds.sh`) and should take about 5-6 h; the weekend split above is no longer needed. Analysis unchanged.

## Update 2026-10-09: 4 mm with 10 networks (N10-LS4)

N10-LS gave a pooled G1 at 4 mm on 3 networks, while at 2.83 mm 3 networks had overstated the margin. Networks 4-10 at
4 mm (0 and $10^{-2}$ /s, 84 runs) were submitted to the cluster on 2026-10-09 (jobs 2422356-2422370). Analysis as above;
rule 1 applied to 4 mm: "a 4 mm network computes beyond its input at $10^{-2}$ /s" stands with pooled G1 and $\ge 9/10$
networks, "most devices" if the pooled G1 fails, withdrawn below 7/10.

## Outputs

* `scripts/n10l_seeds.py` (run / report), `results/n10l_seeds.json`, `figures/fig31_n10l_seeds.png` (per-network MC vs
  degree-2 with the G1 thresholds, per size and rate; spread and $\rho$ with intervals).
* docs/RESULTS.md section 27 addendum, a dated report (md / pdf / html), README, MANUAL, NEXT_STEPS, memory.
