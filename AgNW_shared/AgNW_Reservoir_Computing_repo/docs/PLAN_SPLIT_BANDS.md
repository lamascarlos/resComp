# Plan: the noise and size story for the split-band layout (N10-SB)

*Written 2026-10-09, before the runs; the analysis and decision rules below are fixed in advance.*

## Why

Every noise and size result so far (N10-E, N10-M/S/R/RN/K/OR/B/V, N10-L, N10-LS) uses one electrode layout: the
$4 \times 4$ checkerboard of pads inside the network, layout (a). The edge layout (b), the framework's two 5 % edge bands
cut into 8 driven segments (left) and 8 grounded segments (right), was only run in the N10-E0 pilot: one network, no noise
(memory capacity 1.97 vs 2.87 for the pads, degree-2 4.85 vs 5.45, NARMA-10 0.60 vs 0.49). Edge contacts are much easier
to fabricate, and the measured networks we calibrate against (Milano 2025) are contacted at the edges. Before comparing
new edge layouts (alternating input/output segments, "c" and "d"), (a) and (b) are put on equal footing (Juan,
2026-10-09).

## What

Layout `split_bands`, random fixed input weights, readout = the 16 electrode currents, protocol unchanged from N10-E /
N10-L / N10-LS ($\tau_s = 0.3$ s with spread 1, 100 ms per input, 800 inputs, 50 virtual nodes, windows $1.05$ and
$1.25 \times$ the onset with width $0.45 \times$ onset, 2 i.i.d. replicates + 1 sine/square per window and network). Larger
networks: the 1 mm device enlarged geometrically (bands stay 5 % of the side, so the segments grow with it; the pads
counterpart is `pads_scaled`). Onsets from an up-ramp at 0.5 V/s on network 1 (the N10-E0 pilot sweep; at 1 mm it must
reproduce 2.08 V). `scripts/n10sb_layout.py`.

| step | mirrors (pads) | side | rupture rate (/s per junction) | networks | runs |
|---|---|---|---|---|---|
| B1 noise-free | N10-E, N10-LS | 1 mm | 0 | 1-10 | 60 (3 from the pilot) |
| B2 rupture ladder | N10-R, N10-L tolerance | 1 mm | $10^{-4}$, $10^{-3}$, $3 \times 10^{-3}$, $10^{-2}$ | 1-3 ($10^{-2}$: 1-10) | 54 + 60 |
| B3 size | N10-L, N10-LS | 2, 2.83 mm | 0, $10^{-2}$ | 1-10 | 240 |
| B4 (if B2/B3 differ from the pads) | N10-RN, combined noise | | post hoc / 2.83 mm | | |
| B5 bias + small input | N10-B | 1 mm | 0, $10^{-3}$, $10^{-2}$ (bias 1.5 / 2 / 3 $\times$ onset, width 3 / 15 %) | 1-3 | 162 |

1 mm (B1, B2) runs on this machine; 2 and 2.83 mm (B3) on the HPC cluster after N10-LS4. Post hoc and free: online
retraining (N10-OR) on the B2 runs. Later, only after its window re-tuning on the pads: N10-K (backbone) on (b). Not
repeated: N10-M, N10-S, N10-V, the N10-R mitigations (layout-independent reasons for failure).

**Why N10-B on (b) has priority:** on the pads, bias + small input is the best noise-free reservoir but collapses under
ruptures because the conducting paths between neighbouring pads are few. Between edge bands conduction runs through many
long parallel paths, and Milano 2025 run exactly this configuration (edge electrodes, bias + small input).

## Analysis, fixed in advance

Identical to PLAN_N10L_SEEDS.md (same code paths: `agnw_rc.criteria.evaluate`, operating point chosen on validation
data): pooled G0-G2 over all runs of a group, networks passing G1 individually (Clopper-Pearson 95 %), spread between
networks of MC + degree-2 (bootstrap 95 %), capacity kept $\rho$ against the rupture-free twin (median, bootstrap 95 %),
collapses ($\rho < 0.5$), and for B2 the tolerance $r_{75}$ (median $\rho$ crosses 0.75, as `scripts/n10l_figures.py`).

**Decision rules (stated before the data), (b) against (a) at the same size and rupture rate (10 networks):**

* **On par**: the 95 % bootstrap interval of the difference of the network means of MC + degree-2, (b) $-$ (a), contains
  0. **Better / worse**: the interval lies entirely above / below 0.
* **Noise tolerance**: at $10^{-2}$ /s, (b) is more (less) robust if its median $\rho$ interval lies entirely above (below)
  the pads' interval; the number of networks passing G1 is reported for both.
* **Size**: the N10-LS claims are re-tested for (b) with the same rules (headline at 2.83 mm and $10^{-2}$ /s: pooled G1
  and $\ge 9/10$ networks; spread ratio 1 / 2.83 mm interval excludes 1; $\rho$ intervals overlap across sizes).
* Whatever the outcome, the alternating-edge layouts (c, d) come next (they differ from (b) in the input-output distance).

## Outputs

`scripts/n10sb_layout.py` (sweep / run; report to be added on the first data, same analysis as `n10l_seeds.py`),
`results/n10sb_sweep.json`, `results/n10sb.json`, a figure, RESULTS section 28 (N10-SB; reproduction becomes 29),
NEXT_STEPS, memory.
