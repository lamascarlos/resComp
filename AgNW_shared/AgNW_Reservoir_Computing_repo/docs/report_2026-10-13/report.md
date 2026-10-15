# Larger networks and edge electrodes under junction noise

*Status report, 13 October 2026. Follows the report of 9 October (ten simulated devices per size; at a realistic
junction noise level 9 of 10 devices of 2.83 mm compute beyond their input; 4 mm passed on 3 devices). This report
completes 4 mm with 10 devices and compares the edge-electrode layout (split bands) with the pad array on the same
tests. Simulations of random Ag nanowire networks at the working density (2000 wires of 70 µm per mm$^2$), full
Kirchhoff circuit with wire resistance (AgNW_FDT_Memristor_Framework), N10 volatile junctions; 690 runs, most on the
CCT-Rosario HPC cluster.*

> **Legend — color code.** <span style="color:#1a7f37">**green = worked / positive**</span>, <span style="color:#cf222e">**red = failed / negative**</span>, <span style="color:#0969da">**blue = running or to be tested**</span>, <span style="color:#6e40c9">**violet = still to be determined**</span>.
>
> **Jargon in 30 seconds.** A *reservoir computer* uses a fixed physical network as memory and trains only a linear
> readout. **MC** = memory capacity (past inputs recalled); **degree-2** = nonlinear capacity; **NARMA-10** = a standard
> task, reported as error (lower is better; input alone 0.84). **G1** = "computes beyond the input": MC $\ge 1$,
> degree-2 $\ge 1$, NARMA $\le 0.70$, each with a 2-sd margin over all runs (pooled), or, per device, on the mean of
> its 2 replicates. A **rupture** is a random, complete break of a junction's volatile filament, at rate $r$ per
> junction; real junctions are estimated at $r \approx 0.005\text{–}0.03$ per second. **Capacity kept** $\rho$ = (MC +
> degree-2) of a run relative to the same run without ruptures.

## Answer

* <span style="color:#cf222e">**4 mm does not confirm.**</span> With 10 devices at $10^{-2}$ ruptures per second, **6 of 10** 4 mm devices pass G1
  and the pooled criterion fails. The 3-device pass of the last report was again luck of the first devices; by the
  rule fixed in advance the 4 mm claim is withdrawn.
* <span style="color:#6e40c9">**The pass rate does not depend on size.**</span> Over 1, 2, 2.83 and 4 mm, 8, 8, 9 and 6 of 10 devices pass
  (31 of 40). From 2 mm on, memory and nonlinear capacity pass in **every** device; the failures are all on
  **NARMA-10**, whose error sits just below the 0.70 threshold (0.56-0.66 on average) with a run-to-run spread of about
  0.1.
* <span style="color:#cf222e">**Edge electrodes (split bands) give a weaker reservoir than the pad array**</span>: about 30 % less capacity
  without noise, 15 % less at realistic noise, fewer devices passing G1 (5-7 of 10).
* <span style="color:#1a7f37">**But they are more robust to ruptures, and more so the larger the network**</span>: they keep 63 % of their
  capacity at 2.83 mm (pads 49 %) and almost never collapse (1 of 20 runs, pads 10 of 20).
* <span style="color:#cf222e">**Bias + small input does not help the edge layout**</span> under ruptures (all runs collapse at $10^{-2}$ per second).

## What was run

| test | layout | sides | rupture rate (per s per junction) | devices | runs |
|---|---|---|---|---|---|
| 4 mm completed | pads | 4 mm | 0 and $10^{-2}$ | 10 (7 new) | 84 |
| split bands, size | edge bands | 1, 2, 2.83 mm | 0 and $10^{-2}$ | 10 | 408 |
| split bands, rupture ladder | edge bands | 1 mm | $10^{-4}$, $10^{-3}$, $3 \times 10^{-3}$ | 3 | (in the above) |
| split bands, bias + small input | edge bands | 1 mm | 0, $10^{-3}$, $10^{-2}$ | 3 | 162 |
| input window at 1.5 $\times$ onset | both | 1 mm | 0 and $10^{-2}$ | 3 | 36 |

Pads: the $4 \times 4$ array of the previous reports (8 driven, 8 grounded, inside the network). Split bands: two edge
strips (5 % of the side) cut into 8 driven segments on the left and 8 grounded segments on the right. Readout = the 16
electrode currents in both. Each device: 2 independent streams of 800 inputs at 100 ms per input. **The analysis and the
decision rules were written down before the runs** (docs/PLAN_N10L_SEEDS.md, docs/PLAN_SPLIT_BANDS.md).

## 4 mm with 10 devices

![N10-LS](../../figures/fig31_n10l_seeds.png)

*Each point is one device (mean of its 2 runs). Grey: no ruptures; red: $10^{-2}$ ruptures per second. Filled = passes
G1, open = fails. Open points above both dashed lines fail on NARMA-10 only.*

| side | ruptures /s | memory capacity | degree-2 | NARMA-10 | pooled G1 | devices passing | capacity kept |
|---|---|---|---|---|---|---|---|
| 2.83 mm | $10^{-2}$ | $1.99 \pm 0.33$ | $3.09 \pm 0.55$ | 0.56 | <span style="color:#cf222e">no</span> | 9/10 | 0.49 |
| 4 mm | 0 | $3.54 \pm 0.43$ | $7.23 \pm 0.53$ | 0.45 | <span style="color:#1a7f37">yes</span> | 10/10 | |
| 4 mm | $10^{-2}$ | $1.94 \pm 0.26$ | $3.41 \pm 0.31$ | 0.66 | <span style="color:#cf222e">no</span> | 6/10 | 0.51 |

*Mean $\pm$ sd over the 20 runs. The 4 mm devices have the best memory and nonlinear margins of any size and the
smallest spread between devices; four of them miss NARMA-10 (0.76-0.90).*

## Split bands against pads

![N10-SB](../../figures/fig32_n10sb.png)

*MC + degree-2 per device (filled = passes G1) for pads (grey) and split bands (blue), without and with $10^{-2}$
ruptures per second. Right: capacity kept vs rupture rate at 1 mm (3 devices).*

| side | ruptures /s | MC + degree-2, bands vs pads | devices passing G1, bands / pads | capacity kept, bands / pads | collapsed runs, bands / pads |
|---|---|---|---|---|---|
| 1 mm | 0 | 7.5 vs 10.4 (<span style="color:#cf222e">worse</span>) | 7/10 / 10/10 | | |
| 1 mm | $10^{-2}$ | 3.7 vs 4.0 (on par) | 3/10 / 8/10 | 0.49 / 0.36 | 12 / 14 of 20 |
| 2 mm | $10^{-2}$ | 4.2 vs 4.9 (<span style="color:#cf222e">worse</span>) | 5/10 / 8/10 | <span style="color:#1a7f37">0.57 / 0.46</span> | 3 / 14 of 20 |
| 2.83 mm | 0 | 6.9 vs 10.7 (<span style="color:#cf222e">worse</span>) | 7/10 / 10/10 | | |
| 2.83 mm | $10^{-2}$ | 4.3 vs 5.1 (<span style="color:#cf222e">worse</span>) | 7/10 / 9/10 | <span style="color:#1a7f37">0.63 / 0.49</span> | 1 / 10 of 20 |

*"Worse" / "on par" / "more robust" by the 95 % bootstrap interval of the difference (rules fixed in advance).*

Why: between edge strips the current spreads over many long parallel paths, so a single rupture changes little; between
neighbouring pads the paths are few and short and one rupture can reroute much of the current. The same long paths give
the edge layout less memory and nonlinearity to start with, and the robustness does not make up for it.

**Bias + small input on split bands** (the operating point of the measured devices of Milano et al. 2025): without noise it
raises capacity by about 40 % (MC 3.2, degree-2 7.2), still well below the biased pads (5.8, 10.9); with ruptures every
run collapses, as with the pads.

**A side finding.** For the pads, an input window at 1.5 $\times$ the switching onset (beyond the two windows used so far)
gives more capacity, also with ruptures (3 devices: MC 2.05, degree-2 3.36, NARMA 0.61, all 3 passing). The pad results
so far may therefore be slightly conservative. For the split bands it changes nothing.

## What this means

* The size claim narrows: <span style="color:#6e40c9">**at a realistic junction noise level, about 3 of 4 simulated devices of 1-4 mm compute
  beyond their input, at about half their noise-free capacity**</span>. Size makes the devices more alike in capacity but does
  not raise the share that pass; NARMA-10 decides.
* <span style="color:#cf222e">Edge electrodes, the easiest to fabricate, cost capacity</span>; <span style="color:#1a7f37">their tolerance to ruptures is better and improves with
  size</span>, which matters for centimetre devices.
* Ruptures remain the limiting physics for both layouts.

## Next

1. <span style="color:#0969da">Alternating edge layouts (input and output segments interleaved along the edges): edge contacts with a shorter
   input-output distance, which might combine the robustness of the bands with the capacity of the pads (planned).</span>
2. <span style="color:#0969da">Re-check the pads at the 1.5 $\times$ onset window with 10 devices (cheap; could restore a pooled pass).</span>
3. <span style="color:#0969da">Manuscript: literature positioning, journal.</span>
4. <span style="color:#6e40c9">Measurements (docs/MEASUREMENT_REQUEST.md).</span>

## Names used in this report

| name | what it is | outcome |
|---|---|---|
| N10-LS | 10 devices per size (pads), 1-4 mm | <span style="color:#6e40c9">6-9/10 pass at $10^{-2}$ /s, no size trend</span> |
| N10-LS4 | the 4 mm part with 10 devices | <span style="color:#cf222e">6/10, pooled fails, claim withdrawn</span> |
| N10-SB | split bands vs pads (size, rupture ladder, bias, window) | <span style="color:#cf222e">less capacity</span>, <span style="color:#1a7f37">more robust from 2 mm</span> |
| N10-B | bias + small input (pads, earlier) | best noise-free, fragile |

*Full write-up: docs/RESULTS.md section 27 (N10-LS addendum, update 2026-10-13) and section 28 (N10-SB); plans
docs/PLAN_N10L_SEEDS.md, docs/PLAN_SPLIT_BANDS.md; scripts `scripts/n10l_seeds.py`, `scripts/n10sb_layout.py`; data
`results/n10l_seeds.json`, `results/n10sb.json`, `results/n10sb_bias.json`, `results/n10sb_win15.json`. Previous
reports: docs/report_2026-10-09/, docs/report_2026-10-07/, docs/report_2026-10-06/, docs/report_2026-09-29/.*

*The results presented in this work have been obtained by using the facilities of the CCT-Rosario Computational Center, member of the High Performance Computing National System (SNCAD, MincyT- Argentina).*
