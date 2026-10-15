# AgNW reservoir computing: where we stand (13 Oct 2026)

**Question.** Can a random Ag nanowire network (framework geometry and Kirchhoff circuit, unchanged) work as a reservoir
computer? We design junction model, electrodes, input and readout. **G1** = computes beyond its input (MC $\ge 1$,
degree-2 $\ge 1$, NARMA-10 $\le 0.70$; input alone 0.80); **G2** = $\ge 50\%$ of an equal-size ESN. Operating point chosen on
validation data; later decision rules fixed before the runs.

## What we did

* **A. Framework as is** (25-28 Sep): binary junction, one electrode pair (E1-E3); noise removed (N1, N2); averaging,
  multi-electrode (N5, N7); Fig. 4 filament junction (N4), slow inputs (E4).
* **B. New junction** (28 Sep-1 Oct): N10 volatile junction from measured AgNW data; 32 probes (N10-1); 16 electrodes
  (N10-E); memory limit (N10-M); settings (N10-S).
* **C. Realistic noise** (2-6 Oct): ruptures (N10-R); rate from Milano 2025 data (N10-C); readout noise (N10-RN);
  remedies: backbone (N10-K), retraining (N10-OR), bias + small input (N10-B), matched voltage scale (N10-V).
* **D. Size and layout** (7-13 Oct): 1-4 mm, 10 networks per size, HPC cluster (N10-L, N10-LS); edge bands vs pads (N10-SB).

## What we found

1. **Framework junctions: no reservoir.** Scores at input-only level (MC $\le 0.26$): switching noise (binary) or a frozen
   hold band (Fig. 4 filament). No fix helped.
2. **N10 junction: a weak reservoir.** 16 pads: MC 3.5, degree-2 6.7, NARMA-10 0.40, sine/square 1.00 $\to$ G1, G2 loose
   only. Memory capped at about 4 inputs. Insensitive to our chosen settings.
3. **Ruptures break it.** Tolerance about $10^{-4}$ /s per junction (1 mm); failures are all-or-nothing collapses (current
   rerouting). Readout noise harmless. No remedy restores G1; bias + small input best without noise (MC 5.8) but most fragile.
4. **Real junctions are noisier.** Milano 2025 data imply $r^\ast \approx 0.005\text{–}0.03$ /s per junction, 50-300 times the tolerance.
5. **Size and layout at $10^{-2}$ /s.** Networks passing G1: 8, 8, 9, 6 of 10 at 1, 2, 2.83, 4 mm (about 3 in 4, no
   trend); pooled G1 fails everywhere, on NARMA-10. About half the capacity lost at every size; size makes devices more
   alike, not more successful. Edge bands: 15-30 % less capacity than pads, but more robust from 2 mm (63 % vs 49 % kept).

**In one sentence:** with realistic junctions a nanowire network computes beyond its input, but ruptures at the measured
rate halve its capacity and make about 1 device in 4 fail; neither size (to 4 mm) nor any remedy tried removes this.

## What could be improved

* **Junction model** (main weakness): built from published behaviour, not fitted to single-junction data; rupture rate
  inferred indirectly (mostly one size, few jumps).
* **Protocol:** input windows likely below the optimum (pads better at 1.5 $\times$ onset, 3 networks).
* **Coverage:** networks to 4 mm vs about 1 cm devices; one input encoding; no application tasks; several remedies on
  3 networks only. **Packaging:** no literature positioning, journal or publication figures yet; no git.

## What could be done next

**Model, without new measurements** (all untested so far):

* **State-dependent ruptures.** Now every junction ruptures at the same constant rate. Make the rate grow with the
  junction's power (thermally activated, like the existing rupture above 1 mW): hot junctions on the main paths break
  more. Redo N10-C with it: the same measured jumps may imply a different per-junction rate and tolerance.
* **Joint fit to the Milano 2025 data we already have.** Fit voltage scale, time constants and rupture rate together to
  their recordings, instead of hand-setting them (N10 switches at about 10 times their voltage).
* **More heterogeneity**: spread of switching thresholds, not only of time constants.
* **Path redundancy from the network graph**: count independent current paths between electrodes; may explain why
  edge bands are more robust, predict tolerance vs size, guide layout design.

**Readout and analysis, on existing runs (no new simulations):**

* **Collapse-proof features**: a rupture shifts the baseline current; high-pass filtered or normalised currents may
  remove the step (periodic retraining did not).
* **Why NARMA-10 fails**: memory or number of readout channels.

**New simulations:** pads at 1.5 $\times$ onset with 10 networks (pre-registered); alternating-edge layouts (c, d);
backbone (N10-K) with re-tuned windows; more electrodes on larger networks; 7 mm (costly).

**Paper:** literature positioning, journal, figures (thesis A chosen). **Experiments:** send docs/MEASUREMENT_REQUEST.md.

*Details: docs/RESULTS.md (sections 1-28), docs/NEXT_STEPS.md, reports in docs/report_2026-xx-xx/.*
