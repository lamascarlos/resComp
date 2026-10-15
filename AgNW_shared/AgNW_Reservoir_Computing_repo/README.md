# AgNW Reservoir Computing

Can the stochastic silver-nanowire (AgNW) network simulated by
[`AgNW_FDT_Memristor_Framework`](../../Trabajo/CalculoNumerico/ProgsNeuro/0_Codes/Neuromorphic_Computing/AgNW_FDT_Memristor_Framework)
work as a physical reservoir computer?

## Answer

**In short: yes in simulation, conditionally for real devices.** A volatile-junction network read through 16 pad
electrodes is a reproducible, modest reservoir (memory of about 4 inputs); whether real AgNW networks behave the same
depends on their junction noise: random filament ruptures above about $10^{-4}$ per second per junction break it.
Pinning that rate is the decisive step. N10-C (`docs/RESULTS.md` §21) maps the measured jump statistics
of Milano et al. 2025 onto it; the estimate, 0.005-0.03 per second, is 50-300 times above the tolerance.
No remedy tried so far restores the result at that noise level: a consolidated backbone, online retraining of the
readout, a bias + small input operating point (the best noise-free reservoir, memory capacity 5.8, but the most
fragile) and a junction voltage scale matched to the measured network. **Network size does work, mostly**: at $10^{-2}$ ruptures per second per
junction, inside the real-junction range, 9 of 10 simulated 2.83 mm networks compute beyond their input (8 of 10 at
2 mm, 6 of 10 at 4 mm: about 3 in 4 at every size, no trend); the pooled criterion fails at every size on the
NARMA-10 margin (N10-L, N10-LS). Realistic noise still costs about half the noise-free capacity at every size; what size
buys is a smaller spread in capacity across networks. Edge electrodes (split bands, N10-SB) give about 15-30 % less
capacity than the pad array but lose a smaller fraction to ruptures (63 % kept at 2.83 mm vs 49 %). Measured
devices (about 1 cm) are larger still
(reports: `docs/report_2026-10-13/report.md` on 4 mm and edge electrodes, `docs/report_2026-10-09/report.md` on 10 networks per size, `docs/report_2026-10-07/report.md` on network size, `docs/report_2026-10-06/report.md` on the noise question, `docs/report_2026-09-29/report.md` on the
main result).
**In progress (N11, since 2026-10-13):** instead of hand-set junction parameters and a size-extrapolated rupture
rate, the junction is being fitted to the measured Milano et al. 2025 records on a strip with the measured gap, and the
data choose between a constant and a power-dependent rupture law (docs/PLAN_N11_CALIBRATION.md; stage 0 in
`docs/RESULTS.md` §29). The reservoir tests are then repeated with the fitted junction.

**With the framework's junction model: no. With a volatile junction built from measured AgNW behaviour (N10): yes,
weakly.** The framework's binary junctions (Lamas 2026 preset) and a graded filament junction calibrated on Lamas
Fig. 4 (N4) never compute more than the current input. Junctions that relax spontaneously on a time $\tau_s$ matched to
the input rate (N10) make the network a weak but reproducible reservoir, the first readout to pass criterion G1.

## 1. Methodology

Everything below is the same for every model, so results are comparable. Design: `docs/PLAN.md`. Full results:
`docs/RESULTS.md`.

### 1.1 Fixed network (imported, never copied)

Geometry, wires, and Kirchhoff solver come unchanged from `agnw_sim` (Lamas 2026 preset, framework commit `f6b6951`):

* $N = 2000$ wires, length 70 $\mu$m, on 1 mm$^{2}$; wire diameter 90 nm.
* Dual-node circuit with wire resistance; solved with the framework's cached solver (multi-electrode layouts use our
  own multi-terminal solve, tested to match the framework matrix to round-off).
* Integration step 1 ms (framework binary) or 2 ms (N4 filament, N10 volatile; both validated against 1 ms).
* 3 network seeds $\times$ 2 noise replicates per setting; all numbers are test scores at a validation-chosen
  operating point, mean $\pm$ sd with a 2-sd margin (see §1.5).

### 1.2 Interface: drive, electrodes, readout

* **Drive.** Scalar input $u_k \sim U(0,1)$ mapped to a piecewise-constant electrode voltage
  $V_k = V_{\mathrm{lo}} + (V_{\mathrm{hi}} - V_{\mathrm{lo}}) u_k$, held for `hold` steps. Swept: hold time
  (10 ms to 3 s per input) and window position/width relative to the conduction onset. Multi-electrode encodings:
  same signal everywhere ("direct"), fixed random weights per electrode ("weights",
  $V_e(t) = V_c + \tfrac{1}{2} w_e (2u(t)-1)$), or delayed copies $u(t-d_e)$ across electrodes.
* **Electrodes.** (a) Two bands (framework layout: left 5% driven, right 5% grounded). (b) Split bands
  (8 + 8 segments). (c) $4 \times 4$ array of 50 $\mu$m pads (8 driven + 8 grounded), plus a `pads_float` variant
  (4 driven + 4 grounded + 8 floating read pads). Windows are re-centred per layout from a quasi-static onset sweep
  (bands $\approx 2.1$ V, pads $\approx 0.35$ V).
* **Readouts.** Total `current` sampled densely within each hold (10--50 virtual nodes, so the transient is seen);
  `electrodes` (all 16 electrode currents, plus floating-pad voltages where present); 32 internal voltage `probes`
  (need micro-contacts; reference only); all `junctions` states (oracle upper bound, not measurable). Features below
  the $10^{-5}$ resolution floor (10 $\mu$V / 10 nA) are dropped.
* **Readout training.** Ridge regression (SVD, unpenalized intercept, features z-scored) on the first 60% of the
  stream; ridge $\alpha$ and operating point (hold, window) chosen on the next 20% (validation); all reported scores
  on the last 20% (test). Washout $\ge 3\times$ the slowest time constant precedes scoring, plus a drift check on the
  ON fraction.

### 1.3 Tasks, baselines, criteria

* **Tasks.** Linear memory capacity MC (sum of $r^2(u_{t-k})$, $k = 1..30$, only lags above the noise floor);
  degree-2 capacity (Legendre $P_2(u_{t-k})$ and products $u_{t-k}u_{t-l}$); NARMA-10 NMSE; sine/square per-sample
  accuracy; consistency (mean correlation across the two replicates). Consistency is measured on the *state part*
  ($I/V_{\mathrm{in}}$, $V_p/V_{\mathrm{in}}$) and on its sample-to-sample *increments*, so shared drift and the
  $V_{\mathrm{in}}$ echo do not count.
* **Baselines (no nanowires).** Input only, linear delay line (32 taps), quadratic delay line, ESNs (16/32/200 units),
  Gaussian-noise floor. Same streams, same splits.
* **Go/no-go criteria** (fixed 2026-09-27 before the decisive runs, `agnw_rc/criteria.py`):
  **G0** reproducible (increment consistency $\ge 0.8$);
  **G1** computes beyond the input (MC $\ge 1$ with $\ge 2$ lags above floor, **and** degree-2 $\ge 1$,
  **and** NARMA-10 $\le 0.70$ vs input-only $\approx 0.80$--$0.84$);
  **G2** useful (MC + degree-2 $\ge 50\%$ of an ESN of the same size, **and** sine/square $\ge 0.95$).
  For multi-electrode readouts: **strict** (headline; ESN as large as the feature count incl. virtual nodes) and
  **loose** (one ESN unit per physical electrode; ESN-16 total 15.4, so $\ge 7.7$).

## 2. Models explored: what failed, what worked

The wires never change. What changes is the **junction** (plus interface where noted). Each row below is
self-contained: code, what the junction physically is, what interface it ran on, and the verdict.

**Nomenclature.** **E** = experiment planned upfront in `docs/PLAN.md` (E1 main sweep, E2 relaxation-rate check,
E3 slow inputs, E4 decisive time-scale-matched test). **N** = follow-up work item from the diagnosis in
`docs/NEXT_STEPS.md` (N1 mean-field ablation, N2 heterogeneous forgetting, N3 masks, N4 filament junction, N5
multi-electrode, N7 noise averaging, N8 trade-off map, N9 virtual nodes, N10 volatile measured-data junction).
Suffixes refine an item: M1 = N1 runs, N10-1 = first N10 campaign (two bands, 32 probes), N10-E = N10 on 16
electrodes, N10-M = memory/noise limits, N10-S = sensitivity, N10-R = ruptures, N10-C = calibration on open data
(deferred).

**Colour code used in this section and in §3.** <span style="color:green"><b>green</b></span> = worked / positive
result (passes G1 or robust). <span style="color:red"><b>red</b></span> = failed / negative result (at or
near input-only level, or breaks G1). <span style="color:blue"><b>blue</b></span> = to be tested in the future.
<span style="color:purple"><b>violet</b></span> = parameter still to be pinned by measurement or literature
(e.g. relaxation time, voltage scale, rupture rate).

| model (self-contained) | idea tested | verdict |
|---|---|---|
| **Framework binary stochastic** (Lamas 2026 preset; runs E1--E3; two bands, total current / 32 probes / all junctions) | Reference junction: binary ON/OFF, voltage-driven SET, current-driven RESET, one relaxation rate $p_{\mathrm{relax}} = 10^{-4}$ (10 s). | <span style="color:red"><b>fails.</b></span> MC $\le 0.26$ (0.76 at 1 s holds), degree-2 $\le 0.13$, NARMA $\ge 0.73$, chance accuracy. Junction states irreproducible (consistency 0.00); only the global conductance partly responds. |
| **N1 mean-field** (runs M1; same grids as E1, own windows 3.5--5.5 V) | Ablation, not a device: each junction carries its ON probability with the exact framework rates, no coin flips. Asks whether noise is the blocker. | <span style="color:red"><b>diagnostic, partial.</b></span> Reproducible (conductance consistency 1.00), nonlinear (degree-2 2.2, sine/square 1.00), but MC only 1.8 vs ESN-32 13.8. Noise blocks nonlinearity; the single slow rate still blocks memory. |
| **N2 heterogeneous forgetting** (mean-field + per-junction $p_{\mathrm{relax}}$ spread) | Same as N1 but each junction gets its own forgetting rate (log-uniform over 1--3 decades). Tests whether a spread of time scales fixes memory. | <span style="color:red"><b>helps $\sim$2x, then capped.</b></span> MC 1.8 $\to$ 2.6, degree-2 1.9 $\to$ 4.3. An ideal leaky-bank reference shows purely relaxational single-input memory is noise-limited ($\approx 9$ noise-free, 2--3 at realistic SNR). |
| **N7a replicate averaging / N7b voltage confound** (framework binary, one E1 point / M1 windows) | N7a: average $R = 1$--8 identical devices to wash out noise. N7b: rerun stochastic model at the high mean-field voltages to rule out a voltage confound. | <span style="color:red"><b>fails.</b></span> Junction/probe consistency stays $\le 0.015$ ($\sim$100+ devices needed); high voltage makes it worse. N1 gains come from removing noise, not from voltage. |
| **N5a multi-electrode, random weights (binary junction)** (8 + 8 split bands / $4 \times 4$ pads, 16 electrode currents) | Same binary junctions, richer interface: 8 inputs with fixed random weights, 8 grounded outputs. Tests the single-electrode-pair bottleneck. | <span style="color:red"><b>fails on the binary junction.</b></span> All scores at input-only level (MC $\approx 0$, chance accuracy). A richer interface cannot extract what a noisy state does not hold. |
| **N4 continuous filament, Fig. 4-calibrated** (runs N4-1; two bands, 10--100 ms holds) | Physical graded junction: filament $\Lambda$ grows above $V_{\mathrm{set}} = 0.06$ V, dissolves below $V_{\mathrm{reset}} = 0.03$ V over $\tau_d \approx 25$ s, Simmons tunnelling conductance; fitted to Lamas Fig. 4 I--V loop. | <span style="color:red"><b>fails (frozen).</b></span> Reproducible (junction consistency 0.69--0.83) but static: MC 0, degree-2 $\le 0.02$. Filaments saturate in $\sim$0.1 s or persist 30 s; most junction voltages sit in the hold band where nothing happens. |
| **E4 time-scale-matched inputs (decisive, pre-registered G0--G2)** (binary + calibrated N4; holds 0.3--3 s, washout $3\times\tau$, 50 virtual nodes) | Same two junctions, inputs slowed to the network's own forgetting time. First run with validation-chosen operating point and 2-sd margins. | <span style="color:red"><b>fails G1 for both.</b></span> Binary current at 3 s: MC 0.94, degree-2 0.50, NARMA 0.71 (G0 yes, G1 no). N4 frozen at every hold (0 capacity, conductance varies $\le 0.4\%$). |
| **N8 short: hold band vs forgetting** (N4 scan: $\tau_d = 0.3$--25 s $\times$ band ratio $r = V_{\mathrm{reset}}/V_{\mathrm{set}} = 0.5/0.9$; 1 net, 400 samples) | Disentangles why N4 freezes: is it slow forgetting or the wide hold band? Narrow band $r = 0.9$ keeps a Fig. 4 fit (loss 0.08) while unfreezing. | <span style="color:red"><b>diagnostic: band is the cause.</b></span> Wide band frozen at every $\tau_d$ ($\sim$20/5953 junctions move); narrow band moves 700--1700/sample, MC $\approx 1$, degree-2 $\approx 1$. Still far from G2. <span style="color:blue">Full E4-style check of narrow-band N4 (N4-nb) still to be run.</span> |
| **N10 volatile junction from measured data (runs N10-1; two bands, 32 probes + current)** | Junction built from measured AgNW behaviour (Milano 2022 rate law, non-polar, relaxes spontaneously at zero bias with <span style="color:purple">$\tau_s = 0.03$--3 s</span>, spread across junctions, no hold band; power-gated consolidation + rupture). Settings <span style="color:purple">$v_{\mathrm{scale}} = 5$, $k_{\mathrm{max}} = 100$ s$^{-1}$</span> set onset $\approx 2.2$ V and switching $\ge 10$ ms. Best hold $\approx \tau_s/3$. | <span style="color:green"><b>works weakly: first G1 pass.</b></span> Probes at $\tau_s = 0.3$--3 s: MC $\approx 2.0$--2.4, degree-2 $\approx 3.4$--4.1, NARMA $\approx 0.48$--0.51, sine/square 1.00, consistency 1.00 over 3 nets. G2 no (6 vs 14). Current-only readout just below G1. |
| **N10-E: N10 on 16 pads, electrode currents (chip-compatible)** ($4 \times 4$ pads, 8 driven with random weights + 8 grounded; 100 ms inputs; 3 nets) | Same volatile N10 junction (<span style="color:purple">$\tau_s = 0.3$ s</span>), but 16 spatially distinct electrodes replace the internal probes; readout = 16 electrode currents with virtual nodes. | <span style="color:green"><b>works: G1 + G2 loose.</b></span> MC 3.48, degree-2 6.74, NARMA-10 0.40 ($\approx$ ESN-16 0.37), sine/square 1.00, consistency 1.00. 3x memory and 4x nonlinearity of the two-terminal current. G2 strict no (10.2 vs 25). Floating read pads add nothing. |
| **N10-M: memory and noise limits on N10-E** (same pads; slower <span style="color:purple">$\tau_s$</span>, spread, delayed inputs, Gaussian filament noise <span style="color:purple">$\sigma$</span>) | Stresses the working N10-E setup: slower junctions with matched holds, wider $\tau_s$ spread, delayed input copies across electrodes, additive filament noise. | <span style="color:green"><b>limits mapped (mixed).</b></span> <span style="color:green">Own memory capped $\approx 4$ inputs for any $\tau_s$; $\approx 1\%$-range noise tolerated (G1 holds).</span> <span style="color:red">$\approx 4\%$ noise breaks G0/G1 (consistency 0.47); delays add only interface memory (MC 7.6 = delay line 7 + 0.6).</span> |
| **N10-S: sensitivity to chosen settings** (N10-E setup; voltage scale 3--8, switching cap 30--300 s$^{-1}$) | Repeats N10-E while varying the two hand-chosen N10 settings. Each setting gets its own onset sweep. | <span style="color:green"><b>robust.</b></span> All scores within 1 sd of N10-E, all G0--G2 verdicts unchanged. Onset $\propto 1/v_{\mathrm{scale}}$. What still matters: <span style="color:purple">$\tau_s$</span> (sets usable input rate) and <span style="color:purple">junction noise</span>. |
| **N10-R: realistic rupture noise + mitigations** (N10-E setup; random complete ruptures <span style="color:purple">$r = 10^{-5}$--3 /s per junction</span>; bigger pads, longer holds, double presentation) | Replaces smooth noise by abrupt Poisson filament ruptures (as in measured relaxation), and tries three mitigations at the breaking noise level. | <span style="color:red"><b>breaks above $\approx 10^{-4}$ /s; no mitigation restores G1.</b></span> <span style="color:green">$10^{-5}$ /s safe; up to $3 \times 10^{-4}$ /s the typical run is unaffected</span>, but single runs collapse from $10^{-4}$ /s (a rupture on a bottleneck junction invalidates the trained readout); from $10^{-3}$ /s every run loses about half its capacity. Larger pads worse. <span style="color:blue">Online retraining and consolidation-on N10 still to be run.</span> |
| **N10-C: calibration on Milano 2025 measured data** (their constant-voltage protocol on N10 two-band networks, 1--2.83 mm, sampled every 0.6 s like the instrument) | Turns their 0.08 detected conductance jumps per second (7 mm network) into a per-junction rupture rate: count the junctions whose rupture is detectable, extrapolate with network size. | <span style="color:red"><b>real junctions too noisy for a 1 mm reservoir.</b></span> Data reproduced (0.080 jumps/s, half up/half down). <span style="color:purple">$r^\ast \approx 0.005$--0.03 /s per junction</span>: 50--300x above the N10-R tolerance (extrapolated; rests on the largest networks, few jumps). N10 junctions switch at $\sim 10\times$ their voltage (onset 14 V vs 2.4 V at 7 mm). |
| **N10-RN: measured fluctuations on the readout** (Milano 2025 network-level noise, OU + jumps, on the 16 electrode currents of N10-E; 1--30x measured; no new simulations) | Separates the size of the fluctuations the readout sees from what ruptures do inside the network. | <span style="color:green"><b>harmless at the measured level.</b></span> G0/G1/G2-loose hold at 1x; G1 up to 3x (per electrode) and 30x (common). The rupture damage in N10-R comes from rerouting inside a small network. |
| **N10-K: consolidated backbone vs ruptures** (N10 with the consolidated filament $y$ on, 10 s forming drive; backbones 1.7--12.7 % of junctions; ruptures 0--$10^{-2}$ /s) | Protects the bottleneck junctions with stable (thick) filaments, as in measured STP $\to$ LTP consolidation. | <span style="color:red"><b>no G1.</b></span> <span style="color:green">Collapses gone (1 of 72 runs; ruptures cost 4--25 %)</span>, but the backbone itself halves memory and cuts degree-2 by 60--70 %: robustness and computation compete for the same junctions. <span style="color:blue">Input windows not yet re-tuned after forming.</span> |
| **N10-OR: online retraining of the readout** (readout retrained every 10--40 inputs on the last 150--600, N10-R runs) | Lets the readout follow the network after a rupture reroutes the current. | <span style="color:red"><b>no help.</b></span> Never better than a static readout; short windows are too few samples for 800 features. |
| **N10-B: bias + small input** (bias 1.5--3x onset, input 3--15 % of the bias, as Milano 2025) | Operates deep in the conducting state, as the measured reservoir does. | <span style="color:green"><b>best noise-free reservoir so far</b>: MC 5.8, degree-2 10.9, NARMA 0.36 (> ESN-16).</span> <span style="color:red">Much more fragile: 32/36 runs collapse at $10^{-3}$ /s; measured-size readout noise alone breaks G1.</span> |
| **N10-V: voltage scale matched to Milano** (`v_scale` 40) | Junctions switch at the measured network's voltage. | <span style="color:green">Reservoir unchanged, slightly more robust;</span> calibration ($r^\ast$) unchanged; G1 still fails at $\ge 10^{-3}$ /s. |
| **N10-L: larger networks** (N10-E enlarged to 1.41, 2 and 2.83 mm, same density, pads scaled; ruptures 0--$10^{-2}$ /s) | Tests whether the rupture tolerance grows with size (each rupture matters less in a bigger network). | <span style="color:green"><b>yes: 2.83 mm meets G1 at $10^{-2}$ /s</b> (MC 2.2, degree-2 3.4, NARMA 0.54), inside the real-junction range; 2 mm up to $3 \times 10^{-3}$ /s.</span> Size buys reliability (spread between networks shrinks ~3x), not capacity: about half the noise-free capacity is lost at every size. |
| **N10-LS: 10 networks per size** (networks 4-10 at 1, 2, 2.83 mm, 0 and $10^{-2}$ /s; ruptures + junction + readout noise at 2.83 mm; 4 mm, 3 networks; HPC cluster, 348 runs) | Checks the N10-L headline with publication-level statistics; decision rules fixed in advance (docs/PLAN_N10L_SEEDS.md). | <span style="color:purple"><b>9/10 networks pass G1 at 2.83 mm, 8/10 at 2 mm; pooled G1 fails (NARMA margin)</b></span>. Size halves the spread between networks (ratio 1.98 [1.15, 4.20]); about half the capacity lost at every size; combined noise ~ ruptures alone. 4 mm with 10 networks (N10-LS4, 2026-10-13): 6/10 pass, pooled fails, claim withdrawn; failures are on NARMA-10 only. |
| **N10-SB: split-band edge layout vs pads** (1, 2, 2.83 mm, 10 networks; rupture ladder, bias + small input, 1.5 x onset window; 606 runs) | Edge contacts are easier to fabricate and match the measured devices; same tests and rules as the pads (docs/PLAN_SPLIT_BANDS.md). | <span style="color:red"><b>~30 % less capacity noise-free, ~15 % less at $10^{-2}$ /s, 3-7/10 networks pass G1</b></span>; <span style="color:green">more robust from 2 mm (capacity kept 0.63 vs 0.49 at 2.83 mm, 1/20 collapses)</span>; bias does not help. |
| **N11: junction calibration on measured data** (fit `v_scale`, $\tau_s$, spread, $k_\mathrm{max}$, then the rupture law (constant, power or thermal) to Milano 2025 records on a 15.6 mm strip; started 2026-10-13) | Replaces hand-set junction parameters and the size-extrapolated rupture rate by a fit at the measured gap. | <span style="color:blue"><b>running.</b></span> Stage 0 (measured statistics, validated estimators): onset 2.48 V, jump rate 0.006-0.15 /s flat in voltage, relaxation about 27 s. Stage 1 (`v_scale` root find) on the cluster. |

Why the framework model fails: seen through its electrodes the network behaves as **one noisy, slowly relaxing
integrator**. Which junctions switch is set by noise, so individual junction states and local probe voltages are
not reproducible, and the network forgets over 10 s while inputs change every 10--100 ms. Why N10 works: volatile
filaments with no hold band and $\tau_s$ matched to the input give many reproducible, diverse, fading variables;
16 spatially distinct electrodes expose them.

![electrode layouts](figures/figS3_electrode_layouts.png)

*Layouts: two bands (E1--E4, N10-1; total current + 32 internal probes, left), split bands and $4 \times 4$ pads
(N5a, N10-E/M/S/R; 16 electrode currents, middle/right). Orange driven, blue grounded, black contacts, yellow probes.*

## 3. Headline numbers (same colour code)

800-sample streams, test data, operating point chosen on validation, means over 3 networks $\times$ 2 replicates.
<span style="color:green"><b>Green rows</b></span> pass G1; <span style="color:red"><b>red rows</b></span> do not
compute beyond the input.

| best readout and setting | memory capacity | degree-2 capacity | NARMA-10 NMSE | sine/square accuracy |
|---|---|---|---|---|
| input only (no network) | 0.0 | 0.0 | 0.84 | 0.50 |
| <span style="color:red">framework binary junctions (E4, current, 3 s per input)</span> | 0.94 | 0.50 | 0.71 | 0.53 |
| <span style="color:red">Fig. 4-calibrated filament junctions (N4, E4)</span> | 0.0 | 0.0 | 0.84 | 0.50 |
| <span style="color:green">**volatile N10 junctions (N10-1: volatile Milano-law filament, no hold band, $\tau_s = 3$ s; 32 probes, 1 s per input)**</span> | **2.35** | **4.13** | **0.48** | **1.00** |
| <span style="color:green">**volatile N10 junctions on 16 pads (N10-E: same junction on a $4 \times 4$ pad array, 16 electrode currents, 0.1 s per input)**</span> | **3.48** | **6.74** | **0.40** | **1.00** |
| ESN, 32 units (digital reference) | 13.5 | 14.2 | 0.23 | 1.00 |
| linear delay line, 32 taps (memory reference) | 30.0 | 0.0 | 0.15 | 0.99 |

N10 meets G0 and G1; G2 only per physical contact (vs ESN-16), not per recorded number (strict).

**How to read the figures.** Each figure plots one score (MC, degree-2, NARMA error) vs the varied knob
(hold time, $\tau_s$, noise). Higher is better for capacities and accuracy, lower is better for NARMA NMSE.
Red dotted lines = G1/G0 thresholds; magenta = G2 loose (dotted) / strict (dashed); grey = reference
(two-band N10-1 or noise-free N10-E). Good = green-level points sitting above red with small error bars
(3 nets); bad = red-level points at the input-only floor.

![N10-1](figures/fig15_n10_1.png)
*Fig. 15 (N10-1: volatile junction, two bands). Top: total current; bottom: 32 probes; x = hold time;
solid/dashed = no spread / spread $\tau_s$. Read: probes at 0.1--1 s holds clear all three red G1 lines with
consistency 1.00 — the first green result; current stays red (just below G1). Longer 3 s holds do not help.*

![N10-E](figures/fig17_n10e.png)
*Fig. 17 (N10-E: same junction on 16 pads). Orange = 16 electrode currents (chip-measurable), green = 32 internal
probes for reference. Read: orange clears G1 on all three panels and the magenta loose-G2 line, but not the
strict-G2 line — green per contact, red per recorded number. Memory panel: 3.5 vs ESN-16 9.4 is the weak point.*

![N10-M](figures/fig18_n10m.png)
*Fig. 18 (N10-M limits on the pad setup). Grey = N10-E reference. Read, left to right: slower $\tau_s$ barely moves
memory (capped $\approx 4$ — green but flat); delays lift MC to 7.6 but only to the delay-line level (interface memory,
not network memory); noise $\sigma \ge 0.1$ ($\approx 4\%$ of filament range) drops all panels to red and consistency
below G0.*

![N10-R](figures/fig20_n10r.png)
*Fig. 20 (N10-R ruptures). Grey = noise-free N10-E; orange = rupture rates / mitigations; green = each input shown
twice and averaged. Read: every orange point from 0.01 to 3 ruptures/s sits red (G1 broken, capacities halved);
double presentation recovers only 10--40%. Lower rates (RESULTS §20): tolerance about $10^{-4}$ /s per junction;
<span style="color:blue">online retraining</span> is still open.*

A project review (2026-09-27) corrected consistency and memory-cap statements;
see [docs/REVIEW_2026-09-27.md](docs/REVIEW_2026-09-27.md). Work continues in [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md):
next steps are noise, consolidation and more electrodes on the N10 junction, and measured relaxation data to pin
its parameters.

Full write-up with figures: **[docs/RESULTS.md](docs/RESULTS.md)**. Design and definitions: [docs/PLAN.md](docs/PLAN.md).
Junction physics and framework audit: [docs/SWITCHING_MECHANISMS.md](docs/SWITCHING_MECHANISMS.md).
N10 design: `docs/N10_DESIGN.md`. Measurement request: `docs/MEASUREMENT_REQUEST.md`.

## 4. What is fixed and what this project adds

* **Fixed (from the framework, imported unchanged):** network geometry (N = 2000 wires, 70 µm, 1 mm$^{2}$), dual-node
  Kirchhoff circuit with 90 nm wires, stochastic SET/RESET dynamics with the Lamas 2026 parameters, exponential
  relaxation $p_{\mathrm{relax}} = 10^{-4}$, and the cached solver. Simulations run through `agnw_sim.simulate.run_simulation`.
  `agnw_rc.config` holds the exact settings, and a test checks them against the framework preset.
* **Added here:**
  * input encoding: piecewise-constant electrode voltage, with a sweep over hold time and voltage window; random-weight
    and delayed multi-electrode encodings;
  * readout scenarios: two-terminal current with virtual nodes, 16 electrode currents (split bands / pads /
    pads with floating reads), 32 voltage probes, and all junction states as an oracle;
  * junction models: mean-field ablation (N1/N2), continuous filament calibrated on Fig. 4 (N4), volatile
    measured-data junction with consolidation, rupture and noise (N10 family);
  * multi-terminal Kirchhoff solver and layouts (`electrodes.py`, `multicircuit.py`, `multidrive.py`);
  * a ridge readout with validation-selected regularization and held-out test scores;
  * tasks: memory capacity, degree-2 information-processing capacity, integration (EMA), NARMA-10, sine/square
    classification, and increment consistency;
  * no-nanowire baselines: input only, delay lines, quadratic delay line, echo-state networks, and a noise floor;
  * go/no-go criteria G0--G2 with validation-based operating point and drift check (`criteria.py`).

## 5. Layout

```
agnw_rc/            reservoir layer on top of agnw_sim
  config.py         framework working model + experiment grids (E1-E4, M1, N2, N5a, N4, N10-1/E/M/S/R)
  drive.py          input -> run_simulation -> observables (current, probe voltages, junction states)
  tasks.py          input streams and targets (memory, Legendre degree-2, EMA, NARMA-10, sine/square)
  readout.py        SVD ridge regression, alpha chosen on a validation segment
  metrics.py        capacity, NMSE, accuracy, consistency (incl. increments)
  baselines.py      input only, delay lines, ESN, noise features
  evaluate.py       scoring of reservoir runs and baselines
  meanfield.py      noise-free (mean-field) junction dynamics + fast solver (N1/N2)
  electrodes.py     electrode layouts: two bands (framework), split bands, pad array (N5, N10-E)
  multicircuit.py   multi-terminal Kirchhoff solver on the framework's dual-node graph (N5, N10-E)
  multidrive.py     multi-electrode drive/readout time loop, input encodings (N5, N10-E/M/R)
  filament.py       continuous-filament junction model (N4): rate law, Simmons tunnelling, noise
  fig4.py           Lamas Fig. 4 protocol and calibration loss for N4
  criteria.py       go/no-go criteria G0-G2, validation-based operating point, drift check (review plan)
  volatile.py       N10 volatile two-time-scale junction (Milano rate law + consolidation, ruptures, noise;
                    N11: constant / power / thermal rupture laws)
  strip.py          N11 rectangular strip at the measured gap (square = framework network bit for bit)
  series_stats.py   N11 conductance time-series estimators (onset, theta, relaxation, noise, jumps), validated
  n11sim.py         N11 measured protocols on the strip (constant bias, relaxation at 10 mV)
  n11loss.py        N11 losses, AIC, calibration and rupture-law decision rules
scripts/
  pilot.py, meanfield_pilot.py, multi_pilot.py   operating-point pilots -> results/*.npz
  run_sims.py       parallel, resumable simulations (e1 e2 e3 m1 n2 n7a n7b n5a n4_1 e4 n10_1 n10m;
                    --holds/--reverse/--limit split work across machines) -> results/raw/*.npz (git-ignored)
  e4_report.py, n10_report.py, n10m_report.py, n10s_sensitivity.py, n10r_noise.py, n10e_pilot.py, ...
                    go/no-go reports -> results/*criteria.json, figures/fig1*.png
  n10c_data.py, n10c_sim.py   N10-C: Milano 2025 open data (data/milano2025/) and their protocol on N10
  n10e_readout_noise.py, n10k_consolidation.py   N10-RN measured noise on the readout; N10-K consolidated backbone
  n10r_retrain.py, n10b_bias.py, n10v_vscale.py   N10-OR online retraining; N10-B bias + small input; N10-V matched v_scale
  n10l_size.py      N10-L larger networks (RunSpec.side, pads_scaled; --share splits runs across machines)
  n10l_seeds.py, n10sb_layout.py   N10-LS 10 networks per size; N10-SB split bands vs pads
  n11_data.py       N11 stage 0: measured statistics -> results/n11_data.json, fig33
  n11_fit.py        N11 stages 1-2 (stage1 vscale/grid/refine/checks/report, stage2 plan/run/refine/report, bench)
cluster/            SLURM job scripts for the CCT-Rosario cluster (run_seeds.sh, run_n10sb.sh, run_n11.sh, time_n11.sh)
  evaluate_all.py   all scores -> results/metrics.json, results/summary.csv
  n4_stride_check.py, n4_substep_check.py, n10_kmax_check.py, n10_stride_check.py, ...
                    integration-step validation
  n8_short.py       N4 hold-band vs forgetting scan -> results/n8_short.json
  effective_dimension.py, leaky_bank_reference.py, noise_averaging.py, switching_rates.py, ...
                    references and diagnostics
  make_figures.py   all figures (--only, --mode paper|presentation) -> figures/*.pdf|png
figures/README.md   index: each figure -> drawing function, data, generating script
tests/              pytest (139 tests): framework contract, drive alignment, readout, metrics, tasks, baselines,
                    mean-field, filament, volatile, multi-electrode, criteria, N11 (strip, estimators, fit)
docs/PLAN.md        question, design, experiments, tests
docs/RESULTS.md     results, figures, interpretation, limitations
docs/NEXT_STEPS.md  diagnosis + ranked work items with status (start here when resuming)
docs/SWITCHING_MECHANISMS.md  junction ON<->OFF mechanisms, their rates at the operating points, framework docs audit
docs/N4_DESIGN.md, docs/N10_DESIGN.md  junction models: equations, parameters, calibration and experiment plan
docs/REVIEW_2026-09-27.md  full project review: bugs, corrected conclusions, evidence table
docs/LITERATURE_NOTES.md, docs/MEASUREMENT_REQUEST.md, docs/report_*/  literature, experiment request, reports
docs/PLAN_N11_CALIBRATION.md  N11 spec (targets, losses, decision rules, budget; section 12 = changes)
docs/superpowers/plans/  N11 implementation plans + ledgers (rulings made during implementation)
docs/CLUSTER_TUTORIAL.md  running jobs on the CCT-Rosario cluster
docs/summary_2026-10-13/  one-page overview (what was done, results, what next)
memory/             portable copy of the project memory (loaded via CLAUDE.md)
legacy/             earlier prototype, kept for reference only (its memory-capacity numbers are invalid)
```

## 6. Setup

The framework is installed in editable mode from its own checkout. Nothing is copied from it.

```bash
FRAMEWORK=../AgNW_FDT_Memristor_Framework     # sibling checkout; adjust if it lives elsewhere
python3 -m venv .venv
.venv/bin/pip install -e "$FRAMEWORK"
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest            # 139 tests, ~7 min
```

## 7. Reproduce

```bash
.venv/bin/python scripts/pilot.py
.venv/bin/python scripts/run_sims.py e1 --procs 14
.venv/bin/python scripts/run_sims.py e2 --hold 100 --v-center 1.75
.venv/bin/python scripts/run_sims.py e3 --v-center 1.75
.venv/bin/python scripts/meanfield_pilot.py
.venv/bin/python scripts/run_sims.py m1 --procs 14     # N1 mean-field ablation
.venv/bin/python scripts/run_sims.py n2 --procs 14     # N2 heterogeneous forgetting
.venv/bin/python scripts/evaluate_all.py --procs 14
.venv/bin/python scripts/run_sims.py e4 --model stochastic --procs 12  # E4 binary, ~6 h
.venv/bin/python scripts/run_sims.py e4 --model filament --procs 16    # E4 N4 at 2 ms, ~5 h
.venv/bin/python scripts/e4_report.py
.venv/bin/python scripts/run_sims.py n10_1 --procs 16  # N10-1, 216 runs, ~12 h
.venv/bin/python scripts/n10_report.py
.venv/bin/python scripts/n10e_pilot.py sweep; .venv/bin/python scripts/n10e_pilot.py run; .venv/bin/python scripts/n10e_pilot.py report
.venv/bin/python scripts/n10e_pilot.py full --procs 18; .venv/bin/python scripts/n10e_pilot.py full_report
.venv/bin/python scripts/run_sims.py n10m --procs 18; .venv/bin/python scripts/n10m_report.py
.venv/bin/python scripts/make_figures.py            # --mode presentation for talk-sized fonts
.venv/bin/python scripts/n11_data.py                # N11 stage 0 (~2 min)
.venv/bin/python scripts/n11_fit.py stage1 vscale   # N11 stage 1-2: cluster/run_n11.sh, about 2000 core-h (see RESULTS §30)
```

Each script sets one BLAS thread per process, because the scripts already run in parallel over processes. Leaving
this out oversubscribes the CPU badly. The raw runs (~6 GB, without junction states; full files archived on an external drive, see RESUME_FROM_HERE.md) are git-ignored.
`results/metrics.json` and `results/*criteria.json` keep every score. Full command list: `docs/RESULTS.md` §30.

## 8. Next steps

Tracked in **[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md)**: the diagnosis (stochastic binary switching, a single
forgetting time scale, a single electrode pair), work items ranked by gain/effort, their status, and a log.
Decisive unknown: the per-junction rupture rate (tolerance about $10^{-4}$ /s). N10-C calibration on Milano 2025
open data gives 0.005-0.03 /s, far above it (Gaussian part $\sigma \lesssim 0.05$, at the edge). Remedies tried and failed:
consolidated backbone, online retraining, bias + small input, matched voltage scale (RESULTS 23-26). Larger networks
(N10-L/LS, to 4 mm) raise reliability, not the pass rate. Now running: **N11**, the junction fitted to the measured
data with the rupture law chosen by AIC, then the reservoir re-test. Single-junction / small-network measurements per
`docs/MEASUREMENT_REQUEST.md` would settle the rupture rate directly.
