# AgNW FDT Memristor Framework

Simulator of the stochastic silver-nanowire (AgNW) memristor network of Lamas et al. (2026), plus a fluctuation-dissipation test (FDTest) for memory (non-Markovianity) in its dynamics. Part of a broader project connecting computational neuroscience, neuromorphic hardware, and non-equilibrium statistical physics.

**Where to start reading the science:** [`docs/report_2026-10-02/report_brief.md`](docs/report_2026-10-02/report_brief.md) (collaborator-facing, no repo-internal references) or [`docs/report_2026-10-02/report.md`](docs/report_2026-10-02/report.md) (full, internal). PDFs of both are built with [`docs/md_to_pdf.sh`](#building-the-progress-report-pdfs). [`AgNW_FDT_Memristor_Framework.md`](AgNW_FDT_Memristor_Framework.md) is the original theory document that motivated this work (largely superseded by the FDTest approach — see the report's §5.1). Current switching equations: [`docs/switching_model.md`](docs/switching_model.md). Theory audit: [`docs/theory_audit/THEORY_AUDIT.md`](docs/theory_audit/THEORY_AUDIT.md).

> **Legend (color code, fixed hexes).** <span style="color:#1a7f37">**green** = worked / pass</span> · <span style="color:#cf222e">**red** = failed / fail</span> · <span style="color:#0969da">**blue** = future work / planned test</span> · <span style="color:#6e40c9">**violet** = parameter still to be determined (measure or ask authors)</span>. Color is emphasis only — every verdict also says **pass**, **fail**, **planned**, or **unconfirmed** in words.
>
> **30-second jargon.** **AgNW network** (random sticks = wires; crossings = memristor junctions) · **SET/RESET** (voltage-driven OFF$\to$ON / current-driven ON$\to$OFF switching) · **$g_{\mathrm{eff}} = I_{\mathrm{total}}/V_{\mathrm{in}}$** (whole-network conductance, our main observable) · **FDTest** (perturbation test for memory: kick the voltage, compare relaxation vs. spontaneous fluctuations) · **$\Delta^2$** (its violation score: 0 = memoryless/**Markov**, clearly above noise floor = memory/**non-Markov**) · **$M$** (averaged perturbation cycles, $\approx 1000$) · **$S/N$** (response $\div$ noise; calls need $S/N \ge 3$).

## Contents

- [Methodology](#methodology)
- [Switching variants: what was tried, what works](#switching-variants-what-was-tried-what-works)
- [Headline results](#headline-results)
- [Figures: what, how, so-what](#figures-what-how-so-what)
- [Limitations](#limitations)
- [Status: where we are / doing / next](#status-where-we-are--doing--next)
- [Install](#install)
- [Repository layout](#repository-layout)
- [Running a simulation](#running-a-simulation)
- [Running the FDTest](#running-the-fdtest)
- [Other analyses](#other-analyses)
- [Building the progress report PDFs](#building-the-progress-report-pdfs)
- [Claude Code workflow files](#claude-code-workflow-files)

## Methodology

**Fixed simulator.** **Random stick network** (wires of length $70\,\mu$m dropped with random position/orientation in a $1\,\mathrm{mm}^2$ square; crossings form **junctions**; wires in a 5% edge band are the electrodes; $N=2000$ wires $\to$ $\approx 6000$ junctions), **dual-node Kirchhoff circuit** per Lamas et al. §2.2 (each junction = two nodes joined by $G_j \in \{G_{\mathrm{OFF}}, G_{\mathrm{ON}}\} = \{10^{-9}, 10^{-3}\}$ S, wire segments resistive with $\rho_{\mathrm{Ag}} = 1.59\times10^{-8}\,\Omega$m and diameter $D$), **stochastic junction dynamics** stepped at $\Delta t = 1$ ms. Import `agnw_sim` and never copy it: `from agnw_sim.dynamics import SimParams`, `from agnw_sim.simulate import run_simulation`.

**Junction dynamics (working model).** With $V_{\mathrm{mem}} = |V_a - V_b|$ (voltage drop across the junction edge) and $I_{\mathrm{mem}} = |G_j V_{\mathrm{mem}}|$ (current through it):

$$P_{\mathrm{SET}} = \Delta t\,P_0\,[1-e^{-\alpha(V_{\mathrm{mem}}-V_{\mathrm{th}})}]_+, \qquad P_{\mathrm{RESET}} = \min(1,\,\Delta t\,P_{\mathrm{decay}}\,(I_{\mathrm{mem}}/I_{\mathrm{unit}})^2),$$

plus **spontaneous ON$\to$OFF relaxation** (each ON junction turns OFF with probability $p_{\mathrm{relax}}$/step, independent of voltage/current/history; recalled from a Lamas talk, not in the paper).

**Withdrawn/superseded — see corrected rule below.** Before 2026-09-19 the sim treated Table 1's 0.4 / 0.01 as *rates* and multiplied by $\Delta t$ again, making SET $1000\times$ too slow (the $P_{\mathrm{decay}} \approx 3\times10^8$ finding and avalanche table from that period are invalid). **Correction:** Table 1 lists *per-step products* $\Delta t\,P_0 = 0.4$, $\Delta t\,P_{\mathrm{decay}} = 0.01$ — always build with `SimParams.from_step_probs(0.4, 0.01)`. Anything under `validation_output/old_model/` or `docs/on_hold/validate_p_decay/` uses the mis-scaled model — do not run or build on it.

**Fig. 4 calibration.** The paper's Fig. 4 I-V hysteresis (**sine drive**, 3.6 V amplitude, 50000 steps/cycle, at **network sizes** $N=2000/3000$) is the target; metrics are **$I_{\max}$** (peak current: paper 3.24 mA at $N=2000$), **onset** (first voltage with $I > 0.05$ mA: paper $\approx 1.9$ V), **loop areas $A_\pm$** (positive/negative half-cycle $\oint I\,dV$: paper $A_+ = 2.96$ mA V, $A_-/A_+ = 0.44$ at $N=2000$). Four quantities could not be used as printed and were inferred by matching Fig. 4 (see table below).

**FDTest engine.** The theory document's **$X$** (response/correlation ratio) is unmeasurable here (no $k_BT$, no voltage-conductance coupling; several of its equations are wrong — see theory audit). Instead the **FDTest violation $\Delta^2$** (Geli, Monti et al.; same measure as the brain-dynamics side of the project): kick the voltage by **perturbation $\varepsilon$** ($\varepsilon\times1 \approx 13$ mV, $\times3 = 40$ mV, $\times6 = 80$ mV), compare relaxation vs. spontaneous autocorrelation over 1400 lags; $\Delta^2 > 0$ = memory. **OFF/ON/BOTH** = switch-off mirror / switch-on mirror / average estimators. Protocol: fixed-voltage pilots to pick the **operating point $V_0$** (baseline voltage, 2.5 V), chained-cycle campaign ($T_{\mathrm{relax}} = 30000$, $T_{\mathrm{samp}} = 15000$; **$M$** = total cycles $= 12 \times n_{\mathrm{seeds}}$), analysis (`fdtest_geff` light / `fdtest_analyze` full / `fdtest_observables`), Monte Carlo decision-rule calibration, acceptance gate, **null control** (no-perturbation run: $\Delta^2 \approx 1$ by construction, gate must stay silent).

**Validation.** Exact two-state/renewal benchmarks validate the estimator; a direct memory diagnostic on $g_{\mathrm{eff}}(t)$ and an exactly enumerable 11-junction toy circuit cross-check the violation; $\Delta^2$ vs. $M$ out to $M=2016$ confirms saturation.

## Switching variants: what was tried, what works

Self-contained: each row states the variant, what it means in plain words, and the tagged verdict — no need to open the report first.

| variant | in plain words | result (with conditions) | verdict |
|---|---|---|---|
| SET, $V_{\mathrm{th}} = 0.01$ V (as printed) | junctions switch ON almost as soon as any voltage appears | onset near 0.35 V vs. paper 1.9 V ($N=2000$, RESET off); only the $V_{\mathrm{th}}$ sweep moves the onset | <span style="color:#cf222e">**fail** — printed value wrong</span> |
| SET, <span style="color:#6e40c9">**$V_{\mathrm{th}} = 0.1$ V (inferred, unconfirmed)**</span> | junctions need a sizable local drop before switching | onset $\approx 1.55$ V; same value works at $N=3000$ (gap 1.55 vs. 1.9 V still open) | <span style="color:#6e40c9">**unconfirmed fit** — brackets paper at both sizes</span> |
| Ideal wires ($D = 0$) | wires conduct perfectly, only junctions resist | $\approx 1.2$–$1.3\times$ too much current ($N=2000$) | <span style="color:#cf222e">**fail** — resistance needed</span> |
| <span style="color:#6e40c9">**Wire diameter $D \approx 80$–$90$ nm (inferred, unconfirmed)**</span> | thin wires add series resistance, lowering current without moving onset | peak current crosses paper's 3.24 mA at $D \approx 80$–$90$ nm ($N=2000$) | <span style="color:#6e40c9">**unconfirmed fit** — diameter never stated</span> |
| RESET literal ($\Delta t\,P_{\mathrm{decay}} = 0.01$) | current-driven ON$\to$OFF with the paper's coefficient | $A_-/A_+ = 0.15$ vs. 0.44; nearly inactive at Fig. 4 currents ($N=2000$) | <span style="color:#cf222e">**fail** — cannot make the negative loop</span> |
| RESET scaled $3\times$–$300\times$ | stronger current-driven turn-off | never the right loop with the right currents | <span style="color:#cf222e">**fail** — systematically wrong</span> |
| <span style="color:#1a7f37">**Spontaneous relaxation $p_{\mathrm{relax}} = 10^{-4}$/step (working model)**</span> | ON junctions spontaneously decay (mean life $10^4$ steps $\approx 10$ s), the missing negative-loop mechanism | $A_-/A_+ = 0.53$ vs. 0.44 ($N=2000$, 4 seeds), 0.56 vs. 0.49 ($N=3000$); stationary in $\sim 10^3$ steps at fixed $V$ | <span style="color:#1a7f37">**pass**</span> (mechanism itself <span style="color:#6e40c9">**unconfirmed**</span>: talk, not paper) |
| $p_{\mathrm{relax}} = 3\times10^{-5}$ / $2\times10^{-4}$ / $10^{-3}$ | weaker / stronger / collapsing decay | $A_-/A_+ = 0.34$ / 0.68 / 0.99 ($N=2000$); $10^{-3}$ collapses the current | <span style="color:#cf222e">**fail** as working values — brackets only, $10^{-4}$ kept</span> |
| Fixed lifetime $L = 20000$ | every ON junction dies exactly $L$ steps after SET | fits loops ($A_-/A_+ = 0.54$) but population collapses synchronize at multiples of $L$ at fixed $V$ (cyclostationary) | <span style="color:#cf222e">**fail** as working model — unphysical for FDTest</span> |
| <span style="color:#0969da">**Gamma lifetime $k \in \{1,2,4,8,16\}$, mean $L$ (planned)**</span> | lifetime family joining exponential ($k=1$) to fixed ($k\to\infty$); a thinning filament should have increasing hazard, i.e. intermediate $k$ | not yet run; expects smooth metric variation, best $k$ = inferred shape | <span style="color:#0969da">**planned** next — cheapest candidate mechanism</span> |
| <span style="color:#0969da">**Multi-level filaments (planned)**</span> | filaments form in stages, not clean binary switch | not yet implemented; would make $g_{\mathrm{eff}}$ memory real, not a projection artefact | <span style="color:#0969da">**planned** — most interesting, most expensive</span> |
| <span style="color:#0969da">**Bipolar RESET (deferred)**</span> | reset only under reverse bias, not $I^2$ regardless of sign | needs literature/data check first; touches RESET everywhere | <span style="color:#0969da">**planned** — deferred</span> |
| <span style="color:#0969da">**Thermal cross-talk (deferred)**</span> | heat diffusion couples neighbouring junctions = real spatial memory | breaks the independent-junction speed assumption | <span style="color:#0969da">**planned** — most expensive, deferred</span> |
| <span style="color:#0969da">**Random telegraph noise (lowest priority)**</span> | formed-filament flicker, cheap to add | no clear link to the memory question | <span style="color:#0969da">**planned** — lowest priority</span> |
| Theory-doc $X$ (Eqs. 7,8,10,12–14,16) | original response/correlation ratio | wrong sign, wrong $g$-dependence, no $1/N$ (audit findings F3/F6/F10) | <span style="color:#cf222e">**fail** — superseded by FDTest $\Delta^2$</span> |

Working preset: `agnw_sim.presets.lamas2026_fig4` + $p_{\mathrm{relax}} = 10^{-4}$ + $D = 90$ nm. Open with authors: $I_{\mathrm{unit}}$ (1 mA assumed), real RESET activity, relaxation form, onset offset, Fig. 5A plateau (see `docs/lamas_questions_es.md`).

## Headline results

**Fig. 4 replication** ($N=2000$ 4 seeds, $N=3000$ 1 seed; sine 3.6 V). <span style="color:#1a7f37">**Pass**</span>: $V_{\mathrm{th}} = 0.1$ V + $D \approx 90$ nm + $p_{\mathrm{relax}} = 10^{-4}$ reproduces shape, size, and asymmetry without retuning across sizes. Remaining offsets: onset 1.55 vs. 1.9 V, $I_{\max}$ +5–10%; Fig. 4 and Fig. 5A share no single parameter set yet (<span style="color:#6e40c9">**unconfirmed**</span> with authors).

**FDTest on $g_{\mathrm{eff}}$** ($N=2000$, $V_0 = 2.5$ V, seed 1). <span style="color:#1a7f37">**Pass (non-Markov)**</span>: $\Delta^2 = 0.10$ $[0.02, 0.13]$ vs. floor 0.018 at $\varepsilon\times3$ ($M = 924$, $S/N \approx 9$–$15$); 0.14–0.30 with switch-on included; $\varepsilon\times6$ ON 0.42 $[0.31, 0.50]$. Null control $\Delta^2 \approx 0.97$–$0.99$, correctly no call (<span style="color:#1a7f37">**pass**</span>). $\Delta^2$ flat at 0.10–0.12 from $M \approx 600$ through $M = 2016$ — saturated, not data-starved (<span style="color:#1a7f37">**pass**</span>). Junction-class $\Delta^2$ (0.03–0.1) is sampling-noise artefact (same with no perturbation). ON fraction and cross-fitted linear score carry no response. Two qualifications (see Limitations): one gate check <span style="color:#cf222e">**fails**</span> (odd/even 0.174 vs. 0.106 — likely cycle 0 in the even half); switch-on $\approx 2\times$ switch-off at every $\varepsilon$, unexplained.

**What the violation does/does not show.** $\Delta^2 > 0$ says **$g_{\mathrm{eff}}$ is non-Markov** (its relaxation differs from its autocorrelation by 30–60% RMS) — not that the dynamics have memory (the full junction state is exactly Markov). Two cross-checks: toy-circuit coarse-graining reaches only 0.013 vs. measured 0.10 (not enough); a direct memory diagnostic on real $g_{\mathrm{eff}}(t)$ finds real memory at every lag, same order as $\Delta^2$, with clean nulls on memoryless synthetics. Remaining candidates: size effect or a physical mechanism (Gamma relaxation cheapest to test).

## Figures: what, how, so-what

All in `docs/report_2026-10-02/` (committed PNGs), generated by `make_figures.py` / `make_dynamics_figures.py`.

- `network_N2000.png` — **What:** the $N=2000$ network (5953 junctions); left whole network, right 0.1 mm zoom; red = $V_{\mathrm{in}}$ electrode wires, blue = ground wires, black dots = junctions. **How:** check orientations look random, electrodes only at edges, junctions wherever wires cross. **So-what:** this is the exact geometry all FDTest runs use (network seed 1).
- `dyn_conductance_traces.png` — **What:** conductance of 20 of 1000 recorded junctions over 60 s (60000 steps) at constant $V_{\mathrm{in}} = 2.5$ V, all-OFF start, $G_{\mathrm{OFF}} = 10^{-9}$ S / $G_{\mathrm{ON}} = 10^{-3}$ S. **How:** flat lines = idle, jumps = SET/RESET/relaxation events. **So-what:** junctions flip rarely and stay put for seconds — the network has slow elements (<span style="color:#1a7f37">**pass**</span>: the timescale the FDTest needs exists).
- `dyn_current_trace.png` — **What:** left total current after switching on $V_{\mathrm{in}} = 2.5$ V (60 s); right $g_{\mathrm{eff}}$ over the last 30 s. **How:** look for the rise ending and the trace going flat. **So-what:** current rises $\sim 10$ s then fluctuates around a flat level — that flat part is the stationary state the FDTest requires (<span style="color:#1a7f37">**pass**</span>).
- `dyn_autocorrelation.png` — **What:** left ensemble ON/OFF-state autocorrelation (stationary part); right normalized $g_{\mathrm{eff}}$ autocorrelation. **How:** fast initial drop vs. slow tail. **So-what:** two timescales (fast tens of steps + slow tail), consistent with the rare-switching junctions above.
- `dyn_network_state.png` — **What:** junction state after 60 s at 2.5 V; red = ON, grey = OFF; 311/1000 recorded junctions (31%) ON. **How:** count the red fraction. **So-what:** the FDTest operating point is well into the conducting regime (saturated switching, not onset) — a generalizability caveat (see Limitations).
- `onset_sweep.png` — **What:** conduction onset (first $V$ with $I > 0.05$ mA) for SET-parameter variations at $N=2000$, RESET off; baseline ($V_{\mathrm{th}} = 0.01$ V) vs. sweeps; red dashed = paper ($\approx 1.9$ V). **How:** only the $V_{\mathrm{th}}$ curve reaches the red line; $V_{\mathrm{th}} = 0.5$ V never conducts. **So-what:** the printed 0.01 V must be wrong (<span style="color:#cf222e">**fail**</span>); $V_{\mathrm{th}} = 0.1$ V brackets the paper (<span style="color:#1a7f37">**pass**</span>, <span style="color:#6e40c9">**unconfirmed**</span> with authors).
- `wire_diameter_sweep.png` — **What:** peak rising-branch current vs. wire diameter $D$ ($N=2000$; $D = 0$ = ideal wires); red dashed = paper 3.24 mA. **How:** the curve crosses the red line at $D \approx 80$–$90$ nm; onset does not move. **So-what:** the two calibrations are independent (<span style="color:#1a7f37">**pass**</span>; <span style="color:#6e40c9">**unconfirmed**</span> — diameter never stated).
- `relaxation_scan.png` — **What:** $A_-/A_+$, $A_+$, $I_{\max}$ at $N=2000$ (purple) vs. $p_{\mathrm{relax}}$; red dashed = paper; green square/orange dotted = $N=3000$ at $10^{-4}$ and paper's $N=3000$ ratio; grey line = working $10^{-4}$. **How:** the asymmetry ratio climbs with relaxation strength while $I_{\max}$ stays put; $10^{-4}$ is where purple meets red at both sizes. **So-what:** $10^{-4}$ fits at $N=2000$ and $N=3000$ (<span style="color:#1a7f37">**pass**</span>; mechanism <span style="color:#6e40c9">**unconfirmed**</span>).
- `gallery_seeds_N2000.png` / `gallery_N3000.png` — **What:** simulated loops (purple) vs. digitized paper (red): 4 seeds at $N=2000$ (top fixed $L = 20000$, bottom exponential $10^{-4}$), 1 seed at $N=3000$; paper $A_-/A_+ = 0.44/0.49$, $I_{\max} = 3.24/10.66$ mA. **How:** shape and size match at every seed. **So-what:** calibration is not a one-seed accident (<span style="color:#1a7f37">**pass**</span>); loops alone cannot tell exponential from fixed lifetime.
- `fixed_v_overview.png` — **What:** long constant-$V_{\mathrm{in}}$ runs (60000 steps, $V_{\mathrm{th}} = 0.1$ V); left $G_{\mathrm{eff}}$, right ON fraction; rows: literal RESET / exponential / fixed lifetime / last two at $N=2500$. **How:** top row never flattens, second flattens in $\sim 10^3$ steps, third crashes periodically. **So-what:** literal RESET gives no stationary state (<span style="color:#cf222e">**fail**</span> $\to$ no FDTest); exponential is stationary (<span style="color:#1a7f37">**pass**</span> $\to$ FDTest uses it); fixed lifetime is unphysical (<span style="color:#cf222e">**fail**</span>).
- `fdtest_protocol.png` — **What:** chained-cycle schedule with a real $g_{\mathrm{eff}}(t)$ trace (seed 1, mid-run stationary cycles) and the $M$-cycle ensemble average. **How:** the perturbation is invisible in one trace, clear in the average. **So-what:** this is why the FDTest must average over many cycles.
- `geff_switch_eps3_M924.png` — **What:** $g_{\mathrm{eff}}$ at $\varepsilon = 40$ mV, $M = 924$; left switch-off, right switch-on; blue = response, orange = correlation, grey = noise, red/black dashed = 5% and noise-based cutoffs. **How:** blue visibly separates from orange both ways. **So-what:** the memory signal, $\Delta^2 = 0.10$ / 0.18 (<span style="color:#1a7f37">**pass**</span>).
- `geff_switch_null.png` — **What:** same with no perturbation ($M = 220$). **How:** both curves are grey-level noise. **So-what:** control behaves correctly, $\Delta^2 \approx 1$, gate silent (<span style="color:#1a7f37">**pass**</span>).
- `fdtest_geff_eps3_M1008.png` — **What:** full-analysis response/correlation curves, $\varepsilon = 40$ mV, $M = 1008$, 1400 lags; grey line = cutoff $\tau_c$. **How:** response above correlation out to the grey line, then both in noise. **So-what:** same gap with the full campaign (<span style="color:#1a7f37">**pass**</span>).
- `geff_delta2_vs_m.png` — **What:** $\Delta^2(g_{\mathrm{eff}})$ vs. $M$ (log-log) to $M = 2016$; dashed = $1/M$-scaled floor. **How:** steep drop to $M \approx 600$ (0.80 $\to$ 0.11), then flat 0.10–0.12; floor stays well below. **So-what:** a still-falling curve would mean "need more data"; this one is flat — 0.10 is the answer, not an upper bound (<span style="color:#1a7f37">**pass**</span>).
- `fdtest_benchmarks.png`, `markov_tests_tiny.png`, `markov_tests_geff.png` — **What:** estimator vs. exact two-state/renewal curves; toy-circuit $\Delta^2$ (full state exactly 0, ON-count 0.013); direct diagnostic on real $N=2000$ data at every lag + memoryless-synthetic nulls. **How:** estimator hugs the exact curves; toy 0.013 $\ll$ measured 0.10; diagnostic finds memory at every lag, nothing on synthetics. **So-what:** estimator validated and two artefact explanations ruled out (<span style="color:#1a7f37">**pass**</span>).

## Limitations

- <span style="color:#6e40c9">**Unconfirmed (needs authors):**</span> everything rests on the working relaxation model and inferred $V_{\mathrm{th}}, D$ — none confirmed by the authors.
- <span style="color:#0969da">**Planned (statistical power):**</span> $M \approx 500$–$2016$ here vs. $\approx 5000$ in the brain-dynamics analogue — a smaller real effect could go undetected.
- <span style="color:#0969da">**Planned (generalizability):**</span> one network, one operating point (2.5 V, near-saturated switching); untested at 1.6 V near onset or other seeds; regime comparison and network-level controls (rewired nulls, isolated-junction) are future work.
- <span style="color:#cf222e">**Fail (acceptance gate):**</span> odd- vs. even-numbered cycles give inconsistent $\Delta^2$ (0.174 vs. 0.106) — likely cycle 0 (empty-network start) sitting in the "even" half; not yet re-confirmed with cycle 0 excluded.
- <span style="color:#cf222e">**Fail-or-bias (switch-on/off asymmetry):**</span> switch-on $\Delta^2$ consistently $\approx 2\times$ switch-off at every $\varepsilon$ tested, no identified mechanism — real physics or analysis bias.
- Noise floor covers only $\langle Z \rangle$ noise and runs $\approx 10\%$ low against a Markov Monte Carlo; the $\Delta^2$ vs. $M$ curve (flat at 0.10–0.12 from $M \approx 600$) bounds this.
- Slow junctions (relaxation up to $10^4$ steps) are only partly covered by the 14000-step window.
- A direct response/correlation cross-check (voltage-step nudge, watch $g_{\mathrm{eff}}$ relax) is <span style="color:#0969da">**planned**</span> as an independent check.

## Status: where we are / doing / next

- **Where we are:** simulator + FDTest campaign done at one operating point ($N=2000$, $V_0 = 2.5$ V): $g_{\mathrm{eff}}$ is non-Markov ($\Delta^2 = 0.10$, saturated to $M = 2016$, null control clean), Fig. 4 reproduced at two sizes with the working model.
- **What we are doing:** <span style="color:#0969da">**planned next:**</span> Gamma-lifetime scan ($k \in \{1,2,4,8,16\}$, mean $L = 20000$) as the cheapest physical-memory candidate, or a mid-size Monte Carlo bridge for the size-effect explanation (Juan's call).
- **What to do next + data asks:** <span style="color:#0969da">**planned:**</span> exclude cycle 0 and re-run the gate; explain ON $>$ OFF; operating point near onset (1.6–1.8 V); fixed-lifetime network positive control; more $\varepsilon\times6$ statistics; $N = 1500/3000$ size dependence; $S/N \ge 3$ gate in analyzers. <span style="color:#6e40c9">**Ask authors:**</span> $I_{\mathrm{unit}}$, real RESET activity, relaxation form, $V_{\mathrm{th}}$, $D$, Fig. 5A plateau (see `docs/lamas_questions_es.md`).

## Install

Requires Python ≥3.10.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Dependencies: `numpy`, `scipy`, `h5py`, `matplotlib` (runtime), `pytest` (dev). See [`pyproject.toml`](pyproject.toml).

Run the test suite with the venv's Python, not a bare `pytest` — `scripts` needs to be importable as a package:

```bash
.venv/bin/python -m pytest -q
```

(291 tests as of this writing, ~60 s.)

## Repository layout

```
agnw_sim/        the installable package: circuit, dynamics, simulator, FDTest estimator
scripts/         standalone CLI drivers (python -m scripts.<name>), not imported by the package
tests/           pytest suite for agnw_sim/ and scripts/
docs/            theory document, progress reports, design specs, figures, results
validation_output/   git-ignored: everything scripts/ writes (run data, figures, JSON) — fully regenerable
memory/, RESUME_FROM_HERE.md, docs/superpowers/   Claude Code session-continuity files (see below)
```

### Package: `agnw_sim/`

`agnw_sim/__init__.py` re-exports nothing — always import submodules directly, e.g.
`from agnw_sim.dynamics import SimParams`, `from agnw_sim.simulate import run_simulation`.

| module | purpose | key names |
|---|---|---|
| `geometry.py` | random stick network generation, wire intersections | `Network`, `build_network(n_wires, area, length, rng, ...)` |
| `circuit.py` | sparse Kirchhoff solve on the plain (non-dual-node) graph | `solve_circuit`, `total_current`, `junction_currents` |
| `dualnode.py` | the dual-node circuit of Lamas et al. Sec. 2.2 (each junction $\to$ two nodes + resistive wire segments) | `DualNodeCircuit` |
| `cached_solver.py` | opt-in solver acceleration (rescale/rank-1 update instead of a full refactorization every step) | `CachedSolver` (6.4x faster at stationarity) |
| `dynamics.py` | stochastic SET/RESET/spontaneous-relaxation junction switching (described in `docs/switching_model.md`) | `SimParams`, `p_set`, `p_reset`, `update_dynamics`, `update_dynamics_lifetime` |
| `presets.py` | named parameter sets matched to the paper's Fig. 4 | `lamas2026_fig4(dt=1e-3)` |
| `waveforms.py` | drive-voltage generators | `triangle`, `sine`, `step`, `ramp`, `white_noise_probe` |
| `simulate.py` | the time-stepping loop | `run_simulation(...)`, `SimulationResult` |
| `io.py` | HDF5 trajectory storage | `save_result`, `load_result`, `reconstruct_network` |
| `visualize.py` | sanity-check plots | `plot_network`, `plot_iv_hysteresis`, `plot_conductance_trace` |
| `analysis.py` | trajectory analysis helpers | `autocorrelation`, `hysteresis_loop_area`, `entropy_production` |
| `benchmarks.py` | exact FDTest curves for analytically solvable junctions, used to validate the estimator | `two_state_fdtest`, `renewal_fdtest` |
| `fdtest.py` | **the FDTest estimator** (pure functions) | `select_cutoff`, `delta_squared`, `binary_fdtest`, lag-binning helpers |
| `fdtest_protocol.py` | chained-cycle schedule, stage extraction, pilot analysis, acceptance-gate checks | `Schedule`, `extract_stage_arrays`, `choose_v0`/`choose_epsilon`, `gate_*` |
| `fdtest_observables.py` | alternative observables (ON fraction, cross-fitted linear score) | `on_fraction`, `fit_lda_direction`, `crossfit_scores` |
| `exact_chain.py` | exact one-step transition data of a small circuit (every 2^J state solved) | `dynamic_junctions`, `state_data`, `bridge_network` |
| `markov_tests.py` | memory diagnostics (regression gain, kNN conditional mutual information) and exact finite-chain FDTest quantities | `regression_memory_gain`, `cmi_test`, `transition_matrix`, `exact_delta2` |

### Drivers: `scripts/`

Run every driver as `python -m scripts.<name>` from the repo root, with `.venv/bin/python`. Each script's own
docstring (`python -m scripts.<name> --help`, or just read the top of the file) has the full argument list; this is
a map of what each one is for.

| script | what it does |
|---|---|
| `fdtest_pilot.py` | fixed-voltage pilot runs, to choose the FDTest operating point ($V_0$) |
| `fdtest_run.py` | generates one seed's worth of chained FDTest cycles — the actual simulation |
| `fdtest_analyze.py` | full analysis: $g_{eff}$ + per-junction classes, bootstrap intervals, noise floors, acceptance gate, figures. Memory scales with $M\times K$ (junctions); needs ~9 GB at $M=480$, $K=1000$ |
| `fdtest_geff.py` | lighter $g_{eff}$-only analysis (switch-on mirror + noise-based cutoff), no per-junction memory cost |
| `fdtest_observables.py` | ON-fraction and cross-fitted linear-score analysis |
| `fdtest_interval_calibration.py` | Monte Carlo calibration of the decision rule against synthetic junctions with a known answer |
| `fdtest_tau_slow.py` | slowest per-junction relaxation time from a pilot run |
| `fdtest_common.py` | *not a driver* — shared setup helper reused by the other `fdtest_*` scripts |
| `geff_delta2_vs_m.py` | $\Delta^2(g_{eff})$ vs. number-of-cycles $M$ convergence curve, reusing already-recorded runs (no re-simulation) |
| `fixed_v_stationarity.py` | long fixed-voltage runs, to check a model variant reaches a stationary state |
| `replicate_lamas_fig4.py` | reproduces the paper's Fig. 4 I-V hysteresis loops under a sine drive |
| `explore_fig4_onset.py` | SET-parameter sweep, used to diagnose the conduction-onset mismatch against the paper |
| `explore_wire_resistance.py` | wire-diameter sweep (found $D\approx90$-$100$ nm) |
| `tiny_circuit_exact.py` | exact FDTest of a tiny, fully enumerable circuit (the "Markov test 2" of the report) |
| `markov_geff_test.py` | direct memory diagnostic on real $g_{eff}(t)$ data (the "Markov test 4" of the report) |

### `docs/`

| path | contents |
|---|---|
| `report_YYYY-MM-DD/` | one directory per report, named by the date of its latest edition: `report.md` / `.pdf`, figures and their generator scripts side by side |
| `report_2026-10-02/` | simulation progress reports: `report.md` (full, the primary internal reference for results and methods), `report_brief.md` / `.pdf` (collaborator-facing: no repo-internal file/script references, plain prose), their PNGs and the `make_figures.py`/`make_dynamics_figures.py` generators |
| `report_2026-10-08/` | first measured data (FDTest on the lab measurement of 2026-10-08): `report.md` / `.pdf`, figures, `make_figures.py` |
| `results/` | analysis outputs meant to be kept (unlike `validation_output/`): `interval_calibration.md`, `markov_tests.md` |
| `theory_audit/` | audit of the original theory document + Fig. 4 replication analysis; has its own README |
| `on_hold/optimization_framework/` | an optimization-framework implementation, parked because its core formula is unvalidated |
| `on_hold/validate_p_decay/` | a validation script moved out of `scripts/`: it uses the pre-fix mis-scaled model (double-applied $\Delta t$); do not run it |
| `superpowers/{specs,plans}/` | design specs and task plans from past work sessions — a paper trail of *why*, not needed to run anything |
| `switching_model.md` | the current junction switching model (SET, RESET, spontaneous relaxation): equations, per-step update, working-model parameters and their provenance |
| `lamas_questions_es.md` | draft list of open questions for the paper's authors |
| `lag_binning_design.md`, `alt_observables_design.md` | design docs for two implemented analysis extensions |
| `response_correlation_options.md` | a deferred design doc (not implemented) |
| `md_to_pdf.sh` | builds a report PDF from its Markdown source — see [below](#building-the-progress-report-pdfs) |

### `tests/`

Standard pytest layout, one file per package module plus a few smoke tests for the `scripts/` drivers. Run with
`.venv/bin/python -m pytest -q`.

## Running a simulation

```python
import numpy as np
from agnw_sim.presets import lamas2026_fig4
from agnw_sim.waveforms import sine
from agnw_sim.simulate import run_simulation

params, extra = lamas2026_fig4(dt=1e-3)          # SimParams, {"wire_diameter": 90e-9}
# waveform_fn takes the simulation time t (seconds) and returns V_in(t); waveforms.py's functions
# (sine, triangle, step, ramp) take t as their first argument, so wrap the fixed ones in a lambda:
waveform_fn = lambda t: sine(t, amplitude=3.6, frequency=1 / (50_000 * params.dt))

result = run_simulation(
    n_wires=2000, area=1e-6, length=70e-6,        # 1 mm^2 square, 70 um wires
    waveform_fn=waveform_fn, n_steps=50_000, dt=params.dt, params=params,
    rng=np.random.default_rng(1), wire_diameter=extra["wire_diameter"],
)
# result.I_total, result.G_sampled, ... — see agnw_sim.simulate.SimulationResult
```

For a constant-drive run instead, use `lambda t: step(t, amplitude=V0, t_on=0)` (from `agnw_sim.waveforms`). Add
`solver="cached"` to `run_simulation(...)` for the 6.4x-faster solver (dual-node graph only, i.e. `wire_diameter`
set) once the network has largely stopped switching.

## Running the FDTest

The FDTest needs a *campaign* of chained-cycle runs at one operating point, then an analysis pass over them. This
is the exact, previously-validated sequence (see `RESUME_FROM_HERE.md` for the full history):

```bash
# 1. Pick the operating point (V0) with fixed-voltage pilots at a few candidate voltages...
for v in 1.4 1.6 1.8 2.0 2.5; do
    .venv/bin/python -m scripts.fdtest_pilot run 2000 "$v" --n-steps 60000
done
# ...then compare their mean g_eff, drift and ON-probability spread:
.venv/bin/python -m scripts.fdtest_pilot summarize 2000

# 2. Generate the campaign: one call per seed (repeat for as many seeds as you want M = 12 * n_seeds cycles)
.venv/bin/python -m scripts.fdtest_run 2000 --v0 2.5 --eps 0.04 --t-relax 30000 --t-samp 15000 \
    --cycles 12 --seed 1 --n-lags 600 --out validation_output/my_run
# ... --seed 2, --seed 3, ... (independent processes; run many in parallel)

# 3. Analyze
.venv/bin/python -m scripts.fdtest_geff 2000 --out validation_output/my_run --n-boot 200 --n-lags 1400
.venv/bin/python -m scripts.fdtest_analyze 2000 --out validation_output/my_run --n-lags 1400
.venv/bin/python -m scripts.fdtest_observables 2000 --out validation_output/my_run --n-lags 1400
```

`fdtest_analyze.py` is the full analysis (network + per-junction classes, acceptance gate); `fdtest_geff.py` is a
lighter $g_{eff}$-only pass with no per-junction memory cost — use it first, or when `fdtest_analyze.py`'s RAM
footprint at your $M$ doesn't fit. All three read the same run files under `--out` and never re-simulate.

To calibrate the decision rule itself (percentile vs. reverse-percentile intervals, false-call rates):

```bash
.venv/bin/python -m scripts.fdtest_interval_calibration --reps 60 --Ms 120,240,480 --procs 3
```

## Other analyses

```bash
# Fig. 4 I-V hysteresis replication (lamas_epsE = exponential relaxation, probability E per step; the working value is 1e-4)
.venv/bin/python -m scripts.replicate_lamas_fig4 2000 lamas_eps1e-4 1

# Fixed-voltage stationarity check (same VARIANT tokens as replicate_lamas_fig4.py)
.venv/bin/python -m scripts.fixed_v_stationarity 2000 lamas_eps1e-4 2.5

# Exact tiny-circuit FDTest (Markov test 2)
.venv/bin/python -m scripts.tiny_circuit_exact --out validation_output/markov_tests/tiny

# Direct memory diagnostic on real g_eff(t) data (Markov test 4; needs a campaign from step 2 above first)
.venv/bin/python -m scripts.markov_geff_test --dir validation_output/my_run --N 2000

# Delta^2(g_eff) vs. M convergence curve (reuses an existing campaign, no re-simulation)
.venv/bin/python -m scripts.geff_delta2_vs_m --dir validation_output/my_run --N 2000

# Dynamics-under-constant-drive figures (conductance traces, current, autocorrelation) with the working model
# (needs validation_output/fdtest/pilot_N2000_V2.5.npz from a prior `fdtest_pilot run 2000 2.5` call)
PYTHONPATH=. .venv/bin/python docs/report_2026-10-02/make_dynamics_figures.py
```

`docs/on_hold/validate_p_decay/validate_p_decay.py` uses an early, mis-scaled model (double-applies $\Delta t$ to
the switching probabilities) superseded on 2026-09-18 — do not run it; see its own `README.md` and
`validation_output/old_model/README.md`. It was moved out of `scripts/` for exactly this reason.

## Building the report PDFs

`md_to_pdf.sh` resolves the figures' image paths relative to the current directory, so it must be run **from
inside the report's directory**, not the repo root — running it elsewhere silently produces a PDF with every figure
missing (warnings only, exit status 0):

```bash
cd docs/report_2026-10-02
bash ../md_to_pdf.sh report.md
bash ../md_to_pdf.sh report_brief.md
```

## Claude Code workflow files

`memory/`, `RESUME_FROM_HERE.md`, and `docs/superpowers/` are continuity artifacts for Claude Code sessions working
on this project across time — a persistent memory store, a "resume with zero context" handoff document, and design
specs/plans from past work sessions, respectively. None of them is needed to install, run, or understand the
simulator or the FDTest; skip them unless you're picking up the project inside a Claude Code session.
