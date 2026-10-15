# Junction switching mechanisms (ON <-> OFF)

How a junction's conductance switches in the framework model (`agnw_sim`, AgNW_FDT_Memristor_Framework, commit
`286cb4d`), how often each mechanism fires at the reservoir operating points, what this project adds, and how well
the framework documents all of it. Written 2026-09-25.

## 1. The junction state

Each junction is **binary**: G $\in$ {$G_{\mathrm{OFF}}$ = $10^{-9}$ S, $G_{\mathrm{ON}}$ = $10^{-3}$ S}. There is no intermediate conductance and no
filament variable. In the dual-node circuit, the junction is the memristor edge between the two nodes it has on its
two wires, and its voltage drop dV is taken across that edge only.

## 2. The mechanisms

| # | Transition | Name | Per-step probability | Depends on | Source | Code | Working-model values |
|---|---|---|---|---|---|---|---|
| 1 | OFF -> ON | **SET** (voltage-driven) | $p_\mathrm{set} = \Delta t\,P_0\,\max\!\big(0,\ 1 - e^{-\alpha(\lvert dV\rvert - V_\mathrm{th})}\big)$ | $\lvert dV \rvert$ across the junction | Lamas 2026, Eq. 1 | `agnw_sim/dynamics.py:48` `p_set` | $\Delta t\,P_0 = 0.4$, $\alpha = 1\ \mathrm{V^{-1}}$, $V_{\mathrm{th}} = 0.1$ V (paper prints 0.01 V; 0.1 inferred from Fig. 4) |
| 2 | ON -> OFF | **RESET** (current-driven) | $p_\mathrm{reset} = \min\!\big(1,\ \Delta t\,P_\mathrm{decay}\,(I/I_\mathrm{unit})^2\big)$, $I = G_\mathrm{ON}\,dV$ | junction current | Lamas 2026, Eq. 2 (Eq. 4 in some notes) | `dynamics.py:55` `p_reset` | $\Delta t\,P_\mathrm{decay} = 0.01$, $I_\mathrm{unit} = 1$ mA (unit not stated in the paper; inferred) |
| 3 | ON -> OFF | **Spontaneous relaxation** (exponential) | $p_{\mathrm{relax}}$ per step, constant | nothing (independent of V, I, history) | not in the paper; recalled from a Lamas talk, unconfirmed | `dynamics.py:38` `SimParams.p_relax`, combined at `dynamics.py:69-71` | $p_{\mathrm{relax}} = 10^{-4}$ (mean lifetime $10^{4}$ steps = 10 s); calibrated on Fig. 4 loops |
| 3' | ON -> OFF | **Fixed lifetime** (deterministic variant of 3) | OFF exactly `lifetime_steps` after SET | age since SET | same talk, alternative reading | `dynamics.py:85` `update_dynamics_lifetime` | not used: synchronizes the population (collapses at multiples of L) |
| 3'' | ON -> OFF | **Gamma-distributed lifetime** (planned) | lifetime ~ Gamma(k, mean L) drawn at SET; k = 1 is #3, k -> $\infty$ is #3' | age since SET | framework decision memo 2026-09-22 | not implemented | — |

**One time step** (`update_dynamics`, synchronous for all junctions):

1. The circuit is solved with the current states $G(t)$. This gives each junction's $dV$ and $I = G\,dV$.
2. An OFF junction turns ON with probability p_set(|dV|).
3. An ON junction turns OFF with probability $p_{\mathrm{off}} = 1$ - (1 - p_reset(I))(1 - $p_{\mathrm{relax}}$). RESET and relaxation are
   independent channels, and the junction survives only if neither fires.
4. All draws use the states at the start of the step: SET is evaluated only for OFF junctions, RESET and relaxation
   only for ON junctions.

The model is symmetric in polarity (only |dV| and I$^{2}$ enter), and junctions interact only through the circuit.

**Physical picture (qualitative; the framework model is phenomenological).** SET corresponds to the voltage-driven
formation of a conducting Ag filament across the insulating (PVP) coating at a wire crossing (electrochemical
metallization). RESET corresponds to current-induced (Joule-heating) rupture of the filament, which motivates the
I$^{2}$ dependence. Spontaneous relaxation corresponds to the volatile, diffusive dissolution of thin Ag filaments with
no drive. This behaviour is known from Ag-based diffusive memristors (e.g. Wang et al., *Nat. Mater.* 16, 101 (2017)).
Candidate refinements already listed in the framework's decision memo are multi-level filaments, polarity-dependent
reset, thermal cross-talk between junctions, and telegraph noise.

## 3. How often each mechanism fires at the reservoir operating points

`scripts/switching_rates.py`, 40 snapshots per run of E1 (i.i.d. input, hold 30, network seed 1). Junction voltages
are recomputed from each snapshot and the probabilities evaluated with the framework's functions. Output:
`results/switching_rates.json`.

| input window | ON fraction | SET events/step | RESET events/step | relaxation events/step | OFF junctions with $\lvert dV \rvert > V_{\mathrm{th}}$ | largest ON-junction current |
|---|---|---|---|---|---|---|
| 0.75-1.75 V | 0.21 | 0.12 | 0.009 | 0.12 | 0.3 % | 0.11 mA |
| 1.25-2.25 V | 0.26 | 0.16 | 0.023 | 0.16 | 0.4 % | 0.17 mA |
| 1.75-2.75 V | 0.32 | 0.23 | 0.043 | 0.19 | 0.6 % | 0.20 mA |

### Conclusions

* SET balances the two OFF channels (stationary state). Relaxation is the main OFF channel. RESET is secondary but
  **not negligible**: 8-23 % of the relaxation rate, growing with voltage. The preset docstring's "RESET essentially
  inactive" (`agnw_sim/presets.py:11-12`) refers to the Fig. 4 conditions and should be qualified for other
  operating points.
* Switching is rare and concentrated. Only ~0.15-0.23 events per step happen in the whole network of ~6000
  junctions, and only 0.3-0.6 % of OFF junctions are above threshold at any moment. One input sample of 30 steps
  therefore changes the state by ~5-7 junction flips, about half of them relaxation events on randomly chosen ON
  junctions, independent of the input.
* This is the microscopic reason the reservoir results find junction states to be noise: per input sample, a
  handful of stochastic flips carry the input's effect, and an equal number of input-independent relaxation flips
  overwrite it (RESULTS.md sections 2, 9, 10).

## 4. Variants used in this project (not in the framework)

| Variant | What changes | Where | Used in |
|---|---|---|---|
| Mean-field | Each junction carries its ON probability $P_j$, updated with the exact expectation of steps 2-4: $P \leftarrow P\,(1 - p_\mathrm{off}) + (1 - P)\,p_\mathrm{set}$, with $p_\mathrm{reset}$ evaluated at $I_\mathrm{ON} = G_\mathrm{ON}\,dV$. Circuit solved with $G_\mathrm{OFF} + P_j\,(G_\mathrm{ON} - G_\mathrm{OFF})$. | `agnw_rc/meanfield.py` (tested against Monte Carlo of `update_dynamics`) | N1 (M1), N2 |
| Heterogeneous relaxation | $p_{\mathrm{relax}}$ per junction, log-uniform in [$10^{-4}$, $p_{\mathrm{max}}$], fixed per network | `agnw_rc/config.py` `junction_p_relax` | N2 (mean-field only) |

## 5. Documentation status in the framework

| Framework document | Covers | Status |
|---|---|---|
| `agnw_sim/dynamics.py` docstrings | All three mechanisms: rates vs per-step products (`from_step_probs`), `I_unit`, `p_relax`, `lifetime_steps`, the combination rule (comment at lines 69-71) | **Current and complete**. This is the authoritative description, but the combination rule appears only as a code comment. |
| `agnw_sim/exact_chain.py` docstring (line 32) | $p_{\mathrm{off}} = 1$ - (1 - $p_{\mathrm{reset}}$)(1 - $p_{\mathrm{relax}}$) | Current. |
| `README.md` module table (line 68) | Names SET/RESET/spontaneous relaxation | Current, but no equations or parameter values. |
| `docs/superpowers/specs/2026-09-17-agnw-network-simulator-design.md` §2.4 (lines 117-136) and parameter table | SET and RESET only | **Stale.** No relaxation or lifetime. $p_{\mathrm{reset}}$ has no $I_{\mathrm{unit}}$. The table still lists $V_{\mathrm{th}} = 0.01$ V, $P_0$ as "0.1-0.4 per $\Delta t$" and $P_\mathrm{decay} = 0.01\ \mathrm{s\,A^{-2}}$, the convention that caused the $\Delta t$ double counting fixed on 2026-09-19. Unlike the wire-resistance section (line 97), §2.4 has no "Superseded" note. |
| `docs/superpowers/specs/2026-09-22-switching-mechanisms-candidates.md` | The three current mechanisms plus six candidates; Gamma lifetime approved | Current as a decision record. The approved Gamma lifetime is **not implemented** yet. |
| `docs/REPORT_FDTest_progress*.md` | Working model (dual-node, $V_{\mathrm{th}} = 0.1$ V, D = 90 nm, $p_{\mathrm{relax}} = 10^{-4}$) and its Fig. 4 calibration | Current. |
| `docs/lamas_questions_es.md` | Open questions: Q1 unit of I, Q2 real RESET activity, Q3 third mechanism, Q4 form of P_RESET, Q14 update order | Current, all **unanswered**. |
| `memory/lamas-third-mechanism-relaxation.md` | History and calibration of the relaxation channel, and the 2026-09-20 decision | Current (project memory, not user-facing docs). |

### Conclusions

* The mechanisms are **correctly implemented and documented in the code**. The history and rationale are spread
  over the decision memo, the reports and the memory notes.
* There is **no single up-to-date, user-facing description** of the full switching model: equations, combination
  rule, update order, parameter values and their provenance. The closest document, the original design spec
  (§2.4 and its parameter table), is stale and not marked as such.
* The "RESET essentially inactive" statement in `presets.py` is true for the Fig. 4 conditions it was written for.
  At the reservoir operating points, RESET is 8-23 % of the OFF events.
* **Fixes applied 2026-09-25** (framework working tree, not committed; 291 framework tests pass): (1)
  "Superseded 2026-09-25" notes on spec §2.4 and its parameter table, pointing to the new page; (2) new
  `docs/switching_model.md` (equations, update rule, parameters with provenance, activity, open questions), linked
  from the framework README; (3) the RESET remark in `presets.lamas2026_fig4` is qualified.
