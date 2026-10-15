# Junction switching model (current)

The single up-to-date description of how a junction switches between ON and OFF in `agnw_sim`. The code is
authoritative (`agnw_sim/dynamics.py`). This page collects the equations, the per-step update, the parameter values
of the working model and where each one comes from. Written 2026-09-25. It supersedes §2.4 and the parameter table
of `superpowers/specs/2026-09-17-agnw-network-simulator-design.md`.

## 1. State

Each junction is binary: G ∈ {G_OFF = 1e-9 S, G_ON = 1e-3 S}. There is no intermediate conductance or filament
variable. In the dual-node circuit (`agnw_sim/dualnode.py`), the junction is the memristor edge between its two
nodes, and its voltage drop dV is taken across that edge only.

## 2. Mechanisms

| # | Transition | Name | Per-step probability | Code | Source |
|---|---|---|---|---|---|
| 1 | OFF -> ON | SET (voltage-driven) | p_set = Δt·P0·max(0, 1 - exp(-α(\|dV\| - V_th))) | `dynamics.p_set` | Lamas 2026, Eq. 1 |
| 2 | ON -> OFF | RESET (current-driven) | p_reset = min(1, Δt·P_decay·(I/I_unit)²), I = G·dV | `dynamics.p_reset` | Lamas 2026, Eq. 2 |
| 3 | ON -> OFF | Spontaneous relaxation (exponential) | p_relax, constant; independent of V, I and history | `SimParams.p_relax`, combined in `update_dynamics` | not in the paper; recalled from a Lamas talk; unconfirmed (lamas_questions_es.md Q3) |
| 3' | ON -> OFF | Fixed lifetime (deterministic variant of 3) | OFF exactly `lifetime_steps` steps after SET | `update_dynamics_lifetime` | same talk, alternative reading; not used (synchronized collapses at multiples of L, report §5) |
| 3'' | ON -> OFF | Gamma-distributed lifetime | lifetime ~ Gamma(k, mean L), drawn at SET | **not implemented** | approved in `superpowers/specs/2026-09-22-switching-mechanisms-candidates.md` |

**Rates vs per-step products.** `SimParams.P0` and `P_decay` are *rates* (multiplied by Δt inside `p_set` /
`p_reset`). Lamas 2026 Table 1 lists the per-step products Δt·P0 = 0.4 and Δt·P_decay = 0.01, so build parameters
with `SimParams.from_step_probs(0.4, 0.01, dt=...)`. Passing 0.4 / 0.01 as P0 / P_decay applies Δt twice; this bug
was fixed on 2026-09-19.

## 3. One time step (`update_dynamics`)

1. The circuit is solved with the states G(t) at the start of the step. This gives each junction's dV and
   I = G·dV (an ON junction carries G_ON·dV).
2. Each OFF junction turns ON with probability p_set(|dV|).
3. Each ON junction turns OFF with probability p_off = 1 - (1 - p_reset(I))(1 - p_relax). RESET and relaxation are
   independent channels, and the junction stays ON only if neither fires.
4. The update is synchronous. All draws use the start-of-step states: SET is evaluated only for OFF junctions,
   RESET and relaxation only for ON ones. A junction cannot switch twice in one step.

The model is polarity-symmetric (only |dV| and I² enter). Junctions interact only through the circuit.

## 4. Working-model parameters (`presets.lamas2026_fig4` + p_relax)

| Parameter | Value | Provenance |
|---|---|---|
| Δt | 1e-3 s | paper |
| Δt·P0 | 0.4 | paper, Table 1 |
| α | 1 V⁻¹ | paper, Table 1 |
| V_th | 0.1 V | **inferred** from Fig. 4 (Table 1 prints 0.01 V, which puts the onset near 0.35 V) |
| Δt·P_decay | 0.01 | paper, Table 1 |
| I_unit | 1 mA | **inferred** (unit of I in the RESET law not stated; lamas_questions_es.md Q1). `SimParams` default is 1.0 A |
| G_ON / G_OFF | 1e-3 / 1e-9 S | framework defaults |
| p_relax | 1e-4 per step (mean lifetime 1e4 steps) | **working value**, calibrated on Fig. 4 loops (decision 2026-09-20; report §5) |
| wire diameter (circuit) | 90 nm | **inferred** from Fig. 4 peak currents |

## 5. How active each mechanism is

This depends on the operating point. Under the Fig. 4 protocol (sine drive, 3.6 V amplitude), RESET contributes
little, and relaxation is what reproduces the negative loop (report §5). Under a steady random drive of 0.75-2.75 V
(N = 2000, hold 30 steps), measured in the reservoir-computing project (`AgNW_Reservoir_Computing`,
`scripts/switching_rates.py`, `docs/SWITCHING_MECHANISMS.md`):

* about 0.12-0.23 SET events per step in the whole network (~6000 junctions), balanced by relaxation (0.12-0.19
  per step);
* RESET at 8-23 % of the relaxation rate, growing with voltage; secondary but not negligible;
* only 0.3-0.6 % of OFF junctions above V_th at any moment.

## 6. Open questions

Q1 (unit of I), Q2 (real RESET activity), Q3 (third mechanism), Q4 (functional form of P_RESET) and Q14 (update
order) in `lamas_questions_es.md` are unanswered. Candidate extensions (multi-level filaments, polarity-dependent
reset, thermal cross-talk, telegraph noise, Gamma lifetime) are costed in the 2026-09-22 decision memo.
