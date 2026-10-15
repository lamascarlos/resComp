# N10: a volatile, two-time-scale junction constrained by measured AgNW data

Status: **implemented 2026-09-27** (`agnw_rc/volatile.py`, model "volatile" in `multidrive`, tests/test_volatile.py); phenomenology and N10-1 next. Evidence base:
[LITERATURE_NOTES.md](LITERATURE_NOTES.md), section "Measured AgNW network dynamics". Context: E4 showed that
neither the binary framework junction nor the Fig. 4-calibrated filament junction (N4) computes, even with inputs
held 0.3-3 s (RESULTS section 13). Lamas Fig. 4 is a simulation that is only qualitatively similar to measurements,
so it is a secondary check here, not the calibration target.

## 1. What the model must reproduce (measured phenomenology)

| # | phenomenon | measured in | N4 | binary (framework) |
|---|---|---|---|---|
| M1 | **Spontaneous relaxation** after the stimulus ends, at any voltage below activation (no hold band) | Diaz-Alvarez 2019; Milano 2023; Diaz Schneider 2024 | no: hold band, decay only below $V_{\mathrm{reset}}$, 25 s | yes ($p_{\mathrm{relax}}$), 10 s, noisy |
| M2 | **Two time scales**: short-term relaxation (tens of ms to ~10 s) and amplitude/duration-dependent consolidation into long-lasting states | Milano 2023; Diaz-Alvarez 2019 | one time scale | one time scale |
| M3 | Threshold activation per junction, **$V_c \approx 0.4$ V** | Diaz Schneider 2024 (model fit); Milano 2022 edge model (fast potentiation above $\approx 0.4\text{–}0.5$ V) | $V_{\mathrm{set}}$ 0.06 V | $V_{\mathrm{th}}$ 0.1 V |
| M4 | **Power/current-driven rupture** at high current | Diaz Schneider 2024 ($P_{\mathrm{cf}} \approx 1$ mW) | no | yes (current-driven RESET) |
| M5 | **Random** relaxation times (between cycles and junctions), abrupt drops and plateaus in the network conductance | Diaz-Alvarez 2019 | no | yes (too strong) |
| M6 | Paired-pulse facilitation: a second pulse within the relaxation time gives a larger response | Milano 2023 | weakly | yes |
| M7 | **Non-polar** switching (both polarities potentiate) | Milano 2022 model ($\lvert V \rvert$); Diaz Schneider 2025 | bipolar | non-polar |
| M8 | The measured devices compute with inputs of **1 ms to ~1 s** | Zhu 2023 (1 ms/pixel); Milano 2022 | frozen | noise-limited |

## 2. Model

Two state variables per junction $j$, both in $[0, 1]$:

* **$x_j$, the volatile filament** (thin, labile). Rate-balance law of Milano et al. (2022), extended by power
  rupture:

$$
\frac{dx_j}{dt} = k_p(V_j)\,\Theta(P_c - P_j)\,(1 - x_j) - \big[k_{d,j}(V_j) + k_P(P_j)\big]\,x_j
$$

$$
k_p(V) = \min\!\big(k_{p0}\,e^{\eta_p s_V \lvert V\rvert},\ k_\mathrm{max}\big), \qquad
k_{d,j}(V) = \frac{e^{-\eta_d s_V \lvert V\rvert}}{\tau_{s,j}}, \qquad
k_P(P) = k_{P0}\,\Theta(P - P_c), \qquad P_j = G_j V_j^2
$$

  with $V_j$ the junction voltage drop, $P_j$ the dissipated power and $\Theta$ the step function. $s_V$ (`v_scale`,
  5) converts Milano's network-level fit to one junction, and $k_\mathrm{max} = 100\ \mathrm{s^{-1}}$ caps the growth
  rate (both set during implementation, see RESULTS section 15). Growth stops while $P_j > P_c$ (Joule melting):
  Milano's $k_p$ is astronomically large at $\lvert V\rvert \gtrsim 1$ V ($5 \times 10^{24}\ \mathrm{s^{-1}}$ at 2 V),
  so a competing rupture rate alone could never limit a filament (found while testing, 2026-09-27). At zero bias
  $x_j$ relaxes with time $\tau_{s,j}$ (M1), drawn log-normally across junctions (median $\tau_s$, spread
  $\sigma_\mathrm{ln}$) for M5.
* **$y_j$, the consolidated filament** (thick, stable). It grows only while the volatile filament is formed and the
  drive is strong, and decays slowly (M2):

$$
\frac{dy_j}{dt} = k_c\,x_j^{\,n}\,(1 - y_j)\,\Theta(P_j - P_{c2}) - \frac{y_j}{\tau_l}, \qquad \tau_l \gg \tau_s
$$

  It is gated on junction power, not voltage: formed junctions self-regulate near threshold (about 0.07–0.1 V at
  any drive), while their power separates strong from weak drive ($\le 4 \times 10^{-8}$ W at 2 V, median
  $2 \times 10^{-6}$ W at 3.5 V; $P_{c2} = 1\ \mu\mathrm{W}$).

* **Conductance**: tunnelling through the remaining gap, approximated by a log-linear interpolation (the gap
  shrinks linearly with the filament length, and tunnelling is exponential in the gap):

$$
G_j = G_\mathrm{off}\left(\frac{G_\mathrm{on}}{G_\mathrm{off}}\right)^{\min(1,\ x_j + y_j)}
$$

  with the framework's $G_\mathrm{on} = 10^{-3}$ S and $G_\mathrm{off} = 10^{-9}$ S, so the network current scale
  stays comparable with the framework and N4.

**Special cases, to be tested exactly:**
* $y \equiv 0$ ($k_c = 0$), no spread, no power term, $s_V = 1$, $k_\mathrm{max} = \infty$: the Milano 2022 edge
  model, but with a log-linear conductance instead of the linear $g\,G_\mathrm{max} + (1 - g)\,G_\mathrm{min}$ (a
  switch `conductance="linear"` reproduces it exactly);
* constant $V$: $x$ relaxes exponentially to $x_\infty = k_p / (k_p + k_d + k_P)$ with rate $k_p + k_d + k_P$.

**Integration.** For piecewise-constant $V$ over a step, the $x$ equation is linear, so it is integrated
**exactly**:

$$
x(t + \Delta t) = x_\infty + \big(x(t) - x_\infty\big)\,e^{-(k_p + k_d + k_P)\,\Delta t}
$$

This is unconditionally stable. The step size is then limited only by how often the circuit must be re-solved
(P0.3 found $\le 2$ ms for N4 because of SET avalanches; to be re-checked for N10).

## 3. Parameters

| parameter | meaning | starting value | source / role |
|---|---|---|---|
| $\eta_p$, $k_{\mathrm{p0}}$ | potentiation threshold and steepness | Milano 2022: $\eta_p = 34.9$ V$^{-1}$, $k_{\mathrm{p0}} = 2.56 \times 10^{-6}$ s$^{-1}$ ($k_p = 3$ s$^{-1}$ at 0.4 V, 97 s$^{-1}$ at 0.5 V; $k_p = k_{\mathrm{d0}}$ at $\lvert V \rvert = 0.49$ V) | M3; default, threshold consistent with $V_c = 0.4$ V |
| $\eta_d$ | voltage suppression of relaxation | 5.59 V$^{-1}$ | Milano 2022 |
| $\tau_s$ (median $1/k_{d0}$) | short-term memory | **scan axis**: 0.015 (Milano 2022 fit as quoted by Cipollini 2024; **not supported**, see below), 0.1, 0.3, 1, 3, 10 s (Diaz-Alvarez ~10 s) | M1; the reservoir-relevant parameter |
| $\sigma_{\mathrm{ln}}$ | spread of $\tau_s$ across junctions | 0, 0.5, 1 | M5; N2 showed that a spread helps ~2x |

*Correction 2026-10-15.* The Milano 2022 values above ($k_{\mathrm{p0}}$, $\eta_p$, $\eta_d$, $k_{\mathrm{d0}} = 64.9$ s$^{-1}$) are
second-hand (Cipollini 2024, Table 1). The authors' own code fits a measured network trace with $k_{\mathrm{d0}} = 0.0487$
s$^{-1}$, i.e. $\tau \approx 20$ s at zero bias, and the measured trace relaxes with 11.5 s and 147 s
(`results/milano2022_relax.json`; docs/LITERATURE_NOTES.md). The 15 ms end of the scan therefore has no measured support;
measured networks relax on tens of seconds. N11 fits $\tau_s$ on the Milano 2025 data.
| $k_c$, n, $P_{\mathrm{c2}}$ | consolidation rate, order, power threshold | $k_c$ such that a 10 s, 3 V pulse consolidates noticeably, a 1 V pulse does not; n = 2 | M2 (Milano 2023), qualitative |
| $\tau_l$ | long-term decay | $10^{3}$ s | M2; only needs $\tau_l \gg \tau_s$ and $\gg$ the scored stream |
| $P_c$, $k_{\mathrm{P0}}$ | power rupture | $P_c = 1$ mW (Diaz Schneider 2024), $k_{\mathrm{P0}} = 10^{3}$ s$^{-1}$ | M4 |
| $G_{\mathrm{on}}$, $G_{\mathrm{off}}$ | conductance range | framework values | keeps the circuit scale |

The per-junction threshold of 0.4 V (M3) is $7 \times$ higher than N4's $V_{\mathrm{set}}$ and $4 \times$ the framework's $V_{\mathrm{th}}$. With the
framework geometry (1 mm, ~15-20 junctions along a path) the network onset should move to several volts. That is
consistent with the measured activation (1-10 V for 500 µm, Diaz-Alvarez 2019), and it must be checked in step 4.

## 4. Implementation plan (in this project, not the framework)

1. `agnw_rc/volatile.py`: `VolatileParams`, `conductance`, `step_state` (exact x update, explicit y update), noise
   option; `config.volatile_params(spec)`.
2. `multidrive.simulate_multi`: model `"volatile"` (state x and y stored like `junction_lambda`).
3. Tests: special cases above; steady state; exact relaxation; the `conductance="linear"` Milano limit; power
   rupture; unconditional stability at large $\Delta t$; network run with $k_c = 0$ matches an independent small-network
   solve.
4. `scripts/n10_phenomenology.py` (single junction + network, figure): M1 relaxation after a pulse, M2 amplitude-
   dependent consolidation (1 V vs 3 V, 10 s pulses scaled to our geometry), M3 activation under constant bias
   (sigmoid, tenths of seconds), M6 paired-pulse facilitation vs interval, and a Fig. 4-like sine loop
   (qualitative only).
5. N10-1 reservoir test: the E4 protocol (two bands, 800 samples, 50 virtual nodes, washout $3 \tau_s$ or 3 s, windows
   around the new onset, validation-selected operating point, G0-G2), for $\tau_s$ $\in$ {0.03, 0.3, 3} s and holds
   $\approx \tau_s$ / 3 and $\tau_s$, with $\sigma_{\mathrm{ln}} = 0$ and 1. Noise-free first; then noise (N4-2 analogue) if G1 is met.
6. Zero-mean input check: with non-polar switching (M7) a zero-mean drive can now form filaments, unlike N4.

Cost: like N4 at a 2 ms step (~5 h wall for a 54-run E4 block on 16 processes); ultron can take a share with
`run_sims.py --holds ... --reverse`.

## 5. Open questions for Juan

* Measured **relaxation curves** (conductance vs time after a pulse) and **pulse-train responses** for AgNW networks
  with a known geometry would pin $\tau_s$ and its spread directly. The Bariloche group (Diaz Schneider, Quinteros, Levy,
  Martínez, CNEA/CONICET) measures exactly these networks (arXiv:2409.01318, 2501.16886); their data, or the Milano
  2023 supplementary relaxation curves, are the best calibration targets available.
* Is the non-polar ($\lvert V \rvert$) potentiation acceptable as the default, with a bipolar variant as an option?
* Should consolidation (y) be in the first version, or added only if N10-1 with x alone is promising? Proposal:
  implement it, but run N10-1 first with $k_c = 0$ (pure volatile) and then with consolidation.
