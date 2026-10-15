# Fluctuation-Dissipation Violations in Stochastic Memristive Junctions: A Generalized Framework Bridging Neuromorphic Hardware and Non-Equilibrium Brain Dynamics

> **⚠️ AUDIT NOTICE (2026-09-19).** Eqs. 7, 8, 10, 12, 13, 14 and 16 (and the definition of $X$ in §6–7,
> §13.2) contain errors or are undefined for the simulated model; verified findings, corrected
> derivations (exact two-state response/correlation ratio $\mathrm{d}\ln(a/b)/\mathrm{d}V$) and
> reproducible SymPy/Monte-Carlo checks are in `docs/theory_audit/THEORY_AUDIT.md`. Do not use
> Eqs. 13–14 as predictions.

**Author:** Juan (CONICET / UNR)  
**Status:** Draft — Theoretical Development  
**Last Updated:** 2026-09-18  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Conceptual Framework](#conceptual-framework)
3. [Theoretical Isomorphisms](#theoretical-isomorphisms)
4. [Memristor Model: Lamas et al. Foundation](#memristor-model-lamas-et-al-foundation)
5. [Single Junction as Statistical Particle](#single-junction-as-statistical-particle)
6. [Fokker-Planck Formalism](#fokker-planck-formalism)
7. [Fluctuation-Dissipation Theorem: Standard vs. AgNW](#fluctuation-dissipation-theorem-standard-vs-agnw)
8. [FDT Ratio Derivation](#fdt-ratio-derivation)
9. [Dissipation Quantification in Hysteresis Cycles](#dissipation-quantification-in-hysteresis-cycles)
10. [Unified Non-Equilibrium Framework](#unified-non-equilibrium-framework)
11. [Percolation as Criticality Control Parameter](#percolation-as-criticality-control-parameter)
12. [History-Dependent Dynamics: HTSA Analogue](#history-dependent-dynamics-htsa-analogue)
13. [Empirical Validation Note: RESET Reachability](#empirical-validation-note-reset-reachability)
14. [Nomenclature & Physical Constants](#nomenclature--physical-constants)
15. [Open Questions & Validation Strategy](#open-questions--validation-strategy)
16. [References](#references)

---

## Executive Summary

This document develops a **unified theoretical framework** connecting three research domains:

1. **Computational Neuroscience:** Fluctuation-dissipation violations in whole-brain resting-state dynamics and entropy production metrics (IV-FDT framework)
2. **Neuromorphic Hardware:** Silver nanowire (AgNW) networks as memristive systems exhibiting stochastic switching and percolation-dependent memory
3. **Non-Equilibrium Statistical Physics:** General theory of systems violating thermal equilibrium assumptions

**Central Thesis:** The stochastic switching dynamics of a single memristive junction, when coupled to network-level voltage/current fields, constitute a **driven non-equilibrium system** that violates the classical fluctuation-dissipation theorem (FDT) through the same mechanism as memory-dependent brain dynamics—competition between multiple timescales with internal state dependence.

**Key Outcome:** Derivation of an explicit **FDT violation ratio** $X$ in terms of microscopic parameters ($P_0$, $\alpha$, $V_{\text{th}}$, $P_{\text{decay}}$) that can be measured or computed from network simulations and directly compared with the IV-FDT metric already developed for brain imaging data.

---

## Conceptual Framework

### 1.1 The Core Isomorphism

Two systems, separated by orders of magnitude in scale, exhibit identical **non-equilibrium signatures**:

#### System A: Whole-Brain Resting-State Dynamics
- **Temporal Hierarchy:** Fast local oscillations (~Hz) coupled to slow collective modes (~0.01 Hz)
- **Memory Signature:** Correlation functions decay slowly; spontaneous reversal of activity direction without external drive
- **FDT Violation:** Effective temperature $T_{\text{eff}}$ derived from fluctuation-dissipation relation exceeds actual temperature due to non-Markovian transitions
- **Quantitative Marker:** Ratio $X > 1$ measuring deviation from equilibrium expectation

#### System B: Silver Nanowire Memristive Junction
- **Temporal Hierarchy:** Fast Joule heating at junction (~ns) coupled to slow filament formation/dissolution (~ms–s)
- **Memory Signature:** Conductance depends on history of voltage/current; hysteretic I–V curves with non-zero loop area when $V \to 0$
- **FDT Violation:** Switching probabilities $P_{\text{SET}}(V), P_{\text{RESET}}(I)$ create effective "friction" that depends on state history
- **Quantitative Marker:** Ratio $X$ of actual dissipation to equilibrium expectation (to be derived)

### 1.2 Why FDT Matters for Both Systems

**Classical FDT (Einstein, Green-Kubo):** For a particle at thermal equilibrium under weak external driving, the linear response function $\chi(t)$ is rigidly determined by the autocorrelation function $C(t) = \langle \Delta x(0) \Delta x(t) \rangle$:

$$\chi(t) = \beta \frac{d C(t)}{dt}$$

where $\beta = 1/(k_B T)$ (inverse temperature).

**Physical Meaning:** Fluctuations *alone* determine how the system responds. No internal energy source needed.

**In Real Systems (Brain, AgNW):** This simple relation **breaks down** because:
1. System is not ergodic (state space not fully explored)
2. Internal state variables exist that are not observable
3. Multiple timescales decouple during measurement window
4. Energy injection occurs at rates comparable to dissipation timescales

**Consequence:** Measurement of $\chi(t)$ and $C(t)$ yields a mismatch quantifiable as $X = \chi_{\text{measured}} / \chi_{\text{FDT}}$.

---

## Theoretical Isomorphisms

### 2.1 State Variable Analogy

| **Aspect** | **Brain (fMRI)** | **AgNW Junction** |
|---|---|---|
| **Macro Observable** | BOLD signal $s(t)$ (neural activity proxy) | Current $I_{ij}(t)$ through junction |
| **Micro State** | Membrane potential $V_m$, synaptic efficacy | Conductance state $g(t) \in \{g_{\text{off}}, g_{\text{on}}\}$ |
| **Hidden Variable** | Glial calcium, dendritic compartments | Position of Ag filament within insulating layer |
| **Driving Force** | Metabolic input, sensory drive | Applied voltage $V_{\text{in}}(t)$ |
| **Friction/Dissipation** | Synaptic time constants $\tau_s$ | Joule heating, filament rupture rate |

### 2.2 Energy Dissipation Pathways

**Brain:**
$$P_{\text{dissipated}} = \int (\nabla \cdot \vec{J}) \, V \, dV + \text{(metabolism)}$$
where $\vec{J}$ = ion current density, $V$ = membrane potential (local).

**AgNW Junction:**
$$P_{\text{dissipated}} = I_{ij}^2 (t) \cdot R_{\text{eff}}(g) + \text{(switching energetics)}$$
where $R_{\text{eff}} = 1/g$ and switching energetics include filament formation work.

**Common Feature:** Energy dissipation is **state-dependent** ($R_{\text{eff}}$ depends on $g$, synaptic strength depends on calcium history). This coupling between state and dissipation creates the FDT violation.

---

## Memristor Model: Lamas et al. Foundation

### 3.1 Dual-Node Graph Representation

A single physical junction at the intersection of nanowires $i$ and $j$ is decomposed into a **memristive edge** connecting two electrical nodes:

- Node $n_i$: electrical contact on nanowire $i$
- Node $n_j$: electrical contact on nanowire $j$
- Edge: memristive interface with dynamic conductance $G_{ij}(t)$

### 3.2 Switching Dynamics: SET Process

The transition from OFF to ON state (insulating to conductive channel) is governed by voltage-dependent stochastic activation:

$$P_{\text{SET}}(V_{\text{mem}}) = \Delta t \, P_0 \, \max\left(0, 1 - \exp\left[-\alpha(V_{\text{mem}} - V_{\text{th}})\right]\right) \quad \text{...(1)}$$

**Physical Interpretation:**
- $V_{\text{mem}} = |V_i - V_j|$ = voltage drop across junction
- $P_0 \Delta t \ll 1$ = small probability per unit time (ensures discrete jumps)
- $\alpha$ = voltage sensitivity (steepness of activation)
- $V_{\text{th}}$ = threshold voltage (must exceed for activation)
- Functional form: soft threshold consistent with ion transport through insulating polyvinylpyrrolidone (PVP) coating

### 3.3 Switching Dynamics: RESET Process

The reverse transition (ON to OFF) occurs through filamentary instability or rupture, modeled as current-dependent decay:

$$P_{\text{RESET}}(I_{\text{mem}}) = \min\left(1, \Delta t \, P_{\text{decay}} \, (I_{\text{mem}})^2\right) \quad \text{...(2)}$$

**Physical Interpretation:**
- $I_{\text{mem}} = |I_{ij}|$ = local current through junction
- $P_{\text{decay}}$ = decay rate coefficient (controls hysteresis width)
- Quadratic dependence: higher current → faster filament rupture (Joule heating)
- Represents competition between thermal instability and structural stability

### 3.4 Conductance State Space

At any instant, $G_{ij}(t) \in \{G_{\text{OFF}}, G_{\text{ON}}\}$:

$$G_{\text{OFF}} = 1 \times 10^{-9} \, \mathrm{S} \quad (\text{insulating state})$$
$$G_{\text{ON}} = 1 \times 10^{-3} \, \mathrm{S} \quad (\text{conductive state})$$

**Ratio:** $G_{\text{ON}} / G_{\text{OFF}} = 10^6$—a stark multi-order contrast driving FDT violations.

### 3.5 Electrical Evolution: Nodal Analysis

The full network voltage distribution is obtained via **Kirchhoff's current law** applied to each node:

$$\mathbf{Y} \, \mathbf{V} = \mathbf{I}_{\text{ext}} \quad \text{...(3)}$$

where:
- $\mathbf{Y}$ = admittance matrix with elements:
$$Y_{pq} = \begin{cases}
\sum_k G_{pk} & \text{if } p = q \\
-G_{pq} & \text{if } p \neq q
\end{cases}$$
- $\mathbf{V}$ = nodal voltage vector
- $\mathbf{I}_{\text{ext}}$ = external current injection (boundary conditions)

**Sparse Solver:** Due to sparsity ($E \ll N_g^2$), a direct method (COLAMD-permuted SuperLU) solves Eq. (3) at each discrete time step $\Delta t$.

### 3.6 Local Current via Ohm's Law

Once $\mathbf{V}$ is solved, the current through memristive edge $(i,j)$ is:

$$I_{ij}(t) = G_{ij}(t) \cdot (V_i(t) - V_j(t)) \quad \text{...(4)}$$

This local current determines the RESET probability via Eq. (2).

---

## Single Junction as Statistical Particle

### 4.1 Coarse-Graining Strategy

To develop a **statistical mechanics framework**, treat one memristive junction in isolation while parameterizing the effect of the surrounding network through effective fields:

- **Effective voltage:** $V_{\text{eff}}(t)$ = voltage drop averaged over the ensemble of similar junctions
- **Effective current:** $I_{\text{eff}}(t)$ = characteristic current through a junction
- **Effective temperature:** $T_{\text{eff}}$ = temperature of the thermal bath (substrate, including joule heating)

### 4.2 Generalized Coordinate: Conductance

Define the **order parameter** as the conductance state:

$$g(t) \in [0, 1]$$

where we normalize:
$$g(t) = \frac{G(t) - G_{\text{OFF}}}{G_{\text{ON}} - G_{\text{OFF}}}$$

Thus:
- $g = 0$ = OFF state
- $g = 1$ = ON state
- Intermediate values represent "effective" partially-activated junctions in an ensemble

### 4.3 Effective Potential

Define a **free energy functional** that encodes the energetic cost of states given external voltage:

$$F_g = -k_B T_{\text{eff}} \log P_{\text{stationary}}(g \mid V_{\text{eff}}, I_{\text{eff}})$$

In the absence of time-dependence, the stationary distribution (if quasi-equilibrium existed) would be:

$$P_{\text{stat}}(g \mid V) \propto \exp\left[-\beta_{\text{eff}} (U(g) - V \cdot q(g))\right]$$

where:
- $U(g)$ = intrinsic structural energy of conductance state
- $q(g)$ = "charge-like" quantity representing coupling to voltage
- $\beta_{\text{eff}} = 1/(k_B T_{\text{eff}})$

For the two-state model, approximate:

$$U(g) = U_0 + U_1 \cdot g^2 - \alpha V_{\text{th}} \cdot g$$

where the quadratic term penalizes intermediate states, and the linear term reflects voltage-lowering of the energy barrier.

### 4.4 Effective Friction

The switching probabilities (Eqs. 1–2) imply an **effective damping term**:

$$\gamma_{\text{eff}}(g, \dot{g}) = \frac{1}{\dot{g}} \left[ k_B T_{\text{eff}} + \text{(active driving)} \right]$$

The "active driving" arises because SET and RESET are **not conjugate**: SET is voltage-driven (deterministic input), while RESET is current-driven (output-dependent).

---

## Fokker-Planck Formalism

### 5.1 Probability Density and Master Equation

Define $\rho(g, t)$ = probability density of finding the junction in state $g$ at time $t$.

The evolution is governed by a **Fokker-Planck equation** with both drift and diffusion terms:

$$\frac{\partial \rho(g, t)}{\partial t} = -\frac{\partial}{\partial g}\left[A(g, t) \rho(g, t)\right] + \frac{\partial^2}{\partial g^2}\left[B(g, t) \rho(g, t)\right] \quad \text{...(5)}$$

where:
- $A(g, t)$ = drift coefficient (deterministic driving)
- $B(g, t)$ = diffusion coefficient (stochastic fluctuations)

### 5.2 Drift Coefficient: SET and RESET

The drift arises from the competitive interaction between SET and RESET processes:

$$A(g, t) = P_{\text{SET}}(V_{\text{eff}}(t)) \cdot (1 - g) - P_{\text{RESET}}(I_{\text{eff}}(t)) \cdot g \quad \text{...(6)}$$

**Physical Meaning:**
- First term: probability of activation times the "room to activate" $(1-g)$
- Second term: probability of deactivation times the current state $g$

### 5.3 Diffusion Coefficient: Intrinsic Noise

The diffusion coefficient represents the stochastic granularity of individual switching events:

$$B(g, t) = \frac{1}{2} \left[ P_{\text{SET}}(V_{\text{eff}}(t)) \cdot (1-g)^2 + P_{\text{RESET}}(I_{\text{eff}}(t)) \cdot g^2 \right] \quad \text{...(7)}$$

This form ensures that the noise term is **self-consistent** with the transition rates (Einstein relation).

### 5.4 Stationary vs. Non-Stationary Regimes

**Equilibrium (if it existed):** Set $\partial_t \rho = 0$ in Eq. (5):

$$A(g) \rho_{\text{stat}}(g) = \frac{d}{dg} [B(g) \rho_{\text{stat}}(g)]$$

**Reality (non-equilibrium):** Due to time-dependent driving $V_{\text{eff}}(t)$ and the asymmetry between SET and RESET, the system never reaches stationarity. Instead, $\rho(g, t)$ evolves through **transient attractors**.

---

## Fluctuation-Dissipation Theorem: Standard vs. AgNW

### 6.1 Classical FDT Statement

**For a particle in thermal equilibrium at temperature $T$:**

The **linear response** to a small external force $F(t)$ is related to the **autocorrelation** of thermal fluctuations:

$$\chi(t) = \beta \frac{d}{dt} C(t) = -\beta \int_0^t ds \, \frac{d C(s)}{ds} \quad \text{...(8)}$$

where:
$$C(t) = \langle \Delta x(0) \, \Delta x(t) \rangle = \langle [x(0) - \langle x \rangle][x(t) - \langle x \rangle] \rangle$$

and $\beta = 1/(k_B T)$.

**Physical Content:** The system's memory of past fluctuations *exactly* determines its ability to respond to new driving. No additional energy or internal state required.

### 6.2 Operationalization for Memristor

Adapt FDT to the conductance observable:

**Linear Response Function:**
$$\chi(t) = \frac{\delta \langle g(t) \rangle}{\delta V_{\text{in}}}$$

i.e., the change in average conductance per unit change in applied voltage.

**Autocorrelation Function:**
$$C(t) = \langle \Delta g(0) \, \Delta g(t) \rangle$$

**FDT Prediction (if system were in equilibrium):**
$$\chi_{\text{FDT}}(t) = \beta_{\text{eff}} \frac{d C(t)}{dt}$$

### 6.3 Why AgNW Violates FDT

**Reason 1: Multi-State Complexity**
- SET and RESET operate on different variables: SET depends on $V_{\text{mem}}$ (voltage), RESET on $I_{\text{mem}}$ (current).
- These are not conjugate; violating symmetry assumption of FDT.

**Reason 2: Timescale Separation**
- Voltage ramps vary on timescale $\tau_V \sim$ (rise time of applied signal)
- Filament formation/rupture operates on $\tau_f \sim$ (ms–s)
- If $\tau_V \ll \tau_f$, system cannot equilibrate to fast driving.

**Reason 3: Non-Conservative Dynamics**
- Work done by voltage $W_V = \int I \cdot V \, dt$ includes **irreversible** components (Joule heating in filament).
- Energy dissipation is **state-dependent**, not a function of configuration alone.

### 6.4 Empirical Signature: Hysteresis Loop

The non-equilibrium nature manifests as **non-zero hysteresis** in the I–V curve:

$$\text{Loop Area} = \oint I \, dV \neq 0$$

Even when the cycle is quasi-static (slow), the loop area reflects the **minimum work irreversibly lost per cycle** due to kinetic friction and history-dependent switching.

---

## FDT Ratio Derivation

### 7.1 Definition: FDT Violation Ratio $X$

Define the **FDT ratio** as:

$$X = \frac{\chi_{\text{measured}}}{\chi_{\text{FDT}}} \quad \text{...(9)}$$

where:
- $\chi_{\text{measured}}$ = actual conductance response measured or computed from simulations
- $\chi_{\text{FDT}}$ = response predicted by classical FDT using measured correlations

**Physical Meaning:**
- $X = 1$: System obeys FDT (equilibrium)
- $X > 1$: System over-responds (non-equilibrium, effective temperature $T_{\text{eff}}^{\text{FDT}} = T \cdot X > T$)
- $X \gg 1$: Strong non-equilibrium driving

### 7.2 Derivation from Fokker-Planck

Start with the Fokker-Planck equation for the mean conductance:

$$\frac{d \langle g \rangle}{dt} = \int_0^1 dg \, g \, \frac{\partial \rho(g,t)}{\partial t}$$

Substitute Eq. (5):

$$\frac{d \langle g \rangle}{dt} = -\int_0^1 dg \, g \, \frac{\partial}{\partial g}[A(g,t) \rho(g,t)] + \int_0^1 dg \, g \, \frac{\partial^2}{\partial g^2}[B(g,t) \rho(g,t)]$$

Integrate by parts (assuming boundary terms vanish):

$$\frac{d \langle g \rangle}{dt} = \int_0^1 dg \, A(g,t) \rho(g,t) - \int_0^1 dg \, \frac{\partial}{\partial g}[B(g,t) \rho(g,t)]$$

$$= \langle A(g, t) \rangle - \frac{\partial}{\partial g} B(g) |_{g=0}^{g=1} \quad \text{...(10)}$$

### 7.3 Approximation: Weak-Coupling Expansion

For small deviations $\delta V = V_{\text{eff}} - V_{\text{th}}$, expand the probabilities:

$$P_{\text{SET}}(V_{\text{th}} + \delta V) \approx \Delta t \, P_0 \, \alpha \delta V \quad \text{(for } \delta V > 0 \text{)}$$

$$P_{\text{RESET}}(I_{\text{eff}}) \approx \Delta t \, P_{\text{decay}} \, I_{\text{eff}}^2$$

The drift coefficient becomes:

$$A(g, t) \approx \Delta t \left[ P_0 \alpha (V_{\text{eff}}(t) - V_{\text{th}}) (1-g) - P_{\text{decay}} I_{\text{eff}}^2(t) \, g \right] \quad \text{...(11)}$$

### 7.4 Linear Response: Green-Kubo Form

The **generalized linear response** (Green-Kubo formula) relates conductance response to internal correlations:

$$\chi(t) = \int_0^t ds \, \langle \dot{g}(0) \dot{g}(s) \rangle / k_B T_{\text{eff}}$$

For our system:

$$\chi(t) = \int_0^t ds \, \langle [P_{\text{SET}} (1-g) - P_{\text{RESET}} g]|_{s=0} \times [P_{\text{SET}} (1-g) - P_{\text{RESET}} g]|_{s=t} \rangle$$

$$\quad / k_B T_{\text{eff}} \quad \text{...(12)}$$

Expanding and keeping leading terms:

$$\chi(t) = \int_0^t ds \left[ \langle P_{\text{SET}}(s) P_{\text{SET}}(0) (1-g)(1-g') \rangle - \langle P_{\text{SET}}(0) P_{\text{RESET}}(s) (1-g) g' \rangle \right.$$
$$\quad \left. - \langle P_{\text{RESET}}(0) P_{\text{SET}}(s) g (1-g') \rangle + \langle P_{\text{RESET}}(0) P_{\text{RESET}}(s) g g' \rangle \right] / k_B T_{\text{eff}}$$

where primes denote variables at time $s$.

### 7.5 Asymmetry Contribution to FDT Violation

The key **FDT-breaking term** arises from the **asymmetry** between SET and RESET:

$$X_{\text{asym}} = \frac{\langle P_{\text{SET}}(0) P_{\text{RESET}}(t) \rangle + \langle P_{\text{RESET}}(0) P_{\text{SET}}(t) \rangle}{\langle P_{\text{SET}}(0) P_{\text{SET}}(t) \rangle + \langle P_{\text{RESET}}(0) P_{\text{RESET}}(t) \rangle}$$

If SET and RESET operated symmetrically (i.e., both were voltage-driven or both current-driven), the numerator and denominator would be identical, and $X_{\text{asym}} = 1$. The asymmetry makes $X_{\text{asym}} \neq 1$.

### 7.6 Explicit Formula for $X$ in Terms of Parameters

Under the assumption that the voltage and current variables are approximately **independent** (weak coupling to network), we obtain:

$$X = 1 + \frac{1}{2} \left[ \frac{\langle I_{\text{eff}}^2 \rangle}{\langle V_{\text{eff}}^2 \rangle} \cdot \frac{P_{\text{decay}}}{P_0 \alpha} + \text{(higher-order terms)} \right] \quad \text{...(13)}$$

**Simplified Form (leading order):**

$$X \approx 1 + \frac{\Delta \tau_f^2}{\tau_V^2} \cdot \frac{P_{\text{decay}}}{P_0 \alpha} \quad \text{...(14)}$$

where:
- $\tau_V$ = characteristic timescale of voltage variation
- $\tau_f$ = filament formation/rupture timescale $\approx 1/(P_0 \alpha V_{\text{th}})$
- The ratio $\tau_f^2 / \tau_V^2$ quantifies **timescale separation**

**Physical Interpretation:** Stronger timescale separation (larger $\tau_f / \tau_V$) and larger current-driven decay rate ($P_{\text{decay}}$) lead to larger FDT violations.

---

## Dissipation Quantification in Hysteresis Cycles

### 8.1 Work and Energy in a Single Cycle

During a hysteretic cycle with applied voltage $V_{\text{in}}(t)$ over period $T_{\text{cycle}}$, the **electrical work** done on the junction is:

$$W_{\text{in}} = \int_0^{T_{\text{cycle}}} I_{ij}(t) \cdot V_{\text{in}}(t) \, dt \quad \text{...(15)}$$

This work is partitioned into:

$$W_{\text{in}} = \Delta E_{\text{cond}} + W_{\text{dissipated}}$$

where:
- $\Delta E_{\text{cond}}$ = change in internal energy of the junction (filament formation/dissolution)
- $W_{\text{dissipated}}$ = irreversible dissipation (Joule heating + entropy production)

### 8.2 Irreversible Dissipation via Loop Area

For a **quasi-static cycle** (slow voltage ramp), the irreversible work per cycle is given by the area enclosed in the I–V plane:

$$W_{\text{diss}} = \oint I \, dV \quad \text{...(16)}$$

This integral, taken over a closed path in voltage space, equals the **loop area** visible in hysteresis plots.

**Non-dimensional Form:**

$$\Sigma_{\text{cycle}} = \frac{W_{\text{diss}}}{I_0 V_0}$$

where $I_0, V_0$ are characteristic current and voltage scales.

### 8.3 Entropy Production

The **entropy production rate** (per unit time) can be estimated from the dissipated power:

$$\dot{\Sigma}(t) = \frac{P_{\text{diss}}(t)}{T_{\text{eff}}} = \frac{I_{ij}^2(t) / g_{ij}(t)}{T_{\text{eff}}} \quad \text{...(17)}$$

Integrated over a cycle:

$$\Sigma_{\text{total}} = \int_0^{T_{\text{cycle}}} \dot{\Sigma}(t) \, dt = \frac{1}{T_{\text{eff}}} \int_0^{T_{\text{cycle}}} \frac{I_{ij}^2(t)}{g_{ij}(t)} \, dt \quad \text{...(18)}$$

### 8.4 Relation to FDT Ratio

The **FDT violation ratio** can be rewritten as:

$$X = \frac{\text{Actual Entropy Production}}{\text{FDT-predicted Entropy Production}} \quad \text{...(19)}$$

Thus:
- **High $X$:** System produces more entropy than FDT predicts $\Rightarrow$ excess dissipation from hidden state transitions (filament dynamics)
- **Low $X$ (near 1):** System dissipates as expected for its observable correlations $\Rightarrow$ closer to equilibrium

### 8.5 Practical Measurement from Simulations

From Lamas et al. simulation output:

1. **Extract I–V curves** for different percolation regimes (wire densities $N = 1000, 2000, 3000$)
2. **Compute loop area** numerically via trapezoidal integration:
$$\text{Area} = \sum_i [V_i - V_{i-1}] \times [I_i + I_{i-1}]/2$$
3. **Normalize** by scale: $W_{\text{diss}} = \text{Area} \times (\text{applied voltage scale})^2 / (\text{junction resistance scale})$
4. **Compare with FDT expectation** from autocorrelation of conductance fluctuations

---

## Unified Non-Equilibrium Framework

### 9.1 Generalized Langevin Equation for $g(t)$

A **phenomenological stochastic differential equation** that incorporates both FDT violations and history-dependence:

$$m_{\text{eff}} \ddot{g}(t) + \int_0^t ds \, \gamma(t-s) \dot{g}(s) + \nabla U(g) = F(t) + \xi(t) \quad \text{...(20)}$$

where:
- $m_{\text{eff}} = $ effective inertia (typically negligible; set $\approx 0$ for overdamped dynamics)
- $\gamma(t-s) = $ memory kernel (frequency-dependent friction)
- $U(g) = $ potential energy landscape
- $F(t) = $ deterministic driving ($\propto V_{\text{eff}}(t)$)
- $\xi(t) = $ colored noise with correlation $\langle \xi(t) \xi(s) \rangle = 2 k_B T_{\text{eff}} \gamma(t-s)$ (generalized FDT)

### 9.2 Mapping to Brain Dynamics

In whole-brain models (e.g., Hopf with global coupling):

$$\dot{x}_i = \omega_i x_i + a_i x_i^2 - x_i^3 + C \sum_j A_{ij} x_j + \sqrt{D} \xi_i(t)$$

where $x_i$ = neural activity of region $i$, the **FDT violations arise from:**
1. **Synaptic plasticity:** Coupling strength $A_{ij}$ fluctuates with history (Hebbian learning)
2. **Slow neuromodulation:** Dopamine, serotonin levels change on timescale much longer than neural dynamics
3. **Glial dynamics:** Astrocytes buffer calcium over seconds; neural events occur over milliseconds

This is **isomorphic** to the SET/RESET asymmetry in AgNW: slow state variables (glial, neuromodulatory) couple to fast observables (neural firing), creating frequency-dependent response.

### 9.3 IV-FDT Metric as Special Case

The **IV-FDT framework** (applied to climate, brain, finance) quantifies FDT violations for **linear Langevin systems**:

$$\dot{x} = -\Gamma x + f(t) + \xi(t)$$

where linearization around a reference state yields a friction matrix $\Gamma$ and noise covariance $Q$ such that:

$$\text{FDT-compatible} \quad \Rightarrow \quad Q = 2 T \Gamma$$

**FDT Ratio in IV-FDT:**

$$X_{\text{IV-FDT}} = \frac{\langle (\dot{x} - \langle \dot{x} \rangle)^2 \rangle}{\text{(spectral proxy of Q)} \times \text{(spectral proxy of)} \Gamma}$$

---

## Percolation as Criticality Control Parameter

### 10.1 Percolation Threshold in AgNW Networks

The **percolation threshold** $N_c$ is the minimum number of nanowires required for a conducting path to span the network:

$$N_c \approx 1200 \text{ wires (in } A = 1 \text{ mm}^2)$$

Below $N_c$: network is **fragmented** (sub-percolated regime).
Above $N_c$: network is **connected** (over-percolated regime).

### 10.2 Maximum Memory Window at Criticality

**Empirical Observation (Lamas et al., Fig. 5):**

$$\text{Memory Window} = \frac{G_{\text{final}}}{G_{\text{initial}}} \quad \text{(peaks at } N \approx 1200 \text{)}$$

**Physical Explanation:**
- At $N < N_c$: Few critical junctions; single switch can disconnect network → large conductance ratio
- At $N = N_c$: Balance between sparsity and connectivity → optimal sensitivity
- At $N > N_c$: Redundant pathways bypass individual switches → smaller effective conductance ratio

### 10.3 Connection to Brain Criticality

**Analogous Observation in Whole-Brain Imaging:**

Resting-state networks exhibit **maximum information flow** (transfer entropy, active information storage) when the network operates near a **dynamical criticality**, not at either extreme of strong or weak coupling.

**Quantified by:** Edge-of-chaos criterion in coupled Hopf oscillators:
$$\lambda_{\text{largest eigenvalue}} = 1 + \epsilon \quad (\epsilon \ll 1)$$

where $\lambda$ depends on global coupling strength $C$. Optimal information processing occurs when $C$ is tuned so that $\lambda \approx 1$.

### 10.4 Unified Picture: Percolation = Dynamical Criticality

**Hypothesis:** In AgNW networks, the **structural percolation threshold** acts as the **physical analogue** of the **dynamical criticality** in neural systems.

Both systems maximize:
1. **Sensitivity** (response to small perturbations)
2. **Information capacity** (ability to store and transmit information)
3. **Energy efficiency** (minimum dissipation per bit processed)

at a **critical point** where topological structure transitions from fragmented to connected (structural percolation) or dynamical behavior transitions from quiescent to chaotic (Lyapunov exponent).

---

## History-Dependent Dynamics: HTSA Analogue

### 11.1 History-State Entanglement Analysis (HTSA)

**Definition (from computational neuroscience context):**

HTSA quantifies the **information content** in the trajectory history required to explain the current state:

$$I_{\text{HTSA}} = \sum_{\tau_{\min}}^{\tau_{\max}} I[H(t-\tau) ; S(t)]$$

where:
- $H(t-\tau)$ = state history at lag $\tau$
- $S(t)$ = current state
- Sum over relevant timescales $\tau_{\min}$ to $\tau_{\max}$

**Physical Meaning:** How far back in time must we look to have predictive power for the present?

### 11.2 Application to Single Memristor Junction

Define analogous measure for conductance:

$$I_{\text{hist}}^{\text{junction}} = \sum_{\tau=1}^{\tau_{\max}} \left[ I(G(t-\tau) ; g(t)) + \text{(cross-information with neighbors)} \right] \quad \text{...(21)}$$

where:
- $G(t-\tau)$ = vector of conductances in neighboring junctions at time $t-\tau$
- $g(t)$ = conductance state of target junction at time $t$
- $I(\cdot ; \cdot)$ = mutual information

### 11.3 Expected Scaling with Network Density

**Prediction:** $I_{\text{hist}}$ should **peak near the percolation threshold**:

- **Sparse regime** ($N < N_c$): Strong history dependence because current state of a few critical junctions determines global conductance
- **Percolation threshold** ($N \approx N_c$): Maximum HTSA because transitions in one junction can trigger cascading events
- **Dense regime** ($N > N_c$): Reduced HTSA because individual junction states are "masked" by parallel pathways

### 11.4 Relation to Voltage-Dependent Switching

The HTSA metric directly quantifies the **non-Markovian** nature of switching:

$$P(g(t) = \text{ON} \mid V(t), g(t-\Delta t), g(t-2\Delta t), \ldots) \neq P(g(t) = \text{ON} \mid V(t), g(t-\Delta t))$$

A **strongly non-Markovian** process has high HTSA; a Markovian process has HTSA $ \approx 0$.

---

## Empirical Validation Note: RESET Reachability

> **⚠️ SUPERSEDED (2026-09-19) — read this before §12.1–12.7.** Table 1 of
> Lamas et al. (2026) lists the **per-step products** $\Delta t P_0 = 0.4$ and
> $\Delta t P_{\text{decay}} = 0.01$. Sections 12.2–12.7 below were produced
> with $P_0 = 0.4$ and $P_{\text{decay}}$ passed as *rates* and multiplied by
> $\Delta t = 10^{-3}$ a second time, so SET was **1000× slower** than in the
> paper (per-step $P_{\text{SET}}^{\max} = 4\times10^{-4}$ instead of $0.4$).
> The "$P_{\text{decay}} \approx 3\times10^{8}$" estimate, the avalanche table
> and the stationarity finding of §12.7 are therefore properties of that
> mis-scaled model, **not** of the paper's model, and must not be used.
> What survives: with the paper's per-step values the network saturates within
> ~50–200 steps at fixed $V_{\text{in}}$ (as in the paper's Fig. 5A), and the
> literal RESET law $0.01\,I^2$ is essentially inactive for $I$ in A or mA
> (unit of $I$ not stated in the paper — open question, see
> `docs/lamas_questions_es.md`). **Provisional (2026-09-19):** replicating the
> paper's Fig. 4 (N=2000, 3000; sine drive 3.6 V, 50,000 steps/cycle) requires
> $V_{\text{th}} \approx 0.1$ V rather than Table 1's 0.01 V (onset 1.5/1.2 V vs.
> paper ~1.8/1.3 V; peak currents 4.3/12.7 mA vs. 3.2/10.5 mA); and
> intra-wire resistance $G_{\text{seg}} = \pi (D/2)^2/(\rho d)$ with $D \approx 90$–$100$ nm
> brings the currents to within ~8% of the paper's (3.3–3.5 mA and 9.2–10.0 mA vs.
> 3.2 and 10.5 mA); to be confirmed with the authors. The paper's
> negative half-cycle (Fig. 4) is reproduced only with an extra spontaneous ON→OFF relaxation of
> lifetime ≈ 2×10⁴ steps (fixed lifetime or exponential; a third mechanism mentioned by Lamas in a
> talk, not in the paper, unconfirmed): $A_-/A_+ = 0.54$ vs. 0.49 and $A_- = 4.14$ vs. 4.14 mA·V at
> $N=3000$. See `docs/theory_audit/` and `RESUME_FROM_HERE.md`. Sections below are kept only as a record of the mis-scaled run.

### 12.1 Motivation

The framework's non-equilibrium picture (§6.3, §7.5) depends on SET and RESET
being **simultaneously active, competing processes** — the asymmetry between
a voltage-driven SET and a current-driven RESET is the mechanism generating
the FDT violation ratio $X$ (Eq. 13–14). This requires the junction
population to reach a **stationary fluctuating state**, not a monotone
one-way transition. This note reports simulation evidence, from the
`agnw_sim` network simulator (Section 3's model, implemented and validated
independently — see repository `agnw_sim/`), that the nomenclature table's
original $P_{\text{decay}} = 0.01\ \text{s·A}^{-2}$ (§13.1) makes RESET
**structurally unreachable**, and quantifies the correction needed.

### 12.2 Finding 1: RESET is unreachable at the documented default

Simulating a percolating network ($N = 1500$ wires, $A = 1\,\text{mm}^2$,
$L = 70\,\mu\text{m}$, 3407 junctions) at $V_{\text{in}} = 5\,\text{V}$ with
the framework's own default parameters ($P_0 = 0.1$–$0.4$, $\alpha =
1\,\text{V}^{-1}$, $V_{\text{th}} = 0.01\,\text{V}$, $P_{\text{decay}} =
0.01\,\text{s·A}^{-2}$) produces **zero RESET events across 20,000
timesteps** while 1057 SET events occur. The switching dynamics are a pure
one-way ratchet to saturation, not the competing-process picture the
framework requires.

### 12.3 Finding 2: an avalanche transition sets the relevant current scale

Tracking total network current $I_{\text{total}}(t)$ as junctions switch ON
reveals a sharp transition, not a gradual rise:

| ON junctions | $I_{\text{total}}$ | Regime |
|---|---|---|
| $< 400$ | $\sim 10^{-9}$–$10^{-8}\,\text{A}$ | pre-avalanche: individual ON junctions are bottlenecked in series by the surrounding still-OFF network |
| $\sim 400$–$550$ | transition | avalanche: a low-resistance path completes |
| $> 550$ | $\sim 10^{-4}\,\text{A}$ | post-avalanche: four orders of magnitude higher |

This is consistent with the criticality picture already developed in
§10–11: the interesting non-equilibrium dynamics live near the percolation
point of the **activated subnetwork**, not in the sparse or saturated
regimes. It also means $I_{\text{mem}}$ for any single junction is
capped by whichever OFF junction is the current bottleneck along its
conducting path — raising $V_{\text{in}}$ alone cannot raise individual
junction currents past this cap without first crossing the avalanche.

### 12.4 Finding 3: the corrected $P_{\text{decay}}$ range

Calibrating $P_{\text{decay}}$ so that $P_{\text{RESET}}$ at the
post-avalanche current scale ($I_{\text{mem}} \sim 2\times10^{-4}\,\text{A}$)
lands near $P_{\text{SET}}$'s saturated value ($dt \cdot P_0 \sim
10^{-4}$) gives:

$$P_{\text{decay}} \sim 10^{7}\text{–}10^{8}\ \text{s·A}^{-2}$$

— roughly **nine orders of magnitude larger** than the original default.
Sweeping this range (same network/seed, $V_{\text{in}} = 5\,\text{V}$,
20,000 steps) confirms genuine two-way switching appears and grows with
$P_{\text{decay}}$:

| $P_{\text{decay}}$ (s·A$^{-2}$) | RESET events | SET events | Behavior |
|---|---|---|---|
| $10^{-2}$ (original default) | 0 | 1057 | pure ratchet |
| $10^{5}$ | 10 | 1051 | negligible |
| $10^{6}$ | 37 | 1066 | negligible |
| $10^{7}$ | 73 | 964 | visibly slows net SET growth |
| $10^{8}$ | 115 | 960 | clearly counteracts SET growth |

None of these runs reached full stationarity within 20,000 steps — even
$P_{\text{decay}} = 10^8$ remains net SET-dominated over this window. The
$10^7$–$10^8$ range should be treated as a **starting point for a proper
parameter sweep** (§14.2 Phase 2), not a validated operating point.

### 12.5 Refinement: a longer, wider sweep

§12.4's range was extended and re-tested with longer trajectories (60,000
steps, long enough to separate transient decay from steady behavior) at
five values spanning $P_{\text{decay}} = 10^7$ to $10^9\,\text{s·A}^{-2}$
(same network, seed, and drive as §12.2–12.4), measuring RESET and SET
event counts and net ON-population drift over the trailing half of each
run:

| $P_{\text{decay}}$ (s·A$^{-2}$) | net drift (ON count) | RESET/SET ratio | Two-way switching visible at 2000-step resolution |
|---|---|---|---|
| $10^{7}$ | $+204$ | 68% | No |
| $3\times10^{7}$ | $+149$ | 77% | No |
| $10^{8}$ | $+109$ | 80% | No |
| $\mathbf{3\times10^{8}}$ | $\mathbf{+92}$ | $\mathbf{80\%}$ | **Yes** |
| $10^{9}$ | $+70$ | 87% | No (drift too smooth at this resolution) |

The trend is monotonic and improving but shows **diminishing returns**: a
10× increase from $10^8$ to $10^9$ only moves the RESET/SET ratio from
80% to 87%, while net drift roughly halves. None of these five runs reach
a fully flat (zero-drift) plateau within 60,000 steps. Two explanations
are equally consistent with this data and were not distinguished here:
(a) $P_{\text{decay}}$ needs to go higher still, or (b) the transient from
cold-start simply has not fully decayed by 60,000 steps at any of these
values. Disentangling them needs longer runs, which is left to Sub-Project
2/3's own validation rather than this spike.

**Revised recommendation:** $P_{\text{decay}} \approx 3\times10^{8}\
\text{s·A}^{-2}$ as the practical starting operating point — the only
value tested that showed genuine bidirectional switching even at coarse
(2000-step) time resolution, with a strong 80% RESET/SET ratio. This
supersedes §12.4's $10^7$–$10^8$ range as the best current estimate, but
remains **provisional, not a proven stationary point**.

### 12.6 Implication for Sub-Project 2

The FDT analysis pipeline (autocorrelation $C(\tau)$, linear response
$\chi(t)$, the ratio $X$) presupposes a (quasi-)stationary fluctuating
$g(t)$. Before that sub-project is scoped, validation should confirm
genuine stationarity at (or near) $P_{\text{decay}} \approx 3\times10^8$
— via longer runs than tested here, and/or a joint sweep with $P_0$ and
$V_{\text{in}}$ — rather than assuming §12.5's value is final. This
remains a prerequisite for Sub-Project 2, not a detail internal to it.

### 12.7 First Real Validation Run: Two-Way Switching Confirmed, Stationarity Not Yet

Sub-project 2's first implementation slice (analysis pipeline: $C(\tau)$,
hysteresis loop area, entropy production) was built and run against
$P_{\text{decay}} = 3\times10^8$ at full scale: $N=1500$ wires, $60{,}000$
timesteps (60 s), $V_{\text{in}} = 5\,\text{V}$ constant, $P_0=0.4$,
$\alpha=1.0\,\text{V}^{-1}$, $V_{\text{th}}=0.01\,\text{V}$ — the exact
operating point established in §12.5.

**Confirmed: genuine two-way switching, not a ratchet.** Direct event
counting on the saved trajectory: **96 SET events, 41 RESET events**,
with RESETs spread across every decile of the run (not front-loaded or
absent), and the sampled-junction ON-fraction reaching a **plateau of
$\approx 0.24$** from roughly the third decile onward. $I_{\text{total}}(t)$
shows repeated collapse-to-near-zero-and-recovery — consistent with a
marginally percolating "activated backbone" near the criticality picture
of §10–11, not a smoothly saturating one.

**Not yet confirmed: stationarity.** The validation script's own
autocorrelation figure used $\text{max\_lag} = 2000$ steps (2 s, ~3% of
the run), over which $C(\tau)$ falls only $\approx 19\%$, near-linearly —
the signature of a slowly-varying signal being viewed through too short a
lag window, not a resolved decorrelation. Extending the same analysis to
$\text{lag}=20{,}000$ on the same data:

| lag (steps) | $C(\tau)$ |
|---|---|
| 0 | 0.0329 |
| 2,000 | 0.0237 |
| 5,000 | 0.0114 |
| 10,000 | $-0.0042$ |
| 20,000 | $-0.0227$ |

$C(\tau)$ has **not** reached a clear plateau within the run; the
correlation time is of order $5$–$15\,\text{s}$ — i.e. **10–25% of the
entire 60 s run length**. (The sign flip and growing magnitude past
$\text{lag}\approx n/2$ is the expected variance blow-up of the
$1/(n-\tau)$ unbiased estimator as the overlap count shrinks, not a
physical oscillation — the estimator is only trustworthy out to roughly
$n/10$.)

**Revised implication for Sub-Project 2, part 2 (supersedes §12.6's framing):**
two-way switching at $P_{\text{decay}} \approx 3\times10^8$ is now
empirically confirmed, not just inferred from event-count ratios — but a
run needs to be long enough that $C(\tau)$ resolves to noise **within the
first ~10% of its own length** before $\chi(t)$/$X$ work can trust it. On
current evidence that means **at least 5–10× this run's length**, or a
parameter point with faster dynamics (a shorter correlation time). This
should be an explicit acceptance gate for whatever run is used as input
to the deferred $\chi(t)$/$X$ slice, not an assumption.

A second, structural point for that slice's design: this note's
$C(\tau)$ is a **per-junction** ensemble average (§2.1 of the analysis
pipeline's own design spec), but a small-signal voltage perturbation
naturally measures a **network-level** response. $\chi(t)$ and $C(\tau)$
must be computed on the *same* observable for the FDT ratio $X$ to be
meaningful — deciding whether that observable is the per-junction $g$ or
the network-level $g_{\text{eff}}(t) = I_{\text{total}}(t)/V_{\text{in}}(t)$
(already derivable from the existing saved schema, no schema change
needed) is a design decision for that slice, not a detail to discover
mid-implementation.

---

## Nomenclature & Physical Constants

### 13.1 Electrical Parameters

| Symbol | Name | Value | Units |
|---|---|---|---|
| $G_{\text{ON}}$ | ON conductance | $1 \times 10^{-3}$ | S |
| $G_{\text{OFF}}$ | OFF conductance | $1 \times 10^{-9}$ | S |
| $\rho_{\text{Ag}}$ | Silver resistivity | $1.59 \times 10^{-8}$ | Ω·m |
| $V_{\text{th}}$ | Threshold voltage (SET) | $0.01$ | V |
| $d$ | Junction spacing (intra-wire) | $\sim 1$–$100$ | μm |
| $\alpha$ | Voltage sensitivity | $1$ | V$^{-1}$ |
| $\Delta t\,P_0$ | SET probability **per step** (Lamas 2026, Table 1) | $0.4$ | (dimensionless; $P_0 = 400\ \text{s}^{-1}$ at $\Delta t = 10^{-3}$ s) |
| $\Delta t\,P_{\text{decay}}$ | RESET decay coefficient **per step** (Lamas 2026, Table 1) | $0.01$ (⚠️ unit of $I$ in Eq. 2 not stated in the paper; see §12 and `agnw_sim.dynamics.SimParams.I_unit`) | (dimensionless per step, in units of $I^{-2}$) |
| $\Delta t$ | Simulation time step | $10^{-3}$ | s |

### 13.2 Thermodynamic Parameters

| Symbol | Name | Value | Units |
|---|---|---|---|
| $k_B$ | Boltzmann constant | $1.38 \times 10^{-23}$ | J·K$^{-1}$ |
| $T$ | Substrate temperature | $300$ | K |
| $T_{\text{eff}}$ | Effective temperature (FDT context) | $300 \times X$ | K |
| $\beta$ | Inverse temperature | $\beta = 1/(k_B T)$ | J$^{-1}$ |

### 13.3 Network Parameters

| Symbol | Name | Value | Units |
|---|---|---|---|
| $N$ | Number of nanowires | $1000$–$3000$ | (dimensionless) |
| $N_c$ | Percolation threshold | $\sim 1200$ | (dimensionless) |
| $L$ | Nanowire length | $70$ | μm |
| $A$ | Substrate area | $1$ | mm$^2$ |
| $N_g$ | Number of graph nodes | $\sim 6000$ | (dimensionless) |

### 13.4 Derived Quantities

| Symbol | Definition | Units |
|---|---|---|
| $V_{\text{mem}} = \|V_i - V_j\|$ | Junction voltage drop | V |
| $I_{\text{mem}} = G_{ij} (V_i - V_j)$ | Junction current | A |
| $I_{\text{total}} = \sum_{ij} I_{\text{mem}}$ | Total network current | A |
| $G_{\text{eff}} = I_{\text{total}} / V_{\text{in}}$ | Network conductance | S |
| $X$ | FDT violation ratio | (dimensionless) |
| $\tau_f = 1/(P_0 \alpha V_{\text{th}})$ | Filament timescale | s |
| $\tau_V$ | Voltage ramp timescale | s |

---

## Open Questions & Validation Strategy

### 14.1 Fundamental Open Questions

1. **How precisely does the timescale ratio $\tau_f / \tau_V$ predict $X$?**
   - Current derivation (Eq. 14) is leading-order; higher corrections needed
   - Non-Gaussian noise corrections?

2. **Is there a **universal scaling function** $X(N/N_c, P_{\text{decay}}, \tau_f/\tau_V)$ valid across all parameter regimes?**

3. **How does network topology (graph structure, spatial embedding) modify the FDT ratio?**
   - Current treatment assumes mean-field; loop corrections?

4. **Can brain data (fMRI, EEG) be used to constrain AgNW design parameters (e.g., $P_{\text{decay}}$) for optimal neuromorphic computing?**

### 14.2 Computational Validation Roadmap

**Phase 1: Numerical Verification (from Lamas et al. simulations)**

- Extract full $I(t), V(t)$ trajectories from Lamas simulations at multiple wire densities
- Compute autocorrelation $C(\tau) = \langle \Delta G(0) \Delta G(\tau) \rangle$ for each $N$
- Compute response $\chi(\tau)$ via perturbation analysis (add small $\delta V$ probes)
- Calculate ratio $X = \chi_{\text{measured}} / \chi_{\text{FDT}}$ empirically
- **Test:** Does $X$ match prediction from Eq. (14)?

**Phase 2: Parameter Sweep**

- Vary $P_{\text{decay}}$, $P_0$, $\alpha$ systematically
- Measure effect on $X$ and memory window $G_{\text{final}}/G_{\text{initial}}$
- **Test:** Is there a one-to-one mapping between $X$ and memory window?

**Phase 3: Brain Data Comparison**

- Apply IV-FDT metric to fMRI resting-state data (existing dataset)
- Extract $X_{\text{brain}}$ from spectral analysis
- **Hypothesis:** $X_{\text{brain}} \sim 1.5$–$2.5$ (known from prior work)
- Compare with AgNW-predicted $X$ under similar "criticality" conditions
- **Test:** Can AgNW hardware achieve brain-like FDT violations with physically plausible parameters?

**Phase 4: Information-Theoretic Metrics**

- Compute **transfer entropy** (Zhu et al. 2021) for AgNW simulations
- Compute **active information storage** (AIS) as function of $N/N_c$
- Compute **transfer entropy** and AIS for brain data
- **Test:** Do metrics correlate similarly in both systems?

### 14.3 Experimental Predictions (for collaboration with experimentalists)

1. **Measure $X$ directly** in real AgNW networks via:
   - AC conductance measurements (small-signal response)
   - White noise injection and frequency-resolved correlations
   
2. **Scan percolation regime** systematically in manufactured samples

3. **Compare with thermal bath control:**
   - Vary substrate temperature $T$
   - Does $X(T)$ scale as predicted by theory?

---

## References

### Primary References (Lamas et al. framework)

1. **Lamas, C.A., Medina, A., Arlego, M., Matera, J.M.** (2026).
   "Stochastic Modeling of Silver Nanowire Networks for Neuromorphic Computing."
   *Memorias de las JAIIO—Revista de la Sociedad Argentina de Informática e Investigación Operativa*, 2451–7496.

2. **Milano, G., Michieletti, F., Pilati, D., Ricciardi, C., Miranda, E.** (2025).
   "Self-organizing neuromorphic nanowire networks as stochastic dynamical systems."
   *Nature Communications*, **16**, 3509.

3. **Diaz-Schneider, J.I., Quinteros, C.P., Levy, P., Martínez, E.D.** (2024).
   "Two-junction model in different percolation regimes of silver nanowires networks."
   *Advanced Functional Materials*, **34**(52), 2410766.

4. **Zhu, R., Hochstetter, J., Loeffler, A., et al.** (2021).
   "Information dynamics in neuromorphic nanowire networks."
   *Scientific Reports*, **11**, 13047.

### FDT and Non-Equilibrium Statistical Mechanics

5. **Evans, D.J., Searles, D.J.** (2002).
   "The fluctuation theorem."
   *Advances in Physics*, **51**(7), 1529–1585.

6. **Lebowitz, J.L., Spohn, H.** (1978).
   "Irreversible thermodynamics for interacting oscillators."
   *Journal of Statistical Physics*, **19**(6), 633–654.

7. **Seifert, U.** (2012).
   "Stochastic thermodynamics, fluctuation theorems and molecular machines."
   *Reports on Progress in Physics*, **75**(12), 126001.

### Brain Dynamics and FDT Violations

8. **[Author: Juan]** (2024–2026).
   "IV-FDT framework for whole-brain non-equilibrium metrics."
   *Unpublished; CONICET research program.*

9. **Deco, G., Kringelbach, M.L.** (2020).
   "Metastability and coherence in the whole-brain connectome."
   *Nature Neuroscience*, **23**(12), 1593–1599.

10. **Breakspear, M., Jirsa, V.K., Daunizeau, J.** (2010).
    "Schizophrenia and affective disorders: network mechanisms and interventions."
    *Nature Reviews Neuroscience*, **11**(10), 683–696.

### Neuromorphic Computing & Percolation

11. **Kuncic, Z., Nakayama, T.** (2021).
    "Neuromorphic nanowire networks: Principles, progress and future prospects for neuro-inspired information processing."
    *Advances in Physics: X*, **6**(1), 1894234.

12. **Marković, D., Mizrahi, A., Querlioz, D., Grollier, J.** (2020).
    "Physics for neuromorphic computing."
    *Nature Reviews Physics*, **2**(9), 499–510.

13. **Vahl, A., Milano, G., Kuncic, Z., et al.** (2024).
    "Brain-inspired computing with self-assembled networks of nano-objects."
    *Journal of Physics D: Applied Physics*, **57**(50), 503001.

### Complementary Theory

14. **Hohenberg, P.C., Halperin, B.I.** (1977).
    "Theory of dynamic critical phenomena."
    *Reviews of Modern Physics*, **49**(3), 435–479.

15. **Landauer, R., Martin, T.** (1992).
    "Barrier interaction time (traversal time)."
    *Reviews of Modern Physics*, **64**(2), 217–286.

---

## Document Metadata

| Aspect | Value |
|---|---|
| **Document Type** | Theoretical Framework (Draft) |
| **Purpose** | Foundation for Claude Code analysis and derivation expansion |
| **Target Audience** | Researcher (Juan) + AI-assisted computation tools |
| **Sections Completeness** | All major theoretical sections complete; numerical validation deferred to Phase 1 |
| **Next Action** | Feed this document to Claude Code for: (a) rigorous Fokker-Planck derivation, (b) numerical validation pipeline, (c) publication-ready LaTeX notes |

---

**End of Document**

---

### Notes for Future Development

- **Expand Section 7** with detailed Green-Kubo formula derivation and numerical integration schemes
- **Add computational examples** (pseudocode for computing $X$ from simulation data)
- **Create supplementary notebook:** Step-by-step Fokker-Planck solver with visualization
- **Prepare figure pipeline:** Bifurcation diagrams, FDT ratio vs. network density, hysteresis loop analysis
- **Validation figures:** Comparison of predicted vs. observed $X$ across parameter space
