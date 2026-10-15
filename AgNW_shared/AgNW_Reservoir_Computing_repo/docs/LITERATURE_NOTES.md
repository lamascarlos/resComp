# Literature notes: what each paper in `biblio/` means for this project

Verified bibliographic data: [../biblio/references.md](../biblio/references.md).

## Kasdorf et al. 2025, arXiv:2512.12005 (MemNNetSim), read 2026-09-27

**What it is.** An open-source Python simulator of random memristive nanowire networks (MemNNetSim). It offers two
circuit representations: the junction-dominated assumption (JDA, wires as nodes, like our wire-as-node circuit) and
the multi-nodal representation (MNR, like our dual-node graph). It ships three junction models:
* **HP**: linear drift, $dx/dt = i\,h(x)$;
* **Decay HP**: $dx/dt = i\,h(x) - x/\tau$, forgetting with a constant $\tau$;
* **SLT HP** (Chen et al.): a dynamic decay time $\tau$(t) driven by the current, plus a slow retention variable $\varepsilon$, which
  gives short- and long-term memory in the same junction.

**Reservoir result.** A sine -> square waveform transformation, with the readout taken from **all wire voltages**
and trained by BFGS on a 2 s window (sine V0 = 20 V, 5 Hz, dimensionless units). It fails without forgetting (HP:
RNMSE 0.33, "the system ... just kept driving/copying the input signal") and works with a decay of $\tau = 1$ s (Decay
HP: RNMSE 0.14, down to 0.018 in a later run).

**Relevance.**
1. **Independent support for our diagnosis (J2).** Fading memory on a time scale commensurate with the input
   ($\tau = 1$ s vs a 0.2 s input period) is what makes their network compute. Our calibrated models forget over
   10-30 s while the input changes every 10-100 ms. This motivates E4 (time-scale-matched inputs) and a dissolution
   time around 1 s in the N8 trade-off map.
2. **Candidate junction model (new item N10).** The SLT HP combination of fast, activity-dependent forgetting and a
   slow retention variable might reconcile the Fig. 4 calibration (~30 s persistence, needed for the negative-loop
   asymmetry) with short-term memory at reservoir speeds. It is worth adding to the N4 family.
3. **Cross-validation tool.** MemNNetSim's JDA/MNR solvers could be checked against our wire-as-node and dual-node
   circuits on a small network.

**Limits of its reservoir evidence** (why it does not contradict our negative results):
* one task, with the readout on every wire (an oracle, like our "junctions" readout);
* training and scoring on nearby 2 s windows;
* no input-only or delay-line baseline, no memory-capacity or NARMA benchmark;
* no switching noise, no replicate-consistency test;
* dimensionless units at a 20 V drive.

A positive result under these conditions is compatible with our finding that deterministic, forgetting junctions
(our mean-field and N4 variants) respond reproducibly, while memory and competitiveness remain the hard part.

## Mallinson et al. 2024, Adv. Mater. 36, 2402319

Experimental reservoir computing with percolating networks of **nanoparticles**, not nanowires: chaotic time-series
prediction, nonlinear transformation, memory capacity, and spoken digits, plus an emulated parallel-reservoir
architecture. This is the experimental benchmark our G1/G2 criteria should eventually be compared with (their memory
capacity and nonlinear-transformation numbers). *To do: extract their MC / NLT values and input time scales.*

## Measured AgNW network dynamics (search 2026-09-27, for N10)

Lamas Fig. 4 is a simulation that is only qualitatively similar to measured loops (Juan, 2026-09-27), so the next
junction model is constrained by **measured** data. What the accessible literature reports (full texts read via
Europe PMC / arXiv; items marked *secondary* were not read in the original):

| source | system | measured facts relevant to junction dynamics |
|---|---|---|
| Diaz-Alvarez et al. 2019, *Sci. Rep.* 9, 14920 (PMC6797708) | Ag-PVP NW network on SiO2, 0.76 NW/µm$^{2}$, two W probes 500 µm apart | Threshold activation (LCS -> HCS) under constant bias; at 3 V the current follows a sigmoid: negligible, then an exponential rise of > 2 orders of magnitude "in tenths of seconds", then a logarithmic increase. After a 5 V activation pulse, read at 10 mV: the conductance decays with **abrupt drops and irregular plateaus**; decay times are **random between cycles** and ~$10 \times$ longer than a model with a 1 s decay term, i.e. **~10 s**. Repeated I-V cycles (10 Hz) activate at lower thresholds (short-term memory); a final very stable HCS shows long-lasting backbone changes (long-term memory). |
| Milano et al. 2023, *Nat. Commun.* 14, 5723 (PMC10533552) | Ag-PVP NWs, 115 nm diameter, 20-50 µm long, PVP shell 1-2 nm, ~$10^{5}$ NW/mm$^{2}$, $10 \times 10$ mm$^{2}$, 16 probes | Potentiation under a 10 s pulse (1, 2 or 3 V), then **spontaneous relaxation towards the pristine state** (short-term / working memory; read 10 mV). Larger amplitude (2 V vs 1 V) leaves **long-lasting changes** (consolidation into long-term memory, attributed to topology changes). Paired-pulse facilitation for temporally correlated pulses. Time constants are in the supplementary information (not accessed). |
| Milano et al. 2022, *Nat. Mater.* 21, 195 (paywalled; parameters *secondary*, from Cipollini et al. 2024, arXiv:2411.19353, Table 1) | Ag NW network reservoir | Edge model fitted to network measurements: $dg/dt = k_p(V)\,(1 - g) - k_d(V)\,g$, $k_p = k_{p0}\,e^{\eta_p \lvert V\rvert}$, $k_d = k_{d0}\,e^{-\eta_d \lvert V\rvert}$, with **$k_{\mathrm{p0}} = 2.56 \times 10^{-6}$ s$^{-1}$, $k_{\mathrm{d0}} = 64.9$ s$^{-1}$, $\eta_p = 34.9$ V$^{-1}$, $\eta_d = 5.59$ V$^{-1}$** (effective $G_{\mathrm{min}}$ 1.01 mS, $G_{\mathrm{max}}$ 2.72 mS for a specific electrode pair). At zero bias the relaxation rate is $k_{\mathrm{d0}}$, i.e. $\tau \approx 15$ ms; potentiation becomes fast above $\lvert V \rvert \approx 0.4\text{–}0.5$ V per edge. Volatile, single time scale, deterministic. **Correction 2026-10-15: the 15 ms is not supported by the authors' own material.** Their public code (GitHub `MilanoGianluca/Nanowire_Network_Reservoir_Computing`, `Fig_2/fit_testing.py`) fits a $21 \times 21$ grid graph to a measured 2 V pulse with $k_{\mathrm{d0}} = 0.0487$ s$^{-1}$ ($\tau \approx 20.5$ s at zero bias), $k_{\mathrm{p0}} = 8.4 \times 10^{-10}$ s$^{-1}$, $\eta_p = 0.200$ V$^{-1}$, $\eta_d = 158$ V$^{-1}$; the Cipollini Table 1 set does not appear there and its origin is unverified. The measured trace (`data/milano2022/`, `scripts/milano2022_relax.py`, `results/milano2022_relax.json`) relaxes at 10 mV with 11.5 s and 147 s (two exponentials, $\Delta\mathrm{AIC} = 1479$ over one, which gives 38 s); 70 % of the excess conductance is left after 10 s, 26 % after 240 s, and no ms component is visible at 20 Hz sampling. |
| Diaz Schneider, Quinteros, Levy, Martínez 2024, arXiv:2409.01318 (CNEA/CONICET Bariloche) | AgNW networks over a wide density range | Near percolation (P regime): high pristine resistance, electrical activation; the filament is "thermodynamically unstable and labile" and **relaxes spontaneously** when the stimulus stops (resistance vs time at 1 mV); **excess power ruptures filaments** (deactivation), so the I-V shows an HR->LR->HR sequence. Two-junction model: filament forms above **$V_c = 0.4$ V**, burns above **$P_{\mathrm{cf}} = 1$ mW**, wires fuse above $P_{\mathrm{cw}} = 30$ mW, R_open/R_closed = $10^{4}$, $R_{\mathrm{fo}} = 8.5$ M$\Omega$. Over-percolated (OP) regime: electro-fusing, non-polar switching at about $\pm1$ V (also Diaz Schneider et al. 2025, arXiv:2501.16886). |
| Zhu et al. 2023, *Nat. Commun.* 14, 6697 (PMC10620219) | Ag-PVP NW network on a 16-electrode array | Online learning (MNIST 93.4 %, sequence memory) with inputs of **1 ms per pixel**: the measured device responds and keeps useful fading memory at ms-scale drive. |
| Hochstetter et al. 2021, *Nat. Commun.* 12, 4008 (PMC8242064) | simulation (basis of N4) | $\Lambda$ model with a hold band ($V_{\mathrm{reset}} \le \lvert V \rvert \le V_{\mathrm{set}}$: no change); decay only below $V_{\mathrm{reset}}$. The source of our N4 freezing. |

**What the measurements say about the junction model** (the N4 and binary models violate points 1-3):
1. **Volatility without a hold band.** Filaments relax spontaneously once the drive drops (seconds; random, with
   plateaus), not only below a threshold voltage.
2. **Two time scales.** Short-term relaxation (ms to tens of s) plus long-term consolidation that depends on the
   stimulation amplitude/duration (Milano 2023, Diaz-Alvarez 2019 "very stable HCS").
3. **Power/current-driven rupture** at high current (Diaz Schneider 2024; the framework's current-driven RESET is
   a version of this).
4. **Stochasticity** of relaxation times (Diaz-Alvarez 2019) — heterogeneity across junctions and cycles.
5. **Measured devices compute at ms-to-s input rates** (Zhu 2023: 1 ms per input; Milano 2022 reservoir), which
   our models cannot reproduce.

## Published measured data usable for calibration (search 2026-09-29)

* **Milano, Michieletti, Pilati, Ricciardi, Miranda 2025, *Nat. Commun.*, DOI 10.1038/s41467-025-58741-2
  (PMC11994789); dataset CC-BY on Zenodo, DOI 10.5281/zenodo.15050217 (41 text files by figure, 380 MB);
  code on GitHub (MilanoGianluca/Self-organizing_neuromorphic_networks_as_stochastic_dynamical_systems).**
  Ag NWs 20-50 µm long, ~115 nm diameter, drop-cast on ~1 $\times$ 1 cm SiO$_2$, two Au electrodes ~7 mm apart;
  Keithley 6430, sampling ~1.6 Hz. Protocol: 10 h constant-voltage pulses from 0.1 to 6.6 V in 0.1 V steps, each
  followed by 1 h of reading at 10 mV (relaxation to the ground state). Network-level model: an Ornstein-Uhlenbeck
  process $dg = \theta(\tilde g - g)\,dt + \sigma\,dW + \Gamma\,dq$ with $\theta = \kappa_P + \kappa_D$,
  $\tilde g = \kappa_P/(\kappa_P + \kappa_D)$, $\kappa_{P,D} = \kappa_{P0,D0}\,e^{\pm\eta_{P,D} V}$ (the rate law of
  N10). Measured: steady state vs voltage (onset for a 7 mm gap), potentiation and relaxation transients, Gaussian
  conductance noise (disentangled), and jumps as a Poisson process with rate $\lambda \approx 0.082\ \mathrm{s^{-1}}$
  and power-law amplitudes, $p(\Gamma) \propto \Gamma^{-2.78 \pm 0.07}$. They also test NARMA with the model.
  **The best available calibration source for N10**: relaxation time, onset vs electrode gap, Gaussian noise level
  and jump statistics, all at network level. Limits: two-terminal, network level (not per junction); 1.6 Hz
  sampling (switching faster than ~1 s is not resolved); a much larger network than ours.
* Diaz-Alvarez et al. 2019 (above): random relaxation times (~10 s) and the distribution of times between
  switching events (their Fig. S3).
* Milano et al. 2023 (above): relaxation curves after 1-3 V, 10 s pulses (supplementary).
* Milano et al. 2022 (above, correction 2026-10-15): measured 2 V / 100 s pulse and 250 s relaxation at 10 mV, 20 Hz,
  in their GitHub repository (Zenodo 10.5281/zenodo.5153335); copied to `data/milano2022/`. Relaxation 11.5 s + 147 s.
  Used as an independent check of N11 stage 1 (not in its loss; Juan 2026-10-15).
* Diaz Schneider et al. 2024 (above): resistance vs time at 1 mV (Fig. 2b) and persistence vs pulse amplitude,
  duration and repetition (Figs. S2-S3); no time constant in the text. Raw data requested from the group (2026-10-15).
* Summary (2026-10-15): every measured AgNW network relaxes on seconds to minutes, with several time scales and an
  incomplete return (Milano 2022: 11.5 s + 147 s; Milano 2025: 27 s + slow drift; Diaz-Alvarez 2019: ~10 s, random).
  No measurement shows ms relaxation. Network relaxation is not junction relaxation (spread and rerouting slow it).
* Loeffler et al. 2023, *Sci. Adv.* 9, eadg3289 (Zenodo 7633958): working memory and metaplasticity; data mostly
  from their simulation model, to be checked.
* Diaz Schneider, Gomez, Acha, Levy, Martínez, Quinteros 2025, arXiv:2511.01792: dense Ag-based NW networks, DC and
  AC transport, capacitive response after thermal treatment (Bariloche group, possible direct collaborators).

## Modelling junction memory: short-term, long-term, consolidation (search 2026-10-05, for N10-K)

All DOIs / arXiv IDs resolved live (Crossref / arXiv, `lit-positioning` verifier); claims checked against the abstract
or full text (marked *abstract only* where the full text was not accessible).

**Single devices: one filament, two time scales.**
* Ohno et al. 2011, *Nat. Mater.* 10, 591, DOI 10.1038/nmat3054. Ag$_2$S atomic switch: sparse pulses give short-term
  memory that decays spontaneously; frequent pulses move it into a long-term state (STP $\to$ LTP). The classic
  demonstration of the consolidation that N10's $y$ represents.
* Wang et al. 2017 (online 2016), *Nat. Mater.* 16, 101, DOI 10.1038/nmat4756. Diffusive Ag-in-oxide memristors: Ag
  disperses under bias and regroups spontaneously at zero bias (interfacial energy), giving volatile, Ca$^{2+}$-like
  dynamics; short- and long-term plasticity emulated.
* Wang et al. 2019, *Nat. Commun.* 10, 81, DOI 10.1038/s41467-018-07979-0. **Physical basis for two filament types**:
  filament lifetime is limited by surface diffusion and grows with filament diameter, $\tau \propto d_0^{4}$ for thin
  filaments (from ~10 µs below 1 nm to years near 14 nm); the compliance current sets the size, so the same device
  switches between volatile and non-volatile behaviour. In N10 terms: $x$ = thin, labile filament; $y$ = thick, stable one.
* Chen et al. 2014, *Phys. Lett. A* 378, 2924, DOI 10.1016/j.physleta.2014.08.018. Phenomenological HP-type model
  with a diffusion (forgetting) term towards a retention variable $\varepsilon$ that grows with stimulation, plus a
  current-dependent decay time: STM -> LTM transition (*abstract*; equations as reproduced in Kasdorf 2025).

**Network simulations of AgNW / nanowire networks.**
* Kasdorf et al. 2025, arXiv:2512.12005 (MemNNetSim): implements Chen 2014 as the "SLT HP" junction
  ($dx/dt = i - (x - \varepsilon)/\tau$, $d\tau/dt = i\,\theta\,(a - x)$, $d\varepsilon/dt = i\,\sigma\,h(x)$; dimensionless, current
  driven). The closest existing network model to N10's $x + y$; not calibrated on measured AgNW data, no noise.
* Kuncic / Stieg group: Hochstetter et al. 2021, *Nat. Commun.* 12, 4008, DOI 10.1038/s41467-021-24260-z (single filament
  variable with set/reset thresholds and a hold band; basis of N4); Loeffler et al. 2023, *Sci. Adv.* 9, eadg3289,
  DOI 10.1126/sciadv.adg3289 (*abstract only*): simulations show metaplasticity and memory consolidation through
  strengthening and pruning of conductance pathways, i.e. long-term memory emerging at network level.
* Milano / Ricciardi group: Milano et al. 2022, *Nat. Mater.* 21, 195, DOI 10.1038/s41563-021-01099-9 (rate-balance edge
  model, N10's $x$); Montano, Milano, Ricciardi 2022, *Neuromorph. Comput. Eng.* 2, 014007, DOI 10.1088/2634-4386/ac4d86
  (grid-graph model, single rate-balance variable per edge: short-term plasticity, paired-pulse facilitation,
  heterosynaptic plasticity); Milano et al. 2023, *Nat. Commun.* 14, 5723, DOI 10.1038/s41467-023-40939-x (same model;
  long-lasting memory engrams are **emergent**: no separate consolidation variable; regions with large voltage drops form
  non-volatile filaments; measured 1 V pulses relax, 2 V pulses leave long-lasting changes; no time constants given);
  Milano et al. 2025, *Nat. Commun.*, DOI 10.1038/s41467-025-58741-2 (network-level Ornstein-Uhlenbeck model with
  Gaussian noise and Poisson jumps; N10-C).
* Baccetti et al. 2023, arXiv:2310.09530: in memristive-network reservoirs, loss of ergodicity goes with emerging memory
  and **better** performance near the edge between two stable states (a counterpoint to the drift worry for $y$).
* Lamas, Medina, Arlego, Matera 2026, 55 JAIIO (IFLP-CONICET La Plata; no DOI, PDF at 55jaiio.sadio.org.ar): the
  stochastic dual-node framework used here (binary voltage-dependent switching probabilities, percolation regimes);
  no long-term variable.

**What this means for N10-K.**
1. Two time scales in one Ag junction are well established experimentally (Ohno 2011, Wang 2017) and have a physical
   mechanism (filament size, Wang 2019). Single-device STM/LTM models exist (Chen 2014).
2. AgNW **network** models mostly use one variable per junction; long-term memory either emerges from topology and
   strong local drive (Milano 2023, Loeffler 2023) or is added phenomenologically (MemNNetSim SLT HP). In N10 the
   volatile $x$ always relaxes at zero bias, so emergent non-volatility cannot occur: an explicit $y$ is needed.
3. Not found in this search (thorough, not exhaustive): consolidation used to make a nanowire reservoir **robust to
   junction noise**. That is what N10-K tests.
4. A continuous alternative to separate $x$/$y$, suggested by Wang 2019: a relaxation time that grows steeply with
   filament size, e.g. $\tau_{s,j}$ increasing with accumulated growth. Keep as a variant if $x + y$ shows artefacts.
