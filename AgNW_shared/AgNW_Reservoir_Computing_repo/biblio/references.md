> **Checked 2026-09-27.** Every DOI below resolves at doi.org and was matched against its registry record
> (Crossref, or DataCite for arXiv). Two entries were also checked against the PDFs in this folder. Summaries marked *(not checked against the full text)* come from the previous version and have not been verified against the paper.  
> **Updated 2026-10-15:** section 4 added (relaxation-time search; DOIs checked against Crossref / DataCite), local
> copies marked, the Milano 2022 parameter source corrected (Cipollini entry).

### 1. Theoretical Models & Simulations (In-Silico)

* **2D vs. 3D Topology Analysis**
  * **Reference:** Daniels, R. K., Mallinson, J. B., Heywood, Z. E., Bones, P. J., Arnold, M. D., & Brown, S. A. (2022). *Reservoir computing with 3D nanowire networks*. *Neural Networks*, 154, 122-130.
  * **DOI:** [10.1016/j.neunet.2022.07.001](https://doi.org/10.1016/j.neunet.2022.07.001)
  * **Summary:** This study uses simulation models to evaluate how realistic 3D stacking of nanowires during structural fabrication affects computational capacity compared to standard 2D layouts. It evaluates metrics like Memory Capacity (MC) and Nonlinear Transformation (NLT). *(not checked against the full text)*

* **Unified Simulation Frameworks**
  * **Reference:** Kasdorf, M., Simpson-Ochoa, D., Bekhit, A., Ferreira, M. S., Nicola, W., & Gomes da Rocha, C. (2025). *Characterizing Memristive Nanowire Network Models via a Unified Computational Framework*. *arXiv preprint* arXiv:2512.12005 [cond-mat.mes-hall].
  * **DOI:** [10.48550/arXiv.2512.12005](https://doi.org/10.48550/arXiv.2512.12005)
  * **Local copy:** `Kasdorf 2025 - Characterizing Memristive Nanowire Network Models via a Unified Computational Framework - arxiv.pdf`
  * **Summary:** Introduces **MemNNetSim**, an open-source Python simulation package designed to model random memristive nanowire network dynamics, helping researchers evaluate arbitrary junction models under complex spatial-temporal electrical inputs.

* **Unconventional Electrode Layouts**
  * **Reference:** Milano, G., Montano, K., & Ricciardi, C. (2023). *In materia implementation strategies of physical reservoir computing with memristive nanonetworks*. *Journal of Physics D: Applied Physics*, 56(8), 084005.
  * **DOI:** [10.1088/1361-6463/acb7ff](https://doi.org/10.1088/1361-6463/acb7ff)
  * **Summary:** Proves via simulated network physics that self-assembled structures allow the same electrodes to serve as both inputs and outputs simultaneously. This eliminates complex layout demands while preserving high classification accuracy. *(not checked against the full text)*

### 2. Experimental Applications (In-Materia)

* **Fully Memristive Hardware Integration**
  * **Reference:** Milano, G., Pedretti, G., Montano, K., Ricci, S., Hashemkhani, S., Boarino, L., Ielmini, D., & Ricciardi, C. (2022). *In materia reservoir computing with a fully memristive architecture based on self-organizing nanowire networks*. *Nature Materials*, 21(2), 195-202. (Published online 2021.)
  * **DOI:** [10.1038/s41563-021-01099-9](https://doi.org/10.1038/s41563-021-01099-9)
  * **Summary:** A benchmark physical hardware demonstration using silver (Ag) nanowire networks. It achieves real-time speech/pattern recognition and chaotic time-series forecasting by mapping electrical outputs directly onto an integrated memristive ReRAM readout layer. *(not checked against the full text)*
  * **Code and data:** GitHub `MilanoGianluca/Nanowire_Network_Reservoir_Computing`, Zenodo [10.5281/zenodo.5153335](https://doi.org/10.5281/zenodo.5153335). The measured 2 V pulse and relaxation trace (`Fig_2/raw_data/pulse_2V.txt`) is copied to `data/milano2022/`; the authors' grid-graph fit there has $k_{d0} = 0.0487$ s$^{-1}$ ($\tau \approx 20$ s), see `results/milano2022_relax.json`.
  * **Local copy:** none (paywalled). The accepted manuscript is open at iris.inrim.it/bitstream/11696/73114/3/Manuscript_revised_rev_2_v2.pdf, but the server blocks scripts: download it by browser.

* **Adaptive Multi-Reservoir Routing**
  * **Reference:** Rajarajan, R. S., & Kumar, A. (2026). *Adaptive Multi-Reservoir Neuromorphic Computing Using Memristive Nanowire Networks Enabled by Training-Induced Reservoir Diversity and Confidence-Guided Routing*. *Research Square preprint* (posted 11 August 2026; not peer reviewed).
  * **DOI:** [10.21203/rs.3.rs-10538116/v1](https://doi.org/10.21203/rs.3.rs-10538116/v1)
  * **Summary:** Introduces a multi-layered physical nanowire hardware setup that routes computational tasks dynamically. Simple temporal data stops at the first reservoir layer, while highly noisy inputs are pushed deeper into complex network pools to maximize efficiency. *(not checked against the full text)*

* **Nanoparticle Percolating Networks**
  * **Reference:** Mallinson, J. B., Steel, J. K., Heywood, Z. E., Studholme, S. J., Bones, P. J., & Brown, S. A. (2024). *Experimental Demonstration of Reservoir Computing with Self-Assembled Percolating Networks of Nanoparticles*. *Advanced Materials*, 36(29), 2402319.
  * **DOI:** [10.1002/adma.202402319](https://doi.org/10.1002/adma.202402319)
  * **Local copy:** `Mallinson 2024 - Experimental Demonstration of Reservoir Computing with Self‐Assembled Percolating - Advanced Materials.pdf`
  * **Summary:** Shows experimentally that percolating networks of nanoparticles (PNNs, not nanowires) work as physical reservoirs on chaotic time-series prediction, nonlinear transformation and memory capacity, comparing favourably with earlier nanoelectronic reservoirs. It reaches state-of-the-art spoken digit recognition, and an emulated parallel-reservoir architecture improves all tasks further (from the abstract).


### 3. Measured AgNW network dynamics (added 2026-09-27 for N10; DOIs checked via Europe PMC / arXiv)

* **Diaz-Alvarez, A., Higuchi, R., Sanz-Leon, P., Marcus, I., Shingaya, Y., Stieg, A. Z., Gimzewski, J. K., Kuncic, Z., & Nakayama, T. (2019).** *Emergent dynamics of neuromorphic nanowire networks*. *Scientific Reports*, 9, 14920. DOI: [10.1038/s41598-019-51330-6](https://doi.org/10.1038/s41598-019-51330-6) (open access, PMC6797708). Measured activation and random relaxation of Ag-PVP networks (~10 s). Local copy.
* **Milano, G., Cultrera, A., Boarino, L., Callegaro, L., & Ricciardi, C. (2023).** *Tomography of memory engrams in self-organizing nanowire connectomes*. *Nature Communications*, 14, 5723. DOI: [10.1038/s41467-023-40939-x](https://doi.org/10.1038/s41467-023-40939-x) (PMC10533552). Short-term relaxation and amplitude-dependent long-term consolidation.
* **Zhu, R., Lilak, S., Loeffler, A., Lizier, J., Stieg, A., Gimzewski, J., & Kuncic, Z. (2023).** *Online dynamical learning and sequence memory with neuromorphic nanowire networks*. *Nature Communications*, 14, 6697. DOI: [10.1038/s41467-023-42470-5](https://doi.org/10.1038/s41467-023-42470-5) (PMC10620219). Measured device driven at 1 ms per input.
* **Hochstetter, J., Zhu, R., Loeffler, A., Diaz-Alvarez, A., Nakayama, T., & Kuncic, Z. (2021).** *Avalanches and edge-of-chaos learning in neuromorphic nanowire networks*. *Nature Communications*, 12, 4008. DOI: [10.1038/s41467-021-24260-z](https://doi.org/10.1038/s41467-021-24260-z) (PMC8242064). Source of the N4 junction model.
* **Diaz Schneider, J. I., Quinteros, C. P., Levy, P. E., & Martínez, E. D. (2024).** *Two-junction model in different percolation regimes of silver nanowires networks*. arXiv:2409.01318. Measured activation, spontaneous relaxation and power-driven deactivation; junction threshold 0.4 V. No relaxation time constant in the text (Fig. 2b, Figs. S2-S3); raw data requested 2026-10-15. Local copy.
* **Diaz Schneider, J. I., Martínez, E. D., Levy, P. E., & Quinteros, C. P. (2025).** *Unconventional resistive switching in dense Ag-based nanowire networks with brain-inspired perspectives*. arXiv:2501.16886.
* **Cipollini, D., Greatorex, H., Mastella, M., Chicca, E., & Schomaker, L. (2024).** *Fused-MemBrain: a spiking processor combining CMOS and self-assembled memristive networks*. arXiv:2411.19353. Table 1 lists the Milano et al. (2022) edge-model parameters used for N10 (secondary source). **Checked 2026-10-15: not supported by the Milano 2022 code**, which fits $k_{d0} = 0.0487$ s$^{-1}$ (not 64.9), $\eta_p = 0.200$, $\eta_d = 158$ V$^{-1}$ on a grid graph; the origin of the Table 1 set is unverified.
* **Milano, G., Michieletti, F., Pilati, D., Ricciardi, C., & Miranda, E. (2025).** *Self-organizing neuromorphic nanowire networks as stochastic dynamical systems*. *Nature Communications*, 16. DOI: [10.1038/s41467-025-58741-2](https://doi.org/10.1038/s41467-025-58741-2) (PMC11994789). Dataset: [10.5281/zenodo.15050217](https://doi.org/10.5281/zenodo.15050217) (CC-BY). Measured noise and jump statistics, relaxation and onset of Ag NW networks (N11 targets). Local copy.
* **Loeffler, A., et al. (2023).** *Neuromorphic learning, working memory, and metaplasticity in nanowire networks*. *Science Advances*, 9, eadg3289. DOI: [10.1126/sciadv.adg3289](https://doi.org/10.1126/sciadv.adg3289).
* **Diaz Schneider, J. I., Gomez, C., Acha, C., Levy, P. E., Martínez, E. D., & Quinteros, C. P. (2025).** *Thermal tuning of dynamic response in Ag-based nanowire networks*. arXiv:2511.01792.
* **Ohno, T., Hasegawa, T., Tsuruoka, T., Terabe, K., Gimzewski, J. K., & Aono, M. (2011).** *Short-term plasticity and long-term potentiation mimicked in single inorganic synapses*. *Nature Materials*, 10, 591. DOI: [10.1038/nmat3054](https://doi.org/10.1038/nmat3054). STP -> LTP by stimulation rate in an Ag$_2$S atomic switch (N10-K).
* **Wang, Z., et al. (2017; online 2016).** *Memristors with diffusive dynamics as synaptic emulators for neuromorphic computing*. *Nature Materials*, 16, 101. DOI: [10.1038/nmat4756](https://doi.org/10.1038/nmat4756). Volatile Ag filaments that regroup at zero bias.
* **Wang, W., Wang, M., Ambrosi, E., Bricalli, A., Laudato, M., Sun, Z., Chen, X., & Ielmini, D. (2019).** *Surface diffusion-limited lifetime of silver and copper nanofilaments in resistive switching devices*. *Nature Communications*, 10, 81. DOI: [10.1038/s41467-018-07979-0](https://doi.org/10.1038/s41467-018-07979-0). Lifetime $\propto d_0^{4}$: thin filaments volatile, thick ones stable (physical basis of N10 $x$/$y$).
* **Chen, L., Li, C., Huang, T., Ahmad, H. G., & Chen, Y. (2014).** *A phenomenological memristor model for short-term/long-term memory*. *Physics Letters A*, 378, 2924. DOI: [10.1016/j.physleta.2014.08.018](https://doi.org/10.1016/j.physleta.2014.08.018). The SLT model used in MemNNetSim.
* **Montano, K., Milano, G., & Ricciardi, C. (2022).** *Grid-graph modeling of emergent neuromorphic dynamics and heterosynaptic plasticity in memristive nanonetworks*. *Neuromorphic Computing and Engineering*, 2, 014007. DOI: [10.1088/2634-4386/ac4d86](https://doi.org/10.1088/2634-4386/ac4d86).
* **Baccetti, V., Zhu, R., Kuncic, Z., & Caravelli, F. (2023).** *Ergodicity, lack thereof, and the performance of reservoir computing with memristive networks*. arXiv:2310.09530. Local copy.
* **Lamas, C. A., Medina, A., Arlego, M., & Matera, M. (2026).** *Stochastic Modeling of Silver Nanowire Networks for Neuromorphic Computing*. 55 JAIIO (SADIO), La Plata. The stochastic framework this project imports.

### 4. Relaxation-time search (added 2026-10-15; local copy in this folder unless marked)

Context: docs/LITERATURE_NOTES.md (relaxation summary) and docs/PLAN_N11_CALIBRATION.md section 12. Also used from
section 3: Diaz-Alvarez 2019, Diaz Schneider 2024, Milano 2025, Baccetti 2023; from section 2: Milano 2022 (data and code).

* **Milano, G., Pedretti, G., Fretto, M., Boarino, L., Benfenati, F., Ielmini, D., Valov, I., & Ricciardi, C. (2020).** *Brain-Inspired Structural Plasticity through Reweighting and Rewiring in Multi-Terminal Self-Organizing Memristive Nanowire Networks*. *Advanced Intelligent Systems*, 2(8), 2000096. DOI: [10.1002/aisy.202000096](https://doi.org/10.1002/aisy.202000096). Local copy is the preprint, arXiv:1909.02438, titled *Self-organizing memristive nanowire networks with structural plasticity emulate biological neuronal circuits*. Single Ag-PVP junction: volatile, back to high resistance within the 60 s between sweeps (an upper bound, not a measured time).
* **Adejube, B., Gronenberg, O., Steel, J., Mallinson, J., et al. (19 authors, last Vahl, A.) (2025).** *Silver-Based Self-Organized Resistive Switching Nanoparticle Networks with Neural-Like Spiking Behavior: Implications for Neuromorphic Computing*. *ACS Applied Nano Materials*, 8(34), 16680-16693. DOI: [10.1021/acsanm.5c02520](https://doi.org/10.1021/acsanm.5c02520) (CC-BY). Switch-on and relaxation times of Ag nanoparticle networks from constant-voltage series *(not checked against the full text)*. **No local copy**: ACS and the Kiel repository block scripts; download by browser.
* **Lemos, C., Dias, C., Costa, R. S., & Ventura, J. (2026).** *Fabrication of Stochastic Ni@PVP Nanowire Networks for Memristive Platforms*. *Polymers*, 18(6), 746. DOI: [10.3390/polym18060746](https://doi.org/10.3390/polym18060746). Ni@PVP networks: volatile switching, OFF-state stable over 1 h of reading *(from the abstract and search summary)*.
* **Mensing, J., van der Wiel, W. G., & Heuer, A. (2026).** *Nanoparticle Networks for Neuromorphic Computing*. arXiv:2607.27844, DOI: [10.48550/arXiv.2607.27844](https://doi.org/10.48550/arXiv.2607.27844). Memory time constant grows with network size and is tuned to the task *(not checked against the full text)*.
