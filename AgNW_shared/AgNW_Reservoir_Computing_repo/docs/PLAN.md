# Project plan: can an AgNW network be a reservoir computer?

Written 2026-09-25. Supersedes the prototype (moved to `legacy/`).

## 1. Question

Can the stochastic silver-nanowire network simulated by `AgNW_FDT_Memristor_Framework` work as a physical
reservoir? The answer needs three parts:

1. **Consistency / fading memory (echo-state property).** Does the same input produce the same response,
   whatever the initial state and the switching noise? A reservoir that fails this cannot be trained.
2. **Memory.** How many past inputs can a linear readout recover (linear memory capacity)?
3. **Nonlinear processing.** Does the network compute nonlinear functions of its input history that a
   linear readout of the raw input cannot (NARMA-10, degree-2 information-processing capacity, sine/square
   classification)?

Each answer is compared with **baselines that contain no nanowires**: the raw input alone, a linear delay line,
and an echo-state network (ESN) of the same readout size. The network only "works as a reservoir" where it beats
the input-only and delay-line baselines.

## 2. Fixed: the network (taken unchanged from the framework)

The network is built, solved and simulated by the framework's code (`agnw_sim`, imported, never copied) with its
working model (decision of 2026-09-20, `scripts/fdtest_common.build_setup`):

| Item | Value | Source |
|------|-------|--------|
| Geometry | N = 2000 wires, L = 70 um, area $10^{-6}$ m^2, electrode bands 5% | `agnw_sim.geometry.build_network` |
| Circuit | dual-node, wire diameter 90 nm, rho_Ag | `lamas2026_fig4()` run kwargs |
| Dynamics | dt = 1 ms, dt*P0 = 0.4, alpha = 1/V, $V_{\mathrm{th}} = 0.1$ V, dt*$P_{\mathrm{decay}} = 0.01$, $I_{\mathrm{unit}} = 1$ mA | `lamas2026_fig4()` |
| Relaxation | $p_{\mathrm{relax}} = 10^{-4}$ per step (exponential, framework working value) | `build_setup(p_relax=1e-4)` |
| Simulation | `agnw_sim.simulate.run_simulation(..., solver="cached")` | framework |

The only thing this project adds is **how the input is applied and what is read out**.

## 3. Reservoir interface (what this project designs)

* **Input encoding.** A scalar input sequence $u_k$ in [0, 1] is mapped to the electrode voltage
  $V_k = V_{\mathrm{lo}}$ + ($V_{\mathrm{hi}} - V_{\mathrm{lo}}$) $u_k$ and held for `hold` steps (piecewise constant). Swept: `hold` (time per input
  sample) and the voltage window (where it sits relative to the conduction onset).
* **Readout observables** (three measurement scenarios, from most to least realistic):
  * `current` - two-terminal device: total current sampled at up to 10 evenly spaced instants inside each hold
    block (time-multiplexed "virtual nodes").
  * `probes` - K = 32 voltage probes at fixed random nodes of the network, read at the end of each block.
  * `junctions` - ON/OFF state of every junction at the end of each block (an oracle upper bound; not measurable).
  All three are recovered from `run_simulation` outputs: `I_total` (every step) and `G_snapshots` (full ON/OFF
  state at the end of each block). Probe voltages are recomputed from the snapshot with the framework's circuit
  solver (a test checks that the recomputed current equals the simulated one).
* **Readout.** Ridge regression on z-scored features with an unpenalized intercept, solved via SVD. The ridge
  parameter is chosen on a validation segment. All scores are on a held-out test segment
  (train 60% / validation 20% / test 20%, in time order, after a washout).

## 4. Tasks and metrics

| Task | Input stream | Metric |
|------|--------------|--------|
| Linear memory capacity | i.i.d. U(0,1) | MC = sum over k = 1..30 of test r^2(u_{t-k}); lag 0 reported separately |
| Degree-2 capacity (IPC) | same stream | sum of test r^2 for P2(u_{t-k}) (k = 0..10) and u_{t-k} u_{t-l} (0 <= k < l <= 5), Legendre-normalized input |
| Integration (added during E1) | same stream | test capacity for exponential moving averages (1 - lambda) sum_k lambda^k u_{t-k}, lambda = 0.5 ... 0.995 |
| NARMA-10 | same stream (s = 0.5 u) | test NMSE |
| Sine vs square | random sequence of sine/square periods, 8 samples/period | per-sample test accuracy |
| Consistency | same i.i.d. input, 2 replicates with different noise seeds and random pre-drives | mean correlation of each feature across replicates after washout |

A noise floor for capacities is estimated with Gaussian-noise features of the same dimension. Summed capacities
count only lags whose r^2 exceeds that floor. Consistency is also computed for the state part of each observable
(I/V_in, V_probe/V_in). Raw current and probe voltages contain $V_{\mathrm{in}}$ itself, which makes them look consistent
trivially. Readout features that vary by less than $10^{-5}$ (10 uV / 10 nA / ON flags) over the training segment are
ignored, as a measurement resolution.

## 5. Experiments

* **E1 (main sweep).** hold in {10, 30, 100} steps x voltage window centre in {1.25, 1.75, 2.25} V (width 1 V),
  network seeds {1, 2, 3}, 2 noise replicates. All tasks.
* **E2 (mechanism check).** The point is fixed in advance, not picked from the E1 results, to avoid selection bias:
  hold 100 steps, window 1.25-2.25 V. $p_{\mathrm{relax}}$ in {0, $10^{-4}$, $10^{-3}$, $10^{-2}$}. $p_{\mathrm{relax}} = 0$ is the literal paper model
  (RESET effectively inactive); $10^{-3}$ and $10^{-2}$ lie outside the working model. This tests whether the fading memory
  comes from relaxation. It is a sensitivity study, not part of the framework configuration.
* **E3 (timescale matching).** Added after the first E1 runs. The pilot shows SET within ~10^2 steps but
  relaxation over ~10^4 steps, so the network behaves as a slow leaky integrator. E3 adds hold in {300, 1000}
  steps at 1.25-2.25 V (800 samples per run, 3 seeds x 2 replicates, i.i.d. stream only). The capacity noise floor
  is recomputed for the shorter stream.
* **Baselines** for every task: input-only, delay line (32 taps), ESN with 32 and 200 tanh units (input scaling and
  spectral radius tuned on validation), and the noise floor.

Cost: about 2.4 ms per simulation step with 14 parallel processes; E1 is about 7 M steps (about 25 min wall
time). Raw outputs go to `results/raw/*.npz` and runs that already exist are skipped, so the sweep can resume.

## 6. Code layout

```
agnw_rc/            reservoir layer on top of agnw_sim (framework)
  config.py         canonical framework configuration + experiment grid
  drive.py          input -> run_simulation -> observables (current, probes, junction states)
  tasks.py          input streams and targets (i.i.d., NARMA-10, sine/square, Legendre/IPC targets)
  readout.py        SVD ridge with validation-selected alpha
  metrics.py        nmse, accuracy, capacities, consistency, noise floor
  baselines.py      input-only, delay line, ESN
  evaluate.py       one raw run -> metrics dict
scripts/
  pilot.py          operating-point pilot (response vs voltage window)
  run_sims.py       E1/E2/E3 simulations (parallel, resumable)
  evaluate_all.py   raw runs -> results/metrics.json + summary CSV
  make_figures.py   figures/*.png|pdf
tests/              pytest: framework contract, drive alignment, readout, metrics, tasks, baselines, smoke
docs/PLAN.md        this file
docs/RESULTS.md     results write-up with figures
legacy/             the previous prototype (reference only; its MC numbers are invalid)
```

## 7. Tests (written with the code)

* **Framework contract:** the canonical config equals the framework working model (preset, $p_{\mathrm{relax}}$, D, geometry).
* **Drive alignment:** the waveform gives $u_k$ at every step of block k. Snapshots are taken at the last step of
  each block. The current recomputed from snapshot + $V_{\mathrm{in}}$ equals `run_simulation`'s `I_total` at that step.
* **Readout:** the SVD ridge equals the closed form. The intercept is unpenalized, the fit is scale invariant, and
  constant features are dropped.
* **Metrics:** an ideal delay line of depth D gives MC = D, a memoryless readout gives MC of about 0, noise gives
  about 0. The same checks apply to the degree-2 capacity. NMSE and accuracy are checked on known cases.
* **Tasks:** the NARMA-10 recurrence is checked against a direct implementation. Sine/square labels and period
  lengths are checked, and the Legendre targets are orthonormal.
* **Baselines:** the ESN spectral radius is as requested and the ESN has the echo-state property (two initial
  states converge).
* **Smoke:** a short end-to-end run produces every metric.

## 8. Deliverables

README (question, answer, how to run, layout), `docs/RESULTS.md` (methods, figures, tables, limitations),
`figures/`, `results/metrics.json`, tests passing.

## 9. Figures

1. Setup: the network with electrodes and probes, plus an example drive and response ($V_{\mathrm{in}}$, I_total, ON fraction).
2. Consistency: replicate traces overlaid, and consistency across the E1 grid.
3. Memory: memory function r^2(k) per observable vs baselines, and an MC heatmap over (hold, window).
4. Nonlinear processing: NARMA-10 NMSE, degree-2 capacity and sine/square accuracy vs baselines.
5. Mechanism: MC / NARMA-10 / consistency vs p_relax.
