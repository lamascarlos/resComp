# Theory audit: `AgNW_FDT_Memristor_Framework.md` §5–8, §10, §13

**Date:** 2026-09-19. **Scope:** Eqs. 5–19 and the parameter table, checked against the model that
`agnw_sim` actually simulates (a Bernoulli two-state junction, Lamas 2026 Eqs. 3–4).
**Not audited:** §2, §4.3–4.4 (effective potential/friction), §9 (GLE, brain mapping), §11 (HTSA),
§14 (roadmap). **Reproduce:** `python3 docs/theory_audit/verify_symbolic.py` (SymPy, 51 checks),
`.venv/bin/python docs/theory_audit/verify_mc.py` (Monte Carlo with the package's own
`update_dynamics`) and `.venv/bin/python docs/theory_audit/verify_relaxation.py` (exact + Monte Carlo
for the relaxation mechanisms of §6). Outputs are in `results_symbolic.txt`, `results_mc.txt`,
`results_relaxation.txt`. **§6 was added 2026-09-20** (spontaneous relaxation).

**Verification honesty.** "Symbolic" means SymPy reduced the difference to zero. Some
"refutation" checks fell back to numeric spot-checks (marked *numeric*). The MC checks are
statistical (errors quoted). The Kramers–Moyal variance is verified by MC exactly, and
symbolically only to leading order. The critical-density literature value in §10.1
(5.637) is quoted **from memory** and should be checked against a reference before it is cited.

---

## 1. Summary of findings

| # | Where | Finding | Severity | Verified by |
|---|---|---|---|---|
| F1 | §6.2, §7.1, §13.2 | $X$ is **not defined**: the dynamics contain no $k_BT$ and no coupling $q$ of $V$ to $g$, so $\chi_{\text{FDT}}=\beta\,\mathrm{d}C/\mathrm{d}t$ has no meaning; $T_{\text{eff}}=300\,X$ is circular | critical | derivation (§3), SymPy |
| F2 | §7 | For one junction at fixed $(V,I)$ the response/correlation ratio is a **constant** $\mathrm{d}\ln(a/b)/\mathrm{d}V$: no time-dependent "violation" exists at single-junction level (true for the Markov model, also with exponential relaxation; **not** for a fixed lifetime, see F13) | critical | SymPy + MC |
| F3 | Eq. 8 | $\chi=+\beta\,\mathrm{d}C/\mathrm{d}t$ has the wrong sign (correct: impulse response $R=-\beta\,\mathrm{d}C/\mathrm{d}t$; step response $\beta[C(0)-C(t)]$) | high | SymPy |
| F4 | Eq. 12 | "Green–Kubo" $\int\langle\dot g\dot g\rangle/k_BT$ is not a response function to $V$ (units mismatch; $V$ does not enter as an energy). Correct linear response for a rate perturbation is derived in §3 | high | SymPy (units) |
| F5 | Eqs. 13–14, §13.1 | $X-1\propto P_{\text{decay}}/(P_0\alpha)$ is **dimensionally inconsistent** (units V/A² and 1/V) and is not derived from anything; the printed unit of $P_{\text{decay}}$ (s·A⁻²) contradicts Eq. 2 (must be s⁻¹A⁻²) | high | SymPy (units) |
| F6 | Eq. 7 | Diffusion coefficient has the wrong $g$-dependence $(1-g)^2$, $g^2$ and no $1/N$; MC variance is 28× smaller than the doc's | high | MC + SymPy |
| F7 | Eqs. 5, 10 | A Fokker–Planck equation for a **binary** variable is ill-posed; the boundary term in Eq. 10 ($\partial_g B|_0^1$) is not what integration by parts gives | medium | SymPy (abstract functions) |
| F8 | Eq. 16, §8.2 | $\oint I\,\mathrm{d}V$ is **not** dissipated energy (units A·V = W; zero for a linear resistor that dissipates). `hysteresis_loop_area` in `agnw_sim/analysis.py` repeats the claim in its docstring | high | SymPy + units |
| F9 | Eqs. 17–19 | Two inconsistent definitions of $X$ (Eq. 9 vs Eq. 19); $\dot\Sigma=IV/T$ is Joule heating over the bath temperature and is non-negative by construction | medium | derivation |
| F10 | Eq. 11 | Linearization in $\delta V$ needs $\alpha\,\delta V\ll1$; with $\alpha=1\,\mathrm{V^{-1}}$ and $V_{\text{mem}}\sim0.1$–$1$ V it fails | low | inspection |
| F11 | §10.1–10.2, §13, §14.3 | $N_c\approx1200$ vs. 1150 from the stick-percolation value; memory-window peak quoted at 1200 vs. 1500 in Lamas Fig. 5B; brain target $X\sim1.5$–$2.5$ vs. optimization doc "$X\sim10$–$1000$" | low | SymPy (arithmetic) |
| F12 | §6, relaxation (exp.) | With spontaneous relaxation $\epsilon$ (extra ON→OFF, bias-independent) the exact ratio is $\delta s/s-\delta b/b$: the SET side is unchanged, the RESET-driven part is diluted by $r/(r+\epsilon)$ | info | SymPy + MC |
| F13 | §6, relaxation (fixed lifetime) | A fixed lifetime makes the junction an alternating renewal process: detailed balance is violated by construction, **yet the SET-side ratio is still exactly $\delta s/s$** (all $n$); the **RESET-side ratio $Y(n)$ is time-dependent** and $\ne 1$ when $L\,r\lesssim 1$ (e.g. 0.39 at $n=1$, 0.09 at $n=L$ for $L=20000$, $r=2.5\times10^{-5}$) with echoes at multiples of $L$ | high (changes what "violation" means) | SymPy ($L=3,4$) + exact chain to $L=20000$ + MC |

What survives: the qualitative picture (SET is voltage-driven, RESET current-driven; two timescales;
hysteresis from a persistent ON population) and the identification of $g$ as a two-state order
parameter. What must be replaced: the definition and the analytic prediction of $X$.

---

## 2. Setting

**Model** (Lamas 2026 Eqs. 3–4, as simulated). Each junction is $g\in\{0,1\}$. At each step an OFF
junction turns ON with probability $s=\Delta t\,P_0\,[1-e^{-\alpha(V_{\text{mem}}-V_{\text{th}})}]_+$; an ON
junction turns OFF with probability $r=\min(1,\Delta t\,P_{\text{decay}}I_{\text{mem}}^2)$.

**Assumption (A1).** For single-junction results, $V_{\text{mem}}$ and $I_{\text{mem}}$ are held fixed
(no network feedback). **(A2)** Junctions are independent when $N$ of them are pooled.

---

## 3. Derivations (correct replacements)

### 3.1 Exact single-junction statistics

With $p=P(g=1)$: $p_{n+1}=(1-r)p_n+s(1-p_n)$, stationary $p^*=s/(s+r)$, $\lambda=s+r$, so

$$C(n)=\langle\delta g_0\,\delta g_n\rangle=p^*(1-p^*)(1-\lambda)^n .\tag{1}$$

Perturbing the SET probability by $\delta s$ from $n=0$ (linearized recursion
$\delta p_{n+1}=(1-\lambda)\delta p_n+\delta s\,(1-p^*)$, $\delta p_0=0$):

$$\delta p_n=\delta s\,(1-p^*)\,\frac{1-(1-\lambda)^n}{\lambda},\qquad
\frac{\delta p_n}{\delta s}\Big/\big[C(0)-C(n)\big]=\frac1{s}. \tag{2}$$

Continuous-time limit with rates $a$ (OFF→ON), $b$ (ON→OFF) and $V$-derivatives $a_1,b_1$:
$C(t)=p^*(1-p^*)e^{-(a+b)t}$, $R(t)=[a_1(1-p^*)-b_1p^*]e^{-(a+b)t}$, and

$$\boxed{\;\frac{R(t)}{-\mathrm{d}C/\mathrm{d}t}=\frac{a_1}{a}-\frac{b_1}{b}=\frac{\mathrm{d}}{\mathrm{d}V}\ln\frac{a}{b}\;}\qquad\text{for all }t. \tag{3}$$

### 3.2 Consequence: what "FDT" can mean for this model

* **Equilibrium.** If the rates obey local detailed balance for an energy $-qVg$, $a/b=e^{\beta qV}$, then
  (3) gives $R=-\beta q\,\mathrm{d}C/\mathrm{d}t$: the standard FDT with coupling $q$ (checked symbolically).
* **AgNW junction.** SET depends on $V$; RESET (through $I^2$) does not (at fixed $I$): $b_1=0$, so
  $R/(-\dot C)=\mathrm{d}\ln s/\mathrm{d}V=\alpha/(e^{\alpha(V-V_{\text{th}})}-1)$. This is **constant in
  time**, and $\to 1/(V-V_{\text{th}})$ near threshold.
* **No reference scale.** The model contains neither $T$ nor $q$. The only well-defined dimensionful
  quantity is $1/T_{\text{eff}}^{(V)}\equiv\mathrm{d}\ln(a/b)/\mathrm{d}V$ (units 1/V, an "effective
  temperature in volts over $q$"). Any $X=\chi/\chi_{\text{FDT}}$ needs $\beta q$ from outside the
  model, and $T_{\text{eff}}=300X$ (§13.2) presupposes it. This is F1.
* **Two-state chains are always reversible in the stationary state** (no cycles; a fixed lifetime, §6.2, breaks this), so a single
  junction at fixed $(V,I)$ has no stationary currents and no genuine violation. Time-dependent
  $X(t)$ (Cugliandolo–Kurchan-type) requires collective/network feedback: $I_{\text{mem}}$ depends
  on the other junctions and on $V_{\text{in}}$ (then $b_1\neq0$), or slow hidden variables. This is F2.

*Frozen-network heuristic (not verified).* If all junction states are frozen and the linear
network scales as $V_{\text{mem}},I_{\text{mem}}\propto V_{\text{in}}$, then
$R/(-\dot C)\approx (V_{\text{mem}}/V_{\text{in}})\,\alpha/(e^{\alpha(V_{\text{mem}}-V_{\text{th}})}-1)-2/V_{\text{in}}$
for a global $\delta V_{\text{in}}$ (the $-2/V_{\text{in}}$ is $-\mathrm{d}\ln b/\mathrm{d}V_{\text{in}}$ for
$b\propto I^2$; it can make the ratio negative). It ignores avalanches (other junctions switching
change $I_{\text{mem}}$) and should be treated as a hypothesis to test in the network simulator, not a result.

### 3.3 Fokker–Planck audit (Eqs. 5–7, 10)

For $\partial_t\rho=-\partial_g(A\rho)+\partial_g^2(B\rho)$ on $[0,1]$:

$$\frac{\mathrm{d}\langle g\rangle}{\mathrm{d}t}=\langle A\rangle-\big[gJ\big]_0^1-\big[B\rho\big]_0^1,\qquad J=A\rho-\partial_g(B\rho). \tag{4}$$

With no-flux boundaries $J=0$, so the boundary term is $-[B\rho]_0^1$, not $\partial_gB|_0^1$ as in Eq. 10
(the identity $\mathrm{d}/\mathrm{d}g[-gA\rho+g(B\rho)'-B\rho]=g\,\partial_t\rho-A\rho$ was verified symbolically
after an initial sign error in my own hand derivation was caught by the check).

For $N$ independent junctions the fraction $g=n_{\text{ON}}/N$ has per-step moments
$E[\Delta g]=s(1-g)-rg$ (Eq. 6, correct) and

$$\operatorname{Var}[\Delta g]=\frac{(1-g)\,s(1-s)+g\,r(1-r)}{N}\;\approx\;\frac{s(1-g)+rg}{N}. \tag{5}$$

Eq. 7 ($2B=s(1-g)^2+rg^2$) has the wrong $g$-dependence and no $1/N$; it coincides with (5) only at
$g\in\{0,1\}$ with $N=1$. For a single junction ($N=1$, $g\in\{0,1\}$) the correct object is the master
equation $\dot p=a(1-p)-bp$, not a Fokker–Planck equation.

---

## 4. Verification

### 4.1 Symbolic (SymPy 1.14, 33/33; see `results_symbolic.txt`)

| group | checks | result |
|---|---|---|
| OU reference for Eq. 8 | Lyapunov variance $T/k$; $R'=-\mu kR$; $R=-\beta\dot C$ (PASS); $R=+\beta\dot C$ refuted (*numeric*); step form | PASS |
| Two-state exact | $R'=-\lambda R$, $C'=-\lambda C$; $\int_0^\infty R=\mathrm{d}p^*/\mathrm{d}V$ (independent static check); detailed-balance FDT; $b_1=0$ ratio $=a_1/a$; general ratio $=\mathrm{d}\ln(a/b)/\mathrm{d}V$ | PASS |
| SET rate | $\mathrm{d}\ln s/\mathrm{d}V=\alpha/(e^{\alpha(V-V_{\text{th}})}-1)$; limit $\to1/(V-V_{\text{th}})$ | PASS |
| Discrete time | $p^*$; step-response recursion; $C(n+1)=(1-\lambda)C(n)$; ratio $\delta s/s$ | PASS |
| Fokker–Planck | Eq. 10 identity (abstract functions); Bernoulli variance; Eq. 7 differs from KM (*numeric* refutation), coincides only at $g=0,1$ | PASS |
| Dimensions (`sympy.physics.units`) | Eq. 2 with the printed unit of $P_{\text{decay}}$ is not dimensionless (s²); with s⁻¹A⁻² it is; Eq. 14 $\to$ V/A²; Eq. 13 $\to$ 1/V; Eq. 12 $\to$ mismatch; Eq. 16 $\to$ A·V $\neq$ J | PASS (as refutations) |
| Loop area | linear resistor: $\oint I\,\mathrm{d}V=0$ but $\int IV\,\mathrm{d}t=\pi V_0^2/(R\omega)$ | PASS |
| Percolation | $N_c=5.637\,A/L^2=1150.5$ | PASS (value quoted from memory) |

### 4.2 Monte Carlo vs. `update_dynamics` (`results_mc.txt`)

Fixed $V_{\text{mem}}=0.30$ V, $I_{\text{mem}}=2.5$ mA, $V_{\text{th}}=0.1$ V, $\Delta tP_0=0.4$,
$\Delta tP_{\text{decay}}=0.01$ mA⁻²: $s=0.0725$, $r=0.0625$, $p^*=0.537$.

| quantity | MC | exact |
|---|---|---|
| $E[\Delta g]$ ($N=50$, $g=0.4$) | $0.01854\pm0.00008$ | $0.01850$ |
| $\operatorname{Var}[\Delta g]$ | $1.284\times10^{-3}$ | KM: $1.276\times10^{-3}$; doc Eq. 7: $3.61\times10^{-2}$ |
| $C(n)$, $n=0\ldots40$ | max dev. $3.4\times10^{-4}$ | statistical $\approx1.8\times10^{-4}$ |
| $\delta p_n$ (common random numbers, $\delta V=0.01$ V) | max dev. from the exact *finite-step* solution $1.4\times10^{-4}$ (from the linear formula $2.9\times10^{-4}$; the gap is $\delta s/\lambda=2.4\%$ nonlinearity) | |
| $\chi_{\text{step}}/[C(0)-C(n)]$ | $4.36$–$4.48$ V⁻¹ for $n=3\ldots40$ | constant $4.49$ V⁻¹ (analytic derivative $4.52$) |

The measured ratio is flat in $n$, as (2)–(3) predict, and not $\beta q$-dependent. The response
noise is $\sim7\times10^{-5}$, so deviations of $\sim2\sigma$ at some lags are consistent with statistics.
This also validates `update_dynamics` against the exact Markov-chain results.

---

## 5. Consequences for the project

1. **Redefine $X$.** Use the exactly derived object (3): $\Theta(V)\equiv\mathrm{d}\ln(a/b)/\mathrm{d}V$
   for junction-level statements, and for the network define $X$ operationally as the slope in a
   parametric plot of integrated response vs. $C(0)-C(t)$ (equivalently $R(t)/[-\dot C(t)]$) with the
   coupling $q$ stated explicitly, or report $1/T^{(V)}_{\text{eff}}$ in V⁻¹ without a reference $T$.
   Decide the observable first: per-junction $g$ vs. network $g_{\text{eff}}=I/V_{\text{in}}$ (both
   need their own conjugate coupling).
2. **Use the two-state results as the unit test of the future $\chi/C$ pipeline.** The
   common-random-numbers twin-run protocol in `verify_mc.py` reproduces the exact response with
   $10^6$ independent junctions; its network analogue will decorrelate after avalanches, so it is
   only trustworthy at short lags and needs an explicit check.
3. **Where a real violation can appear:** network feedback ($b_1\neq0$ through $I_{\text{mem}}(V_{\text{in}})$),
   collective avalanches, and slow hidden variables. These need the network simulator, not the mean-field FP.
4. **Drop Eqs. 13–14** as analytic predictions (dimensionally inconsistent, no derivation). The
   dimensionless quantities that do appear are $r/s$ and $\mathrm{d}\ln(a/b)/\mathrm{d}V$.
5. **Fix in the docs/code:** unit of $P_{\text{decay}}$ (s⁻¹A⁻²) in §13.1; Eq. 8 sign; Eq. 7; Eq. 10;
   Eq. 16 wording and the `hysteresis_loop_area` docstring (energy dissipated per cycle is
   $\oint IV\,\mathrm{d}t$, all of which is Joule heat here since the model stores no energy;
   $\oint I\,\mathrm{d}V$ measures branch asymmetry of $G$ weighted by $V$, useful as a memory-window
   metric but not as an entropy or energy).
6. **Optimization framework:** its central function `compute_X_analytical` implements Eq. 14, so it
   is unsupported until $X$ is redefined and validated.

---

## 6. Addendum (2026-09-20): spontaneous relaxation

**Context.** A third switching mechanism, recalled from a talk by Lamas (spontaneous ON→OFF relaxation after some
time; not in the paper, details unconfirmed), reproduces the negative half-cycle of the paper's Fig. 4 when
added to the model. It is implemented in two readings: `SimParams.p_relax` (per-step probability $\epsilon$;
memoryless) and `SimParams.lifetime_steps` (deterministic lifetime $L$ after SET). This section extends §3 to both.
Same setting as §2 (fixed $V_{\text{mem}}$, $I_{\text{mem}}$; independent junctions).

### 6.1 Exponential relaxation

An ON junction turns OFF per step with $b=1-(1-r)(1-\epsilon)$ (independent channels). §3.1 holds with $b$ in place of
$r$: $p^*=s/(s+b)$, $\lambda=s+b$, $C(n)=p^*(1-p^*)(1-\lambda)^n$. For perturbations $\delta s$, $\delta r$
($\delta b=(1-\epsilon)\delta r$):

$$\frac{\chi_{\text{step}}(n)}{C(0)-C(n)}=\frac{\delta s}{s}-\frac{\delta b}{b}\quad\text{for all }n. \tag{6}$$

* SET-side ($\delta r=0$): independent of $\epsilon$ and $r$.
* RESET-side: $\delta b/b\approx\dfrac{r}{r+\epsilon}\dfrac{\delta r}{r}$, so relaxation **dilutes** the current-driven
  contribution. With the network heuristic $\delta r/r=2\,\delta I/I$ of §3.2 this is $2\frac{r}{r+\epsilon}\delta I/I$: for
  $\epsilon\gg r$ the (possibly negative) RESET term disappears and the ratio tends to $\mathrm d\ln s/\mathrm dV$.
* Still a two-state chain, hence reversible in the stationary state; the ratio is constant in time (as in §3).

### 6.2 Fixed lifetime (alternating renewal process)

ON sojourn $D=\min(K,L)$, $K\sim\mathrm{Geom}(r)$; OFF sojourn $\sim\mathrm{Geom}(s)$. Exact description by the extended
Markov chain OFF, (ON, age $j=0..L-1$): $E[D]=\sum_{k=1}^{L}(1-r)^{k-1}=[1-(1-r)^L]/r$,
$\pi_{\text{OFF}}=1/(1+sE[D])$, $\pi_{\text{ON},j}=\pi_{\text{OFF}}\,s(1-r)^j$, $p^*=E[D]/(E[D]+1/s)\to s/(s+r)$ as $L\to\infty$.

1. **Detailed balance is violated by construction** (age never decreases: a directed cycle OFF→ON,0→…→ON,$L$−1→OFF with no reverse
   transitions). The single junction is therefore *not* a reversible two-state chain, unlike §3.
2. **SET-side response is still exactly Markov-like:** for a perturbation $\delta s$ of the SET probability,
$$X(n)\equiv\frac{s\,\delta p_n}{\delta s\,[C(0)-C(n)]}=1\quad\text{for all }n,L. \tag{7}$$
   Verified symbolically for $L=3$, $n=1..4$; exactly (extended chain) for $L$ up to 20000, $n$ up to $6L$; and by
   Monte Carlo. Interpretation (argument, not proved for general $L$): the perturbation only alters the exit from the
   OFF state, whose sojourn is memoryless.
3. **RESET-side response deviates.** For a perturbation $\delta r$ of the current-driven RESET probability define
   $Y(n)=-r\,\delta p_n/(\delta r\,[C(0)-C(n)])$ ($=1$ for a two-state Markov junction). At $n=1$:
$$Y(1)=\frac{r\,(p^*-\pi_{L-1})}{r\,(p^*-\pi_{L-1})+\pi_{L-1}}, \tag{8}$$
   the fraction of the ON-exit flux caused by RESET rather than by expiry (verified from explicit $L=4$ matrices). $Y(n)$ is
   **time-dependent** with echoes at multiples of $L$ (figure `relaxation_ratio.png`). Exact values ($s=0.01$, $r=2.5\times10^{-5}$,
   i.e. $I_{\text{mem}}\approx0.05$ mA): $Y(1)=0.002,\,0.025,\,0.117,\,0.393$ for $L=100,\,1000,\,5000,\,20000$;
   for $L=20000$, $Y$ falls to 0.09 at $n=L$ and jumps back to 0.32 just after; $Y\equiv1$ when RESET dominates ($r=0.0625$).
   The deviation is controlled by $L\,r$.
4. **Network-level combination.** A global $\delta V_{\text{in}}$ perturbs both channels ($\delta s$ through $V_{\text{mem}}$, $\delta r$
   through $I_{\text{mem}}$), and by linearity
$$\frac{R(n)}{-\Delta C(n)}\Big|_{\text{step}}=\frac{\delta s}{s}X(n)-\frac{\delta r}{r}\,Y(n)=\frac{\delta s}{s}-\frac{\delta r}{r}Y(n),\tag{9}$$
   so the **time-dependent FDT violation of a fixed-lifetime junction lives entirely in $Y(n)-1$ times
   $\mathrm d\ln r/\mathrm dV$**. With the frozen-network heuristic of §3.2 ($\delta r/r\approx2\,\delta I/I$) this predicts a
   ratio that changes with lag and shows echoes at $L$ — a *testable prediction* for the network simulator
   (unverified at network level; needs the distribution of $r$ over ON junctions, not yet measured).

### 6.3 Verification (`verify_symbolic.py`: 51/51; `verify_relaxation.py`)

| item | check | result |
|---|---|---|
| exp.: $p^*$, step-response recursion, ratio (6) with $\delta r=0$ and $\ne0$ | SymPy | PASS |
| renewal: $E[D]$, stationary distribution ($L=4$), $p^*$, $L\to\infty$ limit | SymPy | PASS |
| renewal: detailed balance violated; $P(a_2\to a_1)=0$ | SymPy | PASS |
| renewal: $X(n)=1$, $L=3$, $n=1..4$ | SymPy, explicit matrices | PASS |
| renewal: $Y(1)$ eq. (8), $L=4$ | SymPy, explicit matrices | PASS |
| renewal: extended-chain recursion vs dense matrices ($L=8$) | numeric | agree to $10^{-16}$ |
| renewal, $L=8$: $C(n)$, $\delta p_n$ vs. `update_dynamics_lifetime` MC ($M=2\times10^6$, common random numbers) | MC | max dev. $5\times10^{-4}$ / $4\times10^{-4}$ (noise level) |
| renewal, $L=8$: $Y(n)$ vs. MC with perturbed $I_{\text{mem}}$ | MC | 0.351/0.363 (n=1) … 0.207/0.211 (n=40), dev. $\lesssim3\times10^{-4}$ in $\delta p$ |
| exponential $\epsilon=0.02$: $C(n)$ and constant ratio vs. `p_relax` MC | MC | ratio 0.975–0.993 (statistical ≈1–3 %) |

**Not verified:** (7) for general $L$ by proof (numerics + $L=3$ symbolic only); eq. (9) at network level; the size of the
effect in the fitted network regime (needs typical $r=0.01\,(I_{\text{mem}}/\text{mA})^2$ of ON junctions).

### 6.4 Consequences

* The *choice* between the two readings of the third mechanism matters for the physics, not only for the loops: with
  exponential relaxation the model stays a two-state Markov junction with a constant response/correlation ratio; with a
  fixed lifetime it acquires an intrinsic, lag-dependent violation on the RESET channel. One more reason to ask Lamas.
* For the χ/X measurement design: report the SET-side and RESET-side responses separately if possible, and test eq. (9)
  by comparing the measured lag-dependence with $Y(n)$.

---

## 7. Nomenclature (per-step model units; $V$ in volts, $I$ in amperes)

| symbol | meaning | units |
|---|---|---|
| $g\in\{0,1\}$ | junction state (OFF/ON) | – |
| $s$, $r$ | per-step SET / RESET probability | – |
| $a$, $b$ | SET / RESET rates ($s/\Delta t$, $r/\Delta t$) | s⁻¹ |
| $a_1,b_1$ | $\mathrm{d}a/\mathrm{d}V$, $\mathrm{d}b/\mathrm{d}V$ | s⁻¹V⁻¹ |
| $p$, $p^*$ | $P(g=1)$ and its stationary value | – |
| $\lambda$ | relaxation rate $a+b$ (or $s+r$ per step) | s⁻¹ (per step) |
| $C$ | connected autocorrelation of $g$ | – |
| $R$, $\chi_{\text{step}}$ | impulse / step response of $\langle g\rangle$ to $V$ | V⁻¹s⁻¹ / V⁻¹ |
| $\Theta=\mathrm{d}\ln(a/b)/\mathrm{d}V$ | exact response-to-correlation ratio | V⁻¹ |
| $q$ | coupling of $V$ to $g$ ($-qVg$), *not in the model* | C |
| $\beta=1/k_BT$ | inverse bath temperature, *not in the model* | J⁻¹ |
| $P_0$, $P_{\text{decay}}$ | rates (per second) with $\Delta t$ applied inside $s,r$ | s⁻¹; s⁻¹A⁻² |
| $A,B,\rho,J$ | FP drift, diffusion, density, flux | – |
| $N$ | pooled independent junctions | – |
| $\epsilon$ | spontaneous relaxation probability per step (`p_relax`) | – |
| $b$ | total ON→OFF probability $1-(1-r)(1-\epsilon)$ | – |
| $L$ | fixed lifetime after SET (`lifetime_steps`) | steps |
| $D,K$ | ON sojourn $\min(K,L)$; steps until RESET, $K\sim\mathrm{Geom}(r)$ | steps |
| $X(n)$, $Y(n)$ | SET-side / RESET-side response-to-correlation ratios (=1 for Markov) | – |
