"""Canonical network configuration (the framework's working model) and experiment settings.

The network - wire geometry, dual-node circuit, binary switching dynamics - comes from agnw_sim
(AgNW_FDT_Memristor_Framework) exactly as its FDTest drivers build it (scripts/fdtest_common.build_setup, working
model decided 2026-09-20). This project designs the input encoding, the readout and the electrode layouts, and adds
alternative junction models (mean-field ablation, continuous filament) as explicit, separately labelled variants.

Run names format p_relax / p_relax_max / sigma with one significant digit and window centres with two decimals;
tests/test_drive.py checks that the defined experiments never collide.
"""
import dataclasses
import os
import subprocess
from dataclasses import dataclass

import numpy as np

import agnw_sim
from agnw_sim.dynamics import SimParams
from agnw_sim.geometry import Network, build_network
from agnw_sim.presets import lamas2026_fig4

# ---- framework working model
N_WIRES = 2000
AREA = 1e-6            # m^2
LENGTH = 70e-6         # m
DT = 1e-3              # s
P_RELAX = 1e-4         # per step, exponential relaxation (framework working value)
SOLVER = "cached"


def network_params(p_relax: float = P_RELAX) -> tuple[SimParams, float]:
    """(SimParams, wire_diameter) of the working model; p_relax may be overridden for sensitivity studies."""
    params, run_kwargs = lamas2026_fig4(DT)
    return dataclasses.replace(params, p_relax=p_relax), run_kwargs["wire_diameter"]


def make_network(network_seed: int, n_wires: int = N_WIRES, side: float = 0.0) -> Network:
    """The network geometry depends only on network_seed (as in build_setup). side > 0 (mm): a square of that side at
    the same wire density (N10-L), N_WIRES x side^2 wires; side = 0 is the 1 mm working model."""
    if side:
        return build_network(int(round(N_WIRES * side ** 2)), AREA * side ** 2, LENGTH, np.random.default_rng(network_seed))
    return build_network(n_wires, AREA, LENGTH, np.random.default_rng(network_seed))


def framework_commit() -> str:
    """Git commit of the agnw_sim checkout in use ("" if unavailable)."""
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              check=True, cwd=os.path.dirname(agnw_sim.__file__)).stdout.strip()
    except Exception:
        return ""


# ---- reservoir interface / experiment settings
@dataclass(frozen=True)
class RunSpec:
    """One simulation: a network, an input stream, an input encoding, and a noise replicate."""
    stream: str                 # "iid" or "sinesquare"
    hold: int                   # simulation steps per input sample
    v_center: float             # centre of the input voltage window (V)
    network_seed: int
    replicate: int = 0          # dynamics-noise / pre-drive replicate
    v_width: float = 1.0        # width of the voltage window (V)
    p_relax: float = P_RELAX
    n_samples: int = 0          # scored input samples; 0 = N_SAMPLES[stream]
    model: str = "stochastic"   # "stochastic" (framework run_simulation) or "meanfield" (NEXT_STEPS N1)
    p_relax_max: float = 0.0    # > 0: per-junction p_relax log-uniform in [p_relax, p_relax_max] (NEXT_STEPS N2)
    layout: str = ""            # "" = framework two-electrode pipeline; "split_bands" / "pads" / "two_bands" (N5)
    encoding: str = "weights"   # multi-electrode input encoding (N5): "weights" (N5a), "delays" (N5b), "direct"
    sigma: float = 0.0          # N4 filament-state noise amplitude (V s^1/2); only for model == "filament"
    washout: int = 0            # pre-drive length in steps; 0 = legacy WASHOUT_STEPS (review P0.1)
    n_virtual: int = 0          # current samples per hold block; 0 = legacy N_VIRTUAL (review P0.4)
    stride: int = 1             # N4 only: integration step = stride x DT (review P0.3); hold must be a multiple
    tau_s: float = 0.0          # N10 ("volatile"): median zero-bias relaxation time (s); 0 = VolatileParams default
    tau_spread: float = 0.0     # N10: log-normal spread of tau_s across junctions
    consolidation: bool = False # N10: slow consolidated filament on (VOLATILE_K_C) or off (pure volatile)
    v_scale: float = 0.0        # N10: voltage scale override (0 = VOLATILE_V_SCALE)
    k_max: float = 0.0          # N10: growth-rate cap override in 1/s (0 = VOLATILE_K_MAX)
    rupture: float = 0.0        # N10: rate (1/s) of random complete ruptures of the volatile filament (0 = none)
    k_c: float = 0.0            # N10-K: consolidation rate override in 1/s when consolidation is on (0 = VOLATILE_K_C)
    tau_l: float = 0.0          # N10-K: decay time of the consolidated filament in s (0 = VolatileParams default)
    forming: float = 0.0        # N10-K: seconds of constant forming drive before the pre-drive (0 = none)
    forming_v: float = 0.0      # N10-K: forming voltage on every driven electrode (V)
    p_c2: float = 0.0           # N10-K: consolidation power threshold override in W (0 = VolatileParams default 1e-6)
    side: float = 0.0           # N10-L: network side in mm at the working density (0 = the 1 mm working model)

    @property
    def washout_steps(self) -> int:
        return self.washout or WASHOUT_STEPS

    @property
    def n_virtual_eff(self) -> int:
        return self.n_virtual or N_VIRTUAL

    @property
    def length(self) -> int:
        return self.n_samples or N_SAMPLES[self.stream]

    @property
    def v_lo(self) -> float:
        return self.v_center - self.v_width / 2

    @property
    def v_hi(self) -> float:
        return self.v_center + self.v_width / 2

    @property
    def name(self) -> str:
        return (f"{self.stream}_h{self.hold}_v{_volt(self.v_center)}_w{_volt(self.v_width)}"
                f"_pr{self.p_relax:.0e}_n{self.network_seed}_r{self.replicate}"
                + (f"_s{self.n_samples}" if self.n_samples else "")
                + ("_mf" if self.model == "meanfield" else "")
                + (f"_prmax{self.p_relax_max:.0e}" if self.p_relax_max else "")
                + (f"_{self.layout}_{self.encoding}" if self.layout else "")
                + (f"_L{self.side:g}" if self.side else "")
                + ("_fil" + (f"_sig{self.sigma:.0e}" if self.sigma else "") if self.model == "filament" else "")
                + (f"_vol_ts{self.tau_s:g}_sp{self.tau_spread:g}" + ("_con" if self.consolidation else "")
                   + (f"_kc{self.k_c:g}" if self.k_c else "") + (f"_tl{self.tau_l:g}" if self.tau_l else "")
                   + (f"_pc{self.p_c2:g}" if self.p_c2 else "")
                   + (f"_form{self.forming:g}v{self.forming_v:g}" if self.forming else "")
                   + (f"_vs{self.v_scale:g}" if self.v_scale else "") + (f"_km{self.k_max:g}" if self.k_max else "")
                   + (f"_rup{self.rupture:g}" if self.rupture else "")
                   + (f"_sig{self.sigma:.0e}" if self.sigma else "") if self.model == "volatile" else "")
                + (f"_wo{self.washout}" if self.washout else "")
                + (f"_nv{self.n_virtual}" if self.n_virtual else "")
                + (f"_st{self.stride}" if self.stride != 1 else ""))


def _volt(v: float) -> str:
    """Voltage in run names: 2 decimals as always (all runs up to N10-K), more digits only when needed (N10-V windows
    of a few tens of mV), so existing names are unchanged."""
    return f"{v:.2f}" if abs(round(v, 2) - v) < 1e-12 else f"{v:.4g}"


# stream lengths (input samples) and washout (simulation steps, independent of hold: the relaxation time
# 1/p_relax = 1e4 steps sets how long the network needs to forget its all-OFF initial state)
N_SAMPLES = {"iid": 2000, "sinesquare": 1200}
WASHOUT_STEPS = 10_000
N_PROBES = 32
# Store every junction state at every input in the raw run? (2026-10-08, Juan: no unless needed.) They are ~95 % of the
# size of a raw file and feed only the oracle "junctions" readout and a few model checks; scripts that need them pass
# simulate_multi(..., save_states=True). Runs up to 2026-10-08 with states: TOSHIBA_ntfs archive (docs/NEXT_STEPS.md).
SAVE_JUNCTION_STATES = False
N_VIRTUAL = 10                  # current samples per hold block (at most `hold`)
SPLIT = (0.6, 0.2, 0.2)         # train / validation / test, in time order

E1_HOLDS = (10, 30, 100)
E1_CENTERS = (1.25, 1.75, 2.25)
E1_SEEDS = (1, 2, 3)
E1_REPLICATES = (0, 1)
E2_P_RELAX = (0.0, 1e-4, 1e-3, 1e-2)
E3_HOLDS = (300, 1000)          # timescale matching: fewer, longer input samples
E3_N_SAMPLES = 800


def e1_specs() -> list[RunSpec]:
    specs = []
    for hold in E1_HOLDS:
        for vc in E1_CENTERS:
            for seed in E1_SEEDS:
                for rep in E1_REPLICATES:
                    specs.append(RunSpec("iid", hold, vc, seed, rep))
                specs.append(RunSpec("sinesquare", hold, vc, seed, 0))
    return specs


def e2_specs(hold: int, v_center: float) -> list[RunSpec]:
    specs = []
    for pr in E2_P_RELAX:
        for seed in E1_SEEDS:
            for rep in E1_REPLICATES:
                specs.append(RunSpec("iid", hold, v_center, seed, rep, p_relax=pr))
            specs.append(RunSpec("sinesquare", hold, v_center, seed, 0, p_relax=pr))
    return specs


def e3_specs(v_center: float) -> list[RunSpec]:
    """Long holds (i.i.d. stream only, E3_N_SAMPLES samples) at one voltage window."""
    return [RunSpec("iid", hold, v_center, seed, rep, n_samples=E3_N_SAMPLES)
            for hold in E3_HOLDS for seed in E1_SEEDS for rep in E1_REPLICATES]


# N1 mean-field ablation: same holds/seeds/replicates as E1, voltage windows around the mean-field onset (~3-4 V,
# versus ~1.5 V for the stochastic network; see docs/NEXT_STEPS.md)
M1_CENTERS = (3.5, 4.5, 5.5)


def m1_specs() -> list[RunSpec]:
    specs = []
    for hold in E1_HOLDS:
        for vc in M1_CENTERS:
            for seed in E1_SEEDS:
                for rep in E1_REPLICATES:
                    specs.append(RunSpec("iid", hold, vc, seed, rep, model="meanfield"))
                specs.append(RunSpec("sinesquare", hold, vc, seed, 0, model="meanfield"))
    return specs


# N2 heterogeneous forgetting (mean-field): per-junction p_relax log-uniform in [P_RELAX, p_max], with homogeneous
# controls at the same upper rates. Window and holds fixed from the best M1 region (5.5 V, holds 10/30).
N2_P_MAX = (1e-3, 1e-2, 1e-1)
N2_HOMOGENEOUS = (1e-3, 1e-2)
N2_HOLDS = (10, 30)
N2_CENTER = 5.5


def n2_specs() -> list[RunSpec]:
    conds = [dict(p_relax_max=pm) for pm in N2_P_MAX] + [dict(p_relax=pr) for pr in N2_HOMOGENEOUS]
    specs = []
    for hold in N2_HOLDS:
        for cond in conds:
            for seed in E1_SEEDS:
                for rep in E1_REPLICATES:
                    specs.append(RunSpec("iid", hold, N2_CENTER, seed, rep, model="meanfield", **cond))
                specs.append(RunSpec("sinesquare", hold, N2_CENTER, seed, 0, model="meanfield", **cond))
    return specs


def junction_p_relax(spec: RunSpec, n_junctions: int):
    """Per-junction relaxation probabilities of a spec: a scalar (homogeneous) or a fixed log-uniform draw that
    depends only on the network seed (quenched junction-to-junction variability)."""
    if not spec.p_relax_max:
        return spec.p_relax
    if not 0 < spec.p_relax < spec.p_relax_max:
        raise ValueError("heterogeneous p_relax needs 0 < p_relax < p_relax_max")
    rng = np.random.default_rng(50_000 + spec.network_seed)
    return np.exp(rng.uniform(np.log(spec.p_relax), np.log(spec.p_relax_max), n_junctions))


# N7a noise-averaging curve (stochastic framework model): N7A_REPLICATES noise replicates of one E1 point
N7A_REPLICATES = tuple(range(8))
N7A_HOLD, N7A_CENTER = 30, 1.75


def n7a_specs() -> list[RunSpec]:
    return [RunSpec("iid", N7A_HOLD, N7A_CENTER, seed, rep) for seed in E1_SEEDS for rep in N7A_REPLICATES]


# N7b operating-point control: the STOCHASTIC framework model at the mean-field (M1) windows, so that the M1 gains can
# be attributed to noise removal rather than to the higher voltage
N7B_HOLDS = (10, 30)


def n7b_specs() -> list[RunSpec]:
    specs = []
    for hold in N7B_HOLDS:
        for vc in M1_CENTERS:
            for seed in E1_SEEDS:
                for rep in E1_REPLICATES:
                    specs.append(RunSpec("iid", hold, vc, seed, rep))
                specs.append(RunSpec("sinesquare", hold, vc, seed, 0))
    return specs


def zlib_seed(name: str) -> int:
    """Dynamics seed derived from a run name (same rule as drive.dynamics_seed)."""
    import zlib
    return zlib.crc32(name.encode())


# N5a multi-electrode drive, random input weights (canonical encoding). Windows (width 1 V) from the quasi-static
# pilot (scripts/multi_pilot.py): split bands have the two-band onset (~1.5 V), pads conduct from ~0.3-0.5 V.
N5A_CENTERS = {"split_bands": (1.25, 1.75, 2.25), "pads": (0.5, 0.75, 1.0)}
N5A_HOLDS = (10, 30)


def n5a_specs() -> list[RunSpec]:
    specs = []
    for layout, centers in N5A_CENTERS.items():
        for hold in N5A_HOLDS:
            for vc in centers:
                for seed in E1_SEEDS:
                    for rep in E1_REPLICATES:
                        specs.append(RunSpec("iid", hold, vc, seed, rep, layout=layout, encoding="weights"))
                    specs.append(RunSpec("sinesquare", hold, vc, seed, 0, layout=layout, encoding="weights"))
    return specs


# N4 continuous-filament junctions (docs/N4_DESIGN.md). N4_PARAMS holds the dynamic parameters calibrated on Lamas
# Fig. 4, N = 2000 (scripts/n4_calibrate.py; results/n4_calibration.json; 2026-09-27): loss 0.0085, validated on
# 4 network seeds and, unfitted, on N = 3000 (design section 5).
N4_PARAMS = {"V_set": 0.06, "b": 0.02, "lambda_crit": 0.01, "unipolar": False}


def filament_params(spec: RunSpec | None = None, **override):
    """FilamentParams of the N4 model: calibrated dynamics (N4_PARAMS, overridable), V_reset = reset_ratio x V_set
    (default 1/2, as in the source; N8 varies it), lambda_max = 1.5 lambda_crit, the framework's G_on / G_off and
    dt, and the spec's noise amplitude."""
    from .filament import FilamentParams
    fw, _ = network_params()
    d = {**N4_PARAMS, **override}
    stride = spec.stride if spec is not None else int(d.get("stride", 1))
    return FilamentParams(V_set=d["V_set"], V_reset=d["V_set"] * d.get("reset_ratio", 0.5), b=d["b"],
                          lambda_crit=d["lambda_crit"], lambda_max=1.5 * d["lambda_crit"], G_on=fw.G_on, G_off=fw.G_off, dt=stride * DT,
                          sigma=spec.sigma if spec is not None else d.get("sigma", 0.0),
                          unipolar=bool(d.get("unipolar", False)))


# N4-1: the E1 protocol on the calibrated continuous-filament model (two bands, E1 input via encoding "direct"),
# windows around the N4 onset (~2 V, calibration section 5.1), noise-free.
N4_1_CENTERS = (1.75, 2.25, 2.75)


def n4_1_specs() -> list[RunSpec]:
    specs = []
    for hold in E1_HOLDS:
        for vc in N4_1_CENTERS:
            for seed in E1_SEEDS:
                for rep in E1_REPLICATES:
                    specs.append(RunSpec("iid", hold, vc, seed, rep, layout="two_bands", encoding="direct",
                                         model="filament"))
                specs.append(RunSpec("sinesquare", hold, vc, seed, 0, layout="two_bands", encoding="direct",
                                     model="filament"))
    return specs


# N10: volatile two-time-scale junction (docs/N10_DESIGN.md). Consolidation rate when RunSpec.consolidation is on
# (provisional: a 10 s strong drive consolidates noticeably; tuned in scripts/n10_phenomenology.py).
VOLATILE_K_C = 0.1
# Voltage-axis scale from Milano's network-level fit to one junction (VolatileParams.v_scale); calibrated so that the
# network onset is ~2 V, the measured activation range (scripts/n10_phenomenology.py vscale: onset 3.5 / 2.2 / 1.4 V
# at v_scale 3 / 5 / 8, nearly independent of tau_s; per-junction threshold 0.49 / 5 ≈ 0.1 V, like the framework's V_th).
VOLATILE_V_SCALE = 5.0
# Cap on the potentiation rate (switching time >= 10 ms). Milano's k_p reaches 1e3-1e9 1/s above threshold; with
# k dt >> 1 the exact update jumps x to its frozen-voltage equilibrium and the junction-circuit feedback chatters, so
# results depended on the step (22 % in the junction states between 1 and 2 ms). With k_max = 100 1/s, 2 ms agrees
# with 1 ms within 0.2 % (current) / 1.6 % (states); 10 ms does not (scripts/n10_kmax_check.py).
VOLATILE_K_MAX = 100.0


def volatile_params(spec: RunSpec | None = None, **override):
    """VolatileParams of an N10 spec: Milano 2022 rates, the spec's tau_s / tau_spread / consolidation / sigma,
    the framework's G_on / G_off, dt = stride x DT. Keyword overrides go straight to VolatileParams."""
    from .volatile import VolatileParams
    fw, _ = network_params()
    kw = {"G_on": fw.G_on, "G_off": fw.G_off, "v_scale": VOLATILE_V_SCALE, "k_max": VOLATILE_K_MAX}
    if spec is not None:
        kw.update(dt=spec.stride * DT, sigma=spec.sigma, tau_spread=spec.tau_spread,
                  k_c=(spec.k_c or VOLATILE_K_C) if spec.consolidation else 0.0)
        if spec.tau_l:
            kw["tau_l"] = spec.tau_l
        if spec.p_c2:
            kw["P_c2"] = spec.p_c2
        if spec.tau_s:
            kw["tau_s"] = spec.tau_s
        if spec.v_scale:
            kw["v_scale"] = spec.v_scale
        if spec.k_max:
            kw["k_max"] = spec.k_max
        if spec.rupture:
            kw["rupture_rate"] = spec.rupture
    kw.update(override)
    return VolatileParams(**kw)


def model_washout_steps(model: str, p_relax: float = P_RELAX, tau_s: float = 0.0, tau_spread: float = 0.0) -> int:
    """Pre-drive long enough to forget the all-OFF start (review P0.1): 3 x the slowest forgetting time.
    Binary / mean-field: 3 / p_relax steps (at least 30 s). Filament: 3 x the time to dissolve from lambda_max at
    zero voltage, lambda_max / (b V_reset)."""
    if model == "volatile":                 # 3 relaxation times of the slow tail (+2 sd of the spread), >= 3 s
        from .volatile import VolatileParams
        t = (tau_s or VolatileParams.tau_s) * np.exp(2 * tau_spread)
        return int(round(max(3 * t, 3.0) / DT))
    if model == "filament":
        d = N4_PARAMS
        t_dissolve = 1.5 * d["lambda_crit"] / (d["b"] * d["V_set"] / 2)          # s
        return int(round(3 * t_dissolve / DT))
    return int(max(30_000, 3 / p_relax if p_relax else 30_000))


# E4 (review 2026-09-27, decisive plan): time-scale-matched inputs for the binary framework model and the calibrated
# N4, dense current readout, model-dependent washout, 800 samples per stream.
E4_HOLDS = {"stochastic": (300, 1000, 3000), "filament": (300, 1000, 3000)}   # N4 10 s deferred (cost at stride 2)
E4_CENTERS = {"stochastic": (1.75, 2.25), "filament": (2.25, 2.75)}
E4_N_SAMPLES = 800
E4_N_VIRTUAL = 50
E4_STRIDE = 2                   # N4 step 2 ms: Fig. 4 metrics within 1 % of 1 ms; 5 and 10 ms fail (5-12 %, the
                                # circuit must be re-solved during SET avalanches; scripts/n4_stride_check.py,
                                # scripts/n4_substep_check.py)


def e4_specs(models=("stochastic", "filament")) -> list[RunSpec]:
    specs = []
    for model in models:
        wo = model_washout_steps(model)
        extra = (dict(layout="two_bands", encoding="direct", model="filament", stride=E4_STRIDE)
                 if model == "filament" else {})
        for hold in E4_HOLDS[model]:
            wo_m = -(-wo // hold) * hold                     # whole number of hold blocks
            for vc in E4_CENTERS[model]:
                for seed in E1_SEEDS:
                    for rep in E1_REPLICATES:
                        specs.append(RunSpec("iid", hold, vc, seed, rep, n_samples=E4_N_SAMPLES, washout=wo_m,
                                             n_virtual=E4_N_VIRTUAL, **extra))
                    specs.append(RunSpec("sinesquare", hold, vc, seed, 0, n_samples=E4_N_SAMPLES, washout=wo_m,
                                         n_virtual=E4_N_VIRTUAL, **extra))
    return specs


# N10-1 (docs/N10_DESIGN.md section 4, step 5): the E4 protocol on the pure-volatile N10 junction (no consolidation,
# Juan 2026-09-27), for short-term relaxation times tau_s spanning the measured range, holds tau_s / 3 and tau_s,
# homogeneous and log-normally spread tau_s, two windows above the ~2.2 V onset.
N10_TAUS = (0.03, 0.3, 3.0)
N10_SPREADS = (0.0, 1.0)
N10_CENTERS = (2.25, 2.75)
N10_STRIDE = 2                  # 2 ms (scripts/n10_kmax_check.py, n10_stride_check.py)
N10_HOLDS = {0.03: (10, 30), 0.3: (100, 300), 3.0: (1000, 3000)}   # tau_s / 3 and tau_s (3 s block on ultron)


def n10_1_specs(consolidation: bool = False) -> list[RunSpec]:
    specs = []
    for tau in N10_TAUS:
        for hold in N10_HOLDS[tau]:
            stride = N10_STRIDE if hold % N10_STRIDE == 0 else 1
            for spread in N10_SPREADS:
                wo = model_washout_steps("volatile", tau_s=tau, tau_spread=spread)
                wo = -(-wo // hold) * hold
                for vc in N10_CENTERS:
                    common = dict(n_samples=E4_N_SAMPLES, washout=wo, n_virtual=E4_N_VIRTUAL, layout="two_bands",
                                  encoding="direct", model="volatile", tau_s=tau, tau_spread=spread,
                                  consolidation=consolidation, stride=stride)
                    for seed in E1_SEEDS:
                        for rep in E1_REPLICATES:
                            specs.append(RunSpec("iid", hold, vc, seed, rep, **common))
                        specs.append(RunSpec("sinesquare", hold, vc, seed, 0, **common))
    return specs


# N10-M (memory study, 2026-09-29): N10 junction on the 4x4 pads (8 driven + 8 grounded), 16 electrode currents,
# E4 protocol, 3 networks x 2 replicates + sine/square, the two pad windows of N10-E (onset 0.36 V). Reference:
# N10-E (tau_s 0.3 s, spread 1, hold 100 ms, random input weights). Blocks change one thing each.
N10M_WINDOWS = ((0.38, 0.16), (0.45, 0.16))
N10M_BLOCKS = {
    "tau1": [dict(tau_s=1.0, tau_spread=1.0, hold=330, encoding="weights")],
    "tau3": [dict(tau_s=3.0, tau_spread=1.0, hold=1000, encoding="weights")],
    "spread": [dict(tau_s=0.3, tau_spread=sp, hold=100, encoding="weights") for sp in (0.0, 2.0)],
    "delays": [dict(tau_s=0.3, tau_spread=1.0, hold=100, encoding="delays")],
    "noise": [dict(tau_s=0.3, tau_spread=1.0, hold=100, encoding="weights", sigma=sg) for sg in (0.03, 0.1, 0.3)],
}


def n10m_specs(blocks=tuple(N10M_BLOCKS)) -> list[RunSpec]:
    specs = []
    for block in blocks:
        for cfg in N10M_BLOCKS[block]:
            cfg = dict(cfg)
            hold = cfg.pop("hold")
            wo = model_washout_steps("volatile", tau_s=cfg["tau_s"], tau_spread=cfg["tau_spread"])
            common = dict(n_samples=E4_N_SAMPLES, washout=-(-wo // hold) * hold, n_virtual=E4_N_VIRTUAL,
                          layout="pads", model="volatile", stride=N10_STRIDE, **cfg)
            for vc, w in N10M_WINDOWS:
                for seed in E1_SEEDS:
                    for rep in E1_REPLICATES:
                        specs.append(RunSpec("iid", hold, vc, seed, rep, v_width=w, **common))
                    specs.append(RunSpec("sinesquare", hold, vc, seed, 0, v_width=w, **common))
    return specs
