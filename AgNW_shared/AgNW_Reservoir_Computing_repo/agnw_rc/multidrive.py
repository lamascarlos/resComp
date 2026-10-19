"""Multi-electrode drive and readout (NEXT_STEPS N5).

Time loop on the multi-terminal circuit (agnw_rc.multicircuit) with the framework's switching dynamics
(agnw_sim.dynamics.update_dynamics, per step exactly as in agnw_sim.simulate.run_simulation: solve the circuit with
the current states, then update with V_mem = |dV| and I_mem = G dV) or the mean-field update (agnw_rc.meanfield).

Encodings of the scalar input u in [0, 1] onto the driven electrodes (grounded ones stay at 0 V):
  "weights" (N5a): V_e(t) = V_c + (v_width / 2) w_e (2u(t) - 1), w_e ~ U(-1, 1) fixed per network seed;
  "delays"  (N5b): V_e(t) = V_c + (v_width / 2) (2u(t - d_e) - 1), d_e = 0, 1, ..., n_in - 1 input samples;
  "direct":        V_e(t) = V_c + (v_width / 2) (2u(t) - 1) on every driven electrode (two bands: the E1 input).
Junction models: spec.model "stochastic" (framework), "meanfield" (N1), "filament" (N4, agnw_rc.filament),
"volatile" (N10, agnw_rc.volatile; state (x, y) per junction, stored as "junction_x" / "junction_y").
N10-K: spec.forming > 0 applies a constant forming drive (spec.forming_v on every driven electrode, spec.forming s)
before the pre-drive, so that consolidation (spec.consolidation, spec.k_c) can build a stable backbone; the state after
forming is returned as "formed_x" / "formed_y".
Readout, recorded per input sample:
  "current": the currents of ALL electrodes at the virtual-node instants of the hold block (n_virtual x n_electrodes);
  "probe_v": 32 voltage probes at the block's last step;
  junction states at the block's last step (the state that produced that step's voltages).
"""
import time

import numpy as np

from agnw_sim.dynamics import update_dynamics

from . import config as C
from .drive import input_stream, predrive_stream, virtual_offsets
from .electrodes import make_layout
from .filament import conductance as filament_conductance
from .filament import step_lambda
from .meanfield import meanfield_step
from .volatile import conductance as volatile_conductance
from .volatile import junction_kd0, step_state
from .multicircuit import MultiTerminalCircuit


def input_weights(spec: C.RunSpec, n_in: int) -> np.ndarray:
    return np.random.default_rng(60_000 + spec.network_seed).uniform(-1.0, 1.0, n_in)


def electrode_voltages(u: np.ndarray, spec: C.RunSpec, n_in: int, u_before: np.ndarray | None = None) -> np.ndarray:
    """(len(u), n_in) voltages of the driven electrodes for input samples u. For "delays", u_before supplies the
    samples preceding u (the pre-drive), so delayed copies are defined from the first sample."""
    x = 2 * np.asarray(u, dtype=float) - 1                              # in [-1, 1]
    half = spec.v_width / 2
    if spec.encoding == "weights":
        return spec.v_center + half * x[:, None] * input_weights(spec, n_in)[None, :]
    if spec.encoding == "direct":
        return spec.v_center + half * np.repeat(x[:, None], n_in, axis=1)
    if spec.encoding == "delays":
        prev = 2 * np.asarray(u_before if u_before is not None else np.full(n_in, 0.5), dtype=float) - 1
        full = np.concatenate([prev[-(n_in - 1):] if n_in > 1 else prev[:0], x])
        cols = [full[n_in - 1 - d: n_in - 1 - d + len(x)] for d in range(n_in)]
        return spec.v_center + half * np.stack(cols, axis=1)
    raise ValueError(f"unknown encoding {spec.encoding!r}")


def _probe_nodes(circuit: MultiTerminalCircuit, n_probes: int, network_seed: int) -> np.ndarray:
    free = circuit.free
    cand = free[free < 2 * circuit.network.junctions.shape[0]]
    return np.sort(np.random.default_rng(40_000 + network_seed).choice(cand, size=n_probes, replace=False))


def simulate_multi(spec: C.RunSpec, n_samples: int | None = None, washout_steps: int | None = None,
                   fp=None, save_states: bool | None = None) -> dict:
    """Run one multi-electrode spec (spec.layout must be set). Same outputs as drive.simulate, with "current" of
    shape (n_samples, n_virtual * n_electrodes) (virtual-node-major) and "v_electrodes" (n_samples, n_in).
    fp: FilamentParams overriding config.filament_params(spec) (parameter scans, N8; dt must be spec.stride ms).
    save_states: store the junction states of every input ("junction_x"/"junction_y", "junction_lambda", "junction_p",
    "flags_packed"); None = config.SAVE_JUNCTION_STATES (False). "on_fraction" is always stored."""
    if not spec.layout:
        raise ValueError("simulate_multi needs spec.layout")
    t0 = time.time()
    params, wd = C.network_params(spec.p_relax)
    network = C.make_network(spec.network_seed, side=spec.side, geometry=spec.geometry)
    electrodes = make_layout(spec.layout, network)
    circuit = MultiTerminalCircuit(network, wd, electrodes)
    fixed = circuit.fixed_electrodes
    driven = [k for k, i in enumerate(fixed) if electrodes[i].role == "in"]     # positions in the V_fixed vector
    n_in = len(driven)
    rng = np.random.default_rng(C.zlib_seed(spec.name))
    meanfield = spec.model == "meanfield"
    filament = spec.model == "filament"
    volatile = spec.model == "volatile"
    vp = C.volatile_params(spec) if volatile else None
    kd0 = junction_kd0(vp, network.junctions.shape[0], spec.network_seed) if volatile else None
    if fp is None and filament:
        fp = C.filament_params(spec)
    if filament and abs(fp.dt - spec.stride * C.DT) > 1e-12:
        raise ValueError("fp.dt must equal spec.stride x DT")
    p_relax = C.junction_p_relax(spec, network.junctions.shape[0])

    u, labels = input_stream(spec)
    if n_samples is not None:
        u = u[:n_samples]
        labels = None if labels is None else labels[:n_samples]
    u_pre = predrive_stream(spec)
    if washout_steps is not None:
        u_pre = u_pre[:-(-washout_steps // spec.hold)]
    v_pre = electrode_voltages(u_pre, spec, n_in)
    v_in = electrode_voltages(u, spec, n_in, u_before=u_pre)
    stride = spec.stride
    if stride != 1 and not (filament or volatile):
        raise ValueError("stride > 1 is only valid for the deterministic ODE models (filament, volatile)")
    if spec.hold % stride:
        raise ValueError(f"hold {spec.hold} is not a multiple of stride {stride}")
    hold = spec.hold // stride                              # integration steps per input sample
    offsets = virtual_offsets(hold, spec.n_virtual_eff)
    probes = _probe_nodes(circuit, C.N_PROBES, spec.network_seed)
    n_j, n_e = network.junctions.shape[0], len(electrodes)

    state = (np.zeros((2, n_j)) if volatile else np.zeros(n_j) if (meanfield or filament)
             else np.full(n_j, params.G_off))
    V_fixed = np.zeros(len(fixed))
    method = "pcg" if (meanfield or filament or volatile) else "reuse"

    def step(state, v_drive):
        V_fixed[driven] = v_drive
        if meanfield:
            G = params.G_off + state * (params.G_on - params.G_off)
        elif filament:
            G = filament_conductance(state, fp)
        elif volatile:
            G = volatile_conductance(state, vp)
        else:
            G = state
        V = circuit.solve(G, V_fixed, method=method)
        dV = circuit.junction_voltage_drops(V)
        if meanfield:
            new = meanfield_step(state, dV, params, p_relax)
        elif filament:
            new = step_lambda(state, dV, fp, rng)
        elif volatile:
            new = step_state(state, dV, vp, kd0, rng)
        else:
            new = update_dynamics(state, np.abs(dV), G * dV, params, rng)
        return new, V

    formed = None
    if spec.forming and not volatile:
        raise ValueError("forming is only implemented for the volatile (N10) junction")
    if spec.forming:                                        # N10-K: forming drive (constant, all driven electrodes)
        v_form = np.full(n_in, spec.forming_v)
        for _ in range(int(round(spec.forming / (stride * C.DT)))):
            state, _ = step(state, v_form)
        formed = state.copy()
    for v in v_pre:
        for _ in range(hold):
            state, _ = step(state, v)

    current = np.empty((len(u), len(offsets), n_e))
    floats = [k for k, e in enumerate(electrodes) if e.role == "float"]
    v_float = np.empty((len(u), len(offsets), len(floats)))
    probe_v = np.empty((len(u), len(probes)))
    save = C.SAVE_JUNCTION_STATES if save_states is None else save_states
    on_frac = np.empty(len(u))
    jstate = np.empty((len(u) if save else 1, 2, n_j) if volatile else (len(u) if save else 1, n_j),
                      dtype=np.float32 if (filament or volatile) else np.float16 if meanfield else bool)
    for k, v in enumerate(v_in):
        oi = 0
        for s in range(hold):
            before = state
            state, V = step(state, v)
            if oi < len(offsets) and s == offsets[oi]:
                current[k, oi] = circuit.electrode_currents(V)
                if floats:
                    v_float[k, oi] = V[circuit.super[floats]]
                oi += 1
            if s == hold - 1:
                probe_v[k] = V[probes]
                j = k if save else 0
                if filament:
                    jstate[j] = before / fp.lambda_crit
                    on_frac[k] = np.mean(np.abs(jstate[j].astype(float)) >= 1)
                elif volatile:
                    jstate[j] = before
                    on_frac[k] = np.mean(np.minimum(1.0, jstate[j].astype(float).sum(axis=0)) >= 0.5)
                else:
                    jstate[j] = before if meanfield else before > params.G_off
                    on_frac[k] = float(jstate[j].astype(float).mean())

    out = {
        "u": u, "labels": labels if labels is not None else np.empty(0, dtype=int),
        "v_in": v_in.mean(axis=1), "v_electrodes": v_in,
        "current": current.reshape(len(u), -1), "virtual_offsets": offsets * stride, "stride": stride,
        "probe_v": probe_v, "probe_nodes": probes, "n_junctions": n_j, "n_electrodes": n_e,
        "electrode_roles": np.array([e.role for e in electrodes]),
        "electrode_n_wires": np.array([len(e.wires) for e in electrodes]),
        **({"v_float": v_float.reshape(len(u), -1)} if floats else {}),
        "solver_stats": np.array([circuit.stats[k] for k in sorted(circuit.stats)]),
        "seconds": time.time() - t0,
    }
    if formed is not None:
        out["formed_x"], out["formed_y"] = formed[0].astype(np.float32), formed[1].astype(np.float32)
    out["on_fraction"] = on_frac
    if save:
        if volatile:
            out["junction_x"], out["junction_y"] = jstate[:, 0], jstate[:, 1]
        elif filament:
            out["junction_lambda"] = jstate                 # Lambda / lambda_crit (|.| >= 1: filament closed)
        elif meanfield:
            out["junction_p"] = jstate
        else:
            out["flags_packed"] = np.packbits(jstate, axis=1)
    return out
