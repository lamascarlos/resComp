"""Drive the framework's network with an input stream and collect the readout observables.

The simulation itself is agnw_sim.simulate.run_simulation, unchanged. This module only
  * turns input samples u_k in [0, 1] into a piecewise-constant electrode voltage (each held `hold` steps,
    after a random pre-drive of WASHOUT_STEPS that erases the all-OFF initial state),
  * asks run_simulation for the full ON/OFF snapshot at the LAST step of every hold block,
  * derives the observables: total current at N_VIRTUAL instants per block, K probe voltages (recomputed from the
    snapshot with the framework's circuit solver), and the junction states themselves.

A snapshot at step i is the state at the start of step i, i.e. the one run_simulation used to solve the circuit
at step i; so the recomputed voltages are exactly the ones the simulation saw at that step.
"""
import time
import zlib
from pathlib import Path

import numpy as np

from agnw_sim.cached_solver import CachedSolver
from agnw_sim.dualnode import DualNodeCircuit
from agnw_sim.simulate import run_simulation

from . import config as C
from .tasks import iid_stream, sinesquare_stream


def input_stream(spec: C.RunSpec):
    """(u, labels or None) of the scored part. Identical for all replicates of a network seed, so replicates
    can be compared sample by sample."""
    n = spec.length
    if spec.stream == "iid":
        return iid_stream(n, 10_000 + spec.network_seed), None
    if spec.stream == "sinesquare":
        return sinesquare_stream(n, 20_000 + spec.network_seed)
    raise ValueError(f"unknown stream {spec.stream!r}")


def predrive_stream(spec: C.RunSpec) -> np.ndarray:
    """Random i.i.d. pre-drive (washout) samples; differs between replicates."""
    n = -(-spec.washout_steps // spec.hold)
    return iid_stream(n, 30_000 + 100 * spec.network_seed + spec.replicate)


def to_voltage(u, spec: C.RunSpec) -> np.ndarray:
    return spec.v_lo + (spec.v_hi - spec.v_lo) * np.asarray(u, dtype=float)


def virtual_offsets(hold: int, n_virtual: int = C.N_VIRTUAL) -> np.ndarray:
    """Steps inside a hold block at which the current is sampled (evenly spaced, always including the last)."""
    return np.unique(np.round(np.linspace(0, hold - 1, min(n_virtual, hold))).astype(int))


def probe_nodes(network, circuit, n_probes: int, network_seed: int) -> np.ndarray:
    """Fixed random dual-graph nodes (on junctions, connected to the electrodes) used as voltage probes."""
    free = CachedSolver(circuit)._free                   # nodes reachable from the electrodes
    candidates = free[free < 2 * network.junctions.shape[0]]
    rng = np.random.default_rng(40_000 + network_seed)
    return np.sort(rng.choice(candidates, size=n_probes, replace=False))


def dynamics_seed(spec: C.RunSpec) -> int:
    return zlib.crc32(spec.name.encode())


def simulate(spec: C.RunSpec, n_samples: int | None = None, washout_steps: int | None = None) -> dict:
    """Run one spec. n_samples / washout_steps override the config (for tests). Returns a dict of arrays."""
    t0 = time.time()
    params, wire_diameter = C.network_params(spec.p_relax)
    network = C.make_network(spec.network_seed, geometry=spec.geometry)
    u, labels = input_stream(spec)
    if n_samples is not None:
        u = u[:n_samples]
        labels = None if labels is None else labels[:n_samples]
    u_pre = predrive_stream(spec)
    if washout_steps is not None:
        u_pre = u_pre[:-(-washout_steps // spec.hold)]
    hold = spec.hold
    v_steps = np.repeat(to_voltage(np.concatenate([u_pre, u]), spec), hold)
    n_pre_steps = len(u_pre) * hold
    n_steps = len(v_steps)

    def waveform(t):
        return v_steps[min(int(round(t / C.DT)), n_steps - 1)]

    block_end = n_pre_steps + np.arange(len(u)) * hold + hold - 1
    sim = run_simulation(
        C.N_WIRES, C.AREA, C.LENGTH, waveform, n_steps, C.DT, params=params,
        rng=np.random.default_rng(dynamics_seed(spec)), n_sampled_junctions=0,
        wire_diameter=wire_diameter, solver=C.SOLVER, network=network, snapshot_steps=block_end,
    )
    flags = sim.G_snapshots.astype(bool)                               # (n_samples, n_junctions)

    circuit = DualNodeCircuit(network, wire_diameter)
    probes = probe_nodes(network, circuit, C.N_PROBES, spec.network_seed)
    v_in = to_voltage(u, spec)
    probe_v = np.empty((len(u), len(probes)))
    i_check = np.empty(len(u))
    for k in range(len(u)):
        G = np.where(flags[k], params.G_on, params.G_off)
        V = circuit.solve(G, v_in[k])
        probe_v[k] = V[probes]
        i_check[k] = circuit.total_current(V)

    offsets = virtual_offsets(hold, spec.n_virtual_eff)
    block_start = n_pre_steps + np.arange(len(u)) * hold
    current = sim.I_total[block_start[:, None] + offsets[None, :]]      # (n_samples, n_virtual)

    return {
        "u": u, "labels": labels if labels is not None else np.empty(0, dtype=int),
        "v_in": v_in, "current": current, "virtual_offsets": offsets,
        "probe_v": probe_v, "probe_nodes": probes,
        "flags_packed": np.packbits(flags, axis=1), "n_junctions": flags.shape[1],
        "on_fraction": flags.mean(axis=1),
        "i_recomputed": i_check, "i_simulated": sim.I_total[block_end],
        "i_total_scored": sim.I_total[n_pre_steps:].astype(np.float32),
        "i_total_predrive": sim.I_total[:n_pre_steps].astype(np.float32),
        "solver_stats": np.array([sim.solver_stats[k] for k in sorted(sim.solver_stats)]),
        "seconds": time.time() - t0,
    }


def unpack_flags(raw) -> np.ndarray:
    return np.unpackbits(raw["flags_packed"], axis=1, count=int(raw["n_junctions"])).astype(bool)


def raw_path(spec: C.RunSpec, root: Path) -> Path:
    return Path(root) / f"{spec.name}.npz"


def save_raw(spec: C.RunSpec, raw: dict, root: Path) -> Path:
    path = raw_path(spec, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    meta = {f"spec_{k}": v for k, v in spec.__dict__.items()}
    raw = {k: v for k, v in raw.items() if k != "model"}
    np.savez_compressed(tmp, **raw, **meta, framework_commit=C.framework_commit())
    tmp.replace(path)                                  # atomic: a half-written file is never taken as done
    return path


def load_raw(spec: C.RunSpec, root: Path) -> dict:
    with np.load(raw_path(spec, root), allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
