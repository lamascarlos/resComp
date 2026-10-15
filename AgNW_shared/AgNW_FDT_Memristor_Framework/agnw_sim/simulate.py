"""Time-loop orchestration: build the network once, then step circuit +
dynamics forward, recording trajectories."""
from dataclasses import dataclass
import numpy as np

from .geometry import build_network, Network
from .circuit import solve_circuit, junction_currents, total_current
from .dualnode import DualNodeCircuit
from .cached_solver import CachedSolver
from .dynamics import update_dynamics, update_dynamics_lifetime, SimParams


@dataclass
class SimulationResult:
    network: Network
    t: np.ndarray
    V_in: np.ndarray
    I_total: np.ndarray
    G_sample: np.ndarray
    sampled_junction_ids: np.ndarray
    params: SimParams
    initial_rng_state: dict
    band_frac: float
    wire_diameter: float = None   # None = ideal wires (wire-as-node model)
    solver_stats: dict = None     # counters of the cached solver (None for the direct solver)
    G_snapshots: np.ndarray = None    # (len(snapshot_steps), n_junctions) uint8, 1 = ON (None if not requested)
    snapshot_steps: np.ndarray = None


def run_simulation(n_wires, area, length, waveform_fn, n_steps, dt,
                    params=None, rng=None, n_sampled_junctions=200,
                    cell_size=None, band_frac=0.05, wire_diameter=None,
                    solver="direct", network=None, sampled_junction_ids=None,
                    record_every=1, snapshot_steps=None, sample_on_flags=False):
    """Run a full AgNW network simulation and return its trajectories.

    wire_diameter: None (default) treats each wire as one ideal node
    (wire-as-node). A positive value in metres uses the dual-node graph of
    Lamas et al. 2026 with resistive wire segments (see agnw_sim.dualnode);
    then V_mem/I_mem are taken across the memristor edge only.

    solver: "direct" (default) refactorizes the circuit every step; "cached"
    (dual-node only) reuses the factorization with exact rescaling and
    residual-checked rank updates (see agnw_sim.cached_solver), ~5-6x faster in
    steady state and accurate to ~1e-7 in the switching probabilities.

    network: an already built Network. Its wire count must equal n_wires; the
    geometry arguments (area, length, cell_size, band_frac) are then ignored and
    `rng` drives only the dynamics (and the choice of sampled junctions).
    sampled_junction_ids: explicit indices of the junctions recorded in G_sample
    (default: n_sampled_junctions drawn from rng).
    record_every: G_sample keeps every record_every-th step (step i is kept when
    i % record_every == 0); t, V_in and I_total keep full resolution.
    snapshot_steps: step indices at which the full ON/OFF state vector is stored
    in SimulationResult.G_snapshots (state at the start of that step, i.e. before
    its dynamics update).
    sample_on_flags: if True, G_sample is stored as uint8 ON flags
    (G > params.G_off) instead of float conductances (8x less memory)."""
    if solver not in ("direct", "cached"):
        raise ValueError(f"solver must be 'direct' or 'cached', got {solver!r}")
    if solver == "cached" and wire_diameter is None:
        raise ValueError("solver='cached' requires the dual-node model (wire_diameter)")
    if int(record_every) != record_every or record_every < 1:
        raise ValueError(f"record_every must be an integer >= 1, got {record_every!r}")
    record_every = int(record_every)
    if rng is None:
        rng = np.random.default_rng()
    initial_rng_state = rng.bit_generator.state
    if params is None:
        params = SimParams(dt=dt)
    elif params.dt != dt:
        raise ValueError(
            f"Conflicting dt values: dt={dt!r} argument does not match "
            f"params.dt={params.dt!r}. Pass matching values, or omit one."
        )

    if network is None:
        network = build_network(n_wires, area, length, rng, cell_size=cell_size, band_frac=band_frac)
    elif network.n_wires != n_wires:
        raise ValueError(f"network has {network.n_wires} wires but n_wires={n_wires}")
    n_junctions = network.junctions.shape[0]

    circuit = None if wire_diameter is None else DualNodeCircuit(network, wire_diameter)
    cached = CachedSolver(circuit) if solver == "cached" else None

    G = np.full(n_junctions, params.G_off)
    age = np.zeros(n_junctions, dtype=int)     # steps since SET (used only with lifetime_steps)

    if sampled_junction_ids is not None:
        sampled_junction_ids = np.asarray(sampled_junction_ids, dtype=int)
        if sampled_junction_ids.size and (sampled_junction_ids.min() < 0
                                          or sampled_junction_ids.max() >= n_junctions):
            raise ValueError("sampled_junction_ids out of range for this network")
        n_sampled = len(sampled_junction_ids)
    else:
        n_sampled = min(n_sampled_junctions, n_junctions)
        if n_sampled:
            sampled_junction_ids = rng.choice(n_junctions, size=n_sampled, replace=False)
        else:
            sampled_junction_ids = np.empty(0, dtype=int)

    snap_steps = np.unique(np.asarray([] if snapshot_steps is None else snapshot_steps, dtype=int))
    if snap_steps.size and (snap_steps.min() < 0 or snap_steps.max() >= n_steps):
        raise ValueError("snapshot_steps must lie in [0, n_steps)")
    G_snapshots = np.empty((len(snap_steps), n_junctions), dtype=np.uint8)
    next_snap = 0

    t = np.arange(n_steps) * params.dt
    V_in_arr = np.empty(n_steps)
    I_total_arr = np.empty(n_steps)
    G_sample = np.empty((-(-n_steps // record_every), n_sampled),
                        dtype=np.uint8 if sample_on_flags else float)

    for i in range(n_steps):
        V_in_t = float(waveform_fn(t[i]))
        if circuit is None:
            V = solve_circuit(network, G, V_in_t)
            I_total = total_current(network, G, V)
        else:
            V = circuit.solve(G, V_in_t) if cached is None else cached.solve(G, V_in_t)
            I_total = circuit.total_current(V)

        V_in_arr[i] = V_in_t
        I_total_arr[i] = I_total
        if n_sampled and i % record_every == 0:
            g_rec = G[sampled_junction_ids]
            G_sample[i // record_every] = (g_rec > params.G_off) if sample_on_flags else g_rec
        if next_snap < len(snap_steps) and i == snap_steps[next_snap]:
            G_snapshots[next_snap] = G > params.G_off
            next_snap += 1

        if n_junctions:
            if circuit is None:
                k = network.junctions[:, 0]
                l = network.junctions[:, 1]
                dV = V[k] - V[l]
                I_mem = junction_currents(network, G, V)
            else:
                dV = circuit.junction_voltage_drops(V)
                I_mem = G * dV
            V_mem = np.abs(dV)
            if params.lifetime_steps:
                G, age = update_dynamics_lifetime(G, age, V_mem, I_mem, params, rng)
            else:
                G = update_dynamics(G, V_mem, I_mem, params, rng)

    return SimulationResult(
        network=network,
        t=t,
        V_in=V_in_arr,
        I_total=I_total_arr,
        G_sample=G_sample,
        sampled_junction_ids=sampled_junction_ids,
        params=params,
        initial_rng_state=initial_rng_state,
        band_frac=band_frac,
        wire_diameter=wire_diameter,
        solver_stats=None if cached is None else dict(cached.stats),
        G_snapshots=G_snapshots if len(snap_steps) else None,
        snapshot_steps=snap_steps if len(snap_steps) else None,
    )
