"""N1 mean-field ablation: the update is the exact expectation of the framework's stochastic step, and the
solver agrees with the framework's direct circuit solve."""
import numpy as np
import pytest

from agnw_sim.dualnode import DualNodeCircuit
from agnw_sim.dynamics import update_dynamics

from agnw_rc import config as C
from agnw_rc.evaluate import evaluate_iid_run
from agnw_rc.meanfield import MeanFieldSolver, meanfield_step, simulate_meanfield


def test_meanfield_step_is_expectation_of_update_dynamics():
    params, _ = C.network_params(p_relax=1e-2)          # large p_relax so every channel matters
    rng = np.random.default_rng(0)
    n_j, n_rep = 40, 20_000
    P = rng.uniform(0, 1, n_j)
    dV = rng.uniform(0, 1.5, n_j)                       # spans sub- and supra-threshold voltages
    on = rng.random((n_rep, n_j)) < P                   # Bernoulli(P) states
    new_on = np.empty_like(on)
    for r in range(n_rep):
        G = np.where(on[r], params.G_on, params.G_off)
        I = G * dV                                      # ON junctions carry G_on dV, as in the framework loop
        new_on[r] = update_dynamics(G, np.abs(dV), I, params, rng) > params.G_off
    np.testing.assert_allclose(new_on.mean(axis=0), meanfield_step(P, dV, params), atol=0.015)


def test_meanfield_solver_matches_direct_solve():
    params, wd = C.network_params()
    net = C.make_network(1)
    circuit = DualNodeCircuit(net, wd)
    solver = MeanFieldSolver(circuit)
    rng = np.random.default_rng(1)
    P = rng.uniform(0, 0.3, len(net.junctions))
    worst = 0.0
    for _ in range(30):
        P = np.clip(P + rng.normal(0, 0.003, P.shape), 0, 1)
        G = params.G_off + P * (params.G_on - params.G_off)
        worst = max(worst, np.abs(solver.solve(G, 3.0) - circuit.solve(G, 3.0)).max())
    assert worst < 1e-5                                  # below the 10 uV readout resolution
    assert solver.stats["cg_solves"] > 0                 # the fast path was actually used


def test_meanfield_run_smoke():
    spec = C.RunSpec("iid", 5, 4.5, 1, 0, model="meanfield")
    raw = simulate_meanfield(spec, n_samples=120, washout_steps=500)
    assert raw["junction_p"].shape == (120, raw["n_junctions"])
    assert np.all((raw["junction_p"] >= 0) & (raw["junction_p"] <= 1))
    assert raw["current"].shape == (120, 5)
    r = evaluate_iid_run(raw, floor=0.05)
    assert np.isfinite(r["junctions"]["narma10_nmse"])
    assert spec.name.endswith("_mf")


def test_heterogeneous_p_relax_draw():
    spec = C.RunSpec("iid", 10, 5.5, 2, 0, model="meanfield", p_relax_max=1e-2)
    a = C.junction_p_relax(spec, 5000)
    b = C.junction_p_relax(C.RunSpec("iid", 30, 5.5, 2, 1, model="meanfield", p_relax_max=1e-2), 5000)
    np.testing.assert_array_equal(a, b)                  # quenched: depends only on the network seed
    assert a.min() >= 1e-4 and a.max() <= 1e-2
    assert abs(np.log10(a).mean() - (-3.0)) < 0.05       # log-uniform: mean exponent midway
    assert C.junction_p_relax(C.RunSpec("iid", 10, 5.5, 2), 10) == C.P_RELAX
    assert spec.name.endswith("_mf_prmax1e-02")


def test_meanfield_step_accepts_per_junction_rates():
    params, _ = C.network_params()
    P, dV = np.full(3, 0.5), np.zeros(3)                 # no SET, no RESET current: only relaxation acts
    out = meanfield_step(P, dV, params, np.array([0.0, 0.1, 0.5]))
    np.testing.assert_allclose(out, [0.5, 0.45, 0.25])
