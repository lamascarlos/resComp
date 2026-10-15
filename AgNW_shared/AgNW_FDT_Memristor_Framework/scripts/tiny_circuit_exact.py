"""Exact FDTest of a tiny circuit (Markov test 2): is the violation of g_eff a consequence of coarse-graining?

  python -m scripts.tiny_circuit_exact --out DIR [--n-wires 14 --seed 18 --side 2.0] [--search] [--v0 V --eps E]
                                       [--mc-steps 200000 --mc-seeds 2 --trace-steps 1000000 --trace-seeds 4]

A small dual-node circuit (default: 14 wires in a square of side 2 x wire length, seed 18, J = 11 dynamic junctions) has 2^J junction
states. The one-step transition matrix of the working model (Lamas preset + p_relax = 1e-4) is built exactly from the circuit solves
(agnw_sim.exact_chain), and the FDTest quantities <Z(n)>, C_ZZ(n) and delta2 are computed exactly for: the full state (must be 0), the
exact g_eff, g_eff binned to 60 / 20 / 8 bins, and the ON count. The chain is validated against `update_dynamics` (one-step rows) and
against `run_simulation` (per-junction ON fractions). Writes to DIR:
  tiny_circuit_exact.json      strict JSON (non-finite values are null)
  tiny_circuit_exact.png       curves and delta2 per observable
  tiny_circuit_mc_trace.npz    series sampled from the exact chain (control of scripts.markov_geff_test): raw g_eff, and g_eff (20 bins)
                               and ON count each plus i.i.d. uniform jitter of one bin width, `control_key` = the coarse observable with
                               the largest exact delta2
Exit status 0, or 1 with a message (no circuit found, no operating point, the chain does not match the simulator).

Operating point (spec 5.2): V0 = the drive with the most junctions whose stationary ON probability lies in [0.1, 0.9] (scan of --v-grid);
eps = the perturbation with the mean g_eff response closest to 0.5 sigma of the spontaneous fluctuation among those in 0.3-1 sigma.
`--v0`/`--eps` skip the scans (the response is then only reported).
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agnw_sim import exact_chain as ec
from agnw_sim import markov_tests as mt
from agnw_sim.dualnode import DualNodeCircuit
from agnw_sim.dynamics import update_dynamics
from agnw_sim.geometry import build_network
from agnw_sim.presets import lamas2026_fig4
from agnw_sim.simulate import run_simulation
from scripts.fdtest_analyze import sanitize

LENGTH = 70e-6
P_RELAX = 1e-4
OBSERVABLES = ("full", "geff_full", "geff_60", "geff_20", "geff_8", "on_count")
COARSE_CONTROLS = ("geff_20", "on_count")          # observables that the trace file carries
P_MIN_DEFAULT = 1e-6                               # label values with stationary probability below this are excluded from Z ...
P_MIN = {"full": 1e-12, "geff_full": 1e-12}        # ... except for the unbinned observables, whose exactness (delta2 = 0) an exclusion would spoil


# ------------------------------------------------------------------------------------------ circuit and chain
def working_model():
    params, kw = lamas2026_fig4(1e-3)
    return dataclasses.replace(params, p_relax=P_RELAX), kw["wire_diameter"]


def make_circuit(n_wires, seed, side, wire_diameter):
    """(network, dynamic junction ids) of the tiny circuit."""
    net = build_network(n_wires, (side * LENGTH) ** 2, LENGTH, np.random.default_rng(seed))
    return net, ec.dynamic_junctions(net, wire_diameter)


def find_circuits(j_min, j_max, wire_diameter, sides=(2.0, 2.5, 3.0), n_range=range(6, 40, 2), n_seeds=200):
    """All (side, n_wires, seed, J) with j_min <= J <= j_max dynamic junctions (deterministic order)."""
    out = []
    for side in sides:
        for n in n_range:
            for seed in range(n_seeds):
                J = len(make_circuit(n, seed, side, wire_diameter)[1])
                if j_min <= J <= j_max:
                    out.append((side, n, seed, J))
    return out


def chain_at(net, params, D, ids, V):
    p_on, p_off, g = ec.state_data(net, params, D, ids, V)
    T = mt.transition_matrix(p_on, p_off)
    return T, mt.stationary_distribution(T), g


def scan_v0(net, params, D, ids, v_grid):
    """[{V, n_mid}] and the best V (most junctions with stationary ON probability in [0.1, 0.9]; the smallest V among ties)."""
    bits = mt.state_bits(len(ids))
    rows = []
    for V in v_grid:
        pj = chain_at(net, params, D, ids, V)[1] @ bits
        rows.append({"V": float(V), "n_mid": int(np.sum((pj > 0.1) & (pj < 0.9)))})
    best = max(rows, key=lambda r: (r["n_mid"], -r["V"]))
    return rows, best


def scan_eps(net, params, D, ids, V0, eps_grid):
    """[{eps, response}] with response = shift of the mean g_eff in units of its stationary standard deviation at V0, and the entry
    with response in [0.3, 1] closest to 0.5 (None if there is none)."""
    _, pi0, g = chain_at(net, params, D, ids, V0)
    m0, sd = pi0 @ g, np.sqrt(pi0 @ g ** 2 - (pi0 @ g) ** 2)
    rows = [{"eps": float(e), "response": float((chain_at(net, params, D, ids, V0 + e)[1] @ g - m0) / sd)} for e in eps_grid]
    ok = [r for r in rows if 0.3 <= r["response"] <= 1.0]
    return rows, (min(ok, key=lambda r: abs(r["response"] - 0.5)) if ok else None)


# ------------------------------------------------------------------------------------------ observables
def weighted_percentile(values, weights, q):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return np.interp(np.asarray(q) / 100.0, cum / cum[-1], values[order])


def build_labels(g, bits, pi0, pie):
    """{name: integer label per state} for the observables of the table; binned labels use equal-width bins between the weighted 0.5th
    and 99.5th percentile of the pooled (pi_0 + pi_eps) distribution, -1 outside that range."""
    lo, hi = weighted_percentile(g, 0.5 * (pi0 + pie), [0.5, 99.5])

    def binned(n_bins):
        lab = np.digitize(g, np.linspace(lo, hi, n_bins + 1)[1:-1])
        return np.where((g < lo) | (g > hi), -1, lab)
    return {"full": np.arange(len(g)),
            "geff_full": np.unique(np.round(g / g.max(), 12), return_inverse=True)[1].ravel(),
            "geff_60": binned(60), "geff_20": binned(20), "geff_8": binned(8),
            "on_count": bits.sum(axis=1).astype(int)}


def exact_table(T0, pi0, pie, labels, n_start=4000, cap=2 ** 18):
    """Exact curves for every observable, lags doubled from n_start until the window 2 tau_c fits for all of them (or cap).
    Returns (table {name: dict}, curves {name: (z_mean, c_zz)})."""
    names = list(labels)
    maps = [mt.conjugate_state_map(labels[k], pie, pi0, p_min=P_MIN.get(k, P_MIN_DEFAULT)) for k in names]
    Z = np.stack([m[0] for m in maps], axis=1)
    n = n_start
    while True:
        z_mean, c_zz = mt.exact_curves(T0, pie, pi0, Z, n)
        res = [mt.exact_delta2(z_mean[:, i], c_zz[:, i]) for i in range(len(names))]
        if all(np.isfinite(r[3]) for r in res) or n >= cap:
            break
        n *= 2
    table = {k: {"n_labels": int(len(np.unique(labels[k]))), "excluded_mass_0": maps[i][1], "excluded_mass_eps": maps[i][2],
                 "tau_c": int(res[i][0]), "delta2": res[i][1], "delta2_half_window": res[i][2], "delta2_double_window": res[i][3],
                 "hit_end": res[i][4], "n_lags": int(n)} for i, k in enumerate(names)}
    return table, {k: (z_mean[:, i], c_zz[:, i]) for i, k in enumerate(names)}


# ------------------------------------------------------------------------------------------ validation against the simulator
def validate_rows(net, params, D, ids, V0, T0, pi0, rng, n_states=3, n_draws=200_000):
    """One-step frequencies of update_dynamics from the n_states most visited states against the rows of T0, in binomial standard errors."""
    circuit = DualNodeCircuit(net, D)
    bits = mt.state_bits(len(ids))
    J = len(ids)
    worst = []
    for s in np.argsort(pi0)[::-1][:n_states]:
        G = np.full(len(net.junctions), params.G_off)
        G[ids[bits[s] == 1]] = params.G_on
        dV = circuit.junction_voltage_drops(circuit.solve(G, V0))[ids]
        Gd = G[ids]
        new = update_dynamics(np.tile(Gd, n_draws), np.tile(np.abs(dV), n_draws), np.tile(Gd * dV, n_draws), params, rng).reshape(n_draws, J)
        idx = ((new > params.G_off) * (1 << np.arange(J))).sum(axis=1)
        emp = np.bincount(idx, minlength=2 ** J) / n_draws
        sigma = np.sqrt(np.maximum(T0[s] * (1 - T0[s]), 1e-12) / n_draws)
        worst.append(float(np.max(np.abs(emp - T0[s]) / np.maximum(sigma, 1.0 / n_draws))))
    return {"max_deviation_sigma": max(worst), "per_state_sigma": worst, "n_draws": n_draws}


def validate_trajectories(net, params, D, ids, V0, pi0, seeds, n_steps, burn_in=30_000, block=20_000):
    """Per-junction ON fractions of run_simulation trajectories at fixed V0 against the marginals of pi_0 (z-scores from blocked errors)."""
    marg = pi0 @ mt.state_bits(len(ids))
    blocks = []
    for seed in seeds:
        res = run_simulation(net.n_wires, net.area, net.wire_length, lambda t: V0, n_steps + burn_in, params.dt, params=params,
                             rng=np.random.default_rng(seed), wire_diameter=D, network=net, sampled_junction_ids=ids,
                             record_every=1, sample_on_flags=True)
        on = res.G_sample[burn_in:].astype(float)
        nb = len(on) // block
        blocks.append(on[: nb * block].reshape(nb, block, -1).mean(axis=1))
    b = np.concatenate(blocks)
    se = np.maximum(b.std(axis=0, ddof=1) / np.sqrt(len(b)), 1e-4)
    z = (b.mean(axis=0) - marg) / se
    return {"max_abs_z": float(np.abs(z).max()), "z": z.tolist(), "n_blocks": int(len(b)), "n_steps": int(n_steps)}


# ------------------------------------------------------------------------------------------ control trace
def sample_chain(T, pi0, n_steps, rng):
    """State trajectory of the exact chain (start drawn from pi0)."""
    cum = np.cumsum(T, axis=1)
    cum[:, -1] = 1.0
    u = rng.random(n_steps)
    s = int(np.searchsorted(np.cumsum(pi0), rng.random()))
    out = np.empty(n_steps, dtype=np.int64)
    for t in range(n_steps):
        out[t] = s
        s = int(np.searchsorted(cum[s], u[t]))
    return out


def write_trace(path, T0, pi0, g, labels, table, n_steps, seeds):
    """Series for scripts.markov_geff_test: raw g_eff and the coarse controls with i.i.d. uniform jitter of one bin width."""
    keys = {"geff_20": labels["geff_20"], "on_count": labels["on_count"]}
    out = {k: [] for k in ("geff_raw", *(f"{k}_jit" for k in keys))}
    for seed in seeds:
        rng = np.random.default_rng([seed, 77])
        st = sample_chain(T0, pi0, n_steps, rng)
        out["geff_raw"].append(g[st].astype(np.float32))
        for k, lab in keys.items():
            out[f"{k}_jit"].append((lab[st] + rng.uniform(-0.5, 0.5, n_steps)).astype(np.float32))
    control = max(COARSE_CONTROLS, key=lambda k: np.nan_to_num(table[k]["delta2"], nan=-1.0))
    np.savez_compressed(path, **{k: np.stack(v) for k, v in out.items()}, control_key=f"{control}_jit",
                        control_exact_delta2=float(table[control]["delta2"]))
    return control


# ------------------------------------------------------------------------------------------ driver
def plot(table, curves, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, name in zip(axes[:2], ("on_count", "geff_20")):
        z, c = curves[name]
        stride = max(len(z) // 2000, 1)
        ax.plot(np.arange(len(z))[::stride], z[::stride], label=r"$\langle Z(n)\rangle$")
        ax.plot(np.arange(len(c))[::stride], c[::stride], "--", label=r"$C_{ZZ}(n)$")
        ax.axvline(table[name]["tau_c"], color="k", lw=0.8, ls=":")
        ax.set_xscale("symlog", linthresh=10); ax.set_xlabel("lag (steps)"); ax.set_title(f"{name}: delta2 = {table[name]['delta2']:.2e}")
        ax.legend(fontsize=8)
    names = list(table)
    axes[2].bar(range(len(names)), [max(table[k]["delta2"], 1e-12) for k in names])
    axes[2].set_yscale("log"); axes[2].set_xticks(range(len(names))); axes[2].set_xticklabels(names, rotation=45, ha="right")
    axes[2].set_ylabel(r"exact $\Delta^2$ (floor 1e-12)")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def run_tiny(net, ids, params, D, out, v_grid, eps_grid, v0=None, eps=None, min_mid=1, mc_steps=200_000, mc_seeds=(1, 2),
             trace_steps=1_000_000, trace_seeds=(1, 2, 3, 4), n_start=4000, cap=2 ** 18, log=print):
    """Whole computation on a given circuit. Returns the JSON-ready result dict (also written, with the figure and the trace, to `out`)."""
    J = len(ids)
    bits = mt.state_bits(J)
    result = {"circuit": {"n_wires": int(net.n_wires), "n_junctions_total": int(len(net.junctions)), "n_dynamic": int(J), "dynamic_ids": ids.tolist()},
              "params": {**dataclasses.asdict(params), "wire_diameter": D}}
    pilot = None
    if v0 is None:
        pilot, best = scan_v0(net, params, D, ids, v_grid)
        if best["n_mid"] < min_mid:
            sys.exit(f"no drive with at least {min_mid} junction(s) of ON probability in [0.1, 0.9] in the scan (best: {best})")
        v0 = best["V"]
    log(f"V0 = {v0:.4f} V")
    eps_rows = None
    if eps is None:
        eps_rows, pick = scan_eps(net, params, D, ids, v0, eps_grid)
        if pick is None:
            sys.exit("no perturbation with a g_eff response between 0.3 and 1 sigma in the scan; widen --eps-grid or set --eps")
        eps = pick["eps"]
    T0, pi0, g = chain_at(net, params, D, ids, v0)
    Te, pie, ge = chain_at(net, params, D, ids, v0 + eps)
    drive_dep = float(np.abs(ge - g).max() / g.max())
    if drive_dep > 1e-9:
        sys.exit(f"g_eff depends on the drive at fixed state (relative {drive_dep:.2e}); the observable maps assume it does not")
    sd = np.sqrt(pi0 @ g ** 2 - (pi0 @ g) ** 2)
    result["operating_point"] = {"V0": float(v0), "eps": float(eps), "response_sigma": float((pie @ g - pi0 @ g) / sd), "v_pilot": pilot,
                                 "eps_scan": eps_rows, "geff_drive_dependence": drive_dep,
                                 "stationary_residual_0": float(np.abs(pi0 @ T0 - pi0).max()), "stationary_residual_eps": float(np.abs(pie @ Te - pie).max())}
    log(f"eps = {eps:.4f} V, response {result['operating_point']['response_sigma']:.3f} sigma; exact table ...")
    labels = build_labels(g, bits, pi0, pie)
    table, curves = exact_table(T0, pi0, pie, labels, n_start=n_start, cap=cap)
    result["table"] = table
    for k, r in table.items():
        log(f"  {k:10s} labels {r['n_labels']:5d}  tau_c {r['tau_c']:6d}  delta2 {r['delta2']:.3e}" + ("  (window hit the end)" if r["hit_end"] else ""))
    rng = np.random.default_rng(2026)
    result["validation"] = {"rows": validate_rows(net, params, D, ids, v0, T0, pi0, rng)}
    if mc_steps:
        result["validation"]["trajectories"] = validate_trajectories(net, params, D, ids, v0, pi0, mc_seeds, mc_steps)
    control = write_trace(os.path.join(out, "tiny_circuit_mc_trace.npz"), T0, pi0, g, labels, table, trace_steps, trace_seeds)
    result["control_key"] = f"{control}_jit"
    result["curves"] = {k: {"z_mean": z[::20], "c_zz": c[::20], "stride": 20} for k, (z, c) in curves.items()}
    with open(os.path.join(out, "tiny_circuit_exact.json"), "w") as fh:
        json.dump(sanitize(result), fh, indent=1, allow_nan=False)
    plot(table, curves, os.path.join(out, "tiny_circuit_exact.png"))
    return result


def gate(result):
    """Message for every failed check of the chain (empty list = passes)."""
    bad = []
    if result["table"]["full"]["delta2"] > 1e-10:
        bad.append(f"delta2 of the FULL state is {result['table']['full']['delta2']:.3e} (must be 0): stop and find out why")
    if result["validation"]["rows"]["max_deviation_sigma"] > 5:
        bad.append(f"one-step rows deviate from update_dynamics by {result['validation']['rows']['max_deviation_sigma']:.1f} sigma (limit 5)")
    tr = result["validation"].get("trajectories")
    if tr and tr["max_abs_z"] > 4:
        bad.append(f"ON fractions of run_simulation deviate from pi_0 by |z| = {tr['max_abs_z']:.1f} (limit 4)")
    return bad


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--n-wires", type=int, default=14); p.add_argument("--seed", type=int, default=18)
    p.add_argument("--side", type=float, default=2.0, help="substrate side in units of the wire length")
    p.add_argument("--search", action="store_true", help="list circuits with 10-13 dynamic junctions and exit")
    p.add_argument("--v0", type=float); p.add_argument("--eps", type=float)
    p.add_argument("--mc-steps", type=int, default=200_000); p.add_argument("--mc-seeds", type=int, default=2)
    p.add_argument("--trace-steps", type=int, default=1_000_000); p.add_argument("--trace-seeds", type=int, default=4)
    a = p.parse_args(argv)
    params, D = working_model()
    if a.search:
        for side, n, seed, J in find_circuits(10, ec.MAX_JUNCTIONS, D):
            print(f"--side {side} --n-wires {n} --seed {seed}   J = {J}")
        return 0
    net, ids = make_circuit(a.n_wires, a.seed, a.side, D)
    if not 1 <= len(ids) <= ec.MAX_JUNCTIONS:
        sys.exit(f"this circuit has {len(ids)} dynamic junctions (need 1 .. {ec.MAX_JUNCTIONS}); try --search")
    os.makedirs(a.out, exist_ok=True)
    result = run_tiny(net, ids, params, D, a.out, np.arange(0.25, 1.001, 0.025), np.geomspace(1e-3, 0.1, 25), v0=a.v0, eps=a.eps,
                      mc_steps=a.mc_steps, mc_seeds=tuple(range(1, a.mc_seeds + 1)), trace_steps=a.trace_steps,
                      trace_seeds=tuple(range(1, a.trace_seeds + 1)))
    bad = gate(result)
    if bad:
        sys.exit("CHAIN CHECKS FAILED:\n  " + "\n  ".join(bad))
    print("chain checks passed; wrote tiny_circuit_exact.json / .png / tiny_circuit_mc_trace.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
