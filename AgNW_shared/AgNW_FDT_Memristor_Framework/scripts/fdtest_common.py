"""Shared setup of the FDTest drivers: one fixed network, the recorded junctions, the working-model parameters."""
import dataclasses

import numpy as np

from agnw_sim.geometry import build_network
from agnw_sim.presets import lamas2026_fig4

AREA, LENGTH, DT = 1e-6, 70e-6, 1e-3


def build_setup(n_wires, network_seed, k_junctions, p_relax):
    """(network, sampled junction ids, SimParams, wire diameter). The network and the recorded junctions depend
    only on network_seed, so every dynamics seed of a set of runs sees the same system. Working model
    (decision of 2026-09-20): Lamas preset (dual-node, V_th = 0.1 V, D = 90 nm) plus exponential relaxation."""
    params, run_kwargs = lamas2026_fig4(DT)
    params = dataclasses.replace(params, p_relax=p_relax)
    network = build_network(n_wires, AREA, LENGTH, np.random.default_rng(network_seed))
    n_junctions = network.junctions.shape[0]
    k = min(k_junctions, n_junctions)
    ids = np.sort(np.random.default_rng(network_seed + 1000).choice(n_junctions, size=k, replace=False))
    return network, ids, params, run_kwargs["wire_diameter"]
