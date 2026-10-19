"""N12 non-uniform wire placement (agnw_rc/substrate.py, docs/N12_GEOMETRY.md)."""
import numpy as np
import pytest

from agnw_rc import config as C
from agnw_rc.electrodes import make_layout
from agnw_rc.multidrive import simulate_multi
from agnw_rc.substrate import (GEOMETRIES, Cluster, GeometrySpec, build_network, cluster_counts, generate_sticks,
                               network_stats)

SIDE = np.sqrt(C.AREA)


def _same_network(a, b):
    assert a.n_wires == b.n_wires and a.area == b.area and a.wire_length == b.wire_length
    for f in ("wire_endpoints", "junctions", "junction_pos", "electrode_in", "electrode_gnd", "isolated_wires"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def _angles(ends):
    d = ends[:, 1] - ends[:, 0]
    return np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)


def test_default_spec_is_the_framework_network_bit_for_bit():
    for seed, side in ((1, 0.0), (2, 0.0), (3, 2.0)):
        ref = C.make_network(seed, side=side)
        _same_network(C.make_network(seed, side=side, geometry="uniform"), ref)
        _same_network(build_network(ref.n_wires, ref.area, C.LENGTH, np.random.default_rng(seed), GeometrySpec()), ref)


def test_run_names_unchanged_without_geometry():
    base = dict(stream="iid", hold=1000, v_center=0.38, network_seed=1, model="volatile", layout="pads")
    assert "_g" not in C.RunSpec(**base).name
    named = C.RunSpec(**base, geometry="clusters4").name
    assert named.endswith("_gclusters4") or "_gclusters4_" in named


def test_cluster_counts_keep_the_number_of_wires():
    spec = GeometrySpec(clusters=(Cluster(0.2, 0.2, 0.1, 1.0), Cluster(0.8, 0.8, 0.1, 2.0)), background=0.25)
    for n in (1, 7, 2000, 2001):
        c = cluster_counts(n, spec)
        assert c.sum() == n and len(c) == 3
    np.testing.assert_array_equal(cluster_counts(2000, spec), [500, 500, 1000])


def test_clustered_centres_are_inside_and_concentrated_at_the_fixed_positions():
    spec = GEOMETRIES["clusters4"]
    for seed in (1, 2):
        ends = generate_sticks(C.N_WIRES, C.AREA, C.LENGTH, np.random.default_rng(seed), spec)
        c = ends.mean(axis=1) / SIDE
        assert len(c) == C.N_WIRES and c.min() >= 0 and c.max() <= 1
        for cl in spec.clusters:                                  # within 2 sigma: 86 % of the cluster's wires
            near = np.sum(np.hypot(c[:, 0] - cl.x, c[:, 1] - cl.y) < 2 * cl.sigma)
            assert near > 0.8 * 0.7 * C.N_WIRES / 4
    a = C.make_network(1, geometry="clusters4").wire_endpoints.mean(axis=1)
    b = C.make_network(2, geometry="clusters4").wire_endpoints.mean(axis=1)
    assert not np.allclose(a, b)                                  # same cluster positions, different wires


def test_gaussian_orientation_has_the_requested_mean_and_spread():
    for mu in (0.0, np.pi / 2, 2.5):
        spec = GeometrySpec(angle_mu=mu, angle_sigma=0.2)
        th = _angles(generate_sticks(20000, C.AREA, C.LENGTH, np.random.default_rng(0), spec))
        assert th.min() >= 0 and th.max() < np.pi
        dev = np.mod(th - mu + np.pi / 2, np.pi) - np.pi / 2       # signed deviation from mu, modulo pi
        assert abs(dev.mean()) < 0.01 and dev.std() == pytest.approx(0.2, rel=0.03)


def test_alignment_changes_the_number_of_junctions():
    n_uniform = len(C.make_network(1).junctions)
    n_aligned = len(C.make_network(1, geometry="aligned_x").junctions)
    assert n_aligned < 0.7 * n_uniform                            # parallel sticks cross less (~ <sin|dtheta|>)


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError):
        GeometrySpec(clusters=(Cluster(0.5, 0.5, 0.1),), background=1.0)
    with pytest.raises(ValueError):
        GeometrySpec(clusters=(Cluster(1.5, 0.5, 0.1),))
    with pytest.raises(ValueError):
        GeometrySpec(angle_sigma=-0.1)


def test_network_stats_on_the_working_network():
    net = C.make_network(1)
    s = network_stats(net)
    assert s["n_junctions"] == len(net.junctions) and s["percolates"]
    assert s["mean_degree"] == pytest.approx(2 * len(net.junctions) / net.n_wires)
    p = network_stats(net, make_layout("pads", net))
    assert p["percolates"] and p["in_linked"] == "8/8" and p["gnd_linked"] == "8/8"


def test_simulate_multi_runs_on_a_clustered_network():
    spec = C.RunSpec(stream="iid", hold=20, v_center=0.38, v_width=0.16, network_seed=1, n_samples=5,
                     model="volatile", layout="pads", tau_s=3.0, tau_spread=1.0, washout=40, n_virtual=2, stride=2,
                     geometry="clusters4")
    raw = simulate_multi(spec, n_samples=5, washout_steps=40)
    assert raw["current"].shape == (5, 2 * 16) and np.all(np.isfinite(raw["current"]))
