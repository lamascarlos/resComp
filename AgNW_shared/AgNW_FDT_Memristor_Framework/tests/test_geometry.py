import numpy as np
from agnw_sim.geometry import (
    generate_sticks,
    _segments_intersect,
    find_intersections,
    assign_electrodes,
    _find_isolated_wires,
    build_network,
    Network,
)


def test_generate_sticks_shape():
    rng = np.random.default_rng(0)
    endpoints = generate_sticks(n_wires=10, area=1e-6, length=70e-6, rng=rng)
    assert endpoints.shape == (10, 2, 2)


def test_generate_sticks_correct_length():
    rng = np.random.default_rng(1)
    length = 70e-6
    endpoints = generate_sticks(n_wires=50, area=1e-6, length=length, rng=rng)
    lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
    np.testing.assert_allclose(lengths, length, rtol=1e-9)


def test_generate_sticks_reproducible_with_seed():
    endpoints_a = generate_sticks(n_wires=20, area=1e-6, length=70e-6, rng=np.random.default_rng(42))
    endpoints_b = generate_sticks(n_wires=20, area=1e-6, length=70e-6, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(endpoints_a, endpoints_b)


def test_segments_intersect_crossing_point():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([2.0, 2.0])
    p3 = np.array([0.0, 2.0])
    p4 = np.array([2.0, 0.0])
    pt = _segments_intersect(p1, p2, p3, p4)
    np.testing.assert_allclose(pt, [1.0, 1.0])


def test_segments_intersect_parallel_returns_none():
    p1 = np.array([0.0, 0.0]); p2 = np.array([1.0, 0.0])
    p3 = np.array([0.0, 1.0]); p4 = np.array([1.0, 1.0])
    assert _segments_intersect(p1, p2, p3, p4) is None


def test_segments_intersect_no_overlap_returns_none():
    p1 = np.array([0.0, 0.0]); p2 = np.array([1.0, 0.0])
    p3 = np.array([5.0, -1.0]); p4 = np.array([5.0, 1.0])
    assert _segments_intersect(p1, p2, p3, p4) is None


def test_find_intersections_matches_brute_force():
    rng = np.random.default_rng(17)
    wire_endpoints = generate_sticks(n_wires=60, area=1e-6, length=70e-6, rng=rng)

    junctions, positions = find_intersections(wire_endpoints, cell_size=70e-6)
    assert positions.shape[0] == junctions.shape[0]
    found_pairs = set(map(tuple, junctions.tolist())) if junctions.size else set()

    brute_pairs = set()
    for i in range(60):
        for j in range(i + 1, 60):
            if _segments_intersect(wire_endpoints[i, 0], wire_endpoints[i, 1],
                                    wire_endpoints[j, 0], wire_endpoints[j, 1]) is not None:
                brute_pairs.add((i, j))

    assert found_pairs == brute_pairs


def test_find_intersections_empty_network_returns_empty_arrays():
    wire_endpoints = np.empty((0, 2, 2))
    junctions, positions = find_intersections(wire_endpoints, cell_size=70e-6)
    assert junctions.shape == (0, 2)
    assert positions.shape == (0, 2)


def test_assign_electrodes_finds_boundary_wires():
    side = np.sqrt(1e-6)
    # wire 0: fully inside left band; wire 1: fully inside right band; wire 2: middle, touches neither
    wire_endpoints = np.array([
        [[0.0, 0.0], [1e-6, 1e-6]],
        [[side - 1e-6, 0.0], [side, 1e-6]],
        [[side / 2, side / 2], [side / 2 + 1e-6, side / 2 + 1e-6]],
    ])
    electrode_in, electrode_gnd = assign_electrodes(wire_endpoints, area=1e-6, band_frac=0.05)
    assert 0 in electrode_in
    assert 1 in electrode_gnd
    assert 2 not in electrode_in
    assert 2 not in electrode_gnd


def test_find_isolated_wires():
    junctions = np.array([[0, 1]])
    electrode_in = np.array([2])
    electrode_gnd = np.array([3])
    isolated = _find_isolated_wires(5, junctions, electrode_in, electrode_gnd)
    np.testing.assert_array_equal(isolated, [4])


def test_find_isolated_wires_no_junctions():
    junctions = np.empty((0, 2), dtype=int)
    isolated = _find_isolated_wires(3, junctions, np.array([0]), np.array([1]))
    np.testing.assert_array_equal(isolated, [2])


def test_build_network_reproducible_and_consistent():
    net_a = build_network(n_wires=100, area=1e-6, length=70e-6, rng=np.random.default_rng(7))
    net_b = build_network(n_wires=100, area=1e-6, length=70e-6, rng=np.random.default_rng(7))
    assert isinstance(net_a, Network)
    np.testing.assert_array_equal(net_a.wire_endpoints, net_b.wire_endpoints)
    np.testing.assert_array_equal(net_a.junctions, net_b.junctions)
    assert net_a.n_wires == 100
    assert net_a.junctions.shape[1] == 2
    assert net_a.area == 1e-6
    assert net_a.wire_length == 70e-6
