import numpy as np, pytest
from agnw_sim.geometry import build_network
from agnw_sim.dynamics import SimParams
from agnw_sim.simulate import run_simulation

FAST = SimParams(P0=1.0, alpha=1000.0, V_th=0.001, dt=1e-3)
COMMON = dict(n_wires=1500, area=1e-6, length=70e-6, waveform_fn=lambda t: 1.0, n_steps=12, dt=1e-3, params=FAST)


def test_prebuilt_network_matches_internal_build_with_the_same_generator():
    a = run_simulation(rng=np.random.default_rng(3), **COMMON)
    rng = np.random.default_rng(3)
    net = build_network(1500, 1e-6, 70e-6, rng)
    b = run_simulation(network=net, rng=rng, **COMMON)
    np.testing.assert_array_equal(a.network.junctions, b.network.junctions)
    np.testing.assert_array_equal(a.sampled_junction_ids, b.sampled_junction_ids)
    np.testing.assert_array_equal(a.G_sample, b.G_sample)
    np.testing.assert_array_equal(a.I_total, b.I_total)


def test_prebuilt_network_with_wrong_wire_count_raises():
    net = build_network(100, 1e-6, 70e-6, np.random.default_rng(0))
    with pytest.raises(ValueError):
        run_simulation(network=net, rng=np.random.default_rng(1), **COMMON)   # n_wires=1500


def test_record_every_thins_the_sample_record_only():
    a = run_simulation(rng=np.random.default_rng(3), **COMMON)
    b = run_simulation(rng=np.random.default_rng(3), record_every=5, **COMMON)
    np.testing.assert_array_equal(b.G_sample, a.G_sample[::5])
    assert b.G_sample.shape[0] == 3                                    # steps 0, 5, 10 of 12
    np.testing.assert_array_equal(b.I_total, a.I_total)
    assert b.I_total.shape == (12,)


def test_record_every_must_be_a_positive_integer():
    for bad in (0, -2, 1.5):
        with pytest.raises(ValueError):
            run_simulation(rng=np.random.default_rng(3), record_every=bad, **COMMON)


def test_explicit_sampled_junction_ids_are_recorded_and_validated():
    net = build_network(1500, 1e-6, 70e-6, np.random.default_rng(3))
    n_j = net.junctions.shape[0]
    ids = np.array([5, 2, 40])
    r = run_simulation(network=net, sampled_junction_ids=ids, snapshot_steps=[11],
                       rng=np.random.default_rng(1), **COMMON)
    np.testing.assert_array_equal(r.sampled_junction_ids, ids)
    assert r.G_sample.shape == (12, 3)
    np.testing.assert_array_equal(r.G_sample[11] > FAST.G_off, r.G_snapshots[0][ids].astype(bool))
    with pytest.raises(ValueError):
        run_simulation(network=net, sampled_junction_ids=[n_j], rng=np.random.default_rng(1), **COMMON)


def test_snapshots_hold_the_full_state_before_the_update():
    net = build_network(1500, 1e-6, 70e-6, np.random.default_rng(3))
    n_j = net.junctions.shape[0]
    r = run_simulation(network=net, sampled_junction_ids=np.arange(n_j), snapshot_steps=[0, 7, 7, 9],
                       rng=np.random.default_rng(1), **COMMON)
    np.testing.assert_array_equal(r.snapshot_steps, [0, 7, 9])         # sorted, unique
    assert r.G_snapshots.shape == (3, n_j) and r.G_snapshots.dtype == np.uint8
    assert r.G_snapshots[0].sum() == 0                                  # everything starts OFF
    for k, step in enumerate(r.snapshot_steps):
        np.testing.assert_array_equal(r.G_snapshots[k], (r.G_sample[step] > FAST.G_off).astype(np.uint8))
    assert r.G_snapshots[-1].sum() > 0                                  # the fast-switching run did switch


def test_snapshot_steps_out_of_range_raise():
    for bad in ([12], [-1]):
        with pytest.raises(ValueError):
            run_simulation(rng=np.random.default_rng(3), snapshot_steps=bad, **COMMON)


def test_snapshots_are_none_when_not_requested():
    r = run_simulation(rng=np.random.default_rng(3), **COMMON)
    assert r.G_snapshots is None and r.snapshot_steps is None


def test_sample_on_flags_records_uint8_on_flags_matching_the_float_record():
    net = build_network(1500, 1e-6, 70e-6, np.random.default_rng(3))
    kw = dict(network=net, snapshot_steps=[0, 7], **COMMON)
    a = run_simulation(rng=np.random.default_rng(1), **kw)
    b = run_simulation(rng=np.random.default_rng(1), sample_on_flags=True, **kw)
    assert a.G_sample.dtype == np.float64
    assert b.G_sample.dtype == np.uint8
    assert b.G_sample.shape == a.G_sample.shape
    assert (a.G_sample > FAST.G_off).any()                              # junctions actually switched
    np.testing.assert_array_equal(b.G_sample, (a.G_sample > FAST.G_off).astype(np.uint8))
    # the flag changes the recording only
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.V_in, b.V_in)
    np.testing.assert_array_equal(a.I_total, b.I_total)
    np.testing.assert_array_equal(a.snapshot_steps, b.snapshot_steps)
    np.testing.assert_array_equal(a.G_snapshots, b.G_snapshots)


def test_sample_on_flags_with_record_every_and_snapshots():
    net = build_network(1500, 1e-6, 70e-6, np.random.default_rng(3))
    kw = dict(network=net, record_every=5, snapshot_steps=[11], **COMMON)
    a = run_simulation(rng=np.random.default_rng(1), **kw)
    b = run_simulation(rng=np.random.default_rng(1), sample_on_flags=True, **kw)
    assert b.G_sample.dtype == np.uint8 and b.G_sample.shape == (3, a.G_sample.shape[1])   # ceil(12/5)
    np.testing.assert_array_equal(b.G_sample, (a.G_sample > FAST.G_off).astype(np.uint8))
    np.testing.assert_array_equal(a.G_snapshots, b.G_snapshots)
    np.testing.assert_array_equal(a.I_total, b.I_total)
