import numpy as np
from agnw_sim.waveforms import triangle, sine, step, ramp, white_noise_probe


def test_triangle_wave_bounds_and_key_points():
    t = np.linspace(0, 2, 1000)
    y = triangle(t, amplitude=1.0, period=0.5)
    assert np.all(y <= 1.0 + 1e-9)
    assert np.all(y >= -1.0 - 1e-9)
    np.testing.assert_allclose(triangle(np.array([0.0]), 1.0, 0.5), [1.0], atol=1e-9)
    np.testing.assert_allclose(triangle(np.array([0.25]), 1.0, 0.5), [-1.0], atol=1e-9)


def test_sine_wave_matches_formula():
    t = np.array([0.0, 0.25])
    y = sine(t, amplitude=2.0, frequency=1.0)
    np.testing.assert_allclose(y, [0.0, 2.0], atol=1e-9)


def test_step_wave_switches_at_threshold():
    t = np.array([0.0, 0.5, 1.0, 1.5])
    y = step(t, amplitude=3.0, t_on=1.0)
    np.testing.assert_array_equal(y, [0.0, 0.0, 3.0, 3.0])


def test_ramp_linear_in_time():
    t = np.array([0.0, 1.0, 2.0])
    y = ramp(t, rate=0.5)
    np.testing.assert_allclose(y, [0.0, 0.5, 1.0])


def test_white_noise_probe_reproducible_with_seed():
    t = np.linspace(0, 1, 100)
    y1 = white_noise_probe(t, amplitude=0.01, rng=np.random.default_rng(5))
    y2 = white_noise_probe(t, amplitude=0.01, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(y1, y2)
    assert y1.shape == t.shape
