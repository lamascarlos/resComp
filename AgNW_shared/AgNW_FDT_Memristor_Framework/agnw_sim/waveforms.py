"""Electrode driving-voltage waveform generators."""
import numpy as np


def triangle(t, amplitude, period):
    """Triangular wave in [-amplitude, amplitude], value=+amplitude at t=0
    and every full period, -amplitude at the half period."""
    t = np.asarray(t, dtype=float)
    phase = (t % period) / period
    return amplitude * (4 * np.abs(phase - 0.5) - 1)


def sine(t, amplitude, frequency):
    t = np.asarray(t, dtype=float)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def step(t, amplitude, t_on):
    t = np.asarray(t, dtype=float)
    return np.where(t >= t_on, amplitude, 0.0)


def ramp(t, rate):
    t = np.asarray(t, dtype=float)
    return rate * t


def white_noise_probe(t, amplitude, rng):
    t = np.asarray(t, dtype=float)
    return amplitude * rng.standard_normal(t.shape)
