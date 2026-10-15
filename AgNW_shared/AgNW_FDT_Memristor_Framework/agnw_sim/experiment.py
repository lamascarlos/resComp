"""Measured FDTest data (Keithley/Clarius+ export, one sheet per run): long-format table, drift removal, readout-jump
correction and the stage arrays of the FDTest.

Protocol of a sheet (AgNWNs-dec25-M2, 08-10-2026): a voltage list sweep ON / OFF / ON / OFF with ON = V0 + eps = 1.05 V and
OFF = V0 = 1.0 V, about 400 samples per segment at one sample per 28.8 ms, so TWO FDTest cycles per sheet (cycle 1 = segments
A, B; cycle 2 = segments C, D). Between sheets the device is held at 1.0 V but not recorded; the resistance relaxes during that
gap (the next sheet starts about 0.4 Ohm lower instead of the +0.06 Ohm of a switch-on), so the sheets are NOT one continuous
series and the switch-on of cycle 1 (segment A) does not start from the unperturbed steady state.

Observable: x = G / G_trend - 1 with G = I / V. G_trend is linear in time within each sheet, fitted through the means of the
last `tail` samples of the two unperturbed segments (B and D), so the slow drift of the device (132 -> 152 Ohm over the
experiment) is divided out without touching the perturbed segments (the response is the offset of A and C from the trend).
Readout jump: G changes within one sample when V switches (heating / nonlinear I-V); it is a property of reading the state at a
different voltage, not of the state, and is subtracted from the perturbed segments (estimated by readout_jump)."""
import re

import numpy as np
import pandas as pd

COLUMNS = ["run", "sheet", "cycle", "stage", "sample_in_stage", "sample", "t_s", "V", "I", "R"]


def read_xls(path):
    """[(run name, t, I, V)] of the RunN sheets of the Clarius+ export, in run order (columns Time, AI, AV)."""
    x = pd.ExcelFile(path)
    names = sorted((s for s in x.sheet_names if re.fullmatch(r"Run\d+", s)), key=lambda s: int(s[3:]))
    out = []
    for s in names:
        d = x.parse(s)
        out.append((s, d["Time"].to_numpy(float), d["AI"].to_numpy(float), d["AV"].to_numpy(float)))
    return out


def to_long(sheets):
    """Long-format table (COLUMNS) of [(run, t, I, V)]: one row per measured sample, sheet = 1-based position, cycle 1/2 and
    stage perturbed/unperturbed from the voltage segments (V above the midpoint of its two levels = perturbed). Rows without a
    current or voltage (export padding) are dropped; t_s is the time within the sheet (may be NaN where the export lacks it)."""
    frames = []
    for k, (run, t, I, V) in enumerate(sheets):
        ok = np.isfinite(I) & np.isfinite(V)
        t, I, V = t[ok], I[ok], V[ok]
        bounds = np.r_[0, np.flatnonzero(np.diff(V) != 0) + 1, len(V)]
        if len(bounds) != 5:
            raise ValueError(f"{run}: expected 4 voltage segments, found {len(bounds) - 1}")
        seg = np.repeat(np.arange(4), np.diff(bounds))
        perturbed = V > 0.5 * (V.min() + V.max())
        if not (perturbed[bounds[0]] and not perturbed[bounds[1]]):
            raise ValueError(f"{run}: sheet must start with the perturbed segment")
        frames.append(pd.DataFrame({
            "run": run, "sheet": k + 1, "cycle": seg // 2 + 1,
            "stage": np.where(perturbed, "perturbed", "unperturbed"),
            "sample_in_stage": np.arange(len(V)) - bounds[seg], "sample": np.arange(len(V)),
            "t_s": t, "V": V, "I": I, "R": V / I}))
    return pd.concat(frames, ignore_index=True)[COLUMNS]


def sheet_segments(df):
    """Per sheet (in order) a dict run, t: [4 arrays], G: [4 arrays] of the segments A, B, C, D (cycle 1 ON, OFF, cycle 2 ON,
    OFF). Time missing in the export is filled from the sample index and the median sampling interval of the sheet."""
    out = []
    for _, d in df.groupby("sheet", sort=True):
        t = d["t_s"].to_numpy(float)
        n = d["sample"].to_numpy()
        if not np.isfinite(t).all():
            ok = np.isfinite(t)
            dt = np.median(np.diff(t[ok]) / np.diff(n[ok]))
            t = np.where(ok, t, t[ok][0] + (n - n[ok][0]) * dt)
        G = (d["I"] / d["V"]).to_numpy(float)
        key = 2 * (d["cycle"].to_numpy() - 1) + (d["stage"].to_numpy() == "unperturbed")
        out.append({"run": d["run"].iloc[0], "t": [t[key == s] for s in range(4)], "G": [G[key == s] for s in range(4)]})
    return out


def detrend(sh, tail=200):
    """[x_A, x_B, x_C, x_D] = G / G_trend - 1 of one sheet (sheet_segments item); G_trend is the line through the (mean time,
    mean G) of the last `tail` samples of B and of D."""
    (tb, gb), (td, gd) = ((sh["t"][s][-tail:].mean(), sh["G"][s][-tail:].mean()) for s in (1, 3))
    slope = (gd - gb) / (td - tb)
    return [g / (gb + slope * (t - tb)) - 1.0 for t, g in zip(sh["t"], sh["G"])]


def switch_steps(segs):
    """(n_sheets, 3) one-sample steps x(perturbed side) - x(unperturbed side) at the three switches of every sheet
    (A->B off, B->C on, C->D off) of the detrended segments."""
    return np.array([[a[-1] - b[0], c[0] - b[-1], c[-1] - d[0]] for a, b, c, d in segs])


def max_inner_step(segs, skip=20):
    """(n_sheets,) largest |one-sample step| of x inside the segments of every sheet, ignoring the first `skip` samples of each
    segment (the switch transients). Flags sheets with sudden resistance jumps (discrete switching events), which the linear
    trend cannot remove."""
    return np.array([max(np.max(np.abs(np.diff(s[skip:]))) for s in sg) for sg in segs])


def readout_jump(segs):
    """Readout offset of x at the perturbed voltage: mean of switch_steps over all switches."""
    return float(np.mean(switch_steps(segs)))


def stage_arrays(segs, tail=200, n_lags=150, cycles=(1, 2), jump=0.0):
    """(x2, x3, x4, xI) of the FDTest from detrended segments, rows sheet-major then cycle: x2 = last `tail` samples of the
    perturbed segment minus `jump` (P_eps), x3 = first n_lags unperturbed samples (lag 0 = first sample after the switch-off),
    x4 = last `tail` unperturbed samples (P_0), xI = first n_lags perturbed samples minus `jump` (switch-on transient; NaN for
    cycle 1, which starts after the unrecorded gap). Raises ValueError if a transient window would overlap the tail window of
    the same segment."""
    if min(len(s) for sg in segs for s in sg) < tail + n_lags:
        raise ValueError(f"tail ({tail}) + n_lags ({n_lags}) exceeds the shortest segment")
    rows = [[], [], [], []]
    for sg in segs:
        for c in cycles:
            on, off = sg[2 * c - 2] - jump, sg[2 * c - 1]
            rows[0].append(on[-tail:]); rows[1].append(off[:n_lags]); rows[2].append(off[-tail:])
            rows[3].append(on[:n_lags] if c == 2 else np.full(n_lags, np.nan))
    return tuple(np.array(r) for r in rows)
