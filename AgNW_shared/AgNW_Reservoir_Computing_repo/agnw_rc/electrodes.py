"""Electrode layouts for multi-terminal drive and readout (NEXT_STEPS N5).

An electrode is a metal contact: every wire that crosses it is tied to it at one contact point, given as the
distance t along the wire from its first endpoint (the same convention as agnw_sim.dualnode). Each electrode has a
role: "in" (driven, Dirichlet), "gnd" (held at 0 V, current measured) or "float" (free node, voltage measured).

`two_bands` reproduces the framework's two-electrode setup exactly (agnw_sim.geometry.assign_electrodes +
DualNodeCircuit's contact rule), so the multi-terminal circuit can be checked against the framework.
"""
from dataclasses import dataclass

import numpy as np

from agnw_sim.geometry import Network, assign_electrodes


@dataclass(frozen=True)
class Electrode:
    name: str
    role: str                  # "in", "gnd" or "float"
    wires: np.ndarray          # wire indices touching the electrode
    t: np.ndarray              # contact position along each wire, from its first endpoint (m)

    def __post_init__(self):
        if self.role not in ("in", "gnd", "float"):
            raise ValueError(f"unknown electrode role {self.role!r}")
        if len(self.wires) != len(self.t):
            raise ValueError("wires and t must have the same length")


def _band_contact_t(network: Network, wires, left: bool):
    """Framework rule: the contact is the wire end inside the band (smallest x for the left band, largest for the
    right), at t = 0 for the first endpoint and t = L for the second."""
    ends = network.wire_endpoints[wires]
    k = np.argmin(ends[:, :, 0], axis=1) if left else np.argmax(ends[:, :, 0], axis=1)
    return np.where(k == 0, 0.0, network.wire_length), k


def two_bands(network: Network, band_frac: float = 0.05) -> list[Electrode]:
    """The framework layout: V_IN band on the left, GND band on the right."""
    ein, egnd = assign_electrodes(network.wire_endpoints, network.area, band_frac)
    t_in, _ = _band_contact_t(network, ein, left=True)
    t_gnd, _ = _band_contact_t(network, egnd, left=False)
    return [Electrode("in", "in", np.asarray(ein, dtype=int), t_in),
            Electrode("gnd", "gnd", np.asarray(egnd, dtype=int), t_gnd)]


def split_bands(network: Network, n_in: int = 8, n_out: int = 8, band_frac: float = 0.05) -> list[Electrode]:
    """The two framework bands cut into equal segments along y: n_in driven segments on the left, n_out grounded
    segments on the right. A band wire belongs to the segment containing its contact end."""
    ein, egnd = assign_electrodes(network.wire_endpoints, network.area, band_frac)
    side = np.sqrt(network.area)
    out = []
    for wires, left, n, role, tag in ((ein, True, n_in, "in", "L"), (egnd, False, n_out, "gnd", "R")):
        wires = np.asarray(wires, dtype=int)
        t, k = _band_contact_t(network, wires, left)
        y = network.wire_endpoints[wires, k, 1]
        seg = np.clip((y / side * n).astype(int), 0, n - 1)
        for s in range(n):
            m = seg == s
            out.append(Electrode(f"{tag}{s}", role, wires[m], t[m]))
    return out


def _clip_segment_to_box(p1, p2, lo, hi):
    """Liang-Barsky: parameter interval [u0, u1] of p1 + u (p2 - p1), u in [0, 1], inside the box [lo, hi]^2,
    or None if the segment misses the box."""
    d = p2 - p1
    u0, u1 = 0.0, 1.0
    for axis in (0, 1):
        for p, q in ((-d[axis], p1[axis] - lo[axis]), (d[axis], hi[axis] - p1[axis])):
            if p == 0:
                if q < 0:
                    return None
            else:
                r = q / p
                if p < 0:
                    u0 = max(u0, r)
                else:
                    u1 = min(u1, r)
    return (u0, u1) if u0 <= u1 else None


def pad_contacts(network: Network, center, size: float):
    """Wires crossing a square pad (side `size`, centred at `center`) and their contact points, taken at the middle
    of the part of the wire lying on the pad."""
    c = np.asarray(center, dtype=float)
    lo, hi = c - size / 2, c + size / 2
    ends = network.wire_endpoints
    # cheap bounding-box prefilter
    cand = np.nonzero((ends[:, :, 0].max(1) >= lo[0]) & (ends[:, :, 0].min(1) <= hi[0])
                      & (ends[:, :, 1].max(1) >= lo[1]) & (ends[:, :, 1].min(1) <= hi[1]))[0]
    wires, ts = [], []
    for w in cand:
        iv = _clip_segment_to_box(ends[w, 0], ends[w, 1], lo, hi)
        if iv is not None:
            wires.append(w)
            ts.append(0.5 * (iv[0] + iv[1]) * network.wire_length)
    return np.array(wires, dtype=int), np.array(ts, dtype=float)


# N10-E: 4 driven, 4 grounded and 8 floating (voltage-read) pads, interleaved so that every row and column has one
# driven and one grounded pad and each floating pad sits between them. Indexed [i][j] as in pad_grid.
PADS_FLOAT_ROLES = (("in", "float", "gnd", "float"),
                    ("float", "gnd", "float", "in"),
                    ("gnd", "float", "in", "float"),
                    ("float", "in", "float", "gnd"))


def pad_grid(network: Network, n_side: int = 4, pad_size: float = 50e-6, roles=None) -> list[Electrode]:
    """n_side x n_side square pads on a regular grid (centres at (i + 1/2) side / n_side). Default roles are a
    checkerboard: pads with (i + j) even are driven inputs, the others grounded outputs; `roles[i][j]` overrides
    ("in", "gnd" or "float": a floating pad whose voltage is read out)."""
    side = np.sqrt(network.area)
    out = []
    for i in range(n_side):
        for j in range(n_side):
            center = ((i + 0.5) * side / n_side, (j + 0.5) * side / n_side)
            wires, t = pad_contacts(network, center, pad_size)
            role = roles[i][j] if roles is not None else ("in" if (i + j) % 2 == 0 else "gnd")
            out.append(Electrode(f"P{i}{j}", role, wires, t))
    return out


def pads_float(network: Network) -> list[Electrode]:
    """4x4 pads: 4 driven, 4 grounded, 8 floating voltage-read pads (PADS_FLOAT_ROLES), N10-E."""
    return pad_grid(network, roles=PADS_FLOAT_ROLES)


def pads_large(network: Network) -> list[Electrode]:
    """4x4 checkerboard pads of 100 um (4x the area of "pads"): each pad contacts more wires, averaging junction
    noise over more paths (N10 noise mitigation)."""
    return pad_grid(network, pad_size=100e-6)


def pads_scaled(network: Network) -> list[Electrode]:
    """4x4 checkerboard pads with the pad size scaled with the network side (50 um per mm): a larger network is the
    1 mm "pads" device enlarged geometrically (N10-L). Identical to "pads" for a 1 mm network."""
    return pad_grid(network, pad_size=50e-6 * np.sqrt(network.area / 1e-6))


LAYOUTS = {"two_bands": two_bands, "split_bands": split_bands, "pads": pad_grid, "pads_float": pads_float,
           "pads_large": pads_large, "pads_scaled": pads_scaled}


def make_layout(name: str, network: Network) -> list[Electrode]:
    return LAYOUTS[name](network)
