"""N11: a rectangular strip of the working-density network with band electrodes on its two short ends
(docs/PLAN_N11_CALIBRATION.md section 3), to simulate the measured 7 mm gap (gap / wire length 200) directly.

Wires: the framework's stick algorithm (agnw_sim.geometry.generate_sticks) generalised to a rectangle - uniform centres
in [0, L_x] x [0, L_y], uniform angles in [0, pi), fixed length, the same random-number order. Junctions from
agnw_sim.geometry.find_intersections; electrodes as agnw_sim.geometry.assign_electrodes with the band measured on L_x
(driven band at x = 0, grounded band at x = L_x) and the framework's contact rule (electrodes.two_bands). For
L_x = L_y the network and electrodes equal build_network / two_bands bit for bit (tests/test_strip.py).
Network.area is L_x L_y: code that takes sqrt(area) as the side (electrodes.two_bands, pads) must not be used on a strip.
"""
from dataclasses import dataclass

import numpy as np

from agnw_sim.geometry import Network, _find_isolated_wires, find_intersections

from . import config as C
from .electrodes import Electrode, _band_contact_t


@dataclass(frozen=True)
class Strip:
    network: Network
    length: float               # m, electrode to electrode (x)
    width: float                # m (y)
    electrodes: list            # [driven band "in", grounded band "gnd"]


def strip_sticks(n_wires, length, width, wire_length, rng):
    """generate_sticks on a length x width rectangle (endpoints may extend beyond it, as in the framework)."""
    centers = rng.uniform(0, (length, width), size=(n_wires, 2))
    angles = rng.uniform(0, np.pi, size=n_wires)
    direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    half = wire_length / 2.0
    return np.stack([centers - half * direction, centers + half * direction], axis=1)


def make_strip(seed: int, length_mm: float, width_mm: float, band_frac: float = 0.05) -> Strip:
    """Strip of length_mm x width_mm at the working density (C.N_WIRES per mm^2, C.LENGTH wires); seed as
    config.make_network."""
    lx, ly = length_mm * 1e-3, width_mm * 1e-3
    n = int(round(C.N_WIRES * length_mm * width_mm))
    ends = strip_sticks(n, lx, ly, C.LENGTH, np.random.default_rng(seed))
    junctions, pos = find_intersections(ends, C.LENGTH)
    band = band_frac * lx
    ein = np.nonzero(np.any(ends[:, :, 0] <= band, axis=1))[0]
    egnd = np.nonzero(np.any(ends[:, :, 0] >= lx - band, axis=1))[0]
    net = Network(n_wires=n, wire_endpoints=ends, junctions=junctions, junction_pos=pos, electrode_in=ein,
                  electrode_gnd=egnd, isolated_wires=_find_isolated_wires(n, junctions, ein, egnd),
                  area=lx * ly, wire_length=C.LENGTH)
    t_in, _ = _band_contact_t(net, ein, left=True)
    t_gnd, _ = _band_contact_t(net, egnd, left=False)
    els = [Electrode("in", "in", np.asarray(ein, dtype=int), t_in),
           Electrode("gnd", "gnd", np.asarray(egnd, dtype=int), t_gnd)]
    return Strip(net, lx, ly, els)
