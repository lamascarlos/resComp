"""AgNW network geometry: stick generation and intersection detection."""
import numpy as np
from collections import defaultdict
from dataclasses import dataclass


def generate_sticks(n_wires, area, length, rng):
    """Generate n_wires random line segments of fixed length in a
    sqrt(area) x sqrt(area) square substrate.

    Centers are drawn uniformly in [0, side]^2 and orientations uniformly
    in [0, pi); segment endpoints may extend slightly beyond the substrate
    boundary near the edges, which is expected and not corrected.

    Returns:
        wire_endpoints: (n_wires, 2, 2) array, [[x1,y1],[x2,y2]] per wire.
    """
    side = np.sqrt(area)
    centers = rng.uniform(0, side, size=(n_wires, 2))
    angles = rng.uniform(0, np.pi, size=n_wires)
    direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    half = length / 2.0
    p1 = centers - half * direction
    p2 = centers + half * direction
    return np.stack([p1, p2], axis=1)


def _segments_intersect(p1, p2, p3, p4):
    """Return the intersection point of segments (p1,p2) and (p3,p4), or
    None if they are parallel/collinear or don't overlap within both
    segments' extents."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-15:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + t * d1
    return None


def _candidate_pairs(wire_endpoints, cell_size):
    """Spatial-hash binning: return wire index pairs (i<j) whose bounding
    boxes could plausibly intersect, without testing all O(N^2) pairs."""
    n = len(wire_endpoints)
    if n == 0:
        return []
    mins = wire_endpoints.min(axis=1)
    maxs = wire_endpoints.max(axis=1)
    cell_min = np.floor(mins / cell_size).astype(int)
    cell_max = np.floor(maxs / cell_size).astype(int)

    grid = defaultdict(list)
    for i in range(n):
        for cx in range(cell_min[i, 0], cell_max[i, 0] + 1):
            for cy in range(cell_min[i, 1], cell_max[i, 1] + 1):
                grid[(cx, cy)].append(i)

    seen = set()
    pairs = []
    for (cx, cy), wires in grid.items():
        neighbor_wires = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_wires.update(grid.get((cx + dx, cy + dy), []))
        for i in wires:
            for j in neighbor_wires:
                if j <= i or (i, j) in seen:
                    continue
                seen.add((i, j))
                pairs.append((i, j))
    return pairs


def find_intersections(wire_endpoints, cell_size):
    """Find every pair of wires whose segments intersect.

    Returns:
        junctions: (n_junctions, 2) int array of (wire_i, wire_j), i<j.
        junction_pos: (n_junctions, 2) float array of intersection (x,y).
    """
    pairs = _candidate_pairs(wire_endpoints, cell_size)
    junctions = []
    positions = []
    for i, j in pairs:
        pt = _segments_intersect(
            wire_endpoints[i, 0], wire_endpoints[i, 1],
            wire_endpoints[j, 0], wire_endpoints[j, 1],
        )
        if pt is not None:
            junctions.append((i, j))
            positions.append(pt)
    if not junctions:
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=float)
    return np.array(junctions, dtype=int), np.array(positions, dtype=float)


@dataclass(frozen=True)
class Network:
    n_wires: int
    wire_endpoints: np.ndarray
    junctions: np.ndarray
    junction_pos: np.ndarray
    electrode_in: np.ndarray
    electrode_gnd: np.ndarray
    isolated_wires: np.ndarray
    area: float
    wire_length: float


def assign_electrodes(wire_endpoints, area, band_frac=0.05):
    """Assign wires touching a boundary band on the left/right edges of the
    substrate to the V_IN / GND electrode groups respectively."""
    side = np.sqrt(area)
    band = band_frac * side
    left = np.any(wire_endpoints[:, :, 0] <= band, axis=1)
    right = np.any(wire_endpoints[:, :, 0] >= side - band, axis=1)
    electrode_in = np.nonzero(left)[0]
    electrode_gnd = np.nonzero(right)[0]
    return electrode_in, electrode_gnd


def _find_isolated_wires(n_wires, junctions, electrode_in, electrode_gnd):
    """Wires with no junction and no electrode contact -- excluded from the
    circuit solve entirely (spec §9)."""
    connected = np.zeros(n_wires, dtype=bool)
    if junctions.size:
        connected[junctions.ravel()] = True
    connected[electrode_in] = True
    connected[electrode_gnd] = True
    return np.nonzero(~connected)[0]


def build_network(n_wires, area, length, rng, cell_size=None, band_frac=0.05):
    """Generate a full AgNW network: geometry, junctions, electrode
    assignment, and isolated-wire detection."""
    wire_endpoints = generate_sticks(n_wires, area, length, rng)
    if cell_size is None:
        cell_size = length
    junctions, junction_pos = find_intersections(wire_endpoints, cell_size)
    electrode_in, electrode_gnd = assign_electrodes(wire_endpoints, area, band_frac)
    isolated_wires = _find_isolated_wires(n_wires, junctions, electrode_in, electrode_gnd)

    return Network(
        n_wires=n_wires,
        wire_endpoints=wire_endpoints,
        junctions=junctions,
        junction_pos=junction_pos,
        electrode_in=electrode_in,
        electrode_gnd=electrode_gnd,
        isolated_wires=isolated_wires,
        area=area,
        wire_length=length,
    )
