"""N12: non-uniform wire placement on the substrate - dense clusters of wires and preferred wire orientation.

The framework's stick algorithm (agnw_sim.geometry.generate_sticks) draws every wire centre uniformly on the square and
every angle uniformly in [0, pi). Here both distributions are configurable, with the number of wires kept fixed:

* centres: a uniform background fraction plus K Gaussian clusters at FIXED positions (the same on every network seed;
  only the wires inside each cluster change). Cluster centres and widths are given in units of the substrate side, so
  a geometry keeps its shape on larger networks (RunSpec.side). Wire counts per cluster are fixed (largest remainder of
  the weights), not drawn. A Gaussian centre falling outside the square is drawn again (truncated Gaussian).
* angles: uniform in [0, pi), or Gaussian of mean mu and standard deviation sigma (radians) taken modulo pi (a wire at
  theta and at theta + pi is the same wire). mu = 0 is along x, the axis between the left and right band electrodes.

Junctions, band electrodes and isolated wires come from agnw_sim.geometry unchanged, so the result is an ordinary
agnw_sim Network that every layout, circuit and junction model accepts. GeometrySpec() reproduces
agnw_sim.geometry.build_network bit for bit (same random-number order; tests/test_substrate.py).

Named geometries live in GEOMETRIES (used through RunSpec.geometry, like electrodes.LAYOUTS through RunSpec.layout).
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from agnw_sim.geometry import Network, _find_isolated_wires, assign_electrodes, find_intersections


@dataclass(frozen=True)
class Cluster:
    x: float                    # centre, in units of the substrate side (0..1)
    y: float
    sigma: float                # Gaussian standard deviation, in units of the substrate side
    weight: float = 1.0         # relative share of the clustered wires


@dataclass(frozen=True)
class GeometrySpec:
    clusters: tuple = ()        # Cluster, ...; empty = uniform centres
    background: float = 1.0     # fraction of wires with uniform centres (ignored without clusters)
    angle_mu: float = 0.0       # rad, mean wire angle (angle_sigma > 0 only)
    angle_sigma: float = 0.0    # rad; 0 = uniform angles in [0, pi)

    def __post_init__(self):
        if self.clusters and not 0.0 <= self.background < 1.0:
            raise ValueError("background must be in [0, 1) when there are clusters")
        for c in self.clusters:
            if not (0.0 <= c.x <= 1.0 and 0.0 <= c.y <= 1.0 and c.sigma > 0 and c.weight > 0):
                raise ValueError(f"invalid cluster {c}")
        if self.angle_sigma < 0:
            raise ValueError("angle_sigma must be >= 0")


def cluster_counts(n_wires: int, spec: GeometrySpec) -> np.ndarray:
    """Wires per component: [background, cluster 1, ..., cluster K], summing to n_wires (largest remainder)."""
    if not spec.clusters:
        return np.array([n_wires])
    w = np.array([c.weight for c in spec.clusters], dtype=float)
    share = np.concatenate([[spec.background], (1.0 - spec.background) * w / w.sum()]) * n_wires
    counts = np.floor(share).astype(int)
    counts[np.argsort(counts - share)[: n_wires - counts.sum()]] += 1
    return counts


def _truncated_gaussian(rng, n, centre, sigma, side):
    out = np.empty((0, 2))
    while len(out) < n:
        p = rng.normal(centre, sigma, size=(n - len(out), 2))
        out = np.vstack([out, p[np.all((p >= 0) & (p <= side), axis=1)]])
    return out


def generate_sticks(n_wires: int, area: float, length: float, rng, spec: GeometrySpec) -> np.ndarray:
    """(n_wires, 2, 2) wire endpoints, as agnw_sim.geometry.generate_sticks but with spec's centre and angle laws."""
    side = np.sqrt(area)
    counts = cluster_counts(n_wires, spec)
    centers = rng.uniform(0, side, size=(counts[0], 2))
    for c, n in zip(spec.clusters, counts[1:]):
        centers = np.vstack([centers, _truncated_gaussian(rng, n, (c.x * side, c.y * side), c.sigma * side, side)])
    if spec.angle_sigma > 0:
        angles = np.mod(rng.normal(spec.angle_mu, spec.angle_sigma, size=n_wires), np.pi)
    else:
        angles = rng.uniform(0, np.pi, size=n_wires)
    direction = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    half = length / 2.0
    return np.stack([centers - half * direction, centers + half * direction], axis=1)


def build_network(n_wires: int, area: float, length: float, rng, spec: GeometrySpec,
                  band_frac: float = 0.05) -> Network:
    """agnw_sim.geometry.build_network with spec's wire placement."""
    ends = generate_sticks(n_wires, area, length, rng, spec)
    junctions, pos = find_intersections(ends, length)
    ein, egnd = assign_electrodes(ends, area, band_frac)
    return Network(n_wires=n_wires, wire_endpoints=ends, junctions=junctions, junction_pos=pos, electrode_in=ein,
                   electrode_gnd=egnd, isolated_wires=_find_isolated_wires(n_wires, junctions, ein, egnd),
                   area=area, wire_length=length)


def network_stats(network: Network, electrodes=None) -> dict:
    """Connectivity of a network (no simulation): junction counts, largest connected wire cluster and, for an
    electrode list (agnw_rc.electrodes), whether every driven and every grounded electrode touches the wire cluster
    that joins inputs to ground (a percolating path). Without electrodes the framework bands are used."""
    n = network.n_wires
    j = network.junctions
    graph = coo_matrix((np.ones(len(j)), (j[:, 0], j[:, 1])), shape=(n, n)) if len(j) else coo_matrix((n, n))
    _, label = connected_components(graph, directed=False)
    sizes = np.bincount(label)
    if electrodes is None:
        groups = {"in": [network.electrode_in], "gnd": [network.electrode_gnd]}
    else:
        groups = {r: [e.wires for e in electrodes if e.role == r] for r in ("in", "gnd")}
    comps = {r: [set(label[w]) for w in ws] for r, ws in groups.items()}
    shared = set().union(*comps["in"]) & set().union(*comps["gnd"]) if comps["in"] and comps["gnd"] else set()
    linked = {r: sum(bool(c & shared) for c in cs) for r, cs in comps.items()}
    degree = np.bincount(j.ravel(), minlength=n) if len(j) else np.zeros(n, dtype=int)
    return {
        "n_wires": int(n),
        "n_junctions": int(len(j)),
        "mean_degree": float(degree.mean()),
        "isolated_fraction": float(len(network.isolated_wires) / n),
        "largest_cluster_fraction": float(sizes.max() / n),
        "percolates": bool(shared),
        "in_linked": f"{linked['in']}/{len(comps['in'])}",
        "gnd_linked": f"{linked['gnd']}/{len(comps['gnd'])}",
    }


# Named geometries for RunSpec.geometry. Clusters sit at fixed positions (units of the substrate side).
_C4 = (Cluster(0.25, 0.25, 0.08), Cluster(0.75, 0.25, 0.08), Cluster(0.25, 0.75, 0.08), Cluster(0.75, 0.75, 0.08))
_LINE = (Cluster(0.2, 0.5, 0.08), Cluster(0.5, 0.5, 0.08), Cluster(0.8, 0.5, 0.08))
_CHAIN2 = tuple(Cluster(x, y, 0.06) for y in (0.25, 0.75) for x in (0.1, 0.3, 0.5, 0.7, 0.9))
_CHAIN1 = tuple(Cluster(x, 0.5, 0.08) for x in (0.1, 0.3, 0.5, 0.7, 0.9))
GEOMETRIES = {
    "uniform": GeometrySpec(),                                          # the framework network
    "clusters4": GeometrySpec(clusters=_C4, background=0.3),            # 4 dense clusters, 30 % uniform background
    "clusters4_dense": GeometrySpec(clusters=_C4, background=0.1),      # the same, 10 % background
    "clusters_line": GeometrySpec(clusters=_LINE, background=0.3),      # 3 clusters on the line between the bands
    "aligned_x": GeometrySpec(angle_mu=0.0, angle_sigma=0.3),           # wires along x (+-17 deg)
    "aligned_y": GeometrySpec(angle_mu=np.pi / 2, angle_sigma=0.3),     # wires along y, across the band axis
    # clusters4 with more uniform background: with split_bands, 30-40 % background percolates on 3 of 5 seeds,
    # 50 % and more on 5 of 5 (docs/N12_GEOMETRY.md, section 8)
    "clusters4_b50": GeometrySpec(clusters=_C4, background=0.5),        # 1000 uniform + 4 x 250 clustered
    "clusters4_b60": GeometrySpec(clusters=_C4, background=0.6),        # 1200 uniform + 4 x 200
    "clusters4_b70": GeometrySpec(clusters=_C4, background=0.7),        # 1400 uniform + 4 x 150
    # chains of 5 clusters from the left band to the right one (x = 0.1 ... 0.9): with 20 % background the current has
    # to cross cluster after cluster (stages in series); with split_bands both percolate on 5 of 5 seeds
    "cadenas2_b20": GeometrySpec(clusters=_CHAIN2, background=0.2),     # 2 chains (y = 0.25, 0.75), 160 wires/cluster
    "cadena1_b20": GeometrySpec(clusters=_CHAIN1, background=0.2),      # 1 central chain (y = 0.5), 320 wires/cluster
}
