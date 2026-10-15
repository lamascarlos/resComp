"""HDF5 trajectory storage."""
import dataclasses
import json
import os
import subprocess
import h5py
import numpy as np

from .geometry import Network


def _git_commit_hash():
    """Current git commit hash, or "" if unavailable (e.g. not a repo)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip()
    except Exception:
        return ""


def save_result(path, result):
    """Write a SimulationResult to an HDF5 file (schema per sub-project 1
    spec §7, extended in sub-project 2 spec §4.2)."""
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        for k, v in dataclasses.asdict(result.params).items():
            meta.attrs[k] = v
        meta.attrs["n_wires"] = result.network.n_wires
        meta.attrs["area"] = result.network.area
        meta.attrs["wire_length"] = result.network.wire_length
        meta.attrs["initial_rng_state_json"] = json.dumps(result.initial_rng_state)
        meta.attrs["git_commit_hash"] = _git_commit_hash()
        meta.attrs["n_junctions"] = result.network.junctions.shape[0]
        meta.attrs["n_steps"] = len(result.t)
        meta.attrs["band_frac"] = result.band_frac
        # NaN encodes "ideal wires" (wire-as-node); HDF5 attributes cannot hold None.
        meta.attrs["wire_diameter"] = (float("nan") if result.wire_diameter is None
                                       else result.wire_diameter)

        net = f.create_group("network")
        net.create_dataset("wire_endpoints", data=result.network.wire_endpoints)
        net.create_dataset("junctions", data=result.network.junctions)
        net.create_dataset("junction_pos", data=result.network.junction_pos)
        net.create_dataset("electrode_in", data=result.network.electrode_in)
        net.create_dataset("electrode_gnd", data=result.network.electrode_gnd)
        net.create_dataset("isolated_wires", data=result.network.isolated_wires)

        traj = f.create_group("trajectory")
        traj.create_dataset("t", data=result.t)
        traj.create_dataset("V_in", data=result.V_in)
        traj.create_dataset("I_total", data=result.I_total)
        traj.create_dataset("G_sample", data=result.G_sample)
        traj.create_dataset("sampled_junction_ids", data=result.sampled_junction_ids)


def load_result(path):
    """Load a saved simulation into a plain dict of arrays and metadata."""
    with h5py.File(path, "r") as f:
        return {
            "metadata": dict(f["metadata"].attrs),
            "wire_endpoints": f["network/wire_endpoints"][:],
            "junctions": f["network/junctions"][:],
            "junction_pos": f["network/junction_pos"][:],
            "electrode_in": f["network/electrode_in"][:],
            "electrode_gnd": f["network/electrode_gnd"][:],
            "isolated_wires": f["network/isolated_wires"][:],
            "t": f["trajectory/t"][:],
            "V_in": f["trajectory/V_in"][:],
            "I_total": f["trajectory/I_total"][:],
            "G_sample": f["trajectory/G_sample"][:],
            "sampled_junction_ids": f["trajectory/sampled_junction_ids"][:],
        }


def reconstruct_network(data):
    """Build a full Network object from a dict returned by load_result."""
    return Network(
        n_wires=int(data["metadata"]["n_wires"]),
        wire_endpoints=data["wire_endpoints"],
        junctions=data["junctions"],
        junction_pos=data["junction_pos"],
        electrode_in=data["electrode_in"],
        electrode_gnd=data["electrode_gnd"],
        isolated_wires=data["isolated_wires"],
        area=float(data["metadata"]["area"]),
        wire_length=float(data["metadata"]["wire_length"]),
    )


def restore_rng(state_json):
    """Reconstruct a numpy Generator from a JSON-encoded bit_generator
    state, as saved in metadata['initial_rng_state_json']."""
    state = json.loads(state_json)
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng
