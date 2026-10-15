#!/usr/bin/env python3
"""Slim the local raw runs after archiving them (2026-10-08, Juan): drop the per-input junction states.

The full raw files (with junction states) are archived on the external drive TOSHIBA_ntfs, directory
AgNW_Reservoir_Computing_archive/raw_simulation_runs_with_junction_states_2026-10-08/raw/ (README.txt there). The
junction states are ~95 % of the size of a raw file and feed only the oracle "junctions" readout and a few model checks;
every electrode / probe / current analysis keeps working on the slim files (agnw_rc.evaluate skips the "junctions"
readout when the states are absent). To rescore the junction readout, point a script's RAW directory at the archive.

Safety: a local file is rewritten only if the archived copy exists and has the same SHA-256 as the local file; the
slim file is written to a temporary name and renamed. Dropped keys: junction_x, junction_y (N10), junction_p
(mean-field); the small binary flags / filament states are kept. A key "junction_states_archived" names the archive.

    python scripts/slim_raw.py --archive <path to .../raw> [--dry-run]   -> results/slim_raw_log.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
DROP = ("junction_x", "junction_y", "junction_p")
ARCHIVE_NAME = "TOSHIBA_ntfs:AgNW_Reservoir_Computing_archive/raw_simulation_runs_with_junction_states_2026-10-08/raw"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(archive: Path, dry: bool):
    files = sorted(f for f in RAW.glob("*.npz") if not f.name.endswith(".tmp.npz"))
    log = {"archive": ARCHIVE_NAME, "slimmed": [], "skipped_no_states": 0, "refused": [], "bytes_before": 0,
           "bytes_after": 0}
    for i, f in enumerate(files, 1):
        with np.load(f) as z:
            keys = z.files
            if not any(k in keys for k in DROP):
                log["skipped_no_states"] += 1
                continue
        a = archive / f.name
        if not a.exists() or a.stat().st_size != f.stat().st_size or sha256(a) != sha256(f):
            log["refused"].append(f.name)                       # no verified archive copy: leave untouched
            continue
        before = f.stat().st_size
        if not dry:
            with np.load(f, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files if k not in DROP}
            data["junction_states_archived"] = np.array(ARCHIVE_NAME)
            tmp = f.with_suffix(".slim.tmp.npz")
            np.savez_compressed(tmp, **data)
            os.replace(tmp, f)
        log["slimmed"].append(f.name)
        log["bytes_before"] += before
        log["bytes_after"] += f.stat().st_size if not dry else 0
        if i % 100 == 0:
            print(f"{i}/{len(files)} ... slimmed {len(log['slimmed'])}, refused {len(log['refused'])}", flush=True)
    (ROOT / "results" / ("slim_raw_dryrun.json" if dry else "slim_raw_log.json")).write_text(json.dumps(log, indent=1))
    print(f"slimmed {len(log['slimmed'])} files ({log['bytes_before'] / 1e9:.1f} GB -> {log['bytes_after'] / 1e9:.1f} GB), "
          f"no states {log['skipped_no_states']}, refused (no verified archive copy) {len(log['refused'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.archive, a.dry_run)
