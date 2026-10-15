# AgNW Reservoir Computing: shared copy (2026-10-15)

A reduced copy of the project: the code, the documentation, the summarised results and the figures, without the raw
simulation runs (about 11 GB), the run logs and the working notes. It is enough to understand the models, reproduce
the analyses from the saved summaries and run new simulations.

## What to read, in order

1. `docs/summary_2026-10-13/summary.md` (+ pdf): one-page overview of the question, the results and the open problems.
2. `README.md`: project overview, experiment names, how to run.
3. `docs/RESULTS.md` (+ pdf): every experiment with its numbers, figures and conclusions. Start with "Answer in short"
   and "Evaluation criteria (G0, G1, G2)".
4. `docs/MANUAL.md` (+ pdf): the models (network, junction models, electrodes, input encoding, readout, scoring) and the
   code, module by module.
5. Design documents as needed: `docs/N10_DESIGN.md` (the volatile junction used in the main results),
   `docs/SWITCHING_MECHANISMS.md`, `docs/N4_DESIGN.md`, `docs/PLAN_N11_CALIBRATION.md` (calibration on measured data, in
   progress), `docs/PLAN_SPLIT_BANDS.md`, `docs/PLAN_N10L_SEEDS.md`, `docs/PLAN.md` (original design),
   `docs/LITERATURE_NOTES.md`, `biblio/references.md`.
6. `docs/report_2026-10-13/report.md`: the latest report (network size, edge-electrode layout).

## Contents

| path | what |
|---|---|
| `agnw_rc/` | the project package (junction models, multi-electrode circuit, drive, readout, tasks, scoring, N11 calibration) |
| `scripts/` | one entry point per experiment (`run_sims.py`, `n10*_*.py`, `n11_*.py`, ...); each writes `results/*.json` and figures |
| `tests/` | pytest suite |
| `cluster/` | SLURM job scripts (CCT-Rosario cluster; `docs/CLUSTER_TUTORIAL.md`, replace `<user>` with your account) |
| `results/` | all scores and criteria (`*.json`, `summary.csv`, `metrics.json`); **no raw runs** |
| `figures/` | every figure in RESULTS and the reports; `figures/README.md` says which script makes each one |
| `data/milano2025/` | measured data of Milano et al. 2025, *Nat. Commun.* (Zenodo 10.5281/zenodo.15050217, CC-BY), used by N10-C and N11 |
| `docs/`, `biblio/references.md` | documentation, as listed above (no PDFs of papers: see the DOIs in references.md) |

## Setting up

The network itself (wire geometry, Kirchhoff circuit, framework junction models) comes from the separate package
**AgNW_FDT_Memristor_Framework** (`agnw_sim`), which must sit next to this directory. The results were made with framework commit `f6b6951`; later commits (up to
`ffba009`, 2026-10-15) only add two new modules (`agnw_sim/experiment.py`, `agnw_sim/reservoir.py`) and leave the code
used here unchanged.

A reduced copy of the framework is shared with this one as `AgNW_FDT_Memristor_Framework/` (in the zip; `_min` on disk) (code, tests, scripts,
README, the theory audit and the one Lamas Fig. 4 trace that this project reads; snapshot of commit `ffba009`, without
its 1.2 GB of validation output). **Rename it to `AgNW_FDT_Memristor_Framework`** and place it next to this directory:

```
some_folder/
  AgNW_FDT_Memristor_Framework/     # the framework copy, renamed
  AgNW_Reservoir_Computing_repo/    # this copy
```

```bash
cd AgNW_Reservoir_Computing_repo
python3 -m venv .venv                                  # Python >= 3.10 (3.14 used)
.venv/bin/pip install -e ../AgNW_FDT_Memristor_Framework
.venv/bin/pip install -e ".[dev]"                      # numpy, scipy, matplotlib, pytest
.venv/bin/python -m pytest -q                          # full suite, about 7 min
```

Versions used: Python 3.14.7, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1. Parallel scripts set
`OMP_NUM_THREADS=1` (one BLAS thread per process); keep it that way.

## Not included, and what that means

* **Raw runs** (`results/raw/`, about 11 GB). Every number and figure is already in `results/` and `figures/`, but
  re-scoring an existing run needs its raw file; new simulations create their own. Raw files are available on request.
* **Milano et al. 2022 measured trace** (`data/milano2022/`, used by `scripts/milano2022_relax.py`): the authors'
  repository has no license, so it is not redistributed. Get it from GitHub
  `MilanoGianluca/Nanowire_Network_Reservoir_Computing`, file `Fig_2/raw_data/pulse_2V.txt`, into
  `data/milano2022/pulse_2V.txt`.
* Working notes (session log `docs/NEXT_STEPS.md`, resume notes, older reports, internal review). Some documents still
  point to them; the content that matters is in RESULTS and MANUAL.
* The superseded prototype `legacy/` (its numbers are invalid).

Contact: Juan Monti (IFIR, CONICET-UNR).
