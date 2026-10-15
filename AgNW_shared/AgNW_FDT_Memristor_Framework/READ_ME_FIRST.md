# AgNW_FDT_Memristor_Framework: reduced copy (2026-10-15)

Snapshot of commit `ffba009` for use with `AgNW_Reservoir_Computing_repo`: the `agnw_sim` package, tests, scripts,
README, `docs/switching_model.md`, `docs/theory_audit/` and one trace of `validation_output/lamas_fig4/` (read by the
reservoir project). Not included: the other validation output (1.2 GB), reports, the experimental-planning material
and the git history. Rename this folder to `AgNW_FDT_Memristor_Framework` and install it as described in
`AgNW_Reservoir_Computing_repo/READ_ME_FIRST.md`.

Tests (checked 2026-10-15 in a fresh environment): 289 pass. Two tests in `tests/test_io.py` check the git commit hash
and fail here because the copy has no git history (expected, harmless). `tests/test_experiment.py` needs pandas
(`pip install pandas`), which `pyproject.toml` does not list; it covers a module the reservoir project does not use.
