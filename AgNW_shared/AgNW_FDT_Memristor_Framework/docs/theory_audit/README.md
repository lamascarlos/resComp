# docs/theory_audit

Audit of the theory document and the replication analysis of Lamas et al. (2026) Fig. 4.

| file | purpose |
|---|---|
| `THEORY_AUDIT.md` | findings F1-F13, corrected derivations, verification tables (§6: spontaneous relaxation) |
| `verify_symbolic.py` | SymPy checks (51). Run with a Python that has sympy: `python3 docs/theory_audit/verify_symbolic.py` |
| `verify_mc.py` | Monte Carlo vs the package's `update_dynamics` (two-state junction). `.venv/bin/python ...` |
| `verify_relaxation.py` | exact + Monte Carlo audit of the exponential and fixed-lifetime relaxation mechanisms |
| `verify_helpers.py` | copy of the math-derivation skill's SymPy helpers |
| `results_*.txt` | saved outputs of the three verification scripts |
| `digitize_lamas_fig4.py`, `compare_fig4_metrics.py`, `fig4_metrics.json` | digitized paper Fig. 4 and like-for-like envelope metrics |
| `plot_fig4_gallery.py`, `plot_fixed_v.py` | figure galleries (simulation vs paper) and fixed-V long-run overview |
| `relaxation_ratio.png` | response/correlation ratios of the relaxation mechanisms |

**Paper figure crops are not in git** (third-party content). To regenerate them from `biblio/` (the paper PDF is not in git either):

```bash
mkdir -p docs/theory_audit/paper_fig4
pdfimages -png -f 7 -l 7 "biblio/Lamas 2026 - Stochastic Modeling of Silver Nanowire Networks for Neuromorphic Computing.pdf" docs/theory_audit/paper_fig4/p7
rm docs/theory_audit/paper_fig4/p7-001.png docs/theory_audit/paper_fig4/p7-003.png docs/theory_audit/paper_fig4/p7-005.png   # soft masks
```

`p7-002.png` is the N=2000 panel and `p7-004.png` the N=3000 panel. Simulation outputs go to `validation_output/` (ignored);
the drivers are in `scripts/` (`replicate_lamas_fig4.py`, `fixed_v_stationarity.py`, `explore_*.py`).
