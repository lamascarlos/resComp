"""Named parameter sets."""
from .dynamics import SimParams


def lamas2026_fig4(dt=1e-3):
    """Parameters that reproduce Lamas et al. (2026) Fig. 4 (N = 2000 and 3000;
    sine drive of amplitude 3.6 V, 50000 steps/cycle) in our replication.

    Returns (SimParams, run_kwargs) with run_kwargs = {"wire_diameter": ...} for
    run_simulation(). Values taken from the paper's Table 1: dt*P0 = 0.4,
    alpha = 1 V^-1, dt*P_decay = 0.01 (unit of I in Eq. 4 unstated; I_unit =
    1 mA, for which RESET contributes little under the Fig. 4 protocol; under a
    steady 0.75-2.75 V drive it is 8-23% of the relaxation rate, see
    docs/switching_model.md section 5).

    INFERRED, NOT STATED IN THE PAPER (to be confirmed with the authors):
      * V_th = 0.1 V (Table 1 prints 0.01 V; that gives an onset near 0.35 V
        instead of the paper's ~1.8 V),
      * wire diameter 90 nm (G_seg = pi (D/2)^2 / (rho d); reproduces the
        paper's peak currents within ~8%).
    See docs/lamas_questions_es.md and RESUME_FROM_HERE.md.
    """
    params = SimParams.from_step_probs(0.4, 0.01, dt=dt, alpha=1.0, V_th=0.1, I_unit=1e-3)
    return params, {"wire_diameter": 90e-9}
