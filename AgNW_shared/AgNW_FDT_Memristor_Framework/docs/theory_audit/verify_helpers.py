"""
verify.py -- Symbolic (and fallback numeric) verification helpers for analytical
derivations, built on SymPy.

Purpose
-------
A derivation is only "checked" if its critical steps can be independently
confirmed. These helpers wrap the common verification patterns so that each step
in a derivation returns an explicit True / False / residual instead of an
assertion. Every function is deliberately conservative: it tries several
simplification strategies before declaring equality, and it never silently
reports success on an unevaluated expression.

Design principles
-----------------
- Prefer symbolic proof (difference simplifies to 0). Fall back to numeric
  spot-checks ONLY when symbolic closure fails, and label the result as numeric.
- Carry assumptions explicitly via sympy.symbols(..., positive=True, real=True,
  ...). Most spurious `False`/`None` results come from missing assumptions or
  branch cuts -- see references/pitfalls.md.
- Return structured results (VerifyResult) so a caller can report honestly.

Usage
-----
    from verify import check_equal, check_limit, check_series, check_numeric
    import sympy as sp
    x = sp.symbols('x', positive=True)
    check_equal(sp.sin(x)**2 + sp.cos(x)**2, 1)          # -> ok=True (symbolic)

Requires: sympy (pip install sympy).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import sympy as sp


@dataclass
class VerifyResult:
    """Structured verification outcome.

    ok:      True if the check passed, False if it demonstrably failed,
             None if inconclusive (symbolic engine could not decide).
    method:  'symbolic' | 'numeric' | 'inconclusive'
    detail:  human-readable explanation, safe to print into a report.
    residual: optional numeric residual (for numeric checks).
    """
    ok: bool | None
    method: str
    detail: str
    residual: float | None = None

    def __bool__(self) -> bool:  # so `if check_equal(...):` behaves sensibly
        return self.ok is True

    def __repr__(self) -> str:
        tag = {True: "PASS", False: "FAIL", None: "INCONCLUSIVE"}[self.ok]
        r = f", residual={self.residual:.3e}" if self.residual is not None else ""
        return f"<{tag} [{self.method}] {self.detail}{r}>"


# ----------------------------------------------------------------------------
# Symbolic equality
# ----------------------------------------------------------------------------
def _try_symbolic_zero(diff: sp.Expr) -> bool | None:
    """Return True/False if we can decide diff == 0, else None.

    Tries an escalating ladder of simplification strategies, because no single
    SymPy call closes every identity (trig, hyperbolic, radicals, factorials).
    """
    strategies = (
        lambda e: sp.simplify(e),
        lambda e: sp.expand(e),
        lambda e: sp.trigsimp(e),
        lambda e: sp.expand_trig(sp.simplify(e)),
        lambda e: sp.radsimp(e),
        lambda e: sp.powsimp(sp.simplify(e), force=True),
        lambda e: sp.combsimp(e),
        lambda e: sp.simplify(sp.rewrite(e, sp.exp)) if hasattr(e, "rewrite") else e,
        lambda e: sp.cancel(sp.together(e)),
    )
    for strat in strategies:
        try:
            s = strat(diff)
        except Exception:
            continue
        if s == 0:
            return True
        # A nonzero *numeric* constant is a genuine disproof.
        if s.is_number and s.is_zero is False:
            return False
    return None


def check_equal(lhs: Any, rhs: Any, assume_real: bool = True) -> VerifyResult:
    """Verify lhs == rhs by simplifying the difference to zero.

    Falls back to a numeric spot-check if symbolic simplification is
    inconclusive. Use symbols carrying assumptions (positive=True, etc.) for
    best results on identities involving radicals, logs, or powers.
    """
    lhs, rhs = sp.sympify(lhs), sp.sympify(rhs)
    diff = sp.nsimplify(lhs - rhs) if (lhs - rhs).free_symbols == set() else (lhs - rhs)
    verdict = _try_symbolic_zero(sp.expand(lhs - rhs))
    if verdict is True:
        return VerifyResult(True, "symbolic", "lhs - rhs simplifies to 0")
    if verdict is False:
        return VerifyResult(False, "symbolic",
                            f"lhs - rhs simplifies to a nonzero constant: {sp.simplify(lhs-rhs)}")
    # Inconclusive -> numeric fallback
    num = check_numeric(lhs, rhs)
    num.detail = "symbolic inconclusive; " + num.detail
    return num


# ----------------------------------------------------------------------------
# Numeric spot-check
# ----------------------------------------------------------------------------
def check_numeric(lhs: Any, rhs: Any, n_points: int = 40,
                  lo: float = 0.1, hi: float = 3.0, tol: float = 1e-9,
                  seed: int = 0) -> VerifyResult:
    """Evaluate lhs and rhs at random real points and compare.

    A cheap, honest fallback when symbolic proof is unavailable. Substitutes
    every free symbol with random values in (lo, hi) and checks |lhs-rhs| < tol
    in complex modulus (so it survives incidental complex intermediate values).
    Reports the max residual seen.
    """
    lhs, rhs = sp.sympify(lhs), sp.sympify(rhs)
    syms = sorted((lhs - rhs).free_symbols, key=lambda s: s.name)
    rng = random.Random(seed)
    max_res = 0.0
    checked = 0
    for _ in range(n_points):
        subs = {s: rng.uniform(lo, hi) for s in syms}
        try:
            lv = complex(sp.N(lhs.subs(subs)))
            rv = complex(sp.N(rhs.subs(subs)))
        except (TypeError, ValueError):
            continue
        if any(map(lambda z: z != z, (lv.real, rv.real))):  # NaN guard
            continue
        max_res = max(max_res, abs(lv - rv))
        checked += 1
    if checked == 0:
        return VerifyResult(None, "inconclusive",
                            "could not evaluate at any sample point")
    ok = max_res < tol
    return VerifyResult(ok, "numeric",
                        f"{checked} random points, max|lhs-rhs|={max_res:.3e} "
                        f"(tol={tol:.0e})", residual=max_res)


# ----------------------------------------------------------------------------
# Limits
# ----------------------------------------------------------------------------
def check_limit(expr: Any, var: sp.Symbol, point: Any, expected: Any,
                direction: str = "+") -> VerifyResult:
    """Verify that lim_{var -> point} expr == expected."""
    expr, expected = sp.sympify(expr), sp.sympify(expected)
    try:
        got = sp.limit(expr, var, point, dir=direction)
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(None, "inconclusive", f"sympy.limit failed: {exc}")
    eq = _try_symbolic_zero(got - expected)
    if eq is True:
        return VerifyResult(True, "symbolic", f"limit = {got} = expected")
    if eq is False:
        return VerifyResult(False, "symbolic", f"limit = {got} != expected {expected}")
    return VerifyResult(None, "inconclusive",
                        f"limit = {got}, could not confirm equality to {expected}")


# ----------------------------------------------------------------------------
# Series / asymptotic expansion
# ----------------------------------------------------------------------------
def check_series(expr: Any, var: sp.Symbol, point: Any, order: int,
                 expected: Any) -> VerifyResult:
    """Verify the truncated series of expr about `point` equals `expected`.

    `expected` should be the series WITHOUT the O(...) term (i.e. removeO()).
    """
    expr, expected = sp.sympify(expr), sp.sympify(expected)
    try:
        got = sp.series(expr, var, point, order).removeO()
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(None, "inconclusive", f"sympy.series failed: {exc}")
    eq = _try_symbolic_zero(sp.expand(got - expected))
    if eq is True:
        return VerifyResult(True, "symbolic", f"series to O({var}^{order}) matches")
    if eq is False:
        return VerifyResult(False, "symbolic",
                            f"series mismatch: got {got}, expected {expected}")
    return check_numeric(got, expected)


# ----------------------------------------------------------------------------
# ODE / PDE residual
# ----------------------------------------------------------------------------
def check_ode_solution(lhs_minus_rhs: Any) -> VerifyResult:
    """Confirm a candidate solution satisfies an ODE/PDE.

    Pass the fully-substituted residual (LHS - RHS of the equation with the
    candidate solution and its derivatives inserted). Passes iff residual == 0.
    """
    residual = sp.sympify(lhs_minus_rhs)
    verdict = _try_symbolic_zero(residual)
    if verdict is True:
        return VerifyResult(True, "symbolic", "residual simplifies to 0")
    if verdict is False:
        return VerifyResult(False, "symbolic",
                            f"nonzero residual: {sp.simplify(residual)}")
    return check_numeric(residual, 0)


# ----------------------------------------------------------------------------
# Dimensional analysis
# ----------------------------------------------------------------------------
def check_dimensions(expr, expected_dim) -> VerifyResult:
    """Verify the physical dimension of `expr` equals `expected_dim`.

    Uses sympy.physics.units. `expr` should be built from unit-carrying
    quantities; `expected_dim` is a Dimension (e.g. from
    sympy.physics.units.systems.si). This catches the classic 'dropped a factor
    of c / hbar / 2 pi' errors that pure symbolic algebra will not.
    """
    try:
        from sympy.physics.units import Dimension
        from sympy.physics.units.systems.si import dimsys_SI
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(None, "inconclusive", f"units subsystem unavailable: {exc}")
    try:
        got = Dimension(expr)
        same = dimsys_SI.equivalent_dims(got, Dimension(expected_dim))
    except Exception as exc:  # noqa: BLE001
        return VerifyResult(None, "inconclusive", f"dimension check failed: {exc}")
    return VerifyResult(bool(same), "symbolic",
                        f"dimension {'matches' if same else 'MISMATCH'}: "
                        f"got {got}, expected {expected_dim}")


# ----------------------------------------------------------------------------
# Batch reporting
# ----------------------------------------------------------------------------
def report(checks: Sequence[tuple[str, VerifyResult]]) -> str:
    """Format a list of (label, VerifyResult) into a Markdown-friendly block."""
    lines = ["| step | result | method | detail |",
             "|------|--------|--------|--------|"]
    for label, res in checks:
        tag = {True: "PASS", False: "**FAIL**", None: "inconclusive"}[res.ok]
        lines.append(f"| {label} | {tag} | {res.method} | {res.detail} |")
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test: a handful of identities that should pass, and one that fails.
    x = sp.symbols("x", positive=True)
    a = sp.symbols("a", positive=True)
    demo = [
        ("Pythagorean", check_equal(sp.sin(x)**2 + sp.cos(x)**2, 1)),
        ("log product", check_equal(sp.log(a*x), sp.log(a) + sp.log(x))),
        ("geometric limit", check_limit((1 - sp.cos(x))/x**2, x, 0, sp.Rational(1, 2))),
        ("exp series", check_series(sp.exp(x), x, 0, 4,
                                    1 + x + x**2/2 + x**3/6)),
        ("intended failure", check_equal(sp.sin(x), sp.cos(x))),
    ]
    print(report(demo))
