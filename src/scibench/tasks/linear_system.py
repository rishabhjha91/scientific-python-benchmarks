"""Task 1 - Dense linear system solve.

Spec: given a square matrix ``A`` and vector ``b``, return ``x`` such that
``A @ x = b``. A correct solution must be accurate on well-conditioned systems
and must *not* silently return a bogus answer on a singular matrix (it should
raise).
"""

from __future__ import annotations

import numpy as np

from ..verifier import CaseResult, VerifierReport, relative_error

SPEC = {
    "title": "Dense linear system solve (A x = b)",
    "tolerance": 1e-6,
    "complexity": "O(n^3) factorisation expected (e.g., LU).",
    "edge_cases": ["singular matrix must raise, not return a wrong answer"],
}


def reference_solve(A, b):
    """Reference: LU-based solve via NumPy (raises on singular ``A``)."""
    return np.linalg.solve(np.asarray(A, dtype=float), np.asarray(b, dtype=float))


def _cases(seed: int = 0):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(5):
        n = int(rng.integers(3, 8))
        A = rng.normal(size=(n, n)) + n * np.eye(n)  # diagonally dominant, well-conditioned
        x_true = rng.normal(size=n)
        cases.append((f"random_{i}", A, A @ x_true, x_true))
    return cases


def verify(candidate_solve, tol: float = SPEC["tolerance"], seed: int = 0) -> dict:
    """Run ``candidate_solve(A, b)`` against reference cases and edge cases."""
    report = VerifierReport(task="linear_system")

    for name, A, b, x_true in _cases(seed):
        try:
            x = candidate_solve(A, b)
            err = relative_error(x, x_true)
            report.results.append(CaseResult(name, err <= tol, err))
        except Exception as exc:  # noqa: BLE001
            report.results.append(CaseResult(name, False, float("inf"), f"raised {type(exc).__name__}"))

    # Edge case: singular matrix. Correct behaviour is to raise.
    A_sing = np.array([[1.0, 2.0], [2.0, 4.0]])
    b_sing = np.array([1.0, 2.0])
    try:
        candidate_solve(A_sing, b_sing)
        report.results.append(
            CaseResult("singular", False, float("inf"), "did not raise on singular matrix")
        )
    except Exception:  # noqa: BLE001
        report.results.append(CaseResult("singular", True, 0.0, "correctly raised on singular matrix"))

    return report.to_dict()
