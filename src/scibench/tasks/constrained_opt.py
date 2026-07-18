"""Task 2 - Nonnegativity-constrained quadratic program.

Spec: minimise ``0.5 x^T Q x + c^T x`` subject to ``x >= 0`` for a symmetric
positive-definite ``Q``. A correct solution must be feasible (nonnegative) and
achieve an objective no worse than the reference within tolerance.
"""

from __future__ import annotations

import numpy as np

from ..verifier import CaseResult, VerifierReport

SPEC = {
    "title": "Nonnegativity-constrained quadratic program",
    "tolerance": 1e-3,
    "complexity": "Iterative (projected gradient) expected.",
    "edge_cases": ["solution must remain feasible (x >= 0)"],
}


def objective(Q, c, x) -> float:
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ Q @ x + c @ x)


def reference_solve(Q, c, iters: int = 5000):
    """Reference: projected gradient descent with step 1/L (L = largest eigenvalue)."""
    Q = np.asarray(Q, dtype=float)
    c = np.asarray(c, dtype=float)
    n = len(c)
    x = np.zeros(n)
    L = float(np.linalg.eigvalsh(Q).max())
    lr = 1.0 / L
    for _ in range(iters):
        grad = Q @ x + c
        x = np.maximum(0.0, x - lr * grad)
    return x


def _cases(seed: int = 0):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(4):
        n = int(rng.integers(3, 6))
        M = rng.normal(size=(n, n))
        Q = M @ M.T + np.eye(n)  # symmetric positive-definite
        c = rng.normal(size=n)
        cases.append((f"qp_{i}", Q, c))
    return cases


def verify(candidate_solve, tol: float = SPEC["tolerance"], seed: int = 0) -> dict:
    """Run ``candidate_solve(Q, c)`` and check feasibility and objective gap."""
    report = VerifierReport(task="constrained_opt")
    for name, Q, c in _cases(seed):
        x_ref = reference_solve(Q, c)
        try:
            x = np.asarray(candidate_solve(Q, c), dtype=float)
            feasible = bool(np.all(x >= -1e-6))
            gap = objective(Q, c, x) - objective(Q, c, x_ref)
            passed = feasible and gap <= tol
            report.results.append(
                CaseResult(name, passed, float(gap), "" if feasible else "infeasible (x has negatives)")
            )
        except Exception as exc:  # noqa: BLE001
            report.results.append(CaseResult(name, False, float("inf"), f"raised {type(exc).__name__}"))
    return report.to_dict()
