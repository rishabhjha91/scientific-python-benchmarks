"""Task 3 - Dominant eigenpair via power iteration.

Spec: given a symmetric matrix ``A`` with a unique dominant eigenvalue, return
``(lambda, v)`` where ``lambda`` is the largest-magnitude eigenvalue and ``v`` a
corresponding unit eigenvector. Correctness is checked by the residual
``||A v - lambda v||`` and the eigenvalue error against a library reference.
"""

from __future__ import annotations

import numpy as np

from ..verifier import CaseResult, VerifierReport

SPEC = {
    "title": "Dominant eigenpair via power iteration",
    "tolerance": 1e-4,
    "complexity": "O(k n^2) for k iterations.",
    "edge_cases": ["residual ||A v - lambda v|| must be small"],
}


def reference_solve(A):
    """Reference dominant eigenpair via a dense eigensolver."""
    A = np.asarray(A, dtype=float)
    vals, vecs = np.linalg.eig(A)
    idx = int(np.argmax(np.abs(vals)))
    lam = float(vals[idx].real)
    v = vecs[:, idx].real
    v = v / np.linalg.norm(v)
    return lam, v


def power_iteration(A, iters: int = 2000, tol: float = 1e-12):
    """Canonical power-iteration solution (used as the 'good' candidate)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng(0)
    v = rng.normal(size=n)
    v = v / np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = A @ v
        v_new = w / np.linalg.norm(w)
        lam = float(v_new @ A @ v_new)
        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            break
        v = v_new
    return lam, v


def _cases(seed: int = 0):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(4):
        n = int(rng.integers(3, 6))
        M = rng.normal(size=(n, n))
        A = M @ M.T  # symmetric positive-semidefinite -> real eigenvalues
        cases.append((f"sym_{i}", A))
    return cases


def verify(candidate_solve, tol: float = SPEC["tolerance"], seed: int = 0) -> dict:
    """Run ``candidate_solve(A) -> (lambda, v)`` and check eigenvalue + residual."""
    report = VerifierReport(task="power_iteration")
    for name, A in _cases(seed):
        lam_ref, _ = reference_solve(A)
        try:
            lam, v = candidate_solve(A)
            v = np.asarray(v, dtype=float)
            v = v / np.linalg.norm(v)
            eig_err = abs(abs(lam) - abs(lam_ref)) / (abs(lam_ref) + 1e-12)
            resid = float(np.linalg.norm(A @ v - lam * v))
            passed = eig_err <= tol and resid <= 1e-3
            report.results.append(CaseResult(name, passed, float(eig_err), f"residual={resid:.2e}"))
        except Exception as exc:  # noqa: BLE001
            report.results.append(CaseResult(name, False, float("inf"), f"raised {type(exc).__name__}"))
    return report.to_dict()
