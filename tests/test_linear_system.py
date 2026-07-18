import numpy as np

from scibench.tasks import linear_system as ls


def test_reference_passes_all_cases():
    report = ls.verify(ls.reference_solve)
    assert report["pass_rate"] == 1.0
    assert report["passed"] == report["total"]


def test_wrong_candidate_fails():
    def bad_solve(A, b):
        return np.zeros_like(np.asarray(b, dtype=float))

    report = ls.verify(bad_solve)
    assert report["pass_rate"] < 1.0


def test_lstsq_candidate_flagged_on_singular():
    # lstsq silently "solves" a singular system, which the verifier must catch.
    def lstsq_solve(A, b):
        return np.linalg.lstsq(np.asarray(A, float), np.asarray(b, float), rcond=None)[0]

    report = ls.verify(lstsq_solve)
    singular = next(c for c in report["cases"] if c["name"] == "singular")
    assert singular["passed"] is False
