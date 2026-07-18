import numpy as np

from scibench.verifier import VerifierReport, relative_error


def test_relative_error_zero():
    assert relative_error(np.ones(3), np.ones(3)) == 0.0


def test_relative_error_against_zero_vector():
    # exact is zero -> falls back to absolute norm of the difference
    assert relative_error(np.array([3.0, 4.0]), np.zeros(2)) == 5.0


def test_report_pass_rate():
    from scibench.verifier import CaseResult

    r = VerifierReport(task="demo")
    r.results = [CaseResult("a", True, 0.0), CaseResult("b", False, 1.0)]
    d = r.to_dict()
    assert d["total"] == 2
    assert d["passed"] == 1
    assert d["pass_rate"] == 0.5
