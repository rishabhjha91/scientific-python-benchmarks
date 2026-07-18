import numpy as np

from scibench.tasks import constrained_opt as co


def test_reference_passes_all_cases():
    report = co.verify(co.reference_solve)
    assert report["pass_rate"] == 1.0


def test_infeasible_candidate_is_rejected():
    # Ignores the x >= 0 constraint -> should be flagged infeasible on some case.
    def unconstrained(Q, c):
        return np.linalg.solve(np.asarray(Q, float), -np.asarray(c, float))

    report = co.verify(unconstrained)
    assert report["pass_rate"] < 1.0


def test_objective_helper():
    Q = np.eye(2)
    c = np.zeros(2)
    assert co.objective(Q, c, np.array([1.0, 0.0])) == 0.5
