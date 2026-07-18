import numpy as np

from scibench.tasks import power_iteration as pi


def test_canonical_power_iteration_passes():
    report = pi.verify(pi.power_iteration)
    assert report["pass_rate"] == 1.0


def test_random_vector_candidate_fails():
    def bad(A):
        n = A.shape[0]
        return 0.0, np.ones(n)

    report = pi.verify(bad)
    assert report["pass_rate"] < 1.0


def test_reference_residual_small():
    rng = np.random.default_rng(1)
    M = rng.normal(size=(5, 5))
    A = M @ M.T
    lam, v = pi.reference_solve(A)
    v = np.asarray(v)
    assert np.linalg.norm(A @ v - lam * v) < 1e-8
