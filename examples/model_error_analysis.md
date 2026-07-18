# Model error analysis (sample)

A short write-up of the failure modes these benchmarks are designed to expose
when scoring model-generated numerical code. The point of the pack is not just
"did the code run" but "is it numerically correct and safe on the hard cases".

## Task 1 - Linear system solve

- **Silent failure on singular input.** The most common and most dangerous
  failure: a model reaches for `np.linalg.lstsq` (or a hand-rolled normal-equation
  solve) which returns *an* answer for a singular matrix instead of raising. The
  verifier includes a rank-deficient system and requires the candidate to raise,
  so a plausible-looking `lstsq` solution is correctly marked as failing that case.
- **Conditioning.** Naive Gaussian elimination without pivoting drifts on
  moderately conditioned matrices; the `1e-6` relative-error tolerance catches it.

## Task 2 - Nonnegativity-constrained QP

- **Constraint dropped.** A model that solves the *unconstrained* stationarity
  condition (`x = -Q^{-1} c`) often returns a lower objective but with negative
  entries. The verifier checks feasibility (`x >= 0`) first, so an infeasible
  point is rejected even when its objective looks better.
- **Step size / divergence.** Projected gradient with too large a step diverges;
  scoring against the reference objective within tolerance catches non-convergence.

## Task 3 - Power iteration

- **Wrong eigenpair.** Returning a random or non-converged vector yields a large
  residual `||A v - lambda v||`; the verifier requires residual `< 1e-3`.
- **Sign / magnitude confusion.** Comparing `|lambda|` against the reference
  dominant eigenvalue avoids penalising a correct eigenvector reported with a
  flipped sign, while still catching genuinely wrong eigenvalues.

## How to score a model

Point each task's `verify(...)` at a model-produced function and read the
per-case `passed` / `error` fields. Aggregating `pass_rate` across tasks gives a
compact, reproducible discrimination signal between models.
