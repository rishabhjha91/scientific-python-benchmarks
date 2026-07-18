# Scientific Python Benchmark Pack

[![CI](https://github.com/rishabhjha91/scientific-python-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/rishabhjha91/scientific-python-benchmarks/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A small, self-contained pack of **scientific-computing coding benchmarks** used to
evaluate model-generated numerical code. Each task ships a fixed specification, a
reference solution, numerical tolerances, edge cases, and an **automated verifier**
that returns per-case pass/fail plus error metrics - the same structure used for
research-grade coding-agent evaluation.

It is deliberately small and reproducible: no proprietary tasks, no network, no
hidden state. The tasks are built from standard numerical-methods concepts.

## Tasks

| Task | Problem | What it checks |
| --- | --- | --- |
| `linear_system` | Solve `A x = b` | Accuracy on well-conditioned systems; **must raise on singular `A`** rather than silently returning a wrong answer |
| `constrained_opt` | Minimise `0.5 xᵀQx + cᵀx` s.t. `x ≥ 0` | Feasibility and objective gap vs. a projected-gradient reference |
| `power_iteration` | Dominant eigenpair of symmetric `A` | Eigenvalue error and residual `‖A v − λ v‖` |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

Score the reference/canonical solutions (a healthy pack reports `pass_rate == 1.0`):

```bash
python -m scibench.run
```

Score a model-generated function by pointing a verifier at it:

```python
from scibench.tasks import linear_system

def model_solution(A, b):
    ...  # candidate code produced by a model

report = linear_system.verify(model_solution)
print(report["pass_rate"], report["cases"])
```

## Test

```bash
ruff check .
pytest
```

Tests confirm each reference solution passes its own verifier and that
deliberately wrong candidates (e.g., dropping a constraint, using `lstsq` on a
singular system) are correctly failed.

## Why this design

The value of a benchmark is its ability to **discriminate** between a correct
solution and a plausible-but-wrong one. Every task therefore includes a failure
mode that superficially "works" but is numerically unsafe - see
[`examples/model_error_analysis.md`](examples/model_error_analysis.md).

## Layout

```
src/scibench/
  verifier.py            # tolerances, case results, report aggregation
  run.py                 # run all verifiers on reference solutions
  tasks/
    linear_system.py
    constrained_opt.py
    power_iteration.py
tests/                   # pytest: references pass, wrong candidates fail
examples/                # sample model-error analysis
```

## License

MIT
