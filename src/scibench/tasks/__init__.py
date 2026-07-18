"""Benchmark tasks: each exposes a reference solution and a ``verify`` function."""

from . import constrained_opt, linear_system, power_iteration

TASKS = {
    "linear_system": linear_system,
    "constrained_opt": constrained_opt,
    "power_iteration": power_iteration,
}

__all__ = ["TASKS", "linear_system", "constrained_opt", "power_iteration"]
