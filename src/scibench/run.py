"""Run every task verifier against its reference/canonical solution and print JSON.

This doubles as a smoke test: a healthy pack reports pass_rate == 1.0 for all
tasks. Point the verifiers at a model-generated function to score that model
instead.
"""

from __future__ import annotations

import json

from .tasks import constrained_opt, linear_system, power_iteration


def run_all() -> dict:
    return {
        "linear_system": linear_system.verify(linear_system.reference_solve),
        "constrained_opt": constrained_opt.verify(constrained_opt.reference_solve),
        "power_iteration": power_iteration.verify(power_iteration.power_iteration),
    }


def main() -> None:
    print(json.dumps(run_all(), indent=2))


if __name__ == "__main__":
    main()
