"""Shared verification utilities for benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def relative_error(approx, exact, ord=None) -> float:
    """Return the relative error ||approx - exact|| / ||exact|| (abs if exact==0)."""
    approx = np.asarray(approx, dtype=float)
    exact = np.asarray(exact, dtype=float)
    denom = float(np.linalg.norm(exact, ord))
    diff = float(np.linalg.norm(approx - exact, ord))
    return diff / denom if denom > 0 else diff


@dataclass
class CaseResult:
    name: str
    passed: bool
    error: float
    detail: str = ""


@dataclass
class VerifierReport:
    task: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        return round(self.passed / self.total, 3) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "cases": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "error": (None if r.error == float("inf") else round(r.error, 8)),
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }
