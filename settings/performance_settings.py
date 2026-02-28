from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerformanceSettings:
    """Performance-related settings."""

    # "off" | "auto" | "fixed"
    parallel_evaluation_mode: str = "auto"

    # Used only when parallel_evaluation_mode == "fixed"
    number_of_workers: int = 2

    # If True, uses multiprocessing. If False, sequential evaluation is forced.
    # This is mainly for quick debugging.
    enable_parallel_evaluation: bool = True
