from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationSettings:
    number_of_impact_points_per_evaluation: int = 30
    time_noise_standard_deviation: float = 0.0005  # 0.5 ms

    coarse_search_step_in_meters: float = 20.0
    fine_search_step_in_meters: float = 6.0
    refinement_radius_in_meters: float = 80.0

    invalid_coverage_penalty: float = 10
    penalty_for_buoys_too_close: float = 1.0
    minimum_distance_between_buoys_in_meters: float = 100.0

    # NEW: stable scenario seed for impact sampling (independent of population/generation)
    global_seed: int = 123

    # NEW: cap number of sensors used per impact in localization (big speed-up)
    max_sensors_per_impact: int = 5

    # tolerance in meters (errors <= tol don't contribute to cost)
    localization_tolerance_meters: float = 5.0
