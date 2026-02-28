from __future__ import annotations

from typing import List, Tuple, Optional
import logging
import random
import numpy as np

from acoustic.arrival_time import calculate_arrival_time
from acoustic.sound_speed_profile import SoundSpeedProfile
from genetic_algorithm.chromosome import chromosome_converter
from genetic_algorithm.penalty import calculate_penalty_for_separation_between_sensors
from geometry.distance import calculate_distance_2d
from geometry.grid_geometry import GridGeometry
from localization.mle_estimator import estimate_impact_position
from models.model import EvaluationReport
from settings.environment_settings import EnvironmentSettings
from settings.simulation_settings import SimulationSettings

logger = logging.getLogger("underwater_sensor_ga.evaluation")


def evaluate_chromosome_with_report(
    chromosome: np.ndarray,
    number_of_sensors: int,
    grid_geometry: GridGeometry,
    environment_settings: EnvironmentSettings,
    simulation_settings: SimulationSettings,
    sound_speed_profile: SoundSpeedProfile,
    random_generator: random.Random,
    impact_points: Optional[List[Tuple[float, float]]] = None,
) -> EvaluationReport:
    sensors = chromosome_converter(chromosome, number_of_sensors, grid_geometry, environment_settings)

    # Vectorized sensor coordinates (fast distance checks)
    sx = np.array([s.position_x for s in sensors], dtype=float)
    sy = np.array([s.position_y for s in sensors], dtype=float)

    separation_penalty = calculate_penalty_for_separation_between_sensors(
        sensors, simulation_settings.minimum_distance_between_buoys_in_meters
    )

    if impact_points is None:
        raise ValueError(
            "impact_points must be provided for consistent evaluation "
            "(expected from evaluate_population)."
        )

    number_of_impacts = len(impact_points)

    coverage_penalty = 0.0
    impacts_without_coverage = 0

    # We will compute both:
    # - raw error (meters): physical, audit-friendly
    # - excess error over tolerance: enforces the tolerance requirement
    raw_errors: List[float] = []

    max_detect = float(environment_settings.maximum_detection_distance)
    k = max(3, int(simulation_settings.max_sensors_per_impact))
    tol = float(simulation_settings.localization_tolerance_meters)  # set to 5.0 in config

    for (impact_x, impact_y) in impact_points:
        # Vectorized distances to decide who "detects"
        dx = sx - impact_x
        dy = sy - impact_y
        dist2 = np.sqrt(dx * dx + dy * dy)

        detected_idx = np.where(dist2 <= max_detect)[0]

        # Need at least 3 sensors for TDOA/MLE
        if detected_idx.size < 3:
            impacts_without_coverage += 1
            coverage_penalty += float(simulation_settings.invalid_coverage_penalty)
            continue

        # Limit to k nearest sensors (reduces estimator cost)
        if detected_idx.size > k:
            nearest_order = np.argsort(dist2[detected_idx])[:k]
            detected_idx = detected_idx[nearest_order]

        sensors_that_detected = [sensors[j] for j in detected_idx.tolist()]

        theoretical_times = np.array(
            [
                calculate_arrival_time(sensor, impact_x, impact_y, sound_speed_profile)
                for sensor in sensors_that_detected
            ],
            dtype=float,
        )

        noise = np.array(
            [
                random_generator.gauss(0.0, simulation_settings.time_noise_standard_deviation)
                for _ in range(len(sensors_that_detected))
            ],
            dtype=float,
        )

        # emission_time=0
        observed_times = theoretical_times + noise

        est_x, est_y = estimate_impact_position(
            sensors_that_detected,
            observed_times,
            grid_geometry,
            sound_speed_profile,
            simulation_settings,
        )

        raw_error = float(calculate_distance_2d(est_x, est_y, impact_x, impact_y))
        raw_errors.append(raw_error)

    # If nothing was localizable, this chromosome is unusable
    if len(raw_errors) == 0:
        total_cost = float(
            number_of_impacts * float(simulation_settings.invalid_coverage_penalty)
            + float(simulation_settings.penalty_for_buoys_too_close) * float(separation_penalty)
        )
        return EvaluationReport(
            total_cost=total_cost,
            mean_localization_error_meters=float("inf"),
            number_of_impacts=number_of_impacts,
            number_of_localizable_impacts=0,
            number_of_impacts_without_coverage=number_of_impacts,
            coverage_penalty=float(number_of_impacts) * float(simulation_settings.invalid_coverage_penalty),
            separation_penalty=float(separation_penalty),
            impact_points=impact_points,
        )

    raw_np = np.array(raw_errors, dtype=float)

    mean_raw_error = float(np.mean(raw_np))

    # Excess over tolerance (0 if <= tol)
    excess = np.maximum(0.0, raw_np - tol)
    mean_excess = float(np.mean(excess))

    # Final cost:
    # - always encourages improving raw accuracy (mean_raw_error)
    # - additionally punishes being worse than tolerance (mean_excess)
    # - plus penalties
    total_cost = (
        mean_raw_error
        + 0.5 * mean_excess
        + float(simulation_settings.penalty_for_buoys_too_close) * float(separation_penalty)
        + float(coverage_penalty)
    )

    return EvaluationReport(
        total_cost=float(total_cost),
        # IMPORTANT: this is the *raw* mean error in meters (audit-friendly)
        mean_localization_error_meters=mean_raw_error,
        number_of_impacts=number_of_impacts,
        number_of_localizable_impacts=len(raw_errors),
        number_of_impacts_without_coverage=impacts_without_coverage,
        coverage_penalty=float(coverage_penalty),
        separation_penalty=float(separation_penalty),
        impact_points=impact_points,
    )