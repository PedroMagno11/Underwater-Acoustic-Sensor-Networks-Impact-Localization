from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from evaluation.cost_function import evaluate_chromosome_with_report
from impacts.impact_point import ImpactPosition
from models.model import EvaluationReport, ImpactCallback
from settings.environment_settings import EnvironmentSettings
from settings.simulation_settings import SimulationSettings
from settings.performance_settings import PerformanceSettings
from acoustic.sound_speed_profile import SoundSpeedProfile
from geometry.grid_geometry import GridGeometry
from utils.seeding import compute_job_seed, compute_generation_impact_seed


@dataclass(frozen=True)
class EvaluationJob:
    chromosome: np.ndarray
    chromosome_index: int
    generation_index: int
    global_seed: int
    impact_points: List[Tuple[float, float]]


def _generate_impacts_for_generation(
    *,
    grid_geometry: GridGeometry,
    number_of_impacts: int,
    impact_seed: int,
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(number_of_impacts):
        x, y = ImpactPosition.generate_impact_position(
            grid_geometry=grid_geometry,
            global_seed=impact_seed,
            target_index=i,
        )
        pts.append((x, y))
    return pts


def _resolve_number_of_workers(performance_settings: PerformanceSettings) -> int:
    if not performance_settings.enable_parallel_evaluation:
        return 1

    mode = (performance_settings.parallel_evaluation_mode or "auto").lower().strip()
    if mode == "off":
        return 1

    cpu_count = os.cpu_count() or 2

    if mode == "auto":
        return max(1, cpu_count - 1)

    if mode == "fixed":
        return max(1, int(performance_settings.number_of_workers))

    return max(1, cpu_count - 1)


def _evaluate_job(
    job: EvaluationJob,
    number_of_sensors: int,
    environment_settings: EnvironmentSettings,
    simulation_settings: SimulationSettings,
    sound_speed_profile: SoundSpeedProfile,
) -> Tuple[int, EvaluationReport]:
    seed = compute_job_seed(job.global_seed, job.generation_index, job.chromosome_index)
    rng = random.Random(seed)

    grid_geometry = GridGeometry(environment_settings)

    report = evaluate_chromosome_with_report(
        chromosome=job.chromosome,
        number_of_sensors=number_of_sensors,
        grid_geometry=grid_geometry,
        environment_settings=environment_settings,
        simulation_settings=simulation_settings,
        sound_speed_profile=sound_speed_profile,
        random_generator=rng,
        impact_points=job.impact_points,
    )
    return job.chromosome_index, report


def _run_jobs_in_executor(
    executor: ProcessPoolExecutor,
    jobs: List[EvaluationJob],
    reports: List[EvaluationReport],
    number_of_sensors: int,
    environment_settings: EnvironmentSettings,
    simulation_settings: SimulationSettings,
    sound_speed_profile: SoundSpeedProfile,
) -> None:
    futures = [
        executor.submit(
            _evaluate_job,
            job,
            number_of_sensors,
            environment_settings,
            simulation_settings,
            sound_speed_profile,
        )
        for job in jobs
    ]
    for f in as_completed(futures):
        idx, rep = f.result()
        reports[idx] = rep


def evaluate_population(
    population: List[np.ndarray],
    number_of_sensors: int,
    environment_settings: EnvironmentSettings,
    simulation_settings: SimulationSettings,
    sound_speed_profile: SoundSpeedProfile,
    generation_index: int,
    global_seed: int,
    performance_settings: PerformanceSettings,
    impact_seed: Optional[int] = None,
    executor: Optional[ProcessPoolExecutor] = None,
    impact_callback: Optional[ImpactCallback] = None,
) -> List[EvaluationReport]:
    """
    Evaluate population (ordered by population index). Supports reuse of an external executor.
    Also generates impact points ONCE per generation and shares them across all chromosomes.
    """

    number_of_workers = _resolve_number_of_workers(performance_settings)

    # Generate impacts once per generation (in main process)
    grid_geometry = GridGeometry(environment_settings)
    if impact_seed is None:
        impact_seed = compute_generation_impact_seed(global_seed)

    print(f'genetic_algorithm/parallel_evaluation.py -> impact seed: {impact_seed} + global seed: {global_seed}')
    impact_points = _generate_impacts_for_generation(
        grid_geometry=grid_geometry,
        number_of_impacts=simulation_settings.number_of_impact_points_per_evaluation,
        impact_seed=impact_seed,
    )

    if impact_callback is not None:
        impact_callback(generation_index, impact_seed, impact_points)

    jobs = [
        EvaluationJob(
            chromosome=np.array(chromosome, dtype=float, copy=True),
            chromosome_index=i,
            generation_index=generation_index,
            global_seed=global_seed,
            impact_points=impact_points,
        )
        for i, chromosome in enumerate(population)
    ]

    reports: List[EvaluationReport] = [None] * len(population)  # type: ignore

    if number_of_workers == 1:
        for job in jobs:
            idx, rep = _evaluate_job(
                job,
                number_of_sensors,
                environment_settings,
                simulation_settings,
                sound_speed_profile,
            )
            reports[idx] = rep
        return reports

    # Backward compatible: create local pool if caller didn't pass one
    if executor is None:
        with ProcessPoolExecutor(max_workers=number_of_workers) as ex:
            _run_jobs_in_executor(
                ex, jobs, reports,
                number_of_sensors, environment_settings, simulation_settings, sound_speed_profile
            )
        return reports

    # Reuse external pool
    _run_jobs_in_executor(
        executor, jobs, reports,
        number_of_sensors, environment_settings, simulation_settings, sound_speed_profile
    )
    return reports
