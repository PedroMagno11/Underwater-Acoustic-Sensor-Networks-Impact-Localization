from __future__ import annotations

from concurrent.futures.process import ProcessPoolExecutor
from typing import List, Optional
import logging
import random
from pathlib import Path

import numpy as np

from genetic_algorithm.chromosome import chromosome_converter
from genetic_algorithm.crossover import perform_crossover_per_sensor
from genetic_algorithm.mutation import apply_mutation
from genetic_algorithm.population import create_random_chromosome
from genetic_algorithm.selection import select_index_for_tournament
from geometry.grid_geometry import GridGeometry
from impacts.impact_saver import make_impact_saver
from models.model import GeneticAlgorithmResult, EvaluationReport, GenerationMetrics
from performance.parallel_evaluation import evaluate_population
from settings.environment_settings import EnvironmentSettings
from settings.genetic_algorithm_settings import GeneticAlgorithmSettings
from settings.simulation_settings import SimulationSettings
from settings.performance_settings import PerformanceSettings
from acoustic.sound_speed_profile import SoundSpeedProfile
from utils.executors import create_executor_if_needed
from utils.results import write_best_reports_jsonl
from utils.seeding import compute_generation_impact_seed

logger = logging.getLogger("underwater_sensor_ga.ga")

def _extract_costs(reports: List[EvaluationReport]) -> np.ndarray:
    return np.array([r.total_cost for r in reports], dtype=float)


def _compute_no_coverage_rates(reports: List[EvaluationReport]) -> np.ndarray:
    return np.array(
        [
            (r.number_of_impacts_without_coverage / r.number_of_impacts) if r.number_of_impacts > 0 else 1.0
            for r in reports
        ],
        dtype=float,
    )


def _computed_average_localization_error(reports: List[EvaluationReport]) -> float:
    finite_errors = [
        r.mean_localization_error_meters
        for r in reports
        if np.isfinite(r.mean_localization_error_meters)
    ]
    return float(np.mean(finite_errors)) if finite_errors else float("inf")

def _compute_generation_metrics(
    generation_index: int,
    reports: List[EvaluationReport],
    best_global_cost: float,
) -> GenerationMetrics:
    costs = _extract_costs(reports)
    no_coverage_rates = _compute_no_coverage_rates(reports)
    avg_error = _computed_average_localization_error(reports)

    return GenerationMetrics(
        generation_index=generation_index,
        cost_min=float(np.min(costs)),
        cost_avg=float(np.mean(costs)),
        cost_median=float(np.median(costs)),
        cost_p90=float(np.percentile(costs, 90)),
        avg_no_coverage_rate=float(np.mean(no_coverage_rates)),
        avg_error_meters=avg_error,
        best_global_cost=best_global_cost,
    )


def run_genetic_algorithm(
    number_of_sensors: int,
    environment_settings: EnvironmentSettings,
    genetic_algorithm_settings: GeneticAlgorithmSettings,
    simulation_settings: SimulationSettings,
    sound_speed_profile: SoundSpeedProfile,
    performance_settings: Optional[PerformanceSettings] = None,
    reports_output_path: Optional[str] = None,
    impact_points_dir: Optional[str] = None,
) -> GeneticAlgorithmResult:
    if performance_settings is None:
        performance_settings = PerformanceSettings()

    ga_seed = int(genetic_algorithm_settings.random_seed)
    scenario_seed = int(simulation_settings.global_seed)

    rng = random.Random(ga_seed)
    logger.info("GA seeds | ga_seed=%d | scenario_seed=%d", ga_seed, scenario_seed)

    grid_geometry = GridGeometry(environment_settings)

    population: List[np.ndarray] = [
        create_random_chromosome(number_of_sensors, grid_geometry, environment_settings, rng)
        for _ in range(genetic_algorithm_settings.population_size)
    ]

    best_chromosome: Optional[np.ndarray] = None
    best_cost: float = float("inf")
    best_global_report: Optional[EvaluationReport] = None

    generation_metrics: List[GenerationMetrics] = []
    best_chromosomes_per_generation: List[np.ndarray] = []
    best_global_chromosomes_per_generation: List[np.ndarray] = []

    if reports_output_path is None:
        reports_output_path = str(
            Path("outputs") / f"sensors_{number_of_sensors}" / "best_reports.jsonl"
        )

    if impact_points_dir is None:
        impact_points_dir = str(
            Path("outputs")/f"sensors_{number_of_sensors}"/"impacts"
        )

    impact_saver = make_impact_saver(Path(impact_points_dir), scenario_seed=scenario_seed)
    executor = create_executor_if_needed(performance_settings)

    try:
        for gen_idx in range(genetic_algorithm_settings.number_of_generations):

            impact_seed = compute_generation_impact_seed(scenario_seed)

            reports = evaluate_population(
                population=population,
                number_of_sensors=number_of_sensors,
                environment_settings=environment_settings,
                simulation_settings=simulation_settings,
                sound_speed_profile=sound_speed_profile,
                generation_index=gen_idx,
                global_seed=simulation_settings.global_seed,
                performance_settings=performance_settings,
                executor=executor,  # IMPORTANT reuse pool
                impact_callback=impact_saver,
                impact_seed=impact_seed,
            )

            costs = np.array([r.total_cost for r in reports], dtype=float)
            idx_best = int(np.argmin(costs))

            gen_best_cost = float(costs[idx_best])
            gen_best_report = reports[idx_best]
            gen_best_chromosome = np.array(population[idx_best], dtype=float, copy=True)

            best_chromosomes_per_generation.append(gen_best_chromosome)

            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_chromosome = np.array(gen_best_chromosome, dtype=float, copy=True)
                best_global_report = gen_best_report  # NEW

            best_global_chromosomes_per_generation.append(
                np.array(best_chromosome, dtype=float, copy=True)
            )

            if best_global_report is None:
                best_global_report = gen_best_report

            write_best_reports_jsonl(
                output_path=reports_output_path,
                generation_index=gen_idx,
                number_of_sensors=number_of_sensors,
                ga_seed=ga_seed,
                scenario_seed=scenario_seed,
                impact_seed=impact_seed,
                best_of_generation=gen_best_report,
                best_global=best_global_report,
                include_impact_points=False,  # set True if you want to store points (bigger file)
            )

            generation_metrics.append(
                _compute_generation_metrics(
                    generation_index=gen_idx,
                    reports=reports,
                    best_global_cost=best_cost,
                )
            )

            # elitism
            sorted_idx = list(np.argsort(costs))
            elites = [
                np.array(population[i], dtype=float, copy=True)
                for i in sorted_idx[: genetic_algorithm_settings.elitism]
            ]

            new_population: List[np.ndarray] = []
            new_population.extend(elites)

            while len(new_population) < genetic_algorithm_settings.population_size:
                father_idx = select_index_for_tournament(costs.tolist(), genetic_algorithm_settings.tournament_size, rng)
                mother_idx = select_index_for_tournament(costs.tolist(), genetic_algorithm_settings.tournament_size, rng)

                father = population[father_idx]
                mother = population[mother_idx]

                if rng.random() < genetic_algorithm_settings.crossover_probability:
                    child = perform_crossover_per_sensor(
                        father_chromosome=father,
                        mother_chromosome=mother,
                        number_of_sensors=number_of_sensors,
                        random_generator=rng,
                    )
                else:
                    child = np.array(father, dtype=float, copy=True)

                apply_mutation(
                    chromosome=child,
                    number_of_sensors=number_of_sensors,
                    environment_settings=environment_settings,
                    grid_geometry=grid_geometry,
                    genetic_algorithm_settings=genetic_algorithm_settings,
                    random_generator=rng,
                )

                new_population.append(child)

            population = new_population

            logger.info(
                "Generation %d/%d | best_global_cost=%.3f | generation_best=%.3f",
                gen_idx + 1,
                genetic_algorithm_settings.number_of_generations,
                best_cost,
                gen_best_cost,
            )

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if best_chromosome is None:
        raise RuntimeError("Unexpected error: best chromosome not found")

    best_sensors = chromosome_converter(best_chromosome, number_of_sensors, grid_geometry, environment_settings)

    outdir = Path("outputs")/f'sensors_{number_of_sensors}'
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / "best_chromosomes_per_generation.npy", np.asarray(best_chromosomes_per_generation, dtype=float))
    np.save(outdir / "best_global_chromosome.npy", np.asarray(best_chromosome, dtype=float))
    np.save(outdir / "best_global_chromosomes_per_generation.npy", np.asarray(best_global_chromosomes_per_generation, dtype=float))

    return GeneticAlgorithmResult(
        number_of_sensors=number_of_sensors,
        best_chromosome=best_chromosome,
        best_cost=best_cost,
        best_sensors=best_sensors,
        generation_metrics=generation_metrics,
        best_chromosomes_per_generation=best_chromosomes_per_generation,
        global_seed=simulation_settings.global_seed
    )
