from __future__ import annotations

import logging
import random
from concurrent.futures.process import ProcessPoolExecutor

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from acoustic.sound_speed_profile import SoundSpeedProfile
from genetic_algorithm.chromosome import chromosome_converter
from genetic_algorithm.population import create_random_chromosome
from geometry.grid_geometry import GridGeometry
from models.model import ParticleSwarmResult, EvaluationReport
from performance.parallel_evaluation import evaluate_population
from settings.environment_settings import EnvironmentSettings
from settings.particle_swarm_settings import ParticleSwarmSettings
from settings.performance_settings import PerformanceSettings
from settings.simulation_settings import SimulationSettings
from utils.executors import create_executor_if_needed
from utils.results import write_best_reports_jsonl
from utils.seeding import compute_generation_impact_seed

logger = logging.getLogger("underwater_sensor_ga.pso")


def _env_circle(env: EnvironmentSettings) -> Tuple[float, float, float]:
    R = float(env.target_region_radius)
    cx = float(env.x_center_in_meters)
    cy = float(env.y_center_in_meters)
    return R, cx, cy


def _env_depth(env: EnvironmentSettings) -> Tuple[float, float]:
    return float(env.minimum_depth_in_meters), float(env.maximum_depth_in_meters)



def _infer_dim_per_sensor(position: np.ndarray, n: int) -> int:
    dim = int(len(position) / n)
    if dim * n != len(position):
        raise ValueError(f"Chromosome length {len(position)} not divisible by n={n}")
    return dim


def _repair_circle_and_depth(pos: np.ndarray, n: int, env: EnvironmentSettings) -> np.ndarray:
    R, cx, cy = _env_circle(env)
    zmin, zmax = _env_depth(env)

    dim = _infer_dim_per_sensor(pos, n)
    if dim not in (2, 3):
        return pos

    out = np.array(pos, dtype=float, copy=True)
    for i in range(n):
        b = i * dim
        x = float(out[b + 0])
        y = float(out[b + 1])

        dx = x - cx
        dy = y - cy
        d = (dx * dx + dy * dy) ** 0.5

        if d > R and d > 1e-12:
            s = R / d
            out[b + 0] = cx + dx * s
            out[b + 1] = cy + dy * s

        if dim == 3:
            z = float(out[b + 2])
            out[b + 2] = min(max(z, zmin), zmax)

    return out

def _build_gene_bounds(example: np.ndarray, n: int, env: EnvironmentSettings) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bounds retangulares para clamp; o repair garante o círculo.
    """
    R, cx, cy = _env_circle(env)
    zmin, zmax = _env_depth(env)

    dim = _infer_dim_per_sensor(example, n)
    if dim not in (2, 3):
        low = np.full_like(example, -np.inf, dtype=float)
        high = np.full_like(example, np.inf, dtype=float)
        return low, high

    low = np.zeros_like(example, dtype=float)
    high = np.zeros_like(example, dtype=float)

    for i in range(n):
        b = i * dim
        low[b + 0] = cx - R
        high[b + 0] = cx + R
        low[b + 1] = cy - R
        high[b + 1] = cy + R
        if dim == 3:
            low[b + 2] = zmin
            high[b + 2] = zmax

    return low, high

def _clamp(pos: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(pos, low), high)


def run_particle_swarm(
        *,
        number_of_sensors: int,
        environment_settings: EnvironmentSettings,
        particle_swarm_settings: ParticleSwarmSettings,
        simulation_settings: SimulationSettings,
        sound_speed_profile: SoundSpeedProfile,
        performance_settings: PerformanceSettings,
) -> ParticleSwarmResult:

    scenario_seed = int(simulation_settings.global_seed)
    pso_seed = int(particle_swarm_settings.random_seed)

    # tentar setar 123 no pso_seed
    rng = random.Random(pso_seed)
    logger.info("PSO seeds | pso_seed=%d | scenario_seed=%d", pso_seed, scenario_seed)

    grid = GridGeometry(environment_settings)

    reports_output_path = str(Path("outputs") / f"sensors_{number_of_sensors}" / "best_reports_pso.jsonl")

    impact_points_dir = str(Path("outputs") / f"sensors_{number_of_sensors}" / "impacts")

    swarm_size = int(particle_swarm_settings.swarm_size)
    iters = int(particle_swarm_settings.number_of_iterations)


    swarm_positions: List[np.ndarray] = [
        create_random_chromosome(number_of_sensors, grid, environment_settings, rng).astype(float)
        for _ in range(swarm_size)
    ]
    swarm_positions = [_repair_circle_and_depth(p, number_of_sensors, environment_settings) for p in swarm_positions]

    example = swarm_positions[0]
    low, high = _build_gene_bounds(example, number_of_sensors, environment_settings)

    gene_range = np.where(np.isfinite(high - low), (high - low), 1.0)
    vmax = float(particle_swarm_settings.vmax_fraction_of_range) * gene_range

    # init velocities pequenas
    swarm_velocities: List[np.ndarray] = []
    for i in range(swarm_size):
        r = np.random.default_rng(pso_seed + 10_000 + i).uniform(-1.0, 1.0, size=len(example))
        swarm_velocities.append((r * vmax).astype(float))

    pbest_positions = [p.copy() for p in swarm_positions]
    pbest_costs = [float("inf")] * swarm_size

    gbest_pos: Optional[np.ndarray] = None
    gbest_cost = float("inf")
    gbest_report: Optional[EvaluationReport] = None

    best_positions_per_iteration: List[np.ndarray] = []
    iteration_best_positions_per_iteration: List[np.ndarray] = []

    executor = create_executor_if_needed(performance_settings)

    try:
        for it in range(iters):
            impact_seed = compute_generation_impact_seed(scenario_seed)

            reports = evaluate_population(
                population=swarm_positions,
                number_of_sensors=number_of_sensors,
                environment_settings=environment_settings,
                simulation_settings=simulation_settings,
                sound_speed_profile=sound_speed_profile,
                generation_index=it,
                global_seed=simulation_settings.global_seed,
                performance_settings=performance_settings,
                executor=executor,
                impact_seed=impact_seed,
            )

            costs = np.array([r.total_cost for r in reports], dtype=float)

            # update personal bests
            for i in range(swarm_size):
                c = float(costs[i])
                if c < pbest_costs[i]:
                    pbest_costs[i] = c
                    pbest_positions[i] = swarm_positions[i].copy()

            # iteration best
            idx_best = int(np.argmin(costs))
            it_best_cost = float(costs[idx_best])
            it_best_report = reports[idx_best]
            it_best_pos = swarm_positions[idx_best].copy()

            iteration_best_positions_per_iteration.append(it_best_pos.copy())

            if it_best_cost < gbest_cost:
                gbest_cost = it_best_cost
                gbest_pos = it_best_pos.copy()
                gbest_report = it_best_report

            if gbest_pos is None:
                gbest_pos = it_best_pos.copy()
                gbest_report = it_best_report
                gbest_cost = it_best_cost

            best_positions_per_iteration.append(gbest_pos.copy())

            # Export JSONL (mantém schema atual do exporter)
            write_best_reports_jsonl(
                output_path=reports_output_path,
                generation_index=it,
                number_of_sensors=number_of_sensors,
                ga_seed=pso_seed,  # compat com schema atual
                scenario_seed=scenario_seed,
                impact_seed=impact_seed,
                best_of_generation=it_best_report,
                best_global=gbest_report,
                include_impact_points=False,
            )

            # PSO update
            w = float(particle_swarm_settings.inertia_w)
            c1 = float(particle_swarm_settings.cognitive_c1)
            c2 = float(particle_swarm_settings.social_c2)

            for i in range(swarm_size):
                r1 = np.random.default_rng(pso_seed + 20_000 + it * 1000 + i).random(len(example))
                r2 = np.random.default_rng(pso_seed + 30_000 + it * 1000 + i).random(len(example))

                vel = swarm_velocities[i]
                pos = swarm_positions[i]

                vel = w * vel + c1 * r1 * (pbest_positions[i] - pos) + c2 * r2 * (gbest_pos - pos)
                vel = np.clip(vel, -vmax, vmax)

                pos = pos + vel
                pos = _clamp(pos, low, high)
                pos = _repair_circle_and_depth(pos, number_of_sensors, environment_settings)

                swarm_velocities[i] = vel
                swarm_positions[i] = pos

            logger.info(
                "PSO iter %d/%d | best_global_cost=%.6f | iter_best=%.6f",
                it + 1, iters, gbest_cost, it_best_cost
            )

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if gbest_pos is None:
        raise RuntimeError("Unexpected error: PSO gbest not found")

    best_sensors = chromosome_converter(gbest_pos, number_of_sensors, grid, environment_settings)

    outdir = Path("outputs") / f"sensors_{number_of_sensors}"
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "pso_best_global_position.npy", np.asarray(gbest_pos, dtype=float))
    np.save(outdir / "pso_best_positions_per_iteration.npy", np.asarray(best_positions_per_iteration, dtype=float))
    np.save(outdir / "pso_iteration_best_positions_per_iteration.npy",
            np.asarray(iteration_best_positions_per_iteration, dtype=float))

    return ParticleSwarmResult(
        number_of_sensors=number_of_sensors,
        best_position=gbest_pos,
        best_cost=gbest_cost,
        best_sensors=best_sensors,
        global_seed=simulation_settings.global_seed,
        best_positions_per_iteration=best_positions_per_iteration,
        iteration_best_positions_per_iteration=iteration_best_positions_per_iteration,
    )