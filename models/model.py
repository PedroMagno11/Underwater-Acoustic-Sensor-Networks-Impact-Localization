from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import numpy as np


@dataclass(frozen=True)
class AcousticSensor:
    """
    Acoustic Sensor is a underwater acoustic sensor, like a buoy that has a hydrophone
    """
    position_x: float
    position_y: float
    depth: float

@dataclass(frozen=True)
class GenerationMetrics:
    generation_index: int
    cost_min: float
    cost_avg: float
    cost_median: float
    cost_p90: float
    avg_no_coverage_rate: float
    avg_error_meters: float
    best_global_cost: float


@dataclass(frozen=True)
class TopologyResult:
    label: str
    number_of_sensors: int
    sensors: List[AcousticSensor]
    mean_error_meters: float
    no_coverage_rate: float
    total_cost: float
    notes: Optional[str] = None


@dataclass(frozen=True)
class GeneticAlgorithmResult:
    number_of_sensors: int
    best_chromosome: np.ndarray
    best_cost: float
    best_sensors: List[AcousticSensor]
    generation_metrics: List[GenerationMetrics]
    best_chromosomes_per_generation: List[np.ndarray]
    global_seed: int


ImpactCallback = Callable[[int, int, List[Tuple[float, float]]], None]

@dataclass(frozen=True)
class EvaluationReport:
    total_cost: float
    mean_localization_error_meters: float
    number_of_impacts: int
    number_of_localizable_impacts: int
    number_of_impacts_without_coverage: int
    coverage_penalty: float
    separation_penalty: float
    impact_points: List[Tuple[float, float]]

@dataclass
class ParticleSwarmResult:
    number_of_sensors: int
    best_position: np.ndarray
    best_cost: float
    best_sensors: list
    global_seed: int

    best_positions_per_iteration: List[np.ndarray]

    iteration_best_positions_per_iteration: List[np.ndarray]
