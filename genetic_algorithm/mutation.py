from __future__ import annotations
import random
import numpy as np

from settings.environment_settings import EnvironmentSettings
from settings.genetic_algorithm_settings import GeneticAlgorithmSettings
from geometry.grid_geometry import GridGeometry

def apply_mutation(
        chromosome: np.ndarray,
        number_of_sensors: int,
        environment_settings: EnvironmentSettings,
        grid_geometry: GridGeometry,
        genetic_algorithm_settings: GeneticAlgorithmSettings,
        random_generator: random.Random
) -> None:
    for index_of_sensor in range(number_of_sensors):
        if random_generator.random() > genetic_algorithm_settings.mutation_probability:
            continue

        base = 3 * index_of_sensor

        chromosome[base + 0] += random_generator.gauss(0.0, genetic_algorithm_settings.standard_deviation_of_xy_mutation)
        chromosome[base + 1] += random_generator.gauss(0.0, genetic_algorithm_settings.standard_deviation_of_xy_mutation)
        chromosome[base + 2] += random_generator.gauss(0.0, genetic_algorithm_settings.standard_deviation_of_z_mutation)

        chromosome[base + 0] = float(np.clip(chromosome[base + 0], grid_geometry.minimum_limit_x, grid_geometry.maximum_limit_x))
        chromosome[base + 1] = float(np.clip(chromosome[base + 1], grid_geometry.minimum_limit_y, grid_geometry.maximum_limit_y))
        chromosome[base + 2] = float(np.clip(chromosome[base + 2], environment_settings.minimum_depth_in_meters, environment_settings.maximum_depth_in_meters))
