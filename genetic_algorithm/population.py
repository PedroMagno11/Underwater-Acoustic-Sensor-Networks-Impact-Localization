from __future__ import annotations
import random
import numpy as np

from settings.environment_settings import EnvironmentSettings
from geometry.grid_geometry import GridGeometry

def create_random_chromosome(number_of_sensors: int,
                             grid_geometry: GridGeometry,
                             environment_settings: EnvironmentSettings,
                             random_generator: random.Random) -> np.ndarray:
    chromosome = np.zeros(3 * number_of_sensors, dtype=float)

    for index_of_sensor in range(number_of_sensors):
        base = 3 * index_of_sensor
        chromosome[base + 0] = random_generator.uniform(grid_geometry.minimum_limit_x, grid_geometry.maximum_limit_x)
        chromosome[base + 1] = random_generator.uniform(grid_geometry.minimum_limit_y, grid_geometry.maximum_limit_y)
        chromosome[base + 2] = random_generator.uniform(environment_settings.minimum_depth_in_meters, environment_settings.maximum_depth_in_meters)

    return chromosome