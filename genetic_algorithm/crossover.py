from __future__ import annotations
import random
import numpy as np

def perform_crossover_per_sensor(
    father_chromosome: np.ndarray,
    mother_chromosome: np.ndarray,
    number_of_sensors: int,
    random_generator: random.Random
) -> np.ndarray:

    child = np.empty_like(father_chromosome)
    for sensor_index in range(number_of_sensors):
        base = 3 * sensor_index
        father = (random_generator.random() < 0.5)
        origin = father_chromosome if father else mother_chromosome
        child[base: base + 3] = origin[base: base + 3]

    return child
