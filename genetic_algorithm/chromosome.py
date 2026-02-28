from __future__ import annotations

from typing import List
import numpy as np

from geometry.grid_geometry import GridGeometry
from models.model import AcousticSensor
from settings.environment_settings import EnvironmentSettings


def chromosome_converter(chromosome: np.ndarray,
                         number_of_sensors: int,
                         grid_geometry:GridGeometry,
                         environment_settings: EnvironmentSettings) -> List[AcousticSensor]:

    sensors: List[AcousticSensor] = []
    for index_of_sensor in range(number_of_sensors):
        base = 3 * index_of_sensor
        position_x = float(chromosome[base + 0])
        position_y = float(chromosome[base + 1])
        depth = float(chromosome[base + 2])

        position_x = float(np.clip(position_x, grid_geometry.minimum_limit_x, grid_geometry.maximum_limit_x))
        position_y = float(np.clip(position_y, grid_geometry.minimum_limit_y, grid_geometry.maximum_limit_y))
        depth = float(np.clip(depth, environment_settings.minimum_depth_in_meters, environment_settings.maximum_depth_in_meters))

        position_x, position_y = grid_geometry.quantize_for_grid_point(position_x, position_y)

        sensors.append(AcousticSensor(position_x, position_y, depth))

    return sensors