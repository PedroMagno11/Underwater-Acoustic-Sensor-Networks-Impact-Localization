from __future__ import annotations

import numpy as np

from geometry.grid_geometry import GridGeometry
from settings.environment_settings import EnvironmentSettings


def create_regular_polygon_chromosome(
    number_of_sensors: int,
    environment_settings: EnvironmentSettings,
    grid_geometry: GridGeometry,
    polygon_radius_meters: float,
    depth_meters: float,
    angle_offset_degrees: float = 0.0,
) -> np.ndarray:
    """Create a chromosome with sensors placed in a regular polygon.

    Sensors are evenly spaced around a circle centered at the target region center.
    Positions are quantized to grid points.
    """

    center_x = environment_settings.x_center_in_meters
    center_y = environment_settings.y_center_in_meters

    chromosome = np.zeros(3 * number_of_sensors, dtype=float)

    offset_rad = np.radians(angle_offset_degrees)
    for i in range(number_of_sensors):
        angle = offset_rad + (2.0 * np.pi * i / number_of_sensors)
        x = center_x + polygon_radius_meters * np.cos(angle)
        y = center_y + polygon_radius_meters * np.sin(angle)

        x, y = grid_geometry.quantize_for_grid_point(x, y)

        base = 3 * i
        chromosome[base + 0] = x
        chromosome[base + 1] = y
        chromosome[base + 2] = depth_meters

    return chromosome
