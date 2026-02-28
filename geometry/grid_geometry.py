from __future__ import annotations

from typing import Tuple
from settings.environment_settings import EnvironmentSettings


class GridGeometry:
    """
    Represents the geometric structure of a 2D square grid centered on a reference point.

    This class is responsible for:
        - Converting continuous positions (in meters) to the nearest grid point
        - Defining the spatial limits of the grid
        - Generating random points inside a circular target region
    """

    def __init__(self, environment_settings: EnvironmentSettings):
        """
        Initializes the grid geometry based on the environment settings
        """
        self.environment_settings = environment_settings
        self.grid_size_in_points = environment_settings.grid_size_in_points
        self.grid_spacing = environment_settings.grid_spacing

        # Total grid length in meters
        side = (self.grid_size_in_points - 1) * self.grid_spacing

        # Compute grid boundaries assuming the grid is centered at (center_x, center_y)
        self.minimum_limit_x = environment_settings.x_center_in_meters - side / 2.0
        self.maximum_limit_x = environment_settings.x_center_in_meters + side / 2.0
        self.minimum_limit_y = environment_settings.y_center_in_meters - side / 2.0
        self.maximum_limit_y = environment_settings.y_center_in_meters + side / 2.0


    def quantize_for_grid_point(self, position_x: float, position_y: float) -> Tuple[float, float]:
        """
        Quantizes a continuous (x, y) position to the nearest valid grid point.

        The method:
        1. Converts the position from meters to grid indices
        2. Rounds to the nearest grid cell
        3. Clamps the indices to ensure they stay inside grid bounds
        4. Converts the indices back to metric coordinates
        """
        index_x_float = (position_x - self.minimum_limit_x) / self.grid_spacing
        index_y_float = (position_y - self.minimum_limit_y) / self.grid_spacing

        # Round to the nearest integer grid index
        index_x = int(round(index_x_float))
        index_y = int(round(index_y_float))

        # Clamp indices to remain within grid bounds
        index_x = max(0, min(self.grid_size_in_points - 1, index_x))
        index_y = max(0, min(self.grid_size_in_points - 1, index_y))

        # Convert grid indices back matric coordinates
        position_x_quantized = self.minimum_limit_x + index_x * self.grid_spacing
        position_y_quantized = self.minimum_limit_y + index_y * self.grid_spacing

        return position_x_quantized, position_y_quantized