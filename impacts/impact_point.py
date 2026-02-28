import random
import numpy as np
from typing import Tuple

from geometry.grid_geometry import GridGeometry
from utils.seeding import seed_for_impact_position


class ImpactPosition:
    @staticmethod
    def generate_impact_position(grid_geometry: GridGeometry, global_seed: int, target_index: int) -> Tuple[float, float]:
        radius = grid_geometry.environment_settings.target_region_radius
        center_x = grid_geometry.environment_settings.x_center_in_meters
        center_y = grid_geometry.environment_settings.y_center_in_meters

        rng = random.Random(seed_for_impact_position(global_seed=global_seed, target_index=target_index))

        #Random variables for uniform sampling inside a circle
        u = rng.random()
        angle = 2.0 * np.pi * rng.random()
        r = np.sqrt(u) * radius

        # Convert polar coordinates to Cartesian coordinates
        x = center_x + r * np.cos(angle)
        y = center_y + r * np.sin(angle)

        return x, y
