from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class EnvironmentSettings:
    grid_size_in_points: int = 1016
    grid_spacing: float = 9.0

    target_region_radius: float = 1500.0
    maximum_detection_distance: float = 2500.0

    minimum_depth_in_meters: float = 0.5
    maximum_depth_in_meters: float = 8.0

    x_center_in_meters: float = 0.0
    y_center_in_meters: float = 0.0