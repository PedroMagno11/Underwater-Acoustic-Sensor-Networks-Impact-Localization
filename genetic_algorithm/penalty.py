from __future__ import annotations

from typing import List

from geometry.distance import calculate_distance_2d
from models.model import AcousticSensor


def calculate_penalty_for_separation_between_sensors(
        sensors: List[AcousticSensor],
        minimum_distance_between_sensors: float
) -> float:
    penalty = 0.0
    for i in range(len(sensors)):
        for j in range(i+1, len(sensors)):
            distance = calculate_distance_2d(sensors[i].position_x, sensors[i].position_y,
                                             sensors[j].position_x, sensors[j].position_y
            )
            excess = max(0.0, minimum_distance_between_sensors - distance)
            penalty += excess * excess

    return penalty
