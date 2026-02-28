from __future__ import annotations

import numpy as np

from acoustic.sound_speed_profile import SoundSpeedProfile
from geometry.distance import calculate_distance_3d
from models.model import AcousticSensor


def calculate_arrival_time(
        sensor: AcousticSensor,
        impact_position_x: float,
        impact_position_y: float,
        sound_speed_profile: SoundSpeedProfile,
        number_of_samples: int = 11
)-> float:
    """
    Computes the acoustic arrival time using a straight-line propagation model.

    This model assumes:
        - straight-line propagation between source and receiver;
        - depth-dependent sound speed c(z);
        - numerical integration of 1 / c(z) along the segment between source (z=0) and sensor (z = sensor.depth).

    Mathematically:
        t ≈ ∫ ds / c(z(s))
    """
    if number_of_samples < 2:
        raise ValueError('number_of_samples must be greater than 2.')

    # --------------------------------------------------
    # Straight-line distance between source and receiver
    # --------------------------------------------------

    total_path_length = calculate_distance_3d(
        sensor.position_x, sensor.position_y, sensor.depth,
        impact_position_x, impact_position_y, 0.0
    )

    total_path_length = float(total_path_length)
    if total_path_length <= 0.0:
        return 0.0

    # ---------------------------------------
    # Parametrization of the propagation path
    # ---------------------------------------
    path_parameter = np.linspace(
        0.0, 1.0, int(number_of_samples), dtype=float
    )

    # Depth varies linearly along the straight path
    depth_along_path = path_parameter * float(sensor.depth)


    # -------------------------------------
    # Sound speed evaluation along the path
    # -------------------------------------

    sound_speed_along_path = np.array(
        [
            sound_speed_profile.sound_speed(depth)
            for depth in depth_along_path
        ], dtype=float,
    )

    # ------------------------------------
    # Numerical integration of travel time
    # ------------------------------------
    # ds = total_path_length * d(path_parameter)
    inverse_sound_speed = 1.0 / sound_speed_along_path

    arrival_time = total_path_length * float(np.trapezoid(inverse_sound_speed, path_parameter))

    return arrival_time