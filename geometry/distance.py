from __future__ import annotations

import numpy as np

def calculate_distance_2d(x1:float, y1:float, x2:float, y2:float) -> float:
    return np.hypot(x1-x2, y1-y2)

def calculate_distance_3d(x1:float, y1:float,z1:float, x2:float, y2:float, z2: float) -> float:
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2

    return np.sqrt(dx*dx + dy*dy + dz*dz)