from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np

from models.model import AcousticSensor
from settings.simulation_settings import SimulationSettings
from geometry.grid_geometry import GridGeometry
from acoustic.sound_speed_profile import SoundSpeedProfile


# ------------------------------------------------------------
# Small module-level cache to avoid rebuilding grids every call
# Keyed by (center_x, center_y, radius, step) OR (local_center_x, local_center_y, radius, step)
# ------------------------------------------------------------
_GRID_CACHE: Dict[Tuple[float, float, float, float], np.ndarray] = {}


def _build_candidates_in_circle(center_x: float, center_y: float, radius: float, step: float) -> np.ndarray:
    """
    Returns array (P,2) of candidate points inside circle centered at (center_x,center_y).
    Uses a cache to avoid re-creating the same grid repeatedly.
    """
    key = (float(center_x), float(center_y), float(radius), float(step))
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    limit = float(radius)
    values = np.arange(-limit, limit + 1e-9, float(step), dtype=float)
    dx, dy = np.meshgrid(values, values, indexing="xy")
    mask = (dx * dx + dy * dy) <= (radius * radius)

    xs = center_x + dx[mask]
    ys = center_y + dy[mask]
    pts = np.stack([xs, ys], axis=1)  # (P,2)

    _GRID_CACHE[key] = pts
    return pts


def estimate_impact_position(
    sensors: List[AcousticSensor],
    observed_times: np.ndarray,
    grid_geometry: GridGeometry,
    sound_speed_profile: SoundSpeedProfile,
    simulation_settings: SimulationSettings
) -> Tuple[float, float]:
    """
    Fast grid-search MLE estimator (vectorized).

    Model:
      observed_i = t0 + theoretical_i(x,y) + noise
    For fixed (x,y), MLE:
      t0_hat = mean(observed - theoretical)
    Residuals:
      r = observed - (t0_hat + theoretical)
    Cost:
      sum(r^2)

    Uses vectorized computation and evaluates all candidates in batch.
    """

    observed = np.asarray(observed_times, dtype=float)
    n = int(observed.size)
    cx = float(grid_geometry.environment_settings.x_center_in_meters)
    cy = float(grid_geometry.environment_settings.y_center_in_meters)

    if n < 3:
        # Not localizable
        return grid_geometry.quantize_for_grid_point(cx, cy)

    # -----------------------------
    # Sensor arrays
    # -----------------------------
    sx = np.array([s.position_x for s in sensors], dtype=float)  # (n,)
    sy = np.array([s.position_y for s in sensors], dtype=float)  # (n,)

    # Depth field name fallback: "depth" or "position_z"
    sz = np.array([float(getattr(s, "depth", getattr(s, "position_z", 0.0))) for s in sensors], dtype=float)  # (n,)

    # Straight-line time: dist3d / c(avg_depth)
    avg_depth = 0.5 * sz
    c = np.array([sound_speed_profile.sound_speed(float(d)) for d in avg_depth], dtype=float)  # (n,)
    inv_c = 1.0 / c  # (n,)

    # Region
    R = float(grid_geometry.environment_settings.target_region_radius)
    R2 = R * R

    # -----------------------------
    # Cost evaluation (batch)
    # -----------------------------
    def evaluate_costs(points_xy: np.ndarray) -> np.ndarray:
        # points_xy: (P,2)
        px = points_xy[:, 0:1]  # (P,1)
        py = points_xy[:, 1:2]  # (P,1)

        # dist3d: sqrt((x-sx)^2 + (y-sy)^2 + sz^2)
        dx = px - sx[None, :]  # (P,n)
        dy = py - sy[None, :]  # (P,n)
        dist = np.sqrt(dx * dx + dy * dy + (sz[None, :] * sz[None, :]))  # (P,n)

        theo = dist * inv_c[None, :]  # (P,n)
        d = observed[None, :] - theo  # (P,n)

        # cost = sum((d - mean(d))^2) = sum(d^2) - n*mean(d)^2
        mean_d = np.mean(d, axis=1)         # (P,)
        sumsq = np.sum(d * d, axis=1)       # (P,)
        return sumsq - float(n) * (mean_d * mean_d)

    # -----------------------------
    # 1) Coarse search over full circle
    # -----------------------------
    coarse_step = float(simulation_settings.coarse_search_step_in_meters)
    coarse_points = _build_candidates_in_circle(cx, cy, R, coarse_step)
    coarse_costs = evaluate_costs(coarse_points)

    best_idx = int(np.argmin(coarse_costs))
    best_x = float(coarse_points[best_idx, 0])
    best_y = float(coarse_points[best_idx, 1])

    # -----------------------------
    # 2) Fine search around best (still clipped to global target circle)
    # -----------------------------
    fine_step = float(simulation_settings.fine_search_step_in_meters)
    refine_R = float(simulation_settings.refinement_radius_in_meters)

    fine_points = _build_candidates_in_circle(best_x, best_y, refine_R, fine_step)

    # Keep only those still inside global circle
    dxg = fine_points[:, 0] - cx
    dyg = fine_points[:, 1] - cy
    fine_points = fine_points[(dxg * dxg + dyg * dyg) <= R2]

    if fine_points.shape[0] > 0:
        fine_costs = evaluate_costs(fine_points)
        best2_idx = int(np.argmin(fine_costs))
        best_x = float(fine_points[best2_idx, 0])
        best_y = float(fine_points[best2_idx, 1])

    qx, qy = grid_geometry.quantize_for_grid_point(best_x, best_y)
    return qx, qy
