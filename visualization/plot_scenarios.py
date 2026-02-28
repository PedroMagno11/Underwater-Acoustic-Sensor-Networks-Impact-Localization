from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

def draw_target_circle(ax, cx, cy, R):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(cx + R * np.cos(t), cy + R * np.sin(t), linewidth=1.5)

def _get_sensor_depth_m(sensor) -> Optional[float]:
    for attr in ("position_z", "position_z_m", "depth_meters", "depth", "z"):
        if hasattr(sensor, attr):
            try:
                return float(getattr(sensor, attr))
            except Exception:
                pass
    return None


def _classify_impacts_by_coverage_2d(
    impacts_xy: np.ndarray,
    sensors,
    max_det: float,
    min_sensors_for_coverage: int = 3,
) -> np.ndarray:
    if impacts_xy.size == 0:
        return np.array([], dtype=bool)

    if len(sensors) == 0:
        return np.zeros((impacts_xy.shape[0],), dtype=bool)

    sx = np.array([float(s.position_x) for s in sensors], dtype=float)
    sy = np.array([float(s.position_y) for s in sensors], dtype=float)

    dx = impacts_xy[:, 0:1] - sx.reshape(1, -1)
    dy = impacts_xy[:, 1:2] - sy.reshape(1, -1)
    d = np.sqrt(dx * dx + dy * dy)

    within = d <= float(max_det)
    count = within.sum(axis=1)
    return count >= int(min_sensors_for_coverage)


def _polygon_cycle_indices_by_angle(
    sx: np.ndarray,
    sy: np.ndarray,
    cx: float,
    cy: float,
) -> List[int]:
    """
    Ordena sensores por ângulo em torno do centro (cx,cy).
    Retorna a sequência de índices para formar um ciclo (polígono) ligando vizinhos.
    """
    angles = np.arctan2(sy - cy, sx - cx)  # [-pi, pi]
    order = np.argsort(angles)
    return [int(i) for i in order]


def _draw_polygon_edges_and_distances(
    ax,
    sx: np.ndarray,
    sy: np.ndarray,
    order: List[int],
    *,
    distance_decimals: int = 1,
    distance_unit: str = "m",
    line_alpha: float = 0.55,
    line_width: float = 1.1,
    line_color: str = "black",
    line_style: str = "--",
):
    """
    Desenha somente as arestas do ciclo:
      order[0]-order[1]-...-order[n-1]-order[0]
    E escreve a distância em cada aresta.
    """
    n = len(order)
    if n < 2:
        return

    fmt = f"{{:.{int(distance_decimals)}f}} {distance_unit}"
    fontsize = 8 if n <= 4 else 7

    for k in range(n):
        i = order[k]
        j = order[(k + 1) % n]

        x1, y1 = sx[i], sy[i]
        x2, y2 = sx[j], sy[j]
        dist = float(np.hypot(x2 - x1, y2 - y1))

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=line_width,
            alpha=line_alpha,
            color=line_color,
            linestyle=line_style,
        )

        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ax.text(
            mx, my,
            fmt.format(dist),
            fontsize=fontsize,
            alpha=0.9,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", alpha=0.12),
        )

def plot_scene(
    *,
    out_png: Path,
    title: str,
    impacts_xy: np.ndarray,
    sensors,
    env,
    show_detection: bool = True,

    # cobertura
    highlight_out_of_coverage: bool = True,
    out_of_coverage_mask: Optional[np.ndarray] = None,
    min_sensors_for_coverage: int = 3,

    # ✅ linhas + distâncias
    show_sensor_polygon_links: bool = True,
    polygon_line_color: str = "black",
    polygon_line_style: str = "--",
    polygon_line_width: float = 1.1,
    polygon_line_alpha: float = 0.55,
    distance_decimals: int = 1,
    distance_unit: str = "m",

    # profundidade
    show_depth_labels: bool = True,

    # cores
    covered_color: str = "gold",
    uncovered_color: str = "tab:red",
):
    cx = float(env.x_center_in_meters)
    cy = float(env.y_center_in_meters)
    R = float(env.target_region_radius)
    max_det = float(env.maximum_detection_distance)

    fig, ax = plt.subplots()

    # região alvo
    draw_target_circle(ax, cx, cy, R)

    # sensores
    sx = np.array([float(s.position_x) for s in sensors], dtype=float)
    sy = np.array([float(s.position_y) for s in sensors], dtype=float)
    ax.scatter(sx, sy, s=60, marker="^", label="Sensors")

    # ✅ liga sensores por vizinhança angular e anota distâncias
    if show_sensor_polygon_links and len(sensors) >= 3:
        order = _polygon_cycle_indices_by_angle(sx, sy, cx=cx, cy=cy)
        _draw_polygon_edges_and_distances(
            ax,
            sx, sy, order,
            distance_decimals=distance_decimals,
            distance_unit=distance_unit,
            line_alpha=polygon_line_alpha,
            line_width=polygon_line_width,
            line_color=polygon_line_color,
            line_style=polygon_line_style,
        )

    # círculos de detecção
    if show_detection:
        t = np.linspace(0, 2 * np.pi, 200)
        for x, y in zip(sx, sy):
            ax.plot(
                x + max_det * np.cos(t),
                y + max_det * np.sin(t),
                linewidth=0.8,
                alpha=0.25,
            )

    # impactos (coberto vs fora)
    if highlight_out_of_coverage:
        if out_of_coverage_mask is None:
            covered_mask = _classify_impacts_by_coverage_2d(
                impacts_xy=impacts_xy,
                sensors=sensors,
                max_det=max_det,
                min_sensors_for_coverage=min_sensors_for_coverage,
            )
            out_of_coverage_mask = ~covered_mask
        else:
            out_of_coverage_mask = np.asarray(out_of_coverage_mask, dtype=bool)

        if out_of_coverage_mask.shape[0] != impacts_xy.shape[0]:
            raise ValueError("out_of_coverage_mask deve ter shape (M,) com M=len(impacts_xy).")

        covered_mask = ~out_of_coverage_mask

        if covered_mask.any():
            ax.scatter(
                impacts_xy[covered_mask, 0],
                impacts_xy[covered_mask, 1],
                s=14,
                c=covered_color,
                label="Impact points covered",
                alpha=0.9,
                linewidths=0,
            )
        if out_of_coverage_mask.any():
            ax.scatter(
                impacts_xy[out_of_coverage_mask, 0],
                impacts_xy[out_of_coverage_mask, 1],
                s=22,
                c=uncovered_color,
                marker="x",
                label="Impact points not covered",
                alpha=0.95,
                linewidths=1.4,
            )
    else:
        ax.scatter(impacts_xy[:, 0], impacts_xy[:, 1], s=14, label="Impact points")

    # profundidade dos sensores
    if show_depth_labels:
        for i, s in enumerate(sensors):
            z = _get_sensor_depth_m(s)
            if z is None:
                continue
            ax.text(
                sx[i], sy[i],
                f" z={z:.1f}m",
                fontsize=8,
                ha="left",
                va="bottom",
                alpha=0.9,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
