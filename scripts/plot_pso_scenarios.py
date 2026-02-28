from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from settings.environment_settings import EnvironmentSettings
from geometry.grid_geometry import GridGeometry

from utils.loaders import load_json_config, load_settings_from_config, load_impact_points_to_plot
from visualization.plot_scenarios import plot_scene


@dataclass(frozen=True)
class _SimpleSensor:
    position_x: float
    position_y: float
    position_z: float


def _infer_dim_per_sensor(vec: np.ndarray, n: int) -> int:
    dim = int(len(vec) / n)
    if dim * n != len(vec):
        raise ValueError(f"len(vec)={len(vec)} not divisible by n={n}")
    return dim


def _vec_to_sensors(vec: np.ndarray, n_sensors: int, env: EnvironmentSettings) -> List[_SimpleSensor]:
    vec = np.asarray(vec, dtype=float).reshape(-1)
    dim = _infer_dim_per_sensor(vec, n_sensors)
    arr = vec.reshape(n_sensors, dim)

    zmin = float(env.minimum_depth_in_meters)
    zmax = float(env.maximum_depth_in_meters)

    sensors: List[_SimpleSensor] = []
    for i in range(n_sensors):
        x = float(arr[i, 0])
        y = float(arr[i, 1])
        z = float(arr[i, 2]) if dim >= 3 else (zmin + zmax) / 2.0
        z = min(max(z, zmin), zmax)
        sensors.append(_SimpleSensor(position_x=x, position_y=y, position_z=z))
    return sensors


def _fmt_metrics_from_row(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return "metrics=n/a"
    cost = float(row.get("total_cost", float("nan")))
    mean_err = float(row.get("mean_error_m", float("nan")))
    no_cov_rate = row.get("no_coverage_rate", None)
    if no_cov_rate is None:
        return f"cost={cost:.3f} | mean_err={mean_err:.2f} m"
    return f"cost={cost:.3f} | mean_err={mean_err:.2f} m | no_cov={100*float(no_cov_rate):.1f}%"


def _read_best_reports_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_reports_by_iter(items: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for it in items:
        k = it.get("generation_index", it.get("generation", None))
        if k is None:
            continue
        try:
            out[int(float(k))] = it
        except Exception:
            pass
    return out


def _load_env_from_config(config_path: str) -> EnvironmentSettings:
    cfg = load_json_config(config_path)
    env, *_rest = load_settings_from_config(cfg)
    return env


def generate_pso_scenario_figures(
    *,
    number_of_sensors: int,
    config_path: str = "experiment_config.json",
    pso_output_root: str = "outputs",
    figures_subdir: str = "figures_pso",
    pso_best_positions_per_iter_rel: str = "pso_best_positions_per_iteration.npy",
    pso_best_global_rel: str = "pso_best_global_position.npy",
    pso_reports_jsonl_rel: str = "best_reports_pso.jsonl",
    impacts_subdir: str = "impacts_pso",
    fallback_impacts_subdir: str = "impacts",

    # MODES
    plot_best_so_far_on_all_iterations: bool = True,   # <-- substitui o antigo best_global em todas
    plot_best_global_only: bool = False,               # <-- se quiser 1 figura do best_global

    plot_snapshots_on_single_iteration_impacts: bool = True,
    snapshots_iteration_index: int = 0,
    snapshot_every: int = 50,
    max_snapshots: int = 14,

    min_sensors_for_coverage: int = 3,
    show_detection: bool = True,
    show_depth_labels: bool = True,
    covered_color: str = "green",
    uncovered_color: str = "tab:red",

    show_sensor_polygon_links: bool = False,
    polygon_line_color: str = "black",
    polygon_line_style: str = "--",
    polygon_line_width: float = 1.2,
    polygon_line_alpha: float = 0.65,
    distance_decimals: int = 1,
    distance_unit: str = "m",

    title_prefix: str = "PSO",
    include_metrics_in_title: bool = True,
) -> None:
    outdir = Path(pso_output_root) / f"sensors_{number_of_sensors}"
    figs_dir = outdir / figures_subdir
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ✅ usa o mesmo env do experimento
    env = _load_env_from_config(config_path)
    _ = GridGeometry(env)

    # impacts dir
    impacts_dir = outdir / impacts_subdir
    if not impacts_dir.exists():
        impacts_dir = outdir / fallback_impacts_subdir
    if not impacts_dir.exists():
        raise FileNotFoundError(f"Missing impacts dir: {impacts_dir}")

    # PSO arrays
    pso_hist_path = outdir / pso_best_positions_per_iter_rel
    pso_best_path = outdir / pso_best_global_rel
    pso_hist = np.load(pso_hist_path).astype(float)
    pso_best_global_vec = np.load(pso_best_path).astype(float)
    n_iters = int(pso_hist.shape[0])

    # reports
    reports_path = outdir / pso_reports_jsonl_rel
    reports_by_iter: Dict[int, Dict[str, Any]] = {}
    if reports_path.exists():
        reports_by_iter = _index_reports_by_iter(_read_best_reports_jsonl(reports_path))

    def metrics_for_iter(it: int, *, which: str) -> Optional[Dict[str, Any]]:
        obj = reports_by_iter.get(int(it))
        if not obj:
            return None

        block = obj.get(which)  # "best_global" ou "best_of_generation"
        if not isinstance(block, dict):
            return None

        num_imp = float(block.get("number_of_impacts", 0) or 0)
        no_cov = float(block.get("number_of_impacts_without_coverage", 0) or 0)

        return {
            "total_cost": block.get("total_cost"),
            "mean_error_m": block.get("mean_localization_error_meters", block.get("mean_error_m")),
            "no_coverage_rate": (no_cov / num_imp) if num_imp > 0 else None,
        }

    # -------------------------
    # MODE A (corrigido): best-so-far da iteração em cima dos impactos daquela iteração
    # -------------------------
    if plot_best_so_far_on_all_iterations:
        impact_files = sorted(impacts_dir.glob("impact_points_gen_*.npy"))
        if not impact_files:
            raise FileNotFoundError(f"No impact_points_gen_*.npy found in {impacts_dir}")

        for p in impact_files:
            it = int(p.stem.split("_")[-1])
            if it < 0 or it >= n_iters:
                continue

            impacts = np.load(p)
            sensors = _vec_to_sensors(pso_hist[it], number_of_sensors, env)

            title = f"{title_prefix} (Best-so-far) on Iter {it}"
            if include_metrics_in_title:
                m = metrics_for_iter(it, which="best_global")
                title += f" — {_fmt_metrics_from_row(m)}"

            plot_scene(
                out_png=figs_dir / f"scene_pso_best_so_far_on_iter_{it:04d}.png",
                title=title,
                impacts_xy=impacts,
                sensors=sensors,
                env=env,
                show_detection=show_detection,
                highlight_out_of_coverage=True,
                out_of_coverage_mask=None,
                min_sensors_for_coverage=min_sensors_for_coverage,
                show_depth_labels=show_depth_labels,
                covered_color=covered_color,
                uncovered_color=uncovered_color,
                show_sensor_polygon_links=show_sensor_polygon_links,
                polygon_line_color=polygon_line_color,
                polygon_line_style=polygon_line_style,
                polygon_line_width=polygon_line_width,
                polygon_line_alpha=polygon_line_alpha,
                distance_decimals=distance_decimals,
                distance_unit=distance_unit,
            )

        print(f"Saved PSO BEST-SO-FAR (per-iter) figures to: {figs_dir}")

    # -------------------------
    # MODE A opcional: 1 figura do best_global
    # -------------------------
    if plot_best_global_only:
        impacts = load_impact_points_to_plot(impacts_dir, snapshots_iteration_index)
        sensors = _vec_to_sensors(pso_best_global_vec, number_of_sensors, env)

        plot_scene(
            out_png=figs_dir / f"scene_pso_best_global_on_impacts_{snapshots_iteration_index:04d}.png",
            title=f"{title_prefix} best_global — impacts@{snapshots_iteration_index}",
            impacts_xy=impacts,
            sensors=sensors,
            env=env,
            show_detection=show_detection,
            highlight_out_of_coverage=True,
            out_of_coverage_mask=None,
            min_sensors_for_coverage=min_sensors_for_coverage,
            show_depth_labels=show_depth_labels,
            covered_color=covered_color,
            uncovered_color=uncovered_color,
            show_sensor_polygon_links=show_sensor_polygon_links,
            polygon_line_color=polygon_line_color,
            polygon_line_style=polygon_line_style,
            polygon_line_width=polygon_line_width,
            polygon_line_alpha=polygon_line_alpha,
            distance_decimals=distance_decimals,
            distance_unit=distance_unit,
        )

    # -------------------------
    # MODE B: snapshots em um único impacts (ok)
    # -------------------------
    if plot_snapshots_on_single_iteration_impacts:
        impacts = load_impact_points_to_plot(impacts_dir, snapshots_iteration_index)

        idxs = list(range(0, n_iters, max(1, int(snapshot_every))))
        if (n_iters - 1) not in idxs:
            idxs.append(n_iters - 1)

        if len(idxs) > max_snapshots:
            keep = [idxs[0]]
            mid = idxs[1:-1]
            if mid:
                step = max(1, len(mid) // max(1, (max_snapshots - 2)))
                keep.extend(mid[::step])
            keep.append(idxs[-1])
            idxs = keep[:max_snapshots]

        for it in idxs:
            sensors = _vec_to_sensors(pso_hist[int(it)], number_of_sensors, env)

            title = f"{title_prefix} snapshot (best-so-far) iter {it} — impacts@{snapshots_iteration_index}"
            if include_metrics_in_title:
                m = metrics_for_iter(it, which="best_global")
                title += f" — {_fmt_metrics_from_row(m)}"

            plot_scene(
                out_png=figs_dir / f"scene_pso_snapshot_iter_{it:04d}_on_impacts_{snapshots_iteration_index:04d}.png",
                title=title,
                impacts_xy=impacts,
                sensors=sensors,
                env=env,
                show_detection=show_detection,
                highlight_out_of_coverage=True,
                out_of_coverage_mask=None,
                min_sensors_for_coverage=min_sensors_for_coverage,
                show_depth_labels=show_depth_labels,
                covered_color=covered_color,
                uncovered_color=uncovered_color,
                show_sensor_polygon_links=show_sensor_polygon_links,
                polygon_line_color=polygon_line_color,
                polygon_line_style=polygon_line_style,
                polygon_line_width=polygon_line_width,
                polygon_line_alpha=polygon_line_alpha,
                distance_decimals=distance_decimals,
                distance_unit=distance_unit,
            )

        print(f"Saved PSO SNAPSHOT figures to: {figs_dir}")


if __name__ == "__main__":
    generate_pso_scenario_figures(
        number_of_sensors=5,
        config_path="experiment_config.json",
        impacts_subdir="impacts_pso",
        fallback_impacts_subdir="impacts",
        plot_best_so_far_on_all_iterations=True,
        plot_best_global_only=False,
        plot_snapshots_on_single_iteration_impacts=True,
        snapshots_iteration_index=0,
        snapshot_every=50,
        max_snapshots=14,
        show_sensor_polygon_links=False,
    )