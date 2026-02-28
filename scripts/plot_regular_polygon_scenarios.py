from __future__ import annotations

import os
import json
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from acoustic.sound_speed_profile_builder import build_ssp
from genetic_algorithm.chromosome import chromosome_converter
from geometry.grid_geometry import GridGeometry
from regular_polygon.regular_polygon import create_regular_polygon_chromosome
from settings.environment_settings import EnvironmentSettings
from settings.logging_settings import setup_logging
from utils.loaders import load_all_settings, load_impact_points_to_plot
from visualization.plot_scenarios import plot_scene

os.environ["MPLBACKEND"] = "Agg"
LOGGER_NAME = "underwater_sensor_ga.plot_regular_polygon_scenarios"


def pick_depth(env: EnvironmentSettings, strategy: str, fixed: Optional[float]) -> float:
    zmin = float(env.minimum_depth_in_meters)
    zmax = float(env.maximum_depth_in_meters)

    if strategy == "mid":
        return (zmin + zmax) / 2.0
    if strategy == "min":
        return zmin
    if strategy == "max":
        return zmax
    if strategy == "fixed":
        if fixed is None:
            raise ValueError("fixed_depth_m must be set when depth_strategy='fixed'.")
        if not (zmin <= fixed <= zmax):
            raise ValueError(f"fixed_depth_m={fixed} out of range [{zmin},{zmax}].")
        return float(fixed)
    raise ValueError("depth_strategy must be one of: mid|min|max|fixed")


def _resolve_polygon_root(polygon_baseline_root: str | Path) -> Path:
    base = Path(polygon_baseline_root)

    if (base / "polygon_baseline_summary.json").exists():
        return base

    candidate = base / "polygon_baseline"
    if (candidate / "polygon_baseline_summary.json").exists():
        return candidate

    return base


def _read_polygon_metrics_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing polygon baseline metrics CSV: {csv_path}")
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _compute_no_cov_rate(row: Dict[str, Any]) -> Optional[float]:
    """
    Supports both schemas:
      - old: no_coverage_rate
      - new: number_of_impacts_without_coverage / number_of_impacts
    """
    direct = _safe_float(row.get("no_coverage_rate"))
    if direct is not None:
        return direct

    total = _safe_float(row.get("number_of_impacts"))
    no_cov = _safe_float(row.get("number_of_impacts_without_coverage"))

    if total is None or no_cov is None or total <= 0:
        return None

    return float(no_cov) / float(total)


def _get_mean_err_m(row: Dict[str, Any]) -> Optional[float]:
    """
    Supports both schemas:
      - old: mean_error_m
      - new: mean_localization_error_meters
    """
    v = _safe_float(row.get("mean_error_m"))
    if v is not None:
        return v
    return _safe_float(row.get("mean_localization_error_meters"))


def _fmt_metrics_from_row(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return "metrics=n/a"

    cost = _safe_float(row.get("total_cost"))
    mean_err = _get_mean_err_m(row)
    no_cov_rate = _compute_no_cov_rate(row)

    if cost is None or mean_err is None or no_cov_rate is None:
        return "metrics=n/a"

    return f"cost={cost:.3f} | mean_err={mean_err:.2f} m | no_cov={100 * no_cov_rate:.1f}%"


def _load_best_offset_from_summary(summary_path: Path, n: int) -> float:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing polygon summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    by_n = obj.get("by_n", {})
    entry = by_n.get(str(n))
    if not entry:
        raise KeyError(f"Summary does not contain by_n[{n}]")
    return float(entry["best_angle_offset_deg"])


def _find_row_by_angle(metrics_rows: List[Dict[str, Any]], angle: float) -> Optional[Dict[str, Any]]:
    for r in metrics_rows:
        a = _safe_float(r.get("angle_offset_deg"))
        if a is None:
            continue
        if abs(a - float(angle)) < 1e-9:
            return r
    return None


def _safe_angle_slug(angle: float) -> str:
    s = f"{float(angle):06.2f}".strip()
    s = s.replace("-", "m").replace(".", "p")
    s = s.replace(" ", "0")
    return s


def build_polygon_chromosome(
    *,
    n_sensors: int,
    env: EnvironmentSettings,
    grid: GridGeometry,
    radius_m: Optional[float],
    angle_offset_deg: float,
    depth_m: float,
) -> np.ndarray:
    if radius_m is None:
        radius_m = float(env.target_region_radius)

    return create_regular_polygon_chromosome(
        number_of_sensors=n_sensors,
        environment_settings=env,
        grid_geometry=grid,
        polygon_radius_meters=float(radius_m),
        depth_meters=float(depth_m),
        angle_offset_degrees=float(angle_offset_deg),
    )


def generate_polygon_scenario_figures(
    *,
    number_of_sensors: int,
    environment_settings: EnvironmentSettings,
    ga_output_root: str = "outputs",
    polygon_baseline_root: str = "outputs/polygon_baseline",
    figures_subdir: str = "figures_polygon",
    plot_best_polygon_on_all_generations: bool = True,
    plot_all_offsets_on_single_generation: bool = True,
    offsets_generation_index: int = 0,
    angle_offsets_deg: Optional[Sequence[float]] = None,
    polygon_radius_meters: Optional[float] = None,
    depth_strategy: str = "mid",
    fixed_depth_m: Optional[float] = None,
    min_sensors_for_coverage: int = 3,
    show_detection: bool = True,
    show_depth_labels: bool = True,
    covered_color: str = "gold",
    uncovered_color: str = "tab:red",
    show_sensor_polygon_links: bool = True,
    polygon_line_color: str = "black",
    polygon_line_style: str = "--",
    polygon_line_width: float = 1.2,
    polygon_line_alpha: float = 0.65,
    distance_decimals: int = 1,
    distance_unit: str = "m",
    title_prefix: str = "Regular Polygon",
    include_baseline_metrics_in_title: bool = True,
) -> Path:
    outdir = Path(ga_output_root) / f"sensors_{number_of_sensors}"
    impacts_dir = outdir / "impacts"
    figs_dir = outdir / figures_subdir
    figs_dir.mkdir(parents=True, exist_ok=True)

    if not impacts_dir.exists():
        raise FileNotFoundError(f"Missing impacts dir: {impacts_dir}")

    env = environment_settings
    grid = GridGeometry(env)
    depth_m = pick_depth(env, depth_strategy, fixed_depth_m)

    poly_root = _resolve_polygon_root(polygon_baseline_root)
    poly_dir = poly_root / f"sensors_{number_of_sensors}"
    metrics_csv = poly_dir / "polygon_baseline_metrics.csv"
    summary_json = poly_root / "polygon_baseline_summary.json"

    if not summary_json.exists():
        raise FileNotFoundError(
            f"polygon_baseline_summary.json not found at: {summary_json}\n"
            f"Run: python -m executable.regular_polygon_runner"
        )

    metrics_rows = _read_polygon_metrics_csv(metrics_csv)

    if angle_offsets_deg is not None:
        offsets = [float(x) for x in angle_offsets_deg]
    else:
        vals: List[float] = []
        for r in metrics_rows:
            a = _safe_float(r.get("angle_offset_deg"))
            if a is not None:
                vals.append(a)
        offsets = sorted(set(vals))

    if not offsets:
        raise RuntimeError(f"No valid angle offsets found in CSV: {metrics_csv}")

    best_offset = _load_best_offset_from_summary(summary_json, number_of_sensors)

    best_chr = build_polygon_chromosome(
        n_sensors=number_of_sensors,
        env=env,
        grid=grid,
        radius_m=polygon_radius_meters,
        angle_offset_deg=best_offset,
        depth_m=depth_m,
    )
    best_sensors = chromosome_converter(best_chr, number_of_sensors, grid, env)

    if plot_best_polygon_on_all_generations:
        impact_files = sorted(impacts_dir.glob("impact_points_gen_*.npy"))
        if not impact_files:
            raise FileNotFoundError(f"No impact_points_gen_*.npy found in {impacts_dir}")

        best_row = _find_row_by_angle(metrics_rows, best_offset)
        best_metrics_str = _fmt_metrics_from_row(best_row) if include_baseline_metrics_in_title else ""

        for p in impact_files:
            try:
                gen = int(p.stem.split("_")[-1])
            except Exception:
                gen = -1

            impacts = np.load(p)

            title = f"{title_prefix} (Best Offset {best_offset:.2f}°) on Gen {gen}"
            if best_metrics_str:
                title += f" — {best_metrics_str}"

            plot_scene(
                out_png=figs_dir / f"scene_polygon_best_on_gen_{gen:04d}.png",
                title=title,
                impacts_xy=impacts,
                sensors=best_sensors,
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

    if plot_all_offsets_on_single_generation:
        impacts = load_impact_points_to_plot(impacts_dir, offsets_generation_index)

        for angle in offsets:
            chr_poly = build_polygon_chromosome(
                n_sensors=number_of_sensors,
                env=env,
                grid=grid,
                radius_m=polygon_radius_meters,
                angle_offset_deg=float(angle),
                depth_m=depth_m,
            )
            sensors = chromosome_converter(chr_poly, number_of_sensors, grid, env)

            row = _find_row_by_angle(metrics_rows, float(angle))
            metrics_str = _fmt_metrics_from_row(row) if include_baseline_metrics_in_title else ""

            title = f"{title_prefix} (Offset {angle:.2f}°) on Gen {offsets_generation_index}"
            if metrics_str:
                title += f" — {metrics_str}"

            angle_slug = _safe_angle_slug(float(angle))

            plot_scene(
                out_png=figs_dir / f"scene_polygon_offset_{angle_slug}_on_gen_{offsets_generation_index:04d}.png",
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

    return figs_dir


def main() -> None:
    setup_logging(log_level=logging.INFO, log_file_path="execution_plot_polygon.log", log_to_console=True)
    logger = logging.getLogger(LOGGER_NAME)

    (
        environment_settings,
        _genetic_algorithm_settings,
        simulation_settings,
        performance_settings,
        _pso_settings,
    ) = load_all_settings("experiment_config.json")

    _ = (simulation_settings, performance_settings)
    _ssp = build_ssp()

    for n in [3, 4, 5]:
        figs_dir = generate_polygon_scenario_figures(
            number_of_sensors=n,
            environment_settings=environment_settings,
            ga_output_root="outputs",
            polygon_baseline_root="outputs/polygon_baseline",
            figures_subdir="figures_polygon",
            plot_best_polygon_on_all_generations=True,
            plot_all_offsets_on_single_generation=True,
            offsets_generation_index=0,
            polygon_radius_meters=None,
            depth_strategy="mid",
            fixed_depth_m=None,
            min_sensors_for_coverage=3,
            show_detection=True,
            show_depth_labels=True,
            covered_color="green",
            uncovered_color="tab:red",
            show_sensor_polygon_links=False,
            title_prefix="Regular Polygon",
            include_baseline_metrics_in_title=True,
        )
        logger.info("Saved polygon figures (N=%d) to: %s", n, figs_dir)


if __name__ == "__main__":
    main()