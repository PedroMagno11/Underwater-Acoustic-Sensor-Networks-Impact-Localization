from __future__ import annotations

import os
import json
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from acoustic.sound_speed_profile_builder import build_ssp
from geometry.grid_geometry import GridGeometry
from regular_polygon.regular_polygon import create_regular_polygon_chromosome
from settings.environment_settings import EnvironmentSettings
from settings.logging_settings import setup_logging
from settings.performance_settings import PerformanceSettings
from settings.simulation_settings import SimulationSettings
from utils.loaders import load_all_settings
from performance.parallel_evaluation import evaluate_population

# IMPORTANT:
# This MUST point to the same evaluator used by GA to produce:
#  - total_cost
#  - mean_error_m
#  - no_coverage_rate
#
os.environ["MPLBACKEND"] = "Agg"
LOGGER_NAME = "underwater_sensor_ga.regular_polygon_baseline_runner"


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


def _sector_deg(n: int) -> float:
    # regular polygon has rotational symmetry of 360/n
    return 360.0 / float(n)


def _build_polygon_chromosome(
    *,
    n_sensors: int,
    env: EnvironmentSettings,
    grid: GridGeometry,
    radius_m: Optional[float],
    depth_m: float,
    angle_offset_deg: float,
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


def _pick_depth(env: EnvironmentSettings, strategy: str, fixed: Optional[float]) -> float:
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


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _best_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # We assume the GA evaluation produces a 'total_cost' that you minimize.
    # If your cost column has another name, change here.
    def key(r: Dict[str, Any]) -> float:
        v = _safe_float(r.get("total_cost"))
        if v is None:
            # push unknown costs to the end
            return float("inf")
        return float(v)

    best = min(rows, key=key)
    if key(best) == float("inf"):
        raise RuntimeError("Could not select best row: all rows have missing total_cost.")
    return best


def _default_offsets(n: int, step_deg: float) -> List[float]:
    # Because of symmetry, you only need [0, 360/n).
    # We include endpoint only if it lands exactly (usually not needed).
    sector = _sector_deg(n)
    count = int(np.floor(sector / step_deg))
    offsets = [round(i * step_deg, 10) for i in range(count + 1)]
    # drop last if equals sector (duplicated configuration)
    if offsets and abs(offsets[-1] - sector) < 1e-9:
        offsets.pop()
    return offsets


def _to_row(*, n: int, angle: float, report: Any) -> Dict[str, Any]:
    total_cost = getattr(report, "total_cost", None)

    mean_error_m = getattr(report, "mean_localization_error_meters", None)

    num_impacts = getattr(report, "number_of_impacts", None)
    num_no_cov = getattr(report, "number_of_impacts_without_coverage", None)

    # no_coverage_rate = fraction of impacts without coverage
    no_coverage_rate: Optional[float] = None
    if num_impacts is not None and num_no_cov is not None and float(num_impacts) > 0:
        no_coverage_rate = float(num_no_cov) / float(num_impacts)

    return {
        "number_of_sensors": n,
        "angle_offset_deg": float(angle),
        "total_cost": total_cost,
        "mean_error_m": mean_error_m,                 # now filled
        "no_coverage_rate": no_coverage_rate,         # now filled
        "number_of_impacts": num_impacts,
        "number_of_localizable_impacts": getattr(report, "number_of_localizable_impacts", None),
        "number_of_impacts_without_coverage": num_no_cov,
        "coverage_penalty": getattr(report, "coverage_penalty", None),
        "separation_penalty": getattr(report, "separation_penalty", None),
    }


def run_polygon_baseline_for_n(
    *,
    n: int,
    environment_settings: EnvironmentSettings,
    simulation_settings: SimulationSettings,
    performance_settings: PerformanceSettings,
    sound_speed_profile: Any,

    output_root: str = "outputs",
    baseline_subdir: str = "polygon_baseline",

    offsets_deg: Optional[Sequence[float]] = None,
    step_deg: float = 1.0,

    polygon_radius_meters: Optional[float] = None,
    depth_strategy: str = "mid",
    fixed_depth_m: Optional[float] = None,

    # Evaluation seed controls
    global_seed: int = 123,
    scenario_seed: int = 123,
    generation_index: int = 0,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """
    Generates:
      <output_root>/<baseline_subdir>/sensors_<n>/polygon_baseline_metrics.csv

    Returns:
      (csv_path, rows)
    """
    logger = logging.getLogger(LOGGER_NAME)

    base_dir = _ensure_dir(Path(output_root) / baseline_subdir)
    n_dir = _ensure_dir(base_dir / f"sensors_{n}")
    csv_path = n_dir / "polygon_baseline_metrics.csv"

    env = environment_settings
    grid = GridGeometry(env)
    depth_m = _pick_depth(env, depth_strategy, fixed_depth_m)

    if offsets_deg is None:
        offsets = _default_offsets(n, step_deg=step_deg)
    else:
        offsets = [float(x) for x in offsets_deg]

    if not offsets:
        raise ValueError("No angle offsets provided/resolved.")

    logger.info("Baseline N=%d | offsets=%d | sector=%.3f° | step=%.3f°", n, len(offsets), _sector_deg(n), step_deg)

    rows: List[Dict[str, Any]] = []

    # Evaluate each angle as a single-individual population (reuses GA pipeline).
    for angle in offsets:
        chr_poly = _build_polygon_chromosome(
            n_sensors=n,
            env=env,
            grid=grid,
            radius_m=polygon_radius_meters,
            depth_m=depth_m,
            angle_offset_deg=float(angle),
        )

        reports = evaluate_population(
            population=[chr_poly],
            number_of_sensors=n,
            environment_settings=env,
            simulation_settings=simulation_settings,
            sound_speed_profile=sound_speed_profile,
            generation_index=generation_index,
            global_seed=global_seed,
            performance_settings=performance_settings,
        )

        if not reports or len(reports) != 1:
            raise RuntimeError(f"Unexpected reports result for angle={angle}: {type(reports)} len={len(reports) if reports else 0}")

        report = reports[0]
        row = _to_row(n=n, angle=float(angle), report=report)
        rows.append(row)

        tc = _safe_float(row.get("total_cost"))
        me = _safe_float(row.get("mean_error_m"))
        nc = _safe_float(row.get("no_coverage_rate"))
        logger.info("N=%d angle=%7.3f° | total_cost=%s | mean_error_m=%s | no_cov=%s",
                    n, angle,
                    f"{tc:.6f}" if tc is not None else "n/a",
                    f"{me:.6f}" if me is not None else "n/a",
                    f"{100*nc:.3f}%" if nc is not None else "n/a")

    _write_csv(csv_path, rows)
    logger.info("Saved baseline CSV: %s", csv_path)
    return csv_path, rows


def write_polygon_baseline_summary(
    *,
    output_root: str = "outputs",
    baseline_subdir: str = "polygon_baseline",
    by_n: Dict[int, Dict[str, Any]],
) -> Path:
    base_dir = _ensure_dir(Path(output_root) / baseline_subdir)
    summary_path = base_dir / "polygon_baseline_summary.json"

    payload = {
        "by_n": {str(k): v for k, v in by_n.items()},
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    logging.getLogger(LOGGER_NAME).info("Saved baseline summary JSON: %s", summary_path)
    return summary_path


def main() -> None:
    setup_logging(log_level=logging.INFO, log_file_path="execution_polygon_baseline.log", log_to_console=True)
    logger = logging.getLogger(LOGGER_NAME)

    (
        environment_settings,
        _genetic_algorithm_settings,
        simulation_settings,
        performance_settings,
        _pso_settings,
    ) = load_all_settings("experiment_config.json")

    ssp = build_ssp()

    output_root = "outputs"
    baseline_subdir = "polygon_baseline"

    # Choose Ns here
    Ns = [3, 4, 5]

    # Angle sweep configuration:
    # step_deg=1.0 => 0..(360/n) with 1 degree step (symmetry reduced search)
    step_deg = 1.0

    # Depth/polygon radius behavior should match your scenario plots
    depth_strategy = "mid"
    polygon_radius_meters = None  # None => env.target_region_radius

    # Seeds used by the evaluator (keep deterministic)
    global_seed = getattr(_genetic_algorithm_settings, "random_seed", 123) if _genetic_algorithm_settings else 123
    scenario_seed = global_seed
    generation_index = 0  # baseline computed for gen 0 impacts (consistent & deterministic)

    by_n_summary: Dict[int, Dict[str, Any]] = {}

    for n in Ns:
        csv_path, rows = run_polygon_baseline_for_n(
            n=n,
            environment_settings=environment_settings,
            simulation_settings=simulation_settings,
            performance_settings=performance_settings,
            sound_speed_profile=ssp,
            output_root=output_root,
            baseline_subdir=baseline_subdir,
            offsets_deg=None,   # auto
            step_deg=step_deg,
            polygon_radius_meters=polygon_radius_meters,
            depth_strategy=depth_strategy,
            fixed_depth_m=None,
            global_seed=int(global_seed),
            scenario_seed=int(scenario_seed),
            generation_index=int(generation_index),
        )

        best = _best_row(rows)
        by_n_summary[n] = {
            "best_angle_offset_deg": float(best["angle_offset_deg"]),
            "best_total_cost": float(best["total_cost"]),
            "best_mean_error_m": float(best["mean_error_m"]) if best.get("mean_error_m") is not None else None,
            "best_no_coverage_rate": float(best["no_coverage_rate"]) if best.get("no_coverage_rate") is not None else None,
            "metrics_csv": str(csv_path.as_posix()),
            "sector_deg": _sector_deg(n),
            "step_deg": float(step_deg),
        }

        logger.info("BEST N=%d => angle=%.3f° | total_cost=%.6f",
                    n, by_n_summary[n]["best_angle_offset_deg"], by_n_summary[n]["best_total_cost"])

    write_polygon_baseline_summary(
        output_root=output_root,
        baseline_subdir=baseline_subdir,
        by_n=by_n_summary,
    )

    logger.info("Polygon baseline finished. Output at: %s", (Path(output_root) / baseline_subdir))


if __name__ == "__main__":
    main()