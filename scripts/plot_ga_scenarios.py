from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from genetic_algorithm.chromosome import chromosome_converter
from geometry.grid_geometry import GridGeometry
from settings.environment_settings import EnvironmentSettings
from utils.loaders import load_best_reports_jsonl, load_impact_points_to_plot
from visualization.plot_scenarios import plot_scene


# =========================
#  Plot helpers
# =========================

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



def _require_keys(rep: dict, keys: List[str], context: str) -> Any:
    for k in keys:
        if k in rep and rep[k] is not None:
            return rep[k]
    raise KeyError(f"Missing keys {keys} in report ({context}). Available keys={list(rep.keys())}")


def fmt_from_report_dict(rep: dict) -> str:
    total_cost = _require_keys(rep, ["total_cost", "cost"], "total_cost")
    mean_err = _require_keys(rep, ["mean_localization_error_meters", "mean_error_meters", "mean_error_m"], "mean_err")
    n_impacts = _require_keys(rep, ["number_of_impacts", "num_impacts"], "num_impacts")
    n_no_cov = _require_keys(rep, ["number_of_impacts_without_coverage", "impacts_without_coverage"], "no_cov")

    n_impacts_i = int(n_impacts)
    n_no_cov_i = int(n_no_cov)
    no_cov_rate = (n_no_cov_i / max(1, n_impacts_i))

    return f"cost={float(total_cost):.3f} | mean_err={float(mean_err):.2f} m | no_cov={100*no_cov_rate:.1f}%"


def extract_row_from_report(gen: int, label: str, rep: dict) -> dict:
    # Campos mínimos
    total_cost = float(_require_keys(rep, ["total_cost", "cost"], "total_cost"))
    mean_err = float(_require_keys(rep, ["mean_localization_error_meters", "mean_error_meters", "mean_error_m"], "mean_err"))
    n_impacts = int(_require_keys(rep, ["number_of_impacts", "num_impacts"], "num_impacts"))
    n_no_cov = int(_require_keys(rep, ["number_of_impacts_without_coverage", "impacts_without_coverage"], "no_cov"))
    no_cov_rate = n_no_cov / max(1, n_impacts)

    localizable = rep.get("number_of_localizable_impacts", rep.get("localizable_impacts", 0))

    return {
        "generation": int(gen),
        "label": str(label),
        "total_cost": total_cost,
        "mean_error_m": mean_err,
        "no_coverage_rate": float(no_cov_rate),
        "localizable_impacts": int(localizable) if localizable is not None else 0,
        "num_impacts": int(n_impacts),
    }


# =========================
#  Main (GA only, plot-only)
# =========================

def generate_ga_scenario_figures(
    *,
    number_of_sensors: int,
    output_root: str = "outputs",

    # visual settings
    figures_subdir: str = "figures_ga_only",
    covered_color: str = "gold",
    uncovered_color: str = "tab:red",
    min_sensors_for_coverage: int = 3,
    show_detection: bool = True,
    show_depth_labels: bool = True,

    # ✅ habilitar/desabilitar linhas + distâncias
    show_sensor_polygon_links: bool = True,
    polygon_line_color: str = "black",
    polygon_line_style: str = "--",
    polygon_line_width: float = 1.2,
    polygon_line_alpha: float = 0.65,
    distance_decimals: int = 1,
    distance_unit: str = "m",

    write_csv: bool = True,
) -> None:

    outdir = Path(output_root) / f"sensors_{number_of_sensors}"
    impacts_dir = outdir / "impacts"
    figs_dir = outdir / figures_subdir

    # arquivos essenciais
    best_gen_path = outdir / "best_chromosomes_per_generation.npy"
    best_global_final_path = outdir / "best_global_chromosome.npy"
    best_global_per_gen_path = outdir / "best_global_chromosomes_per_generation.npy"
    reports_jsonl_path = outdir / "best_reports.jsonl"

    if not best_gen_path.exists():
        raise FileNotFoundError(f"Missing: {best_gen_path}")
    if not best_global_final_path.exists():
        raise FileNotFoundError(f"Missing: {best_global_final_path}")
    if not reports_jsonl_path.exists():
        raise FileNotFoundError(f"Missing: {reports_jsonl_path}")
    if not impacts_dir.exists():
        raise FileNotFoundError(f"Missing impacts dir: {impacts_dir}")

    # settings (para converter cromossomo -> sensores e para plotar cenário)
    env = EnvironmentSettings()
    grid = GridGeometry(env)

    best_chrs = np.load(best_gen_path)                 # (G, L)
    best_global_final = np.load(best_global_final_path)  # (L,)
    num_generations = int(best_chrs.shape[0])

    # global por geração (se existir)
    best_global_per_gen = None
    if best_global_per_gen_path.exists():
        best_global_per_gen = np.load(best_global_per_gen_path)  # (G, L)
        if int(best_global_per_gen.shape[0]) != num_generations:
            raise ValueError(
                f"{best_global_per_gen_path} has {best_global_per_gen.shape[0]} generations, "
                f"but best_chromosomes_per_generation has {num_generations}."
            )

    # métricas prontas do GA
    reports_by_gen = load_best_reports_jsonl(reports_jsonl_path)

    rows: List[dict] = []
    csv_path = figs_dir / "ga_only_comparison_metrics.csv"

    for gen in range(num_generations):
        impacts = load_impact_points_to_plot(impacts_dir, gen)

        # sensores GA best daquela geração
        best_gen_chr = best_chrs[gen]
        ga_sensors_gen = chromosome_converter(best_gen_chr, number_of_sensors, grid, env)

        # sensores GA global
        if best_global_per_gen is not None:
            global_chr = best_global_per_gen[gen]
        else:
            # fallback: global final
            global_chr = best_global_final
        ga_sensors_global = chromosome_converter(global_chr, number_of_sensors, grid, env)

        # títulos vindos do report salvo (sem reevaluar)
        if gen not in reports_by_gen:
            raise KeyError(f"Generation {gen} not found in best_reports.jsonl")

        rep_gen = reports_by_gen[gen]["GA_best_gen"]
        rep_global = reports_by_gen[gen]["GA_best_global"]

        title_gen = f"GA Best (Gen {gen}) — {fmt_from_report_dict(rep_gen)}"
        title_global = f"GA Best Global (on Gen {gen}) — {fmt_from_report_dict(rep_global)}"

        plot_scene(
            out_png=figs_dir / f"scene_ga_best_gen_{gen:04d}.png",
            title=title_gen,
            impacts_xy=impacts,
            sensors=ga_sensors_gen,
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

        plot_scene(
            out_png=figs_dir / f"scene_ga_best_global_on_gen_{gen:04d}.png",
            title=title_global,
            impacts_xy=impacts,
            sensors=ga_sensors_global,
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

        if write_csv:
            rows.append(extract_row_from_report(gen, "GA_best_gen", rep_gen))
            rows.append(extract_row_from_report(gen, "GA_best_global", rep_global))

    if write_csv and rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"Saved {num_generations * 2} GA figures to: {figs_dir}")
    if write_csv and rows:
        print(f"Saved GA comparison CSV to: {csv_path}")


if __name__ == "__main__":
    for n in [3,4,5]:
        generate_ga_scenario_figures(
            number_of_sensors=n,
            output_root="outputs",
            figures_subdir="figures_ga_only",
            covered_color="green",
            uncovered_color="tab:red",
            min_sensors_for_coverage=3,
            show_detection=True,
            show_depth_labels=True,

            show_sensor_polygon_links=False,   # <-- mude pra False para desligar
            polygon_line_color="black",
            polygon_line_style="--",
            polygon_line_width=1.2,
            polygon_line_alpha=0.65,
            distance_decimals=1,
            distance_unit="m",

            write_csv=True,
        )