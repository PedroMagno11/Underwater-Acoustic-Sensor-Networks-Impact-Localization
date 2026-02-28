from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from settings.environment_settings import EnvironmentSettings
from settings.genetic_algorithm_settings import GeneticAlgorithmSettings
from settings.particle_swarm_settings import ParticleSwarmSettings
from settings.performance_settings import PerformanceSettings
from settings.simulation_settings import SimulationSettings

def load_json_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _dataclass_from_dict(cls, data: Dict[str, Any]):
    return cls(**data)

def load_settings_from_config(config: Dict[str, Any]) -> Tuple[
    EnvironmentSettings,
    GeneticAlgorithmSettings,
    SimulationSettings,
    PerformanceSettings,
    ParticleSwarmSettings,   # NEW
]:
    env = _dataclass_from_dict(EnvironmentSettings, config.get("environment", {}))
    ga = _dataclass_from_dict(GeneticAlgorithmSettings, config.get("genetic_algorithm", {}))
    sim = _dataclass_from_dict(SimulationSettings, config.get("simulation", {}))
    perf = _dataclass_from_dict(PerformanceSettings, config.get("performance", {}))
    # NEW
    pso = _dataclass_from_dict(ParticleSwarmSettings, config.get("particle_swarm", {}))

    return env, ga, sim, perf, pso


def load_all_settings(config_path: str) -> Tuple[
    EnvironmentSettings,
    GeneticAlgorithmSettings,
    SimulationSettings,
    PerformanceSettings,
    ParticleSwarmSettings
]:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}. "
            f"Create it or copy the provided 'expirement_config.json"
        )

    config = load_json_config(config_path)
    return load_settings_from_config(config)

def load_impact_points_to_plot(impacts_dir: Path, gen: int) -> np.ndarray:
    p = impacts_dir / f"impact_points_gen_{gen:04d}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Impact file not found: {p}")
    return np.load(p)


def load_best_reports_jsonl(path: Path) -> Dict[int, Dict[str, dict]]:
    """
    Espera linhas JSON com algo como:
      {
        "generation_index": 0,
        "best_of_generation": {... EvaluationReport ...},
        "best_global": {... EvaluationReport ...},
        ...
      }

    Retorna:
      reports[generation]["GA_best_gen"] -> dict
      reports[generation]["GA_best_global"] -> dict
    """
    if not path.exists():
        raise FileNotFoundError(f"best_reports.jsonl not found: {path}")

    out: Dict[int, Dict[str, dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            gen = obj.get("generation_index", obj.get("generation", None))
            if gen is None:
                raise KeyError(f"[line {line_idx}] Missing generation_index/generation. Keys={list(obj.keys())}")

            best_gen = obj.get("best_of_generation", obj.get("best_generation", obj.get("best_gen", None)))
            best_global = obj.get("best_global", obj.get("global_best", None))

            if best_gen is None or best_global is None:
                raise KeyError(
                    f"[line {line_idx}] Missing best_of_generation and/or best_global. Keys={list(obj.keys())}"
                )

            out.setdefault(int(gen), {})
            out[int(gen)]["GA_best_gen"] = best_gen
            out[int(gen)]["GA_best_global"] = best_global

    return out