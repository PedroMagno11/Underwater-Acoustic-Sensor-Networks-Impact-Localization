from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

def make_impact_saver(output_dir: Path, scenario_seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)


    def _save(generation_index: int, impact_seed: int, impact_points: List[Tuple[float, float]]) -> None:
        arr = np.asarray(impact_points, dtype=float)
        np.save(output_dir / f"impact_points_gen_{generation_index:04d}.npy", arr)

        meta = {
            "generation_index": generation_index,
            "scenario_seed": scenario_seed,
            "impact_seed": impact_seed,
            "num_impacts": int(arr.shape[0]),
        }
        (output_dir / f"impact_points_gen_{generation_index:04d}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


    return _save