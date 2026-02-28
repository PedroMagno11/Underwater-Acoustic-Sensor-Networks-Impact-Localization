from __future__ import annotations

def seed_for_impact_position(global_seed: int, target_index: int) -> int:
    x = (global_seed * 1_000_003) ^ (target_index * 100_003)
    return x & 0xFFFFFFFF

def compute_generation_impact_seed(scenario_seed: int) -> int:
    # return ((scenario_seed * 1_000_003) ^ (generation_index * 100_003)) & 0xFFFFFFFF
    return (scenario_seed * 1_000_003) & 0xFFFFFFFF

def compute_job_seed(global_seed: int, generation_index: int, chromosome_index: int) -> int:
    return (
        (global_seed * 1_000_003)
        ^ (generation_index * 100_003)
        ^ (chromosome_index * 10_003)
    ) & 0xFFFFFFFF
