from __future__ import annotations

import random
from typing import List

def select_index_for_tournament(
        costs: List[float],
        tournament_size: int,
        random_generator: random.Random
) -> int:
    candidates = [random_generator.randrange(0, len(costs)) for _ in range(tournament_size)]
    best = min(candidates, key=lambda index: costs[index])
    return best