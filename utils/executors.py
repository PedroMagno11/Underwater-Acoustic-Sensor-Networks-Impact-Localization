from concurrent.futures.process import ProcessPoolExecutor
from typing import Optional

from settings.performance_settings import PerformanceSettings


def create_executor_if_needed(perf: PerformanceSettings) -> Optional[ProcessPoolExecutor]:
    if not perf.enable_parallel_evaluation:
        return None

    mode = (perf.parallel_evaluation_mode or "auto").lower().strip()
    if mode == "off":
        return None

    if mode == "fixed":
        workers = max(1, int(perf.number_of_workers))
    else:
        import os
        workers = max(1, (os.cpu_count() or 2) - 1)

    return ProcessPoolExecutor(max_workers=workers) if workers > 1 else None