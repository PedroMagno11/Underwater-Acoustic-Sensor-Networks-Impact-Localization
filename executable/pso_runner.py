from __future__ import annotations

import os
import logging

from acoustic.sound_speed_profile_builder import build_ssp
from particle_swarm.particle_swarm import run_particle_swarm
from settings.logging_settings import setup_logging
from utils.loaders import load_all_settings

os.environ["MPLBACKEND"] = "Agg"

LOGGER_NAME = "underwater_sensor_ga.pso_runner"

def main() -> None:
    setup_logging(log_level=logging.INFO, log_file_path="execution_pso.log", log_to_console=True)
    logger = logging.getLogger(LOGGER_NAME)

    (
        environment_settings,
        genetic_algorithm_settings,
        simulation_settings,
        performance_settings,
        pso_settings,
    ) = load_all_settings("experiment_config.json")

    ssp = build_ssp()

    for n in [3, 4, 5]:
        logger.info("=" * 80)
        logger.info("Running PSO for number_of_sensors=%d", n)

        result = run_particle_swarm(
            number_of_sensors=n,
            environment_settings=environment_settings,
            simulation_settings=simulation_settings,
            sound_speed_profile=ssp,
            performance_settings=performance_settings,
            particle_swarm_settings=pso_settings
        )

        logger.info("Final PSO best cost (N=%d): %.6f", n, result.best_cost)

    logger.info("All PSO experiments finished. Outputs saved under 'outputs'")


if __name__ == "__main__":
    main()
