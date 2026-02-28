from __future__ import annotations

import os
import logging

from acoustic.sound_speed_profile import SoundSpeedProfile
from acoustic.sound_speed_profile_builder import build_ssp
from executable.pso_runner import LOGGER_NAME
from genetic_algorithm.genetic_algorithm import run_genetic_algorithm
from models.model import GeneticAlgorithmResult
from settings.environment_settings import EnvironmentSettings
from settings.genetic_algorithm_settings import GeneticAlgorithmSettings
from settings.logging_settings import setup_logging
from settings.performance_settings import PerformanceSettings
from settings.simulation_settings import SimulationSettings
from utils.loaders import load_all_settings

os.environ["MPLBACKEND"] = "Agg"

LOGGER_NAME = "underwater_sensor_ga.runner"


def run_experiment_for_n(*, number_of_sensors: int, environment_settings: EnvironmentSettings,
                         genetic_algorithm_settings: GeneticAlgorithmSettings, simulation_settings: SimulationSettings,
                         performance_settings: PerformanceSettings, sound_speed_profile: SoundSpeedProfile) -> None:
    logger = logging.getLogger(LOGGER_NAME)

    logger.info("=" * 80)
    logger.info("Running Genetic Algorithm for number_of_sensors=%d", number_of_sensors)

    result: GeneticAlgorithmResult = run_genetic_algorithm(
        number_of_sensors=number_of_sensors,
        environment_settings=environment_settings,
        genetic_algorithm_settings=genetic_algorithm_settings,
        simulation_settings=simulation_settings,
        sound_speed_profile=sound_speed_profile,
        performance_settings=performance_settings,
    )

    logger.info("Final best cost (N=%d): %.3f", number_of_sensors, result.best_cost)


def main() -> None:
    setup_logging(log_level=logging.INFO, log_file_path="execution.log", log_to_console=True)
    logger = logging.getLogger(LOGGER_NAME)

    (
        environment_settings,
        genetic_algorithm_settings,
        simulation_settings,
        performance_settings,
        _pso_settings,
    ) = load_all_settings("experiment_config.json")

    sound_speed_profile = build_ssp()

    for number_of_sensors in [3, 4, 5]:
        run_experiment_for_n(
            number_of_sensors=number_of_sensors,
            environment_settings=environment_settings,
            genetic_algorithm_settings=genetic_algorithm_settings,
            simulation_settings=simulation_settings,
            performance_settings=performance_settings,
            sound_speed_profile=sound_speed_profile,
        )

    logger.info("All experiments finished. Outputs saved under 'outputs'")


if __name__ == "__main__":
    main()
