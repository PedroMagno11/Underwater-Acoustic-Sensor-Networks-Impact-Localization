from __future__ import annotations

from typing import Sequence
import numpy as np


class SoundSpeedProfile:
    """
    Sound Speed Profile (SSP): maps depth (meters) -> sound speed (m/s)
    """

    def __init__(self, depths: np.ndarray, sound_speeds: np.ndarray):
        if len(depths) < 2:
            raise ValueError("Sound Speed Profile needs at least two points for interpolation.")
        if len(depths) != len(sound_speeds):
            raise ValueError("Depths and Sound Speed needs to be the same size.")

        sorted_indexes = np.argsort(depths)
        self.depths = np.asarray(depths, dtype=float)[sorted_indexes]
        self.sound_speeds = np.asarray(sound_speeds, dtype=float)[sorted_indexes]

        if np.any(np.diff(self.depths) <= 0):
            raise ValueError("Depths must be strictly increasing after sorting.")


    # -----------------------
    # Coppens Equation
    # -----------------------
    @staticmethod
    def _coppens_sound_speed(T_c: float, S: float, depth_m: float) -> float:
        t = T_c / 10.0  # Coppens uses t = T/10
        D = depth_m / 1000.0  # depth in km

        c0 = (
                1449.05
                + 45.7 * t
                - 5.21 * t ** 2
                + 0.23 * t ** 3
                + (1.333 - 0.126 * t + 0.009 * t ** 2) * (S - 35.0)
        )

        c = (
                c0
                + (16.23 + 0.253 * t) * D
                + (0.213 - 0.1 * t) * D ** 2
                + (0.016 + 0.0002 * (S - 35.0)) * (S - 35.0) * t * D
        )
        return float(c)

    @staticmethod
    def from_temperature_salinity_profiles(
            depths_in_meters: Sequence[float],
            temperatures_celsius: Sequence[float],
            salinity_psu: Sequence[float],
    ) -> SoundSpeedProfile:
        """
        Builds a Sound Speed Profile (SSP) from discrete temperature and salinity
        samples as a function of depth, using the Coppens (1981) empirical model.

        Parameters
        ----------
        depths_in_meters : Sequence[float]
            Depth samples (meters), positive downward.
        temperatures_celsius : Sequence[float]
            Water temperature at each depth (°C).
        salinity_psu : Sequence[float]
            Water salinity at each depth (PSU).

        Returns
        -------
        SoundSpeedProfile
            Interpolable sound speed profile c(z).
        """

        depth_array = np.asarray(depths_in_meters, dtype=float)
        temperature_array = np.asarray(temperatures_celsius, dtype=float)
        salinity_array = np.asarray(salinity_psu, dtype=float)

        if depth_array.ndim != 1 or temperature_array.ndim != 1 or salinity_array.ndim != 1:
            raise ValueError("Depth, temperature, and salinity inputs must be one-dimensional sequences.")

        if not (
                len(depth_array) == len(temperature_array) == len(salinity_array)
        ):
            raise ValueError(
                "Depth, temperature, and salinity sequences must have the same length."
            )

        if len(depth_array) < 2:
            raise ValueError(
                "At least two depth points are required to construct a sound speed profile."
            )

        # Ensure monotonic ordering by depth
        sorting_indices = np.argsort(depth_array)

        sorted_depths = depth_array[sorting_indices]
        sorted_temperatures = temperature_array[sorting_indices]
        sorted_salinities = salinity_array[sorting_indices]

        sound_speeds = np.array(
            [
                SoundSpeedProfile._coppens_sound_speed(
                    temperature, salinity, depth
                )
                for temperature, salinity, depth
                in zip(sorted_temperatures, sorted_salinities, sorted_depths)
            ],
            dtype=float,
        )

        return SoundSpeedProfile(sorted_depths, sound_speeds)


    def sound_speed(self, depth: float) -> float:
        """
        Returns interpolated sound speed (m/s) at a given depth (m).
        Clamps depth to the SSP range.
        """
        depth_clamped = float(np.clip(depth, self.depths[0], self.depths[-1]))
        return float(np.interp(depth_clamped, self.depths, self.sound_speeds))