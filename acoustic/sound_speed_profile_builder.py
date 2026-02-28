from acoustic.sound_speed_profile import SoundSpeedProfile


def build_ssp() -> SoundSpeedProfile:
    return SoundSpeedProfile.from_temperature_salinity_profiles(
        depths_in_meters=[0.5, 2.0, 5.0, 8.0],
        temperatures_celsius=[26.5, 26.0, 25.2, 24.8],
        salinity_psu=[35.0, 35.1, 35.2, 35.2],
    )