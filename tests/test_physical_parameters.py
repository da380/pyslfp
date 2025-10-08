import numpy as np
from pyslfp.physical_parameters import (
    EarthModelParameters,
    MEAN_RADIUS,
    MASS,
    WATER_DENSITY,
    ICE_DENSITY,
)


def test_default_initialization():
    """
    Tests that the EarthModelParameters class initializes with the correct
    default SI values and that the non-dimensionalisation scales are 1.0.
    """
    params = EarthModelParameters()

    # Check that scales are unity by default
    assert params.length_scale == 1.0
    assert params.density_scale == 1.0
    assert params.time_scale == 1.0

    # Check that SI properties match the default constants
    assert params.mean_radius_si == MEAN_RADIUS
    assert params.mass_si == MASS
    assert params.water_density_si == WATER_DENSITY


def test_custom_initialization():
    """
    Tests that the class can be initialized with custom physical parameters
    and that the SI properties reflect these custom values.
    """
    custom_radius = 7.0e6
    custom_mass = 6.0e24
    params = EarthModelParameters(mean_radius=custom_radius, mass=custom_mass)

    assert params.mean_radius_si == custom_radius
    assert params.mass_si == custom_mass


def test_standard_non_dimensionalisation_factory():
    """
    Tests the factory method that sets up a standard non-dimensionalisation
    scheme based on Earth's mean radius, water density, and an hour.
    """
    params = EarthModelParameters.from_standard_non_dimensionalisation()

    # Check that the characteristic scales are set correctly
    assert params.length_scale == 6371000.0
    assert params.density_scale == 1000.0
    assert params.time_scale == 3600.0

    # Check that key non-dimensional values are now 1.0
    assert params.mean_radius == 1.0
    assert params.water_density == 1.0

    # Check that other values are correctly non-dimensionalised
    expected_ice_density = ICE_DENSITY / WATER_DENSITY
    assert np.isclose(params.ice_density, expected_ice_density)
