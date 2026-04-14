"""
Test suite for the core physics module: Parameters, Love Numbers, and Earth Model.
"""

import pytest
import numpy as np
from dataclasses import FrozenInstanceError
from pyshtools import SHGrid, SHCoeffs

from pyslfp.core import EarthModelParameters, LoveNumbers, EarthModel
from pyslfp.core import MEAN_RADIUS, MASS, WATER_DENSITY, ICE_DENSITY


# ==================================================================== #
#                          Fixtures                                    #
# ==================================================================== #


@pytest.fixture
def default_params():
    """Provides standard non-dimensionalized Earth parameters."""
    return EarthModelParameters.from_defaults()


@pytest.fixture
def earth_model():
    """Provides a low-resolution EarthModel for fast testing."""
    return EarthModel.from_defaults(lmax=64)


# ==================================================================== #
#                  1. EarthModelParameters Tests                       #
# ==================================================================== #


def test_parameters_default_initialization():
    """
    Tests that initializing without the factory method yields standard
    SI values with non-dimensionalization scales of 1.0.
    """
    params = EarthModelParameters()

    assert params.length_scale == 1.0
    assert params.density_scale == 1.0
    assert params.time_scale == 1.0

    assert params.raw_mean_radius == MEAN_RADIUS
    assert params.raw_mass == MASS
    assert params.raw_water_density == WATER_DENSITY


def test_parameters_custom_initialization():
    """Tests that custom raw values are correctly stored."""
    custom_radius = 7.0e6
    params = EarthModelParameters(raw_mean_radius=custom_radius)
    assert params.raw_mean_radius == custom_radius


def test_parameters_non_dimensionalisation_factory(default_params):
    """
    Tests the factory method that sets up the standard non-dimensionalization
    scheme based on Earth's mean radius, mean density, and surface gravity.
    """
    # Key non-dimensional values should now be exactly 1.0
    assert default_params.mean_radius == 1.0
    assert np.isclose(default_params.gravitational_acceleration, 1.0)

    # Water density is no longer the base scale (1.0).
    # It should be correctly scaled relative to Earth's mean density.
    expected_density_scale = 3 * MASS / (4 * np.pi * MEAN_RADIUS**3)
    assert np.isclose(default_params.density_scale, expected_density_scale)
    assert np.isclose(
        default_params.water_density, WATER_DENSITY / expected_density_scale
    )

    # Ice density should also be correctly scaled relative to Earth's mean density
    expected_ice_density = ICE_DENSITY / expected_density_scale
    assert np.isclose(default_params.ice_density, expected_ice_density)


def test_parameters_are_immutable(default_params):
    """
    Tests that the dataclass is strictly frozen to prevent accidental
    physics modifications mid-run.
    """
    with pytest.raises(FrozenInstanceError):
        default_params.length_scale = 2.0


# ==================================================================== #
#                        2. LoveNumbers Tests                          #
# ==================================================================== #


def test_love_numbers_initialization_and_loading(default_params):
    """
    Tests that LoveNumbers correctly loads the default data file and
    constructs arrays of the exact required length (lmax + 1).
    """
    lmax = 64
    ln = LoveNumbers(lmax, default_params)

    # Core arrays should be lmax + 1 in length
    for array in [ln.h, ln.k, ln.ht, ln.kt]:
        assert isinstance(array, np.ndarray)
        assert len(array) == lmax + 1


def test_love_numbers_lmax_too_large_raises_error(default_params):
    """
    Tests that requesting an lmax higher than the data file supports
    raises a ValueError.
    """
    lmax_too_large = 5000  # Default PREM goes to 4096
    with pytest.raises(ValueError, match="exceeds Love number file max degree"):
        LoveNumbers(lmax_too_large, default_params)


def test_greens_functions_evaluation(default_params):
    """Smoke test to ensure the Green's functions compute finite floats."""
    ln = LoveNumbers(64, default_params)

    # Evaluate at 10 degrees (in radians)
    angle_rad = np.deg2rad(10.0)
    g_disp = ln.displacement_greens_function(angle_rad)
    g_pot = ln.potential_greens_function(angle_rad)

    assert isinstance(g_disp, float)
    assert np.isfinite(g_disp)
    assert isinstance(g_pot, float)
    assert np.isfinite(g_pot)


def test_love_numbers_sub_properties(default_params):
    """Ensure all individual Love number components are exposed correctly."""
    lmax = 64
    ln = LoveNumbers(lmax, default_params)

    for array in [ln.h_u, ln.k_u, ln.h_phi, ln.k_phi]:
        assert isinstance(array, np.ndarray)
        assert len(array) == lmax + 1


def test_greens_functions_peak_at_origin(default_params):
    """Sanity check: Green's functions should be highly concentrated near 0 degrees."""
    ln = LoveNumbers(64, default_params)

    g_disp_0 = ln.displacement_greens_function(0.0)
    g_disp_10 = ln.displacement_greens_function(np.deg2rad(10.0))

    # The absolute magnitude at 0 degrees should be strictly larger than at 10 degrees
    assert abs(g_disp_0) > abs(g_disp_10)


def test_love_numbers_plotting_smoke_test(default_params):
    """Ensure the Matplotlib plotting methods execute without crashing."""
    ln = LoveNumbers(32, default_params)  # Lower lmax for faster plot generation

    fig1, axes1 = ln.plot_greens_functions(n_points=10)
    assert fig1 is not None
    assert len(axes1) == 2

    fig2, axes2 = ln.plot_greens_functions_split(n_points=20)
    assert fig2 is not None
    assert axes2.shape == (2, 2)


# ==================================================================== #
#                         3. EarthModel Tests                          #
# ==================================================================== #


def test_earth_model_grid_logic():
    """Tests the parsing of specific oversampled grid formats."""
    # Standard DH
    em_standard = EarthModel(64, grid="DH")
    assert em_standard.grid == "DH"
    assert em_standard.sampling == 1
    assert em_standard.grid_name == "DH"

    # Oversampled DH2
    em_oversampled = EarthModel(64, grid="DH2")
    assert em_oversampled.grid == "DH"
    assert em_oversampled.sampling == 2
    assert em_oversampled.grid_name == "DH2"


def test_earth_model_zero_generators(earth_model):
    """Tests that the EarthModel generates correctly sized empty arrays."""
    zero_grid = earth_model.zero_grid()
    assert isinstance(zero_grid, SHGrid)
    assert zero_grid.lmax == earth_model.lmax

    zero_coeffs = earth_model.zero_coefficients()
    assert isinstance(zero_coeffs, SHCoeffs)
    assert zero_coeffs.lmax == earth_model.lmax


def test_earth_model_compatibility_checks(earth_model):
    """Tests the strict compatibility checking for incoming grids."""
    valid_grid = earth_model.zero_grid()
    assert earth_model.check_field(valid_grid) is True

    # Incompatible lmax
    incompatible_grid = SHGrid.from_zeros(
        lmax=earth_model.lmax + 10, grid=earth_model.grid
    )
    with pytest.raises(ValueError, match="is not compatible"):
        earth_model.check_field(incompatible_grid)

    # Incompatible grid type
    incompatible_grid_type = SHGrid.from_zeros(lmax=earth_model.lmax, grid="GLQ")
    with pytest.raises(ValueError, match="is not compatible"):
        earth_model.check_field(incompatible_grid_type)


def test_earth_model_integration(earth_model):
    """
    Unit test for the integrate method: integrating a constant field of 1.0
    over the sphere should yield the surface area of the sphere.
    """
    constant_field = earth_model.constant_grid(1.0)
    integral_result = earth_model.integrate(constant_field)

    radius = earth_model.parameters.mean_sea_floor_radius
    expected_surface_area = 4.0 * np.pi * radius**2

    assert np.isclose(integral_result, expected_surface_area, rtol=1e-6)


def test_earth_model_expand_methods(earth_model):
    """Tests expanding a grid to coefficients and back."""
    initial_grid = earth_model.constant_grid(1.0)

    # Grid to Coeffs
    coeffs = earth_model.expand_field(initial_grid)
    assert isinstance(coeffs, SHCoeffs)

    # Coeffs back to Grid
    reconstructed_grid = earth_model.expand_coefficient(coeffs)
    assert isinstance(reconstructed_grid, SHGrid)

    # Constant field should reconstruct almost perfectly
    assert np.allclose(initial_grid.data, reconstructed_grid.data, atol=1e-10)
