"""
Test suite for the EarthState data container and masking logic.
"""

import pytest
import numpy as np
from unittest.mock import patch
from pyshtools import SHGrid
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from pyslfp.core import EarthModel
from pyslfp.state import EarthState


# ==================================================================== #
#                          Fixtures                                    #
# ==================================================================== #


@pytest.fixture(scope="module")
def analytical_state():
    """
    Provides a low-resolution, mathematically smooth EarthState using
    the AnalyticalIceModel. Perfect for fast, stable unit tests.
    """
    return EarthState.for_testing(32)


# ==================================================================== #
#                  1. Initialization & Compatibility                   #
# ==================================================================== #


def test_state_initialization_compatibility():
    """Tests that EarthState strictly enforces grid compatibility."""
    model = EarthModel(32)

    # Use the model to generate a perfectly compatible grid
    valid_grid = model.zero_grid()

    # Create an incompatible grid by explicitly forcing the wrong lmax
    incompatible_grid = SHGrid.from_zeros(
        lmax=64, grid=model.grid, sampling=model.sampling, extend=model.extend
    )

    # Should initialize fine
    _ = EarthState(valid_grid, valid_grid, model)

    # Should catch the incompatible ice grid
    with pytest.raises(ValueError, match="is not compatible"):
        EarthState(incompatible_grid, valid_grid, model)

    # Should catch the incompatible sea level grid
    with pytest.raises(ValueError, match="is not compatible"):
        EarthState(valid_grid, incompatible_grid, model)


def test_ocean_function_evaluation(analytical_state):
    """
    Tests that the lazy ocean function evaluates correctly,
    resulting in a binary mask (0 or 1).
    """
    ocean_func = analytical_state.ocean_function
    assert isinstance(ocean_func, SHGrid)
    assert np.all((ocean_func.data == 0) | (ocean_func.data == 1))

    # Check 1 - ocean function
    inv_ocean_func = analytical_state.one_minus_ocean_function
    assert np.all(inv_ocean_func.data == (1 - ocean_func.data))


def test_ocean_area_calculation(analytical_state):
    """Tests that ocean area is a positive float less than the total Earth area."""
    ocean_area = analytical_state.ocean_area
    total_area = (
        4.0 * np.pi * analytical_state.model.parameters.mean_sea_floor_radius**2
    )

    assert isinstance(ocean_area, float)
    assert ocean_area > 0
    assert ocean_area < total_area


# ==================================================================== #
#                  2. Geographic Projections                           #
# ==================================================================== #


def test_hemisphere_projections(analytical_state):
    """Tests that hemisphere projections perfectly partition the globe."""
    nh_proj = analytical_state.northern_hemisphere_projection(value=0)
    sh_proj = analytical_state.southern_hemisphere_projection(value=0)

    assert np.all((nh_proj.data == 0) | (nh_proj.data == 1))
    assert np.all((sh_proj.data == 0) | (sh_proj.data == 1))

    # Should not overlap
    assert np.max(nh_proj.data * sh_proj.data) == 0


def test_ocean_land_partition(analytical_state):
    """Tests that ocean and land projections completely partition the sphere."""
    ocean_proj = analytical_state.ocean_projection(value=0)
    land_proj = analytical_state.land_projection(value=0)

    # Every point should be strictly ocean or land
    total = ocean_proj.data + land_proj.data
    assert np.all(total == 1)

    # No overlap
    overlap = ocean_proj.data * land_proj.data
    assert np.all(overlap == 0)


def test_ocean_projection_ice_shelves(analytical_state):
    """Tests that excluding ice shelves removes area from the ocean projection."""
    ocean_all = analytical_state.ocean_projection(value=0, exclude_ice_shelves=False)
    ocean_no_shelves = analytical_state.ocean_projection(
        value=0, exclude_ice_shelves=True
    )

    # The projection without shelves must be smaller or equal
    assert np.sum(ocean_no_shelves.data) <= np.sum(ocean_all.data)


@pytest.mark.parametrize(
    "exclude_shelves,exclude_glaciers",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_ice_projection_exclusions(analytical_state, exclude_shelves, exclude_glaciers):
    """Tests all boolean combinations for ice projection exclusions."""
    ice_proj = analytical_state.ice_projection(
        value=0, exclude_ice_shelves=exclude_shelves, exclude_glaciers=exclude_glaciers
    )
    assert np.all((ice_proj.data == 0) | (ice_proj.data == 1))


def test_altimetry_projection_bounds(analytical_state):
    """Ensures altimetry projection strictly obeys the requested latitude bounds."""
    lat_min, lat_max = -60.0, 60.0
    alt_proj = analytical_state.altimetry_projection(
        latitude_min=lat_min, latitude_max=lat_max, value=0
    )

    lats, _ = np.meshgrid(
        analytical_state.lats(), analytical_state.lons(), indexing="ij"
    )

    # Wherever the projection is 1, the latitude must be within bounds
    active_lats = lats[alt_proj.data == 1]
    assert np.all((active_lats > lat_min) & (active_lats < lat_max))


# ==================================================================== #
#                  3. Load Converters & Generators                     #
# ==================================================================== #


def test_load_converters(analytical_state):
    """Tests that thickness changes are correctly scaled to density loads."""
    dummy_change = analytical_state.model.constant_grid(1.0)

    # 1. Ice thickness change -> Direct Load
    ice_load = analytical_state.direct_load_from_ice_thickness_change(dummy_change)
    expected_ice_max = analytical_state.model.parameters.ice_density * 1.0
    assert np.max(ice_load.data) <= expected_ice_max

    # 2. Sea level change -> Direct Load
    sl_load = analytical_state.direct_load_from_sea_level_change(dummy_change)
    expected_sl_max = analytical_state.model.parameters.water_density * 1.0
    assert np.max(sl_load.data) <= expected_sl_max


def test_disk_load(analytical_state):
    """Smoke test for the spherical harmonic cap generator."""
    load = analytical_state.disk_load(10.0, 45.0, 90.0, 1.0)
    assert isinstance(load, SHGrid)
    assert load.lmax == analytical_state.lmax


def test_convenience_loads(analytical_state):
    """Smoke test for built-in convenience loads."""
    load_nh = analytical_state.northern_hemisphere_load(fraction=0.5)
    assert isinstance(load_nh, SHGrid)

    load_grl = analytical_state.greenland_load()
    assert isinstance(load_grl, SHGrid)


@patch("pyslfp.state.EarthState.imbie_ant_projection")
def test_dynamic_convenience_loads(mock_imbie_proj, analytical_state):
    """Test shapefile-driven loads without triggering real shapefile parsing."""
    # Mock the region projection to return a valid grid of ones
    mock_imbie_proj.return_value = analytical_state.model.constant_grid(1.0)

    load = analytical_state.imbie_ant_load("DummyRegion", fraction=0.1)

    assert isinstance(load, SHGrid)
    mock_imbie_proj.assert_called_once_with("DummyRegion", value=0)


# ==================================================================== #
#                  4. Plotting Utilities                               #
# ==================================================================== #


def test_plot_coastline_smoke_test(analytical_state):
    """Ensures the coastline plotting logic executes without Matplotlib crashes."""
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

    try:
        artist = analytical_state.plot_coastline(ax, color="red")
        assert artist is not None
    finally:
        plt.close(fig)
