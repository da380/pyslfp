import pytest
import numpy as np
import os
from pyshtools import SHGrid
from pyslfp.ice_ng import IceNG, IceModel, DATADIR

# Check if the ICE-7G data directory exists to decide if data-dependent
# tests should be run. We only check for one to keep it simple.
DATA_PATH_EXISTS = os.path.isdir(os.path.join(DATADIR, "ice7g"))
REASON_TO_SKIP = "ICE-NG data directory not found."

# ==================================================================== #
#                       Tests for Core Logic                           #
# ==================================================================== #


@pytest.mark.parametrize("version", [IceModel.ICE5G, IceModel.ICE6G, IceModel.ICE7G])
def test_initialization(version):
    """Tests that the IceNG class can be initialized with all model versions."""
    loader = IceNG(version=version)
    assert loader._version == version


@pytest.mark.parametrize(
    "version, date, expected_f1, expected_f2, expected_frac",
    [
        # --- ICE-7G ---
        # Date exactly on a time slice
        (IceModel.ICE7G, 10.0, "9.5.nc", "10.nc", 0.0),
        # Date between two time slices (10.2 is 60% of the way from 10.5 to 10.0)
        (IceModel.ICE7G, 10.2, "10.nc", "10.5.nc", 0.6),
        # Date out of bounds (past) -> should use the last file
        (IceModel.ICE7G, 30.0, "26.nc", "26.nc", 0.0),
        # Date out of bounds (future) -> should use the first file
        (IceModel.ICE7G, -1.0, "0.nc", "0.nc", 0.0),
        # --- ICE-5G ---
        # Date exactly on a time slice (note the different file format)
        (IceModel.ICE5G, 12.0, "11.5k_1deg.nc", "12.0k_1deg.nc", 0.0),
        # Date between two time slices
        (IceModel.ICE5G, 12.2, "12.0k_1deg.nc", "12.5k_1deg.nc", 0.6),
    ],
)
def test_find_files_logic(version, date, expected_f1, expected_f2, expected_frac):
    """
    Tests the _find_files method to ensure it correctly identifies the
    bounding data files and the interpolation fraction for various dates
    and model versions. This test does not require the data files to exist.
    """
    loader = IceNG(version=version)
    file1, file2, fraction = loader._find_files(date)

    assert file1.endswith(expected_f1)
    assert file2.endswith(expected_f2)
    assert np.isclose(fraction, expected_frac)


# ==================================================================== #
#                  Tests Requiring Data Files                          #
# ==================================================================== #


@pytest.mark.skipif(not DATA_PATH_EXISTS, reason=REASON_TO_SKIP)
@pytest.mark.parametrize("version", [IceModel.ICE6G, IceModel.ICE7G])
@pytest.mark.parametrize(
    "grid, sampling",
    [
        ("DH", 1),
        ("DH", 2),
        ("GLQ", 1),  # Sampling is ignored for GLQ but needed for parameterization
    ],
    ids=["DH1", "DH2", "GLQ"],
)
def test_data_loading_types_and_shape(version, grid, sampling):
    """
    Tests the end-to-end data loading for various grid types (DH, DH2, GLQ).
    It verifies that the method returns SHGrid objects with the correct shape
    for each grid configuration. This test requires the data files.
    """
    lmax = 32
    loader = IceNG(version=version)

    # Call the method with the specified grid and sampling options
    ice, topo = loader.get_ice_thickness_and_topography(
        5.2, lmax, grid=grid, sampling=sampling
    )

    assert isinstance(ice, SHGrid)
    assert isinstance(topo, SHGrid)

    assert ice.lmax == lmax
    assert topo.lmax == lmax

    # Determine the expected shape based on the grid type, sampling,
    # and the default extend=True option
    if grid == "DH":
        expected_shape = (2 * lmax + 3, sampling * 2 * (lmax + 1) + 1)
    elif grid == "GLQ":
        expected_shape = (lmax + 1, 2 * lmax + 2)

    assert ice.data.shape == expected_shape


@pytest.mark.skipif(not DATA_PATH_EXISTS, reason=REASON_TO_SKIP)
def test_sea_level_calculation_sanity_check():
    """
    Performs a sanity check on the sea level calculation by loading
    present-day data and checking known geographic grid points.
    """
    lmax = 64
    loader = IceNG(version=IceModel.ICE7G)
    ice, sea_level = loader.get_ice_thickness_and_sea_level(0.0, lmax)

    lats = ice.lats()
    lons = ice.lons()

    # --- Check Greenland (approx. 72°N, 40°W) ---
    target_lat_g = 72
    target_lon_g = -40

    # Latitude index logic is correct
    lat_idx_g = np.argmin(np.abs(lats - target_lat_g))

    # Corrected longitude index logic
    # Convert target to [0, 360) and find the shortest angular distance
    target_lon_g_positive = target_lon_g % 360
    angular_dist = np.minimum(
        np.abs(lons - target_lon_g_positive), 360 - np.abs(lons - target_lon_g_positive)
    )
    lon_idx_g = np.argmin(angular_dist)

    greenland_ice = ice.data[lat_idx_g, lon_idx_g]
    # greenland_sl = sea_level.data[lat_idx_g, lon_idx_g]

    assert greenland_ice > 1000

    # --- Check the Pacific (approx. 0°N, 150°W) ---
    target_lat_p = 0
    target_lon_p = -150

    lat_idx_p = np.argmin(np.abs(lats - target_lat_p))

    # Corrected longitude index logic
    target_lon_p_positive = target_lon_p % 360
    angular_dist = np.minimum(
        np.abs(lons - target_lon_p_positive), 360 - np.abs(lons - target_lon_p_positive)
    )
    lon_idx_p = np.argmin(angular_dist)

    pacific_ice = ice.data[lat_idx_p, lon_idx_p]
    pacific_sl = sea_level.data[lat_idx_p, lon_idx_p]

    assert np.isclose(pacific_ice, 0.0)
    assert pacific_sl > 1000
