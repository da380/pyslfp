"""
Test suite for the ICE-NG data loaders and interpolation logic.
"""

import pytest
import numpy as np
import os
from unittest.mock import patch
import xarray as xr
from pyshtools import SHGrid

from pyslfp.ice.ice_ng import IceNG
from pyslfp.data import DATADIR


@pytest.fixture
def mock_xr_dataset():
    """Generates a tiny, fake xarray Dataset resembling ICE-7G structure."""
    lats = np.array([-90, 0, 90])
    lons = np.array([0, 180, 360])

    # Create fake 3x3 grids for ice and topography
    stgit_data = np.full((3, 3), 1000.0)  # 1000m of ice everywhere
    topo_data = np.full((3, 3), -500.0)  # -500m topography everywhere

    ds = xr.Dataset(
        {
            "stgit": (["lat", "lon"], stgit_data),
            "Topo": (["lat", "lon"], topo_data),
        },
        coords={
            "lat": lats,
            "lon": lons,
        },
    )
    return ds


# Patch ensure_data on the class to prevent it from downloading files
@patch("pyslfp.ice.ice_ng.ensure_data")
class TestIceNG:

    def test_initialization(self, mock_ensure):
        """Tests that the IceNG class correctly stores its version."""
        loader = IceNG(version="ICE6G")
        assert loader.version == "ICE6G"
        mock_ensure.assert_called_once_with("ICE6G")

    @pytest.mark.parametrize(
        "version, date, expected_f1, expected_f2, expected_frac",
        [
            ("ICE7G", 10.0, "9.5.nc", "10.nc", 0.0),
            ("ICE7G", 10.2, "10.nc", "10.5.nc", 0.6),
            ("ICE7G", 30.0, "26.nc", "26.nc", 0.0),
            ("ICE7G", -1.0, "0.nc", "0.nc", 0.0),
            ("ICE5G", 12.0, "11.5k_1deg.nc", "12.0k_1deg.nc", 0.0),
            ("ICE5G", 12.2, "12.0k_1deg.nc", "12.5k_1deg.nc", 0.6),
        ],
    )
    def test_find_files_logic(
        self, mock_ensure, version, date, expected_f1, expected_f2, expected_frac
    ):
        """Tests the logic for bounding data files and the interpolation fraction."""
        loader = IceNG(version=version)
        file1, file2, fraction = loader._find_files(date)

        assert file1.endswith(expected_f1)
        assert file2.endswith(expected_f2)
        assert np.isclose(fraction, expected_frac)

    @patch("pyslfp.ice.ice_ng.IceNG._get_time_slice")
    def test_time_slice_interpolation_blending(self, mock_get_slice, mock_ensure):
        """
        Tests the `else` block in get_ice_thickness_and_topography where
        data is linearly interpolated between two distinct time slices.
        """
        loader = IceNG(version="ICE7G")
        lmax = 16

        # Create two distinct dummy grids
        grid1_ice = SHGrid.from_zeros(lmax)
        grid1_ice.data += 100.0  # 100m ice at time 1
        grid1_topo = SHGrid.from_zeros(lmax)

        grid2_ice = SHGrid.from_zeros(lmax)
        grid2_ice.data += 200.0  # 200m ice at time 2
        grid2_topo = SHGrid.from_zeros(lmax)

        # Make the mock return grid1 the first time it's called, and grid2 the second time
        mock_get_slice.side_effect = [(grid1_ice, grid1_topo), (grid2_ice, grid2_topo)]

        # For ICE7G, time slices are 0.0, 0.5, 1.0...
        # date=0.25 is exactly halfway between 0.0 and 0.5.
        # The blending fraction should evaluate to 0.5.
        ice, topo = loader.get_ice_thickness_and_topography(0.25, lmax)

        # 0.5 * 100 + 0.5 * 200 = 150.0
        assert np.allclose(ice.data, 150.0)

        # Ensure it actually fetched two different files
        assert mock_get_slice.call_count == 2


@patch("pyslfp.ice.ice_ng.ensure_data")
@patch("pyslfp.ice.ice_ng.xr.open_dataset")
def test_get_time_slice_interpolation(mock_open, mock_ensure, mock_xr_dataset):
    """
    Tests loading an xarray dataset, interpolating onto an SHGrid,
    and scaling by the length_scale.
    """
    mock_open.return_value = mock_xr_dataset

    lmax = 16
    length_scale = 100.0
    loader = IceNG(version="ICE7G", length_scale=length_scale)

    ice, topo = loader._get_time_slice(
        "dummy_file.nc", lmax, grid="DH", sampling=1, extend=True
    )

    assert isinstance(ice, SHGrid)
    assert isinstance(topo, SHGrid)

    # 1000m ice / 100 length_scale = 10.0
    assert np.allclose(ice.data, 10.0)
    # -500m topo / 100 length_scale = -5.0
    assert np.allclose(topo.data, -5.0)


@patch("pyslfp.ice.ice_ng.ensure_data")
@patch("pyslfp.ice.ice_ng.xr.open_dataset")
def test_sea_level_and_ice_shelf_logic(mock_open, mock_ensure, mock_xr_dataset):
    """
    Tests the logic that converts topography into sea level and
    accounts for ice shelf flotation displacement.
    """
    mock_open.return_value = mock_xr_dataset

    loader = IceNG(version="ICE7G")
    ice, sl = loader.get_ice_thickness_and_sea_level(0.0, 16)

    # topo is -500 (ocean bed), ice is 1000.
    # Sea level = -topo = 500.
    # Flotation displacement = (917/1028) * 1000.
    expected_sl = 500.0 + (917.0 / 1028.0) * 1000.0

    assert np.allclose(sl.data, expected_sl)


# ==================================================================== #
#                  Integration Test (Real Data)                        #
# ==================================================================== #

# Check if data exists locally to avoid forced downloads during standard test runs
DATA_PATH_EXISTS = os.path.isdir(DATADIR / "ice7g")


@pytest.mark.slow
@pytest.mark.skipif(
    not DATA_PATH_EXISTS,
    reason="Real ICE-7G data not found. Run with '-m slow' to trigger.",
)
def test_iceng_real_data_integration():
    """End-to-end check using real Zenodo datasets."""
    lmax = 64
    loader = IceNG(version="ICE7G")
    ice, sl = loader.get_ice_thickness_and_sea_level(0.0, lmax)

    # Check Greenland (approx 72N, 320E)
    lat_idx = np.argmin(np.abs(ice.lats() - 72))
    lon_idx = np.argmin(np.abs(ice.lons() - 320))
    assert ice.data[lat_idx, lon_idx] > 1500.0
