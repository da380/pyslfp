"""
Test suite for the Regions mixin class.
"""

import pytest
import numpy as np
from unittest.mock import patch
from pyshtools import SHGrid
import geopandas as gpd
from shapely.geometry import Polygon

from pyslfp.core import EarthModel
from pyslfp.regions import Regions


# ==================================================================== #
#                          Host Class & Fixtures                       #
# ==================================================================== #


class MockRegionHost(EarthModel, Regions):
    """
    A lightweight host class combining EarthModel (for spatial grids/coords)
    and Regions (the mixin being tested) for pure isolation.
    """

    def __init__(self, lmax=64):
        EarthModel.__init__(self, lmax, grid="DH")
        Regions.__init__(self)


@pytest.fixture
def region_engine():
    """Provides a low-resolution host instance."""
    return MockRegionHost(lmax=64)


@pytest.fixture
def mock_gdf():
    """Provides a tiny GeoDataFrame with fake polygons to bypass shapefile loading."""
    # Create two simple 10x10 degree boxes
    p1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    p2 = Polygon([(20, 20), (20, 30), (30, 30), (30, 20)])

    # Populate the columns expected by all the different loaders
    gdf = gpd.GeoDataFrame(
        {
            "Subregion": ["AntA", "AntB"],  # IMBIE
            "SUBREGION1": ["GrlA", "GrlB"],  # Mouginot
            "HYBAS_ID": ["101", "102"],  # HydroBASINS
            "NAME": ["SeaA", "SeaB"],  # IHO Seas
            "geometry": [p1, p2],
        },
        crs="EPSG:4326",
    )

    return gdf


# ==================================================================== #
#                  1. Static & Built-in Projections                    #
# ==================================================================== #


def test_glacier_projection_basic(region_engine):
    """Test the static glacier projection mask (North America bounds)."""
    glacier_proj = region_engine.glacier_projection(value=0)
    assert isinstance(glacier_proj, SHGrid)
    assert np.all((glacier_proj.data == 0) | (glacier_proj.data == 1))
    assert np.sum(glacier_proj.data) > 0


def test_caspian_sea_projection(region_engine):
    """Test the static Caspian Sea mask."""
    caspian_proj = region_engine.caspian_sea_projection(value=0)
    assert isinstance(caspian_proj, SHGrid)
    assert np.all((caspian_proj.data == 0) | (caspian_proj.data == 1))
    assert np.sum(caspian_proj.data) > 0


def test_ar6_regionmask_projections(region_engine):
    """Test built-in regionmask (AR6) conveniences and error handling."""
    grl_proj = region_engine.greenland_projection(value=0)
    wais_proj = region_engine.west_antarctic_projection(value=0)
    eais_proj = region_engine.east_antarctic_projection(value=0)

    for proj in [grl_proj, wais_proj, eais_proj]:
        assert np.all((proj.data == 0) | (proj.data == 1))

    # Greenland and West Antarctica should not overlap
    assert np.max(grl_proj.data * wais_proj.data) == 0

    with pytest.raises(ValueError, match="not found in the AR6 dataset"):
        region_engine.regionmask_projection("Atlantis")


def test_ne_ocean_projections(region_engine):
    """Test Natural Earth built-in ocean regions."""
    oceans = region_engine.list_ne_oceans()
    assert len(oceans) > 0

    # Project the first available ocean
    proj = region_engine.ne_ocean_projection(oceans[0], value=0)
    assert np.all((proj.data == 0) | (proj.data == 1))


# ==================================================================== #
#                  2. Dynamic Shapefile Loading & Routing              #
# ==================================================================== #


@patch("pyslfp.regions.ensure_data")
@patch("pyslfp.regions.gpd.read_file")
def test_shapefile_lazy_loading(mock_read_file, mock_ensure, region_engine, mock_gdf):
    """
    Test that shapefile datasets lazily load and trigger the downloader.
    We use the real mock_gdf to satisfy regionmask's strict type checking.
    """
    # 1. Setup the mock to handle the chain and return the real GeoDataFrame
    # This allows self._imbie_ant_gdf = gpd.read_file(path).to_crs(epsg=4326) to work
    mock_read_file.return_value = mock_gdf

    # We must patch 'to_crs', 'dissolve', and 'reset_index' on the mock_gdf itself
    # because the code calls these methods. We make them return 'self' (the gdf).
    with patch.object(gpd.GeoDataFrame, "to_crs", return_value=mock_gdf), patch.object(
        gpd.GeoDataFrame, "dissolve", return_value=mock_gdf
    ), patch.object(gpd.GeoDataFrame, "reset_index", return_value=mock_gdf):

        # 2. Trigger IMBIE load
        # (pyslfp/regions.py:61-66)
        ant_regions = region_engine.list_imbie_ant_regions()
        assert "AntA" in ant_regions
        mock_ensure.assert_any_call("IMBIE_ANT")

        # 3. Trigger Mouginot load
        # (pyslfp/regions.py:77-83)
        grl_regions = region_engine.list_mouginot_grl_regions()
        assert "GrlA" in grl_regions
        mock_ensure.assert_any_call("MOUGINOT_GRL")


@patch("pyslfp.regions.ensure_data")
@patch("pyslfp.regions.gpd.read_file")
def test_shapefile_projection_routing(
    mock_read_file, mock_ensure, region_engine, mock_gdf
):
    """
    Test that _apply_regionmask correctly extracts the region from the
    generated regionmask object and creates a valid SHGrid.
    """
    # Setup mocks
    mock_read_file.return_value.to_crs.return_value = mock_gdf

    # Request a specific region projection
    proj = region_engine.imbie_ant_projection("AntA", value=0)

    assert isinstance(proj, SHGrid)
    assert np.all((proj.data == 0) | (proj.data == 1))

    # Error routing
    with pytest.raises(ValueError, match="not found in ANT"):
        region_engine.imbie_ant_projection("FakeBasin")


def test_invalid_dataset_key(region_engine):
    """Test the internal error catch for bad dataset keys."""
    with pytest.raises(ValueError, match="must be 'ANT', 'GRL'"):
        region_engine._apply_regionmask("FAKE_KEY", "Region", 0.0)


# ==================================================================== #
#                  3. Composite Projections & Groupings                #
# ==================================================================== #


@patch.object(MockRegionHost, "list_imbie_ant_regions", return_value=["A", "B"])
@patch.object(MockRegionHost, "list_mouginot_grl_regions", return_value=["X", "Y"])
def test_ice_basin_groupings(mock_grl, mock_ant, region_engine):
    """Test that the grouping logic successfully organizes the names."""

    # Individual scheme
    individual = region_engine.ice_basin_groupings(scheme="individual")
    assert individual == [["ANT_A"], ["ANT_B"], ["GRL_X"], ["GRL_Y"]]

    # Ice sheets scheme
    sheets = region_engine.ice_basin_groupings(scheme="ice_sheets")
    assert sheets == [["ANT_A", "ANT_B"], ["GRL_X", "GRL_Y"]]

    # Unknown scheme
    with pytest.raises(ValueError, match="Unknown grouping scheme"):
        region_engine.ice_basin_groupings(scheme="fake_scheme")


@patch.object(MockRegionHost, "ice_basin_groupings")
@patch.object(MockRegionHost, "imbie_ant_projection")
@patch.object(MockRegionHost, "mouginot_grl_projection")
def test_grouped_ice_projections(
    mock_grl_proj, mock_ant_proj, mock_groupings, region_engine
):
    """Test that group projections correctly sum underlying mask grids."""

    # Mock the groupings to return one group with two regions
    mock_groupings.return_value = [["ANT_A", "GRL_X"]]

    # Create two fake SHGrids to act as the individual projections
    grid1 = region_engine.zero_grid()
    grid1.data[0, 0] = 1.0  # Mark a single pixel
    mock_ant_proj.return_value = grid1

    grid2 = region_engine.zero_grid()
    grid2.data[1, 1] = 1.0  # Mark a different pixel
    mock_grl_proj.return_value = grid2

    # Execute the grouping
    masks, labels = region_engine.grouped_ice_projections(groupings="dummy_scheme")

    assert labels == ["ANT_A + GRL_X"]
    assert len(masks) == 1

    # The combined mask should have both pixels marked
    combined_grid = masks[0]
    assert combined_grid.data[0, 0] == 1.0
    assert combined_grid.data[1, 1] == 1.0
    assert np.sum(combined_grid.data) == 2.0
