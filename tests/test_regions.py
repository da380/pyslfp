"""
Test suite for the Regions mixin class.
"""

import pytest
import numpy as np
from unittest.mock import patch
from pyshtools import SHGrid
import regionmask
import geopandas as gpd
from shapely.geometry import Polygon

from unittest.mock import MagicMock
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


@pytest.mark.slow
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


@pytest.mark.slow
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


# ==================================================================== #
#                  4. New Composite & Universal Routing                #
# ==================================================================== #


@pytest.mark.slow
def test_ar6_composite_projection(region_engine):
    """Test passing a list of AR6 regions creates a unified mask without double-counting."""
    # Get the individual masks
    grl_proj = region_engine.regionmask_projection("Greenland/Iceland", value=0)
    wais_proj = region_engine.regionmask_projection("W.Antarctica", value=0)

    # Get the composite mask using the new list feature
    composite_proj = region_engine.regionmask_projection(
        ["Greenland/Iceland", "W.Antarctica"], value=0
    )

    # The composite should exactly equal the logical OR of the individual masks
    expected_data = np.where((grl_proj.data == 1) | (wais_proj.data == 1), 1.0, 0.0)

    np.testing.assert_array_equal(composite_proj.data, expected_data)
    assert (
        np.max(composite_proj.data) == 1.0
    )  # Ensures no values of 2.0 (double counting)


@patch("pyslfp.regions.ensure_data")
@patch("pyslfp.regions.gpd.read_file")
def test_shapefile_composite_routing(
    mock_read_file, mock_ensure, region_engine, mock_gdf
):
    """Test that _apply_regionmask handles a list of regions via np.any()."""
    mock_read_file.return_value.to_crs.return_value = mock_gdf

    # Test passing a list to the mocked IMBIE dataset
    proj = region_engine.imbie_ant_projection(["AntA", "AntB"], value=0)

    assert isinstance(proj, SHGrid)
    assert np.all((proj.data == 0) | (proj.data == 1))
    assert np.sum(proj.data) > 0  # Ensure it actually accumulated both regions


@pytest.mark.slow
def test_get_projection_integration(region_engine):
    """
    Test the universal router using built-in datasets (AR6 and Caspian)
    to verify it searches multiple datasets sequentially.
    """
    # Cross-dataset composite: Caspian (special case) + W.Antarctica (AR6)
    composite = region_engine.get_projection(["Caspian Sea", "W.Antarctica"], value=0.0)

    # Verify it's a valid binary grid
    assert isinstance(composite, SHGrid)
    assert np.all((composite.data == 0) | (composite.data == 1))
    assert np.sum(composite.data) > 0

    # Ensure it fails cleanly if ANY region in the list is missing
    with pytest.raises(ValueError, match="Universal lookup failed: Region 'Atlantis'"):
        region_engine.get_projection(["Caspian Sea", "Atlantis"])


@patch.object(MockRegionHost, "iho_sea_projection")
@patch.object(MockRegionHost, "hydrobasin_projection")
def test_get_projection_mocked_routing(mock_hydro, mock_iho, region_engine):
    """
    Test the universal router against mocked shapefile loaders to ensure
    it accumulates data from different methods correctly.
    """
    # Setup mocks to return empty grids with one specific pixel set to 1
    grid_iho = region_engine.zero_grid()
    grid_iho.data[0, 0] = 1.0

    grid_hydro = region_engine.zero_grid()
    grid_hydro.data[1, 1] = 1.0

    # Configure the mocks to succeed only for specific names
    def iho_side_effect(name, value=np.nan):
        if name == "Gulf of Mexico":
            return grid_iho
        raise ValueError

    mock_iho.side_effect = iho_side_effect

    def hydro_side_effect(name, value=np.nan):
        if name == "Amazon":
            return grid_hydro
        raise ValueError

    mock_hydro.side_effect = hydro_side_effect

    # Perform the universal lookup
    composite = region_engine.get_projection(["Gulf of Mexico", "Amazon"], value=0.0)

    # Verify both pixels were accumulated into the final mask
    assert composite.data[0, 0] == 1.0  # Proves IHO was queried and accumulated
    assert composite.data[1, 1] == 1.0  # Proves Hydro was queried and accumulated
    assert np.max(composite.data) == 1.0


# ==================================================================== #
#                  5. Slow Integration Tests (Real Data)               #
# ==================================================================== #


@pytest.mark.slow
def test_real_imbie_ant_projection(region_engine):
    """Test real data loading and projection for IMBIE Antarctica."""
    names = region_engine.list_imbie_ant_regions()
    assert len(names) > 0

    # "G-H" is the Amundsen Sea Embayment
    assert "G-H" in names
    proj = region_engine.imbie_ant_projection("G-H", value=0.0)
    assert np.max(proj.data) == 1.0


@pytest.mark.slow
def test_real_mouginot_grl_projection(region_engine):
    """Test real data loading and projection for Mouginot Greenland."""
    names = region_engine.list_mouginot_grl_regions()
    assert len(names) > 0

    # "NW" is the North-West basin
    assert "NW" in names
    proj = region_engine.mouginot_grl_projection("NW", value=0.0)
    assert np.max(proj.data) == 1.0


@pytest.mark.slow
def test_real_hydrobasin_projection(region_engine):
    """Test real data loading and projection for HydroBASINS."""
    names = region_engine.list_hydrobasins()
    assert len(names) > 0

    # "4030025450" is the Ganges-Brahmaputra-Meghna basin
    assert "4030025450" in names
    proj = region_engine.hydrobasin_projection("4030025450", value=0.0)
    assert np.max(proj.data) == 1.0


@pytest.mark.slow
def test_real_iho_seas_projection(region_engine):
    """Test real data loading and projection for IHO Seas."""
    names = region_engine.list_iho_seas()
    assert len(names) > 0

    assert "Gulf of Mexico" in names
    proj = region_engine.iho_sea_projection("Gulf of Mexico", value=0.0)
    assert np.max(proj.data) == 1.0


@pytest.mark.slow
def test_real_get_projection_composite(region_engine):
    """
    Test the universal router across multiple REAL datasets.
    This is the ultimate end-to-end integration test for the module.
    """
    # This will trigger the loading of AR6, IHO, and Mouginot
    proj = region_engine.get_projection(
        ["W.Antarctica", "Gulf of Mexico", "NW"], value=0.0
    )

    assert isinstance(proj, SHGrid)
    assert np.all((proj.data == 0.0) | (proj.data == 1.0))
    assert np.sum(proj.data) > 0


# ==================================================================== #
#                  6. Plotting & Universal Listing                     #
# ==================================================================== #


@patch("pyslfp.regions.ensure_data")
@patch("pyslfp.regions.gpd.read_file")
def test_list_all_regions(mock_read_file, mock_ensure, region_engine, mock_gdf):
    """Test that the master dictionary returns the expected structure."""
    # Return the raw mock for HydroBASINS (which doesn't use .to_crs())
    mock_read_file.return_value = mock_gdf

    # --- NEW: Prevent regionmask from trying to load Natural Earth internally ---
    dummy_ne = MagicMock()
    dummy_ne.names = ["Mocked Ocean Basin"]
    region_engine._ne_ocean_regions = dummy_ne
    # ----------------------------------------------------------------------------

    # Patch the GeoDataFrame methods so they gracefully return the mock_gdf
    # for the other loaders (IMBIE, Mouginot, etc.)
    with patch.object(gpd.GeoDataFrame, "to_crs", return_value=mock_gdf), patch.object(
        gpd.GeoDataFrame, "dissolve", return_value=mock_gdf
    ), patch.object(gpd.GeoDataFrame, "reset_index", return_value=mock_gdf):
        all_regions = region_engine.list_all_regions()

    assert isinstance(all_regions, dict)
    assert "AR6 (IPCC Climate Regions)" in all_regions
    assert "IMBIE (Antarctica)" in all_regions
    assert "HydroBASINS (Level 3)" in all_regions
    assert "Natural Earth (Oceans)" in all_regions

    # Check that our mocked regions populated the lists
    assert "AntA" in all_regions["IMBIE (Antarctica)"]
    assert "GrlA" in all_regions["Mouginot (Greenland)"]
    assert "101" in all_regions["HydroBASINS (Level 3)"]
    assert "Mocked Ocean Basin" in all_regions["Natural Earth (Oceans)"]


@pytest.mark.slow
@patch.object(regionmask.Regions, "plot")
def test_ar6_and_ne_ocean_plotters(mock_rm_plot, region_engine):
    """Test that the built in regionmask plotters can handle string filtering without crashing."""

    region_engine.plot_ar6_boundaries(None, region_names="W.Antarctica")
    mock_rm_plot.assert_called_once()
    mock_rm_plot.reset_mock()

    # Dynamically pull available ocean names to prevent version-specific KeyErrors
    oceans = region_engine.list_ne_oceans()

    region_engine.plot_ne_ocean_boundaries(None, region_names=[oceans[0], oceans[1]])
    mock_rm_plot.assert_called_once()


@pytest.mark.slow
@patch.object(MockRegionHost, "plot_ar6_boundaries")
@patch.object(MockRegionHost, "plot_iho_sea_boundaries")
def test_universal_plot_routing(mock_plot_iho, mock_plot_ar6, region_engine):
    """Test that plot_boundaries correctly routes requests to dataset specific plotters."""

    # Route an AR6 region and an IHO region
    region_engine.plot_boundaries(None, ["W.Antarctica", "Gulf of Mexico"])

    # Assert that the specific plotters were called exactly once with the correctly grouped lists
    mock_plot_ar6.assert_called_once_with(None, region_names=["W.Antarctica"])
    mock_plot_iho.assert_called_once_with(None, region_names=["Gulf of Mexico"])

    # Test failure on unknown region
    with pytest.raises(ValueError, match="Universal plot failed"):
        region_engine.plot_boundaries(None, ["FakeRegion"])

    # Test warning for Caspian Sea (has no vector boundaries)
    with pytest.warns(UserWarning, match="has no vector boundaries to plot"):
        region_engine.plot_boundaries(None, ["Caspian Sea"])
