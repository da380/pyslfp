"""
Test suite for the Regions mixin class.
"""

import pytest
import numpy as np
from pyslfp.finger_print import FingerPrint


@pytest.fixture(scope="module")
def region_engine():
    """
    Provides a low-resolution FingerPrint instance to act as the
    host for the Regions mixin methods.
    """
    return FingerPrint(lmax=128, grid="DH")


# ====================================================================#
#             Tests for Static Geometric Projections                  #
# ====================================================================#


def test_hemisphere_projections(region_engine: FingerPrint):
    """Test that hemisphere projections are binary and mutually exclusive."""
    nh_proj = region_engine.northern_hemisphere_projection(value=0)
    sh_proj = region_engine.southern_hemisphere_projection(value=0)

    assert np.all((nh_proj.data == 0) | (nh_proj.data == 1))
    assert np.all((sh_proj.data == 0) | (sh_proj.data == 1))
    assert np.max(nh_proj.data * sh_proj.data) == 0


def test_glacier_projection_basic(region_engine: FingerPrint):
    """Test the static glacier projection mask (North America bounds)."""
    glacier_proj = region_engine.glacier_projection(value=0)
    assert np.all((glacier_proj.data == 0) | (glacier_proj.data == 1))
    assert np.sum(glacier_proj.data) > 0


def test_caspian_sea_projection(region_engine: FingerPrint):
    """Test the static Caspian Sea mask."""
    caspian_proj = region_engine.caspian_sea_projection(value=0)
    assert np.all((caspian_proj.data == 0) | (caspian_proj.data == 1))
    assert np.sum(caspian_proj.data) > 0


# ====================================================================#
#             Tests for AR6 (Built-in regionmask) Projections         #
# ====================================================================#


def test_ar6_regionmask_projections(region_engine: FingerPrint):
    """Test AR6 conveniences and error handling."""
    grl_proj = region_engine.greenland_projection(value=0)
    wais_proj = region_engine.west_antarctic_projection(value=0)

    assert np.all((grl_proj.data == 0) | (grl_proj.data == 1))
    assert np.all((wais_proj.data == 0) | (wais_proj.data == 1))
    assert np.max(grl_proj.data * wais_proj.data) == 0

    with pytest.raises(ValueError, match="not found in the AR6 dataset"):
        region_engine.regionmask_projection("Atlantis")


# ====================================================================#
#             Tests for Dynamic Shapefile Datasets                    #
# ====================================================================#


def test_shapefile_region_lists(region_engine: FingerPrint):
    """Test that all dynamic datasets correctly list their available regions."""
    ant_regions = region_engine.list_imbie_ant_regions()
    grl_regions = region_engine.list_mouginot_grl_regions()
    hydro_regions = region_engine.list_hydrobasins()
    iho_regions = region_engine.list_iho_seas()
    ne_oceans = region_engine.list_ne_oceans()

    assert len(ant_regions) > 0
    assert len(grl_regions) > 0
    assert len(hydro_regions) > 0
    assert len(iho_regions) > 0
    assert len(ne_oceans) > 0


def test_shapefile_projections_valid(region_engine: FingerPrint):
    """Test that a valid projection can be extracted from each dataset."""
    # Test one from each to ensure the property lazy-loading and masking works
    datasets = [
        (region_engine.list_imbie_ant_regions, region_engine.imbie_ant_projection),
        (
            region_engine.list_mouginot_grl_regions,
            region_engine.mouginot_grl_projection,
        ),
        (region_engine.list_hydrobasins, region_engine.hydrobasin_projection),
        (region_engine.list_iho_seas, region_engine.iho_sea_projection),
        (region_engine.list_ne_oceans, region_engine.ne_ocean_projection),
    ]

    for list_func, proj_func in datasets:
        regions = list_func()
        if regions:
            test_region = regions[0]
            proj = proj_func(test_region, value=0)

            # Ensure it generates a valid binary mask
            assert np.all((proj.data == 0) | (proj.data == 1))
            assert np.sum(proj.data) > 0


def test_shapefile_projections_invalid(region_engine: FingerPrint):
    """Test error handling for bad region names across all datasets."""
    with pytest.raises(ValueError, match="not found in ANT"):
        region_engine.imbie_ant_projection("FakeBasin")

    with pytest.raises(ValueError, match="not found in HYDRO"):
        region_engine.hydrobasin_projection("FakeRiver")

    with pytest.raises(ValueError, match="not found in OCEAN"):
        region_engine.iho_sea_projection("FakeSea")


def test_known_large_hydro_basin(region_engine: FingerPrint):
    """
    Specifically test a large, well-known basin (e.g., Amazon) to ensure
    the coordinate wrapping is working even at lower resolutions.
    """
    # Look for a HYBAS_ID that likely belongs to the Amazon (starts with 403)
    amazon_ids = [r for r in region_engine.list_hydrobasins() if r.startswith("403")]
    if amazon_ids:
        proj = region_engine.hydrobasin_projection(amazon_ids[0], value=0)
        assert np.sum(proj.data) > 0
