"""
Tests for the water bodies: labelling on a raster, the present-day landlocked
depressions, and the land a state holds whatever the flotation.
"""

import numpy as np
import pytest

from pyslfp.core import EarthState
from pyslfp.ice import AnalyticalIceModel, IceNG
from pyslfp.core import EarthModel
from pyslfp.core.water_bodies import Channel, label_water_bodies


@pytest.fixture(scope="module")
def raster():
    """A one-degree raster: land at 100 m, an ocean south of 30S, and four
    depressions below sea level."""
    lat = -89.5 + np.arange(180.0)
    lon = 0.5 + np.arange(360.0)
    elevation = np.full((180, 360), 100.0)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    elevation[LAT < -30] = -4000.0
    basins = {
        "far": (LAT > 30) & (LAT < 40) & (LON > 100) & (LON < 110),
        "near": (LAT > -29) & (LAT < -20) & (LON > 50) & (LON < 60),
        "channelled": (LAT > 10) & (LAT < 20) & (LON > 200) & (LON < 210),
        "wrapped": (LAT > 50) & (LAT < 55) & ((LON < 3) | (LON > 357)),
    }
    for mask in basins.values():
        elevation[mask] = -50.0
    return lat, lon, elevation, basins


def test_a_basin_cut_off_from_the_ocean_is_landlocked(raster):
    lat, lon, elevation, basins = raster
    bodies = label_water_bodies(elevation, lat, lon)
    for name in basins:
        assert np.all(bodies.landlocked[basins[name]]), name
    ocean = elevation < -1000.0
    assert np.all(bodies.labels[ocean] == bodies.ocean)
    assert not bodies.landlocked[ocean].any()


def test_a_basin_across_the_antimeridian_is_one_body(raster):
    lat, lon, elevation, basins = raster
    bodies = label_water_bodies(elevation, lat, lon)
    assert np.unique(bodies.labels[basins["wrapped"]]).size == 1


def test_a_body_near_the_ocean_is_joined_to_it(raster):
    """The near basin lies one row, about 111 km, from the ocean."""
    lat, lon, elevation, basins = raster
    bodies = label_water_bodies(elevation, lat, lon, connect_within=150.0)
    assert not bodies.landlocked[basins["near"]].any()
    assert np.all(bodies.landlocked[basins["far"]])


def test_a_stated_channel_joins_a_basin(raster):
    lat, lon, elevation, basins = raster
    channel = Channel("test", (15.0, 205.0), (-35.0, 205.0))
    bodies = label_water_bodies(elevation, lat, lon, channels=[channel])
    assert not bodies.landlocked[basins["channelled"]].any()
    assert np.all(bodies.landlocked[basins["far"]])


def test_dry_cells_hold_no_water(raster):
    lat, lon, elevation, basins = raster
    bodies = label_water_bodies(elevation, lat, lon, dry=basins["far"])
    assert np.all(bodies.labels[basins["far"]] == 0)
    assert not bodies.landlocked[basins["far"]].any()


# ==================================================================== #
#                         Land held by a state                         #
# ==================================================================== #


def test_land_is_held_whatever_the_flotation():
    state = EarthState.for_testing(16)
    assert state.land is None
    north = state.northern_hemisphere_projection(value=0.0)
    held = EarthState(state.ice_thickness, state.sea_level, state.model, land=north)
    assert np.all(held.ocean_function.data[north.data > 0] == 0.0)
    south = north.data == 0
    np.testing.assert_array_equal(
        held.ocean_function.data[south], state.ocean_function.data[south]
    )
    np.testing.assert_array_equal(held.land.data, north.data)
    with pytest.raises(ValueError, match="ones and zeros"):
        EarthState(state.ice_thickness, state.sea_level, state.model, land=0.5 * north)


def test_ice_ng_holds_the_landlocked_depressions():
    """By default an ICE-NG state holds the present-day landlocked
    depressions as land; with land=None flotation floods the Caspian."""
    model = EarthModel(64)
    ice_model = IceNG(length_scale=model.parameters.length_scale)
    held = EarthState.from_ice_model(ice_model, 0.0, model)
    flooded = EarthState.from_ice_model(ice_model, 0.0, model, land=None)
    np.testing.assert_array_equal(
        held.land.data, held.landlocked_projection(value=0.0).data
    )
    i = np.argmin(np.abs(held.lats() - 42.0))
    j = np.argmin(np.abs(held.lons() - 50.5))
    assert flooded.ocean_function.data[i, j] == 1.0
    assert held.ocean_function.data[i, j] == 0.0
    assert flooded.land is None


def test_a_model_without_geography_holds_nothing():
    model = EarthModel(16)
    ice_model = AnalyticalIceModel(length_scale=model.parameters.length_scale)
    assert ice_model.get_land(16) is None
    assert EarthState.from_ice_model(ice_model, 0.0, model).land is None
    with pytest.raises(ValueError, match="land must be"):
        EarthState.from_ice_model(ice_model, 0.0, model, land="earth")
