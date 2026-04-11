import pytest
import numpy as np
from pyshtools import SHGrid

from pyslfp.core import EarthModel
from pyslfp.state import EarthState
from pyslfp.ice_ng import IceNG, IceModel  # Used purely to populate real test grids

# ==================================================================== #
#                            Fixtures                                  #
# ==================================================================== #


@pytest.fixture(
    scope="module",
    params=[
        (64, IceModel.ICE7G, "DH"),
        (64, IceModel.ICE6G, "DH2"),
        (64, IceModel.ICE7G, "GLQ"),
    ],
    ids=["lmax64-ICE7G-DH", "lmax64-ICE6G-DH2", "lmax64-ICE7G-GLQ"],
)
def earth_state(request):
    """
    Provides a pre-configured EarthState instance for testing.
    Parameterized across different grid types and ice models to ensure
    pyshtools compatibility.
    """
    lmax, version, grid = request.param

    # FIX: lmax is strictly positional now
    model = EarthModel(lmax)

    # Use IceNG to get physically realistic ice and topography to test masks
    # version and length_scale are strictly keyword
    ice_ng = IceNG(version=version, length_scale=model.parameters.length_scale)

    sampling = 2 if grid == "DH2" else 1
    grid_type = "DH" if grid == "DH2" else grid

    # date and lmax are strictly positional, others are keyword
    ice_thickness, sea_level = ice_ng.get_ice_thickness_and_sea_level(
        0.0, lmax, grid=grid_type, sampling=sampling, extend=True
    )

    return EarthState(ice_thickness, sea_level, model)


def random_grid(state: EarthState) -> SHGrid:
    """Helper to generate a random grid matching the state's specs."""
    data = np.random.uniform(0, 1, size=(state.lats().size, state.lons().size))
    return SHGrid.from_array(data, grid=state.grid)


# ==================================================================== #
#                  Structural & Initialization Tests                   #
# ==================================================================== #


def test_earth_state_immutability(earth_state: EarthState):
    """Proves properties are read-only to prevent accidental mutation."""
    with pytest.raises(AttributeError):
        earth_state.ice_thickness = random_grid(earth_state)

    with pytest.raises(AttributeError):
        earth_state.model = EarthModel(32)  # FIX: positional lmax


def test_incompatible_grid_raises_error():
    """Tests that the state constructor rejects mismatched grids."""
    model = EarthModel(32)  # FIX: positional lmax
    ice = SHGrid.from_zeros(32, grid="DH")
    bad_sea_lmax = SHGrid.from_zeros(64, grid="DH")
    bad_sea_grid = SHGrid.from_zeros(32, grid="GLQ")

    with pytest.raises(ValueError, match="same lmax"):
        EarthState(ice, bad_sea_lmax, model)

    with pytest.raises(ValueError, match="same grid type"):
        EarthState(ice, bad_sea_grid, model)


def test_ocean_function_is_cached(earth_state: EarthState):
    """Ensures the heavy ocean function calculation is memoized."""
    ocean_1 = earth_state.ocean_function
    ocean_2 = earth_state.ocean_function
    assert ocean_1 is ocean_2  # Must be the exact same object in memory


# ==================================================================== #
#                       Integration Tests                              #
# ==================================================================== #


def test_integrate_constant_field(earth_state: EarthState):
    """Integrating a constant field of 1.0 should yield the surface area."""
    constant_data = np.ones((earth_state.lats().size, earth_state.lons().size))
    constant_field = SHGrid.from_array(constant_data, grid=earth_state.grid)

    integral_result = earth_state.integrate(constant_field)
    radius = earth_state.model.parameters.mean_sea_floor_radius
    expected_surface_area = 4.0 * np.pi * radius**2
    assert np.isclose(integral_result, expected_surface_area, rtol=1e-6)


def test_ocean_average_of_ocean_function(earth_state: EarthState):
    """The average of the ocean function over the oceans should be exactly 1.0."""
    integral = earth_state.integrate(earth_state.ocean_function)
    avg = integral / earth_state.ocean_area
    assert np.isclose(avg, 1.0)


# ==================================================================== #
#                    Tests for Projection Operators                    #
# ==================================================================== #


def test_ocean_projection_basic(earth_state: EarthState):
    ocean_proj = earth_state.ocean_projection()
    assert np.all((ocean_proj.data == 1) | np.isnan(ocean_proj.data))

    ocean_proj_zero = earth_state.ocean_projection(value=0)
    assert np.all((ocean_proj_zero.data == 1) | (ocean_proj_zero.data == 0))

    ocean_mask = earth_state.ocean_function.data > 0
    assert np.all(ocean_proj_zero.data[ocean_mask] == 1)


def test_ocean_projection_exclude_ice_shelves(earth_state: EarthState):
    ocean_proj = earth_state.ocean_projection(value=0, exclude_ice_shelves=False)
    ocean_proj_no_shelves = earth_state.ocean_projection(
        value=0, exclude_ice_shelves=True
    )

    assert np.sum(ocean_proj_no_shelves.data) <= np.sum(ocean_proj.data)

    ice_shelf_mask = (earth_state.ocean_function.data > 0) & (
        earth_state.ice_thickness.data > 0
    )
    if np.any(ice_shelf_mask):
        assert np.all(ocean_proj_no_shelves.data[ice_shelf_mask] == 0)


@pytest.mark.parametrize(
    "exclude_ice_shelves,exclude_glaciers",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_ice_projection_exclusions(
    earth_state: EarthState, exclude_ice_shelves: bool, exclude_glaciers: bool
):
    ice_proj = earth_state.ice_projection(
        value=0,
        exclude_ice_shelves=exclude_ice_shelves,
        exclude_glaciers=exclude_glaciers,
    )
    assert np.all((ice_proj.data == 0) | (ice_proj.data == 1))

    if exclude_ice_shelves:
        ice_shelf_mask = (earth_state.ocean_function.data > 0) & (
            earth_state.ice_thickness.data > 0
        )
        if np.any(ice_shelf_mask):
            assert np.all(ice_proj.data[ice_shelf_mask] == 0)

    if exclude_glaciers:
        glacier_proj = earth_state.glacier_projection(value=0)
        glacier_mask = glacier_proj.data == 1
        assert np.all(ice_proj.data[glacier_mask] == 0)


def test_land_projection_exclude_ice(earth_state: EarthState):
    land_proj = earth_state.land_projection(value=0, exclude_ice=False)
    land_proj_no_ice = earth_state.land_projection(value=0, exclude_ice=True)

    assert np.sum(land_proj_no_ice.data) <= np.sum(land_proj.data)

    icy_land_mask = (earth_state.ocean_function.data == 0) & (
        earth_state.ice_thickness.data > 0
    )
    if np.any(icy_land_mask):
        assert np.all(land_proj_no_ice.data[icy_land_mask] == 0)


def test_ocean_land_partition(earth_state: EarthState):
    """Every point is either ocean or land, not both."""
    ocean_proj = earth_state.ocean_projection(value=0)
    land_proj = earth_state.land_projection(value=0)

    total = ocean_proj.data + land_proj.data
    assert np.all(total == 1)

    overlap = ocean_proj.data * land_proj.data
    assert np.all(overlap == 0)


def test_glacier_projection_basic(earth_state: EarthState):
    glacier_proj = earth_state.glacier_projection(value=0)
    assert np.all((glacier_proj.data == 0) | (glacier_proj.data == 1))

    lats, lons = np.meshgrid(earth_state.lats(), earth_state.lons(), indexing="ij")
    expected_glacier_mask = (lats > 30) & (lats < 70) & (lons > 180) & (lons < 270)
    assert np.all(glacier_proj.data == expected_glacier_mask.astype(float))


def test_projection_integration(earth_state: EarthState):
    ocean_proj = earth_state.ocean_projection(value=0)
    land_proj = earth_state.land_projection(value=0)

    ocean_area = earth_state.integrate(ocean_proj)
    land_area = earth_state.integrate(land_proj)

    radius = earth_state.model.parameters.mean_sea_floor_radius
    total_area = 4.0 * np.pi * radius**2

    assert np.isclose(ocean_area + land_area, total_area, rtol=1e-6)


def test_projection_with_random_load(earth_state: EarthState):
    load = random_grid(earth_state)
    ocean_proj = earth_state.ocean_projection(value=0)

    ocean_load = SHGrid.from_array(load.data * ocean_proj.data, grid=earth_state.grid)
    ocean_integral = earth_state.integrate(ocean_load)

    assert np.isfinite(ocean_integral)


# ==================================================================== #
#             Tests for Regional and Hemisphere Projections            #
# ==================================================================== #


def test_hemisphere_projections(earth_state: EarthState):
    nh_proj = earth_state.northern_hemisphere_projection(value=0)
    sh_proj = earth_state.southern_hemisphere_projection(value=0)

    assert np.all((nh_proj.data == 0) | (nh_proj.data == 1))
    assert np.all((sh_proj.data == 0) | (sh_proj.data == 1))
    assert np.max(nh_proj.data * sh_proj.data) == 0


def test_ar6_regionmask_projections(earth_state: EarthState):
    grl_proj = earth_state.greenland_projection(value=0)
    wais_proj = earth_state.west_antarctic_projection(value=0)

    assert np.all((grl_proj.data == 0) | (grl_proj.data == 1))
    assert np.all((wais_proj.data == 0) | (wais_proj.data == 1))
    assert np.max(grl_proj.data * wais_proj.data) == 0

    with pytest.raises(ValueError, match="not found in the AR6 dataset"):
        earth_state.regionmask_projection("Atlantis")


def test_shapefile_region_lists(earth_state: EarthState):
    ant_regions = earth_state.list_imbie_ant_regions()
    grl_regions = earth_state.list_mouginot_grl_regions()
    assert isinstance(ant_regions, list) and len(ant_regions) > 0
    assert isinstance(grl_regions, list) and len(grl_regions) > 0


def test_shapefile_projections(earth_state: EarthState):
    ant_regions = earth_state.list_imbie_ant_regions()
    if ant_regions:
        proj = earth_state.imbie_ant_projection(ant_regions[0], value=0)
        assert np.all((proj.data == 0) | (proj.data == 1))
        assert np.sum(proj.data) > 0

    with pytest.raises(ValueError, match="not found in ANT"):
        earth_state.imbie_ant_projection("FakeBasin")


# ==================================================================== #
#                    Tests for Observation Generators                  #
# ==================================================================== #


def test_altimetry_point_generators(earth_state: EarthState):
    # Coarse spacing to keep test fast
    ocean_pts = earth_state.ocean_altimetry_points(spacing_degrees=10.0)
    ice_pts = earth_state.ice_altimetry_points(spacing_degrees=10.0)

    assert isinstance(ocean_pts, list)
    if ocean_pts:
        assert isinstance(ocean_pts[0], tuple)
        assert len(ocean_pts[0]) == 2
        lats = [p[0] for p in ocean_pts]
        assert np.max(lats) <= 66.0
        assert np.min(lats) >= -66.0

    assert isinstance(ice_pts, list)
    if ice_pts:
        assert isinstance(ice_pts[0], tuple)
        assert len(ice_pts[0]) == 2


# ==================================================================== #
#                    Tests for Load Converters                         #
# ==================================================================== #


def test_load_converters(earth_state: EarthState):
    """Tests that physical changes are correctly converted to mass loads."""
    dummy_change = random_grid(earth_state)

    # 1. Ice Thickness -> Load (Should apply ice density and mask out oceans)
    ice_load = earth_state.direct_load_from_ice_thickness_change(dummy_change)
    expected_ice = (
        earth_state.model.parameters.ice_density
        * earth_state.one_minus_ocean_function.data
        * dummy_change.data
    )
    assert np.allclose(ice_load.data, expected_ice)

    # 2. Sea Level -> Load (Should apply water density and mask out land)
    sl_load = earth_state.direct_load_from_sea_level_change(dummy_change)
    expected_sl = (
        earth_state.model.parameters.water_density
        * earth_state.ocean_function.data
        * dummy_change.data
    )
    assert np.allclose(sl_load.data, expected_sl)

    # 3. Density -> Load (Should scale by background sea level and mask out land)
    dens_load = earth_state.direct_load_from_density_change(dummy_change)
    expected_dens = (
        earth_state.sea_level.data * earth_state.ocean_function.data * dummy_change.data
    )
    assert np.allclose(dens_load.data, expected_dens)


# ==================================================================== #
#                    Tests for Convenience Loads                       #
# ==================================================================== #


def test_disk_load_grid_alignment(earth_state: EarthState):
    """Ensures disk_load respects the specific grid sampling of the state."""
    # FIX: disk_load arguments are purely positional
    d_load = earth_state.disk_load(10.0, 0.0, 0.0, 100.0)

    # Crucial: Must match the exact grid shape to prevent broadcasting errors in the solver
    assert d_load.lmax == earth_state.lmax
    assert d_load.grid == earth_state.grid
    assert d_load.data.shape == earth_state.ice_thickness.data.shape


def test_regional_convenience_loads(earth_state: EarthState):
    """Tests that regional loads are strictly bounded by their geographic masks."""
    fraction = 0.5

    # FIX: fractional loads strictly require fraction as a keyword
    # 1. Northern Hemisphere Load
    nh_load = earth_state.northern_hemisphere_load(fraction=fraction)
    nh_proj = earth_state.northern_hemisphere_projection(value=0)
    # Load must be exactly 0 where the projection mask is 0
    assert np.all(nh_load.data[nh_proj.data == 0] == 0)

    # 2. Greenland Load
    grl_load = earth_state.greenland_load(fraction=fraction)
    grl_proj = earth_state.greenland_projection(value=0)
    assert np.all(grl_load.data[grl_proj.data == 0] == 0)

    # 3. West Antarctic Load
    wais_load = earth_state.west_antarctic_load(fraction=fraction)
    wais_proj = earth_state.west_antarctic_projection(value=0)
    assert np.all(wais_load.data[wais_proj.data == 0] == 0)
