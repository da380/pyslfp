"""
Test suite for the FingerPrint class.
"""

import pytest
import numpy as np
from pyslfp.finger_print import FingerPrint, EarthModelParameters, IceModel
from pyslfp.physical_parameters import EQUATORIAL_RADIUS, WATER_DENSITY
from pyshtools import SHGrid
import pygeoinf as inf

# Define the non-dimensionalisation schemes to be tested
standard_nondim = EarthModelParameters.from_standard_non_dimensionalisation()
equatorial_nondim = EarthModelParameters(
    length_scale=EQUATORIAL_RADIUS / 2, density_scale=WATER_DENSITY, time_scale=3600
)


@pytest.fixture(
    scope="module",
    params=[
        # Test default grid (DH1) with different lmax, ice models, and nondim schemes
        (64, IceModel.ICE7G, "DH", standard_nondim),
        (64, IceModel.ICE7G, "DH", equatorial_nondim),
        (100, IceModel.ICE7G, "DH", standard_nondim),
        (100, IceModel.ICE6G, "DH", standard_nondim),
        # Add specific tests for DH2 and GLQ grids
        (64, IceModel.ICE7G, "DH2", standard_nondim),
        (64, IceModel.ICE7G, "GLQ", standard_nondim),
    ],
    ids=[
        "lmax64-ICE7G-DH-standard",
        "lmax64-ICE7G-DH-equatorial",
        "lmax100-ICE7G-DH-standard",
        "lmax100-ICE6G-DH-standard",
        "lmax64-ICE7G-DH2-standard",
        "lmax64-ICE7G-GLQ-standard",
    ],
)
def fingerprint(request):
    """
    Provides a pre-configured FingerPrint instance for testing.
    This fixture is parameterized to create instances with different
    lmax values, background ice models, grid types, and non-dimensionalisation
    schemes.
    """
    lmax, version, grid, nondim_scheme = request.param
    fp = FingerPrint(
        lmax=lmax,
        grid=grid,
        earth_model_parameters=nondim_scheme,
    )
    fp.set_state_from_ice_ng(version=version)
    return fp


@pytest.fixture(scope="module")
def fast_fingerprint():
    """
    A single, lower-resolution FingerPrint instance dedicated to
    computationally heavy tests (like randomized adjoint checks) to save time.
    """
    fp = FingerPrint(lmax=64, grid="DH")
    fp.set_state_from_ice_ng(version=IceModel.ICE7G)
    return fp


def random_load(fp: FingerPrint) -> SHGrid:
    """Returns a random disk load for use in tests."""
    f = np.random.uniform()
    load1 = fp.northern_hemisphere_load()
    load2 = fp.southern_hemisphere_load()
    return load1 * f + load2 * (1 - f)


def random_angular_momentum(fp: FingerPrint):
    """Returns a random angular momentum jump."""
    b = fp.mean_sea_floor_radius
    omega = fp.rotation_frequency
    load = random_load(fp)
    load_lm = load.expand(lmax_calc=2, normalization="ortho")
    return omega * b**4 * load_lm.coeffs[:, 2, 1]


# ==================================================================== #
#                     Unit and Sanity Check Tests                      #
# ==================================================================== #


def test_zero_load_input(fingerprint: FingerPrint):
    """Sanity check: A zero load should produce a zero response."""
    zero_load = fingerprint.zero_grid()
    slc, disp, gpc, avc = fingerprint(direct_load=zero_load)

    assert np.all(slc.data == 0)
    assert np.all(disp.data == 0)
    assert np.all(gpc.data == 0)
    assert np.all(avc == 0)


def test_mass_conservation(fingerprint: FingerPrint):
    """
    Check for mass conservation: the mass of water added to the ocean
    must equal the mass removed by the direct load (e.g., melted ice).
    """
    direct_load = random_load(fingerprint)
    mass_removed = -fingerprint.integrate(direct_load)
    mean_sea_level_change = fingerprint.mean_sea_level_change(direct_load)
    mass_added = (
        mean_sea_level_change * fingerprint.ocean_area * fingerprint.water_density
    )
    assert np.isclose(mass_removed, mass_added, rtol=1e-5)


def test_integrate_constant_field(fingerprint: FingerPrint):
    """
    Unit test for the integrate method: integrating a constant field of 1.0
    over the sphere should yield the surface area of the sphere.
    """
    constant_field = fingerprint.constant_grid(1.0)
    integral_result = fingerprint.integrate(constant_field)
    radius = fingerprint.mean_sea_floor_radius
    expected_surface_area = 4.0 * np.pi * radius**2
    assert np.isclose(integral_result, expected_surface_area, rtol=1e-6)


def test_ocean_average_of_ocean_function(fingerprint: FingerPrint):
    """
    Unit test for ocean_average: the average of the ocean function
    over the oceans should be exactly 1.0.
    """
    avg = fingerprint.ocean_average(fingerprint.ocean_function)
    assert np.isclose(avg, 1.0)


# ==================================================================== #
#           High-Level Physics-Based Reciprocity Tests                 #
# ==================================================================== #


@pytest.mark.parametrize("rotational_feedbacks", [True, False])
def test_sea_level_reciprocity(
    fast_fingerprint: FingerPrint, rotational_feedbacks: bool
):
    """
    Check the sea level reciprocity relation using random loads.
    This acts as a direct physics test of the __call__ solver.
    """
    direct_load_1 = random_load(fast_fingerprint)
    direct_load_2 = random_load(fast_fingerprint)
    rtol = 1e-9

    sea_level_change_1, _, _, _ = fast_fingerprint(
        direct_load=direct_load_1,
        rotational_feedbacks=rotational_feedbacks,
        rtol=rtol,
    )
    sea_level_change_2, _, _, _ = fast_fingerprint(
        direct_load=direct_load_2,
        rotational_feedbacks=rotational_feedbacks,
        rtol=rtol,
    )

    lhs = fast_fingerprint.integrate(direct_load_2 * sea_level_change_1)
    rhs = fast_fingerprint.integrate(direct_load_1 * sea_level_change_2)

    assert np.isclose(lhs, rhs, rtol=1000 * rtol)


@pytest.mark.parametrize("rotational_feedbacks", [True, False])
def test_generalised_sea_level_reciprocity(
    fast_fingerprint: FingerPrint, rotational_feedbacks: bool
):
    """Test the generalised reciprocity relation using random forces."""
    direct_load_1 = random_load(fast_fingerprint)
    direct_load_2 = random_load(fast_fingerprint)
    displacement_load_1 = random_load(fast_fingerprint)
    displacement_load_2 = random_load(fast_fingerprint)
    gravitational_potential_load_1 = random_load(fast_fingerprint)
    gravitational_potential_load_2 = random_load(fast_fingerprint)

    angular_momentum_change_1 = (
        random_angular_momentum(fast_fingerprint)
        if rotational_feedbacks
        else np.zeros(2)
    )
    angular_momentum_change_2 = (
        random_angular_momentum(fast_fingerprint)
        if rotational_feedbacks
        else np.zeros(2)
    )
    rtol = 1e-9

    (
        sea_level_change_1,
        displacement_1,
        gravity_potential_change_1,
        angular_velocity_change_1,
    ) = fast_fingerprint(
        direct_load=direct_load_1,
        displacement_load=displacement_load_1,
        gravitational_potential_load=gravitational_potential_load_1,
        angular_momentum_change=angular_momentum_change_1,
        rtol=rtol,
    )

    (
        sea_level_change_2,
        displacement_2,
        gravity_potential_change_2,
        angular_velocity_change_2,
    ) = fast_fingerprint(
        direct_load=direct_load_2,
        displacement_load=displacement_load_2,
        gravitational_potential_load=gravitational_potential_load_2,
        angular_momentum_change=angular_momentum_change_2,
        rtol=rtol,
    )

    g = fast_fingerprint.gravitational_acceleration

    lhs_integrand = direct_load_2 * sea_level_change_1 - (1.0 / g) * (
        g * displacement_load_2 * displacement_1
        + gravitational_potential_load_2 * gravity_potential_change_1
    )
    lhs = (
        fast_fingerprint.integrate(lhs_integrand)
        - np.dot(angular_momentum_change_2, angular_velocity_change_1) / g
    )

    rhs_integrand = direct_load_1 * sea_level_change_2 - (1.0 / g) * (
        g * displacement_load_1 * displacement_2
        + gravitational_potential_load_1 * gravity_potential_change_2
    )
    rhs = (
        fast_fingerprint.integrate(rhs_integrand)
        - np.dot(angular_momentum_change_1, angular_velocity_change_2) / g
    )

    assert np.isclose(lhs, rhs, rtol=1000 * rtol)


def test_as_lebesgue_operator_creation(fingerprint: FingerPrint):
    """
    A simple smoke test to ensure the as_lebesgue_operator method runs
    and returns an object of the correct pygeoinf.LinearOperator type.
    """
    op = fingerprint.as_lebesgue_linear_operator()
    assert isinstance(op, inf.LinearOperator)


def test_as_sobolev_operator_creation(fingerprint: FingerPrint):
    """
    A simple smoke test to ensure the as_lebesgue_operator method runs
    and returns an object of the correct pygeoinf.LinearOperator type.
    """
    op = fingerprint.as_sobolev_linear_operator(
        2, 0.2 * fingerprint.mean_sea_floor_radius
    )
    assert isinstance(op, inf.LinearOperator)


@pytest.mark.parametrize(
    "sobolev, order, scale_factor, rotational_feedbacks",
    [
        # 1. Lebesgue, no rotational feedbacks
        (False, None, None, False),
        # 2. Lebesgue, with rotational feedbacks
        (False, None, None, True),
        # 3. Sobolev (Order 1), no rotational feedbacks
        (True, 1.0, 0.1, False),
        # 4. Sobolev (Order 2), with rotational feedbacks
        (True, 2.0, 0.2, True),
    ],
    ids=[
        "Lebesgue-NoRotation",
        "Lebesgue-Rotation",
        "Sobolev-O1-NoRotation",
        "Sobolev-O2-Rotation",
    ],
)
def test_linear_operator_adjoint_identity(
    fast_fingerprint: FingerPrint,  # <-- Use the fast fixture here!
    sobolev: bool,
    order: float,
    scale_factor: float,
    rotational_feedbacks: bool,
):
    """
    Tests the adjoint identity for the LinearOperator using pygeoinf's
    built-in .check() method with spatially regular Gaussian measures.
    """
    rtol = 1e-9
    check_rtol = 1e-4
    check_atol = 1e-4

    # 1. Construct the fingerprint linear operator
    if sobolev:
        A = fast_fingerprint.as_sobolev_linear_operator(
            order,
            scale_factor * fast_fingerprint.mean_sea_floor_radius,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )
    else:
        A = fast_fingerprint.as_lebesgue_linear_operator(
            rotational_feedbacks=rotational_feedbacks, rtol=rtol
        )

    # 2. Domain measure (Load space)
    smoothness_scale = 0.1 * fast_fingerprint.mean_sea_floor_radius
    domain_measure = A.domain.heat_kernel_gaussian_measure(smoothness_scale)

    # 3. Codomain measure (Response space)
    field_space = A.codomain.subspace(0)
    field_measure = field_space.heat_kernel_gaussian_measure(smoothness_scale)

    euclidean_space = A.codomain.subspace(3)
    angular_momentum_std = (
        fast_fingerprint.rotation_frequency * fast_fingerprint.mean_sea_floor_radius**4
    )

    euclidean_measure = inf.GaussianMeasure.from_standard_deviation(
        euclidean_space, angular_momentum_std
    )

    codomain_measure = inf.GaussianMeasure.from_direct_sum(
        [field_measure, field_measure, field_measure, euclidean_measure]
    )

    # 4. Run the comprehensive self-checks
    A.check(
        n_checks=3,
        check_rtol=check_rtol,
        check_atol=check_atol,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


# ====================================================================#
#               Tests for Error Handling and Assertions               #
# ====================================================================#


def test_uninitialized_state_raises_error():
    """
    Tests that accessing properties that depend on the background state
    raises an AttributeError if the state has not been set.
    """
    # Create an instance WITHOUT calling set_state_from_ice_ng
    fp = FingerPrint(lmax=32)

    with pytest.raises(AttributeError, match="Sea level has not been set"):
        _ = fp.sea_level

    with pytest.raises(AttributeError, match="Ice thickness has not been set"):
        _ = fp.ice_thickness

    with pytest.raises(
        AttributeError, match="must be set before computing ocean function"
    ):
        _ = fp.ocean_function


def test_incompatible_grid_raises_error(fingerprint: FingerPrint):
    """
    Tests that methods raise a ValueError if they are passed an SHGrid
    object that is not compatible with the FingerPrint instance's settings.
    """
    # Create a grid with a different lmax
    incompatible_grid = SHGrid.from_zeros(
        lmax=fingerprint.lmax + 1, grid=fingerprint.grid
    )

    with pytest.raises(ValueError, match="not compatible"):
        fingerprint.integrate(incompatible_grid)

    with pytest.raises(ValueError, match="not compatible"):
        fingerprint(direct_load=incompatible_grid)


def test_lmax_too_large_for_love_numbers_raises_error():
    """
    Tests that initializing a FingerPrint instance with an lmax greater
    than what's available in the Love number file raises a ValueError.
    """
    with pytest.raises(ValueError, match="is larger than the maximum degree"):
        # The default Love number file goes up to degree 4096
        _ = FingerPrint(lmax=5000)


def test_coefficient_evaluation_out_of_bounds(fingerprint: FingerPrint):
    """
    Tests that coefficient_evaluation raises a ValueError for out-of-bounds
    degree (l) or order (m).
    """
    grid = fingerprint.zero_grid()

    # Test l > lmax
    with pytest.raises(ValueError, match="is out of bounds"):
        fingerprint.coefficient_evaluation(grid, l=fingerprint.lmax + 1, m=0)

    # Test m > l
    with pytest.raises(ValueError, match="is out of bounds"):
        fingerprint.coefficient_evaluation(grid, l=2, m=3)


# ====================================================================#
#                    Tests for Projection Operators                   #
# ====================================================================#


def test_ocean_projection_basic(fingerprint: FingerPrint):
    """
    Tests that ocean_projection returns 1 over oceans and the specified
    value elsewhere.
    """
    # Test with default NaN value
    ocean_proj = fingerprint.ocean_projection()
    assert np.all((ocean_proj.data == 1) | np.isnan(ocean_proj.data))

    # Test with value=0
    ocean_proj_zero = fingerprint.ocean_projection(value=0)
    assert np.all((ocean_proj_zero.data == 1) | (ocean_proj_zero.data == 0))

    # Check that ocean function matches projection where defined
    ocean_mask = fingerprint.ocean_function.data > 0
    assert np.all(ocean_proj_zero.data[ocean_mask] == 1)


def test_ocean_projection_exclude_ice_shelves(fingerprint: FingerPrint):
    """
    Tests that ocean_projection with exclude_ice_shelves=True properly
    excludes regions with ice shelves.
    """
    ocean_proj = fingerprint.ocean_projection(value=0, exclude_ice_shelves=False)
    ocean_proj_no_shelves = fingerprint.ocean_projection(
        value=0, exclude_ice_shelves=True
    )

    # The projection excluding ice shelves should have equal or fewer ocean points
    assert np.sum(ocean_proj_no_shelves.data) <= np.sum(ocean_proj.data)

    # Ice shelf regions (ocean_function > 0 AND ice_thickness > 0) should be excluded
    ice_shelf_mask = (fingerprint.ocean_function.data > 0) & (
        fingerprint.ice_thickness.data > 0
    )
    if np.any(ice_shelf_mask):
        assert np.all(ocean_proj_no_shelves.data[ice_shelf_mask] == 0)


def test_ice_projection_basic(fingerprint: FingerPrint):
    """
    Tests that ice_projection returns 1 over ice sheets and the specified
    value elsewhere.
    """
    # Test with default parameters (exclude_glaciers=True by default)
    ice_proj = fingerprint.ice_projection()
    assert np.all((ice_proj.data == 1) | np.isnan(ice_proj.data))

    # Test with value=0
    ice_proj_zero = fingerprint.ice_projection(value=0)
    assert np.all((ice_proj_zero.data == 1) | (ice_proj_zero.data == 0))

    # Check that ice is properly identified
    ice_mask = fingerprint.ice_thickness.data > 0
    # Note: some ice may be excluded due to default glacier exclusion
    assert np.sum(ice_proj_zero.data) <= np.sum(ice_mask)


@pytest.mark.parametrize(
    "exclude_ice_shelves,exclude_glaciers",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
)
def test_ice_projection_exclusions(
    fingerprint: FingerPrint, exclude_ice_shelves: bool, exclude_glaciers: bool
):
    """
    Tests all four combinations of ice_projection exclusion parameters.
    """
    ice_proj = fingerprint.ice_projection(
        value=0,
        exclude_ice_shelves=exclude_ice_shelves,
        exclude_glaciers=exclude_glaciers,
    )

    # Result should only contain 0s and 1s
    assert np.all((ice_proj.data == 0) | (ice_proj.data == 1))

    # When excluding ice shelves, regions with both ocean_function and ice_thickness
    # should be excluded
    if exclude_ice_shelves:
        ice_shelf_mask = (fingerprint.ocean_function.data > 0) & (
            fingerprint.ice_thickness.data > 0
        )
        if np.any(ice_shelf_mask):
            assert np.all(ice_proj.data[ice_shelf_mask] == 0)

    # When excluding glaciers, glacier regions should be excluded
    if exclude_glaciers:
        glacier_proj = fingerprint.glacier_projection(value=0)
        glacier_mask = glacier_proj.data == 1
        # Ice in glacier regions should be excluded
        assert np.all(ice_proj.data[glacier_mask] == 0)


def test_ice_projection_glacier_exclusion_effect(fingerprint: FingerPrint):
    """
    Tests that excluding glaciers actually reduces the ice coverage.
    """
    ice_all = fingerprint.ice_projection(value=0, exclude_glaciers=False)
    ice_no_glaciers = fingerprint.ice_projection(value=0, exclude_glaciers=True)

    # Excluding glaciers should result in equal or less ice coverage
    assert np.sum(ice_no_glaciers.data) <= np.sum(ice_all.data)


def test_land_projection_basic(fingerprint: FingerPrint):
    """
    Tests that land_projection returns 1 over land and the specified
    value elsewhere.
    """
    # Test with default parameters
    land_proj = fingerprint.land_projection()
    assert np.all((land_proj.data == 1) | np.isnan(land_proj.data))

    # Test with value=0
    land_proj_zero = fingerprint.land_projection(value=0)
    assert np.all((land_proj_zero.data == 1) | (land_proj_zero.data == 0))

    # Land should be where ocean_function is 0
    land_mask = fingerprint.ocean_function.data == 0
    assert np.all(land_proj_zero.data[land_mask] == 1)


def test_land_projection_exclude_ice(fingerprint: FingerPrint):
    """
    Tests that land_projection with exclude_ice=True properly excludes
    regions with ice.
    """
    land_proj = fingerprint.land_projection(value=0, exclude_ice=False)
    land_proj_no_ice = fingerprint.land_projection(value=0, exclude_ice=True)

    # Excluding ice should result in equal or less land coverage
    assert np.sum(land_proj_no_ice.data) <= np.sum(land_proj.data)

    # Icy land regions (ocean_function == 0 AND ice_thickness > 0) should be excluded
    icy_land_mask = (fingerprint.ocean_function.data == 0) & (
        fingerprint.ice_thickness.data > 0
    )
    if np.any(icy_land_mask):
        assert np.all(land_proj_no_ice.data[icy_land_mask] == 0)


def test_ocean_land_partition(fingerprint: FingerPrint):
    """
    Tests that ocean and land projections partition the sphere
    (every point is either ocean or land, not both).
    """
    ocean_proj = fingerprint.ocean_projection(value=0)
    land_proj = fingerprint.land_projection(value=0)

    # Every point should be either ocean (1,0) or land (0,1)
    total = ocean_proj.data + land_proj.data
    assert np.all(total == 1)

    # No overlap
    overlap = ocean_proj.data * land_proj.data
    assert np.all(overlap == 0)


def test_glacier_projection_basic(fingerprint: FingerPrint):
    """
    Tests that glacier_projection returns expected values.
    """
    glacier_proj = fingerprint.glacier_projection(value=0)

    # Should only contain 0s and 1s
    assert np.all((glacier_proj.data == 0) | (glacier_proj.data == 1))

    # Glacier region should be North American region (30-70N, 180-270E)
    lats, lons = np.meshgrid(fingerprint.lats(), fingerprint.lons(), indexing="ij")
    lat_mask = np.logical_and(lats > 30, lats < 70)
    lon_mask = np.logical_and(lons > 180, lons < 270)
    expected_glacier_mask = np.logical_and(lat_mask, lon_mask)

    assert np.all(glacier_proj.data == expected_glacier_mask.astype(float))


def test_projection_integration(fingerprint: FingerPrint):
    """
    Tests that projections can be integrated to get total area.
    """
    ocean_proj = fingerprint.ocean_projection(value=0)
    land_proj = fingerprint.land_projection(value=0)

    ocean_area = fingerprint.integrate(ocean_proj)
    land_area = fingerprint.integrate(land_proj)

    # Ocean and land areas should sum to total sphere area
    radius = fingerprint.mean_sea_floor_radius
    total_area = 4.0 * np.pi * radius**2

    assert np.isclose(ocean_area + land_area, total_area, rtol=1e-6)


def test_projection_multiplication(fingerprint: FingerPrint):
    """
    Tests that projections can be multiplied to get intersections.
    """
    # Get ice projection without glacier exclusion
    ice_proj = fingerprint.ice_projection(
        value=0, exclude_ice_shelves=False, exclude_glaciers=False
    )
    land_proj = fingerprint.land_projection(value=0)
    glacier_proj = fingerprint.glacier_projection(value=0)

    # Ice on land (excluding ice shelves)
    ice_on_land = ice_proj.data * land_proj.data

    # Glacier ice (intersection of ice and glacier regions)
    glacier_ice = ice_proj.data * glacier_proj.data

    # Both should be binary
    assert np.all((ice_on_land == 0) | (ice_on_land == 1))
    assert np.all((glacier_ice == 0) | (glacier_ice == 1))

    # Ice on land should be non-zero somewhere
    assert np.sum(ice_on_land) > 0


def test_projection_with_random_load(fingerprint: FingerPrint):
    """
    Tests that projections work correctly when applied to a random load.
    """
    load = random_load(fingerprint)
    ocean_proj = fingerprint.ocean_projection(value=0)

    # Apply projection
    ocean_load = SHGrid.from_array(load.data * ocean_proj.data, grid=fingerprint.grid)

    # Integrate over ocean only
    ocean_integral = fingerprint.integrate(ocean_load)

    # Should be finite
    assert np.isfinite(ocean_integral)


# ====================================================================#
#             Tests for Regional and Hemisphere Projections           #
# ====================================================================#


def test_hemisphere_projections(fingerprint: FingerPrint):
    """
    Tests that the northern and southern hemisphere projections are binary
    and mutually exclusive.
    """
    nh_proj = fingerprint.northern_hemisphere_projection(value=0)
    sh_proj = fingerprint.southern_hemisphere_projection(value=0)

    # Check they are binary
    assert np.all((nh_proj.data == 0) | (nh_proj.data == 1))
    assert np.all((sh_proj.data == 0) | (sh_proj.data == 1))

    # Check they do not overlap
    overlap = nh_proj.data * sh_proj.data
    assert np.max(overlap) == 0


def test_ar6_regionmask_projections(fingerprint: FingerPrint):
    """
    Tests the AR6 regionmask convenience methods and error handling.
    """
    grl_proj = fingerprint.greenland_projection(value=0)
    wais_proj = fingerprint.west_antarctic_projection(value=0)

    # Check they are binary
    assert np.all((grl_proj.data == 0) | (grl_proj.data == 1))
    assert np.all((wais_proj.data == 0) | (wais_proj.data == 1))

    # Greenland and West Antarctica should not overlap
    assert np.max(grl_proj.data * wais_proj.data) == 0

    # Test error handling for a non-existent region
    with pytest.raises(ValueError, match="not found in the AR6 dataset"):
        fingerprint.regionmask_projection("Atlantis")


def test_shapefile_region_lists(fingerprint: FingerPrint):
    """
    Tests that the dynamic listing of IMBIE and Mouginot regions works.
    """
    ant_regions = fingerprint.list_imbie_ant_regions()
    grl_regions = fingerprint.list_mouginot_grl_regions()

    assert isinstance(ant_regions, list)
    assert isinstance(grl_regions, list)
    assert len(ant_regions) > 0
    assert len(grl_regions) > 0


def test_shapefile_projections(fingerprint: FingerPrint):
    """
    Tests the dynamic projection generation for IMBIE and Mouginot regions.
    """
    ant_regions = fingerprint.list_imbie_ant_regions()
    if ant_regions:
        # Test a valid Antarctica region
        test_region = ant_regions[0]
        proj = fingerprint.imbie_ant_projection(test_region, value=0)
        assert np.all((proj.data == 0) | (proj.data == 1))
        # It should have some non-zero area
        assert np.sum(proj.data) > 0

    # Test error handling
    with pytest.raises(ValueError, match="not found in ANT"):
        fingerprint.imbie_ant_projection("FakeBasin")


# ====================================================================#
#                    Tests for Observation Generators                 #
# ====================================================================#


def test_altimetry_point_generators(fingerprint: FingerPrint):
    """
    Tests that the altimetry point generators return valid lists of
    (lat, lon) coordinates based on the respective masks.
    """
    # Use a coarse spacing to keep the test fast
    ocean_pts = fingerprint.ocean_altimetry_points(spacing_degrees=10.0)
    ice_pts = fingerprint.ice_altimetry_points(spacing_degrees=10.0)

    # Check ocean points
    assert isinstance(ocean_pts, list)
    if ocean_pts:
        assert isinstance(ocean_pts[0], tuple)
        assert len(ocean_pts[0]) == 2
        # Ensure points respect the default latitude bounds (-66 to 66)
        lats = [p[0] for p in ocean_pts]
        assert np.max(lats) <= 66.0
        assert np.min(lats) >= -66.0

    # Check ice points
    assert isinstance(ice_pts, list)
    if ice_pts:
        assert isinstance(ice_pts[0], tuple)
        assert len(ice_pts[0]) == 2
