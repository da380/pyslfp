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
        (128, IceModel.ICE7G, "DH", standard_nondim),
        (128, IceModel.ICE6G, "DH", standard_nondim),
        # Add specific tests for DH2 and GLQ grids
        (64, IceModel.ICE7G, "DH2", standard_nondim),
        (64, IceModel.ICE7G, "GLQ", standard_nondim),
    ],
    ids=[
        "lmax64-ICE7G-DH-standard",
        "lmax64-ICE7G-DH-equatorial",
        "lmax128-ICE7G-DH-standard",
        "lmax128-ICE6G-DH-standard",
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
def test_sea_level_reciprocity(fingerprint: FingerPrint, rotational_feedbacks: bool):
    """
    Check the sea level reciprocity relation using random loads.
    This test verifies the self-adjoint nature of the sea-level operator.
    """
    direct_load_1 = random_load(fingerprint)
    direct_load_2 = random_load(fingerprint)
    rtol = 1e-9

    sea_level_change_1, _, _, _ = fingerprint(
        direct_load=direct_load_1,
        rotational_feedbacks=rotational_feedbacks,
        rtol=rtol,
    )
    sea_level_change_2, _, _, _ = fingerprint(
        direct_load=direct_load_2,
        rotational_feedbacks=rotational_feedbacks,
        rtol=rtol,
    )

    lhs = fingerprint.integrate(direct_load_2 * sea_level_change_1)
    rhs = fingerprint.integrate(direct_load_1 * sea_level_change_2)

    assert np.isclose(lhs, rhs, rtol=1000 * rtol)


@pytest.mark.parametrize("rotational_feedbacks", [True, False])
def test_generalised_sea_level_reciprocity(
    fingerprint: FingerPrint, rotational_feedbacks: bool
):
    """Test the generalised reciprocity relation using random forces."""
    direct_load_1 = random_load(fingerprint)
    direct_load_2 = random_load(fingerprint)
    displacement_load_1 = random_load(fingerprint)
    displacement_load_2 = random_load(fingerprint)
    gravitational_potential_load_1 = random_load(fingerprint)
    gravitational_potential_load_2 = random_load(fingerprint)
    angular_momentum_change_1 = (
        random_angular_momentum(fingerprint) if rotational_feedbacks else np.zeros(2)
    )
    angular_momentum_change_2 = (
        random_angular_momentum(fingerprint) if rotational_feedbacks else np.zeros(2)
    )
    rtol = 1e-9

    (
        sea_level_change_1,
        displacement_1,
        gravity_potential_change_1,
        angular_velocity_change_1,
    ) = fingerprint(
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
    ) = fingerprint(
        direct_load=direct_load_2,
        displacement_load=displacement_load_2,
        gravitational_potential_load=gravitational_potential_load_2,
        angular_momentum_change=angular_momentum_change_2,
        rtol=rtol,
    )

    g = fingerprint.gravitational_acceleration

    lhs_integrand = direct_load_2 * sea_level_change_1 - (1 / g) * (
        g * displacement_load_2 * displacement_1
        + gravitational_potential_load_2 * gravity_potential_change_1
    )
    lhs = (
        fingerprint.integrate(lhs_integrand)
        - np.dot(angular_momentum_change_2, angular_velocity_change_1) / g
    )

    rhs_integrand = direct_load_1 * sea_level_change_2 - (1 / g) * (
        g * displacement_load_1 * displacement_2
        + gravitational_potential_load_1 * gravity_potential_change_2
    )
    rhs = (
        fingerprint.integrate(rhs_integrand)
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


@pytest.mark.parametrize("sobolev", [False, True])
@pytest.mark.parametrize("rotational_feedbacks", [True, False])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("scale", [0.1, 0.5])
def test_linear_operator_adjoint_identity(
    fingerprint: FingerPrint,
    sobolev: bool,
    order: float,
    scale: float,
    rotational_feedbacks: bool,
):
    """
    Tests the adjoint identity (dot-product test) for the LinearOperator.

    This is a fundamental test that verifies that the implemented formal
    adjoint mapping is indeed the correct adjoint of the forward mapping
    with respect to the defined Hilbert space inner products.
    """

    rtol = 1e-9

    # 1. Create the linear operator with a tight solver tolerance for accuracy

    if sobolev:
        A = fingerprint.as_sobolev_linear_operator(
            order,
            scale * fingerprint.mean_sea_floor_radius,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )
    else:
        A = fingerprint.as_lebesgue_linear_operator(
            rotational_feedbacks=rotational_feedbacks, rtol=rtol
        )

    # 2. Create random elements in the domain and codomain spaces
    direct_load = random_load(fingerprint)
    adjoint_direct_load = random_load(fingerprint)
    adjoint_displacement_load = random_load(fingerprint)
    adjoint_gravitational_potential_load = random_load(fingerprint)

    adjoint_angular_momentum_change = (
        random_angular_momentum(fingerprint) if rotational_feedbacks else np.zeros(2)
    )

    u = direct_load
    v = [
        adjoint_direct_load,
        adjoint_displacement_load,
        adjoint_gravitational_potential_load,
        adjoint_angular_momentum_change,
    ]

    # 3. Calculate the inner products for both sides of the identity
    lhs = A.codomain.inner_product(A(u), v)
    rhs = A.domain.inner_product(u, A.adjoint(v))

    # 5. Assert that the two sides are equal within a relative tolerance
    # A looser tolerance is needed here due to numerical precision accumulating
    # through the iterative solvers in both the forward and adjoint mappings.
    assert np.isclose(lhs, rhs, rtol=1000 * rtol)


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
    incompatible_grid = SHGrid.from_zeros(lmax=16, grid=fingerprint.grid)

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
