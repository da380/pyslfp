import pytest
import numpy as np
from pyslfp.finger_print import FingerPrint, EarthModelParameters, IceModel
from pyshtools import SHGrid


@pytest.fixture(
    scope="module",
    params=[
        (64, IceModel.ICE7G),  # Low resolution, ICE7G
        (128, IceModel.ICE7G),  # High resolution, ICE7G
        (128, IceModel.ICE6G),  # High resolution, ICE6G
    ],
    ids=[
        "lmax64-ICE7G",
        "lmax128-ICE7G",
        "lmax128-ICE6G",
    ],
)
def fingerprint(request):
    """
    Provides a pre-configured FingerPrint instance for testing.
    This fixture is parameterized to create instances with different
    lmax values and background ice models.
    """
    lmax, version = request.param
    fp = FingerPrint(
        lmax=lmax,
        earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng(version=version)
    return fp


def random_load(fp: FingerPrint) -> SHGrid:
    """Returns a random disk load for use in tests."""
    delta = np.random.uniform(10, 30)
    lat = np.random.uniform(-90, 90)
    lon = np.random.uniform(-180, 180)
    amp = np.random.randn()
    return fp.disk_load(delta, lat, lon, amp)


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
    rtol = 1e-6

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

    assert np.isclose(lhs, rhs, rtol=5000 * rtol)


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
    rtol = 1e-6

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

    assert np.isclose(lhs, rhs, rtol=5000 * rtol)
