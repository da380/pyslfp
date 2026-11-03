"""
Test suite for satellite altimetry observation models and operators.
"""

import pytest
import numpy as np
import pygeoinf as inf
from pygeoinf.testing import check_operator

from pyslfp.core import EarthState
from pyslfp.elastic import FingerPrintOperator
from pyslfp.observations import (
    ocean_altimetry_points,
    ice_altimetry_points,
    altimetry_point_operator,
)
from pyslfp.core import geocentric_sea_level_operator


# ==================================================================== #
#                          Fixtures                                    #
# ==================================================================== #


@pytest.fixture(scope="module")
def operator_lmax():
    return 16


@pytest.fixture(scope="module")
def testing_state(operator_lmax):
    return EarthState.for_testing(operator_lmax)


@pytest.fixture(scope="module")
def fingerprint_operator(operator_lmax):
    """
    Provides a FingerPrintOperator with Sobolev spaces.
    A load order of 1.0 yields a response order of 2.0, which is
    mathematically required for continuous point evaluation.
    """
    return FingerPrintOperator.for_testing(
        operator_lmax,
        load_parameters=(2.0, 0.1),
        response_parameters=(3.0, 0.1),
        rotational_feedbacks=True,
    )


@pytest.fixture
def sample_points():
    """Returns a few sample altimetry coordinates."""
    return [(0.0, 180.0), (45.0, -45.0), (-60.0, 90.0)]


# ==================================================================== #
#                  1. Point Generation Utilities                       #
# ==================================================================== #


def test_ocean_altimetry_points(testing_state):
    """Verifies that the ocean altimetry grid generator filters correctly."""
    # Use a huge spacing (30 degrees) so the test runs instantly
    points = ocean_altimetry_points(
        testing_state, spacing=30.0, latitude_min=-60.0, latitude_max=60.0
    )

    assert isinstance(points, list)
    assert len(points) > 0
    assert isinstance(points[0], tuple)

    # Ensure latitude bounds were respected
    lats = [p[0] for p in points]
    assert np.max(lats) <= 60.0
    assert np.min(lats) >= -60.0


def test_ice_altimetry_points(testing_state):
    """Verifies that the ice altimetry grid generator filters correctly."""
    points = ice_altimetry_points(testing_state, spacing=30.0)

    assert isinstance(points, list)
    if points:  # Depends on the analytical mock state having ice coverage
        assert isinstance(points[0], tuple)


# ==================================================================== #
#                  2. Raw Math Operators                               #
# ==================================================================== #


@pytest.mark.parametrize("remove_rot", [True, False])
def test_sea_surface_height_operator(fingerprint_operator, remove_rot):
    """
    Tests the operator that collapses the 4-component physical response
    into a single geocentric sea level change.
    """
    response_space = fingerprint_operator.codomain
    state = fingerprint_operator.state
    b = state.model.parameters.mean_sea_floor_radius

    A = geocentric_sea_level_operator(
        state, response_space, remove_rotational_contribution=remove_rot
    )

    # Domain is the composite SLE response space
    domain_measure = fingerprint_operator.response_measure_for_testing()

    check_operator(
        A,
        rng=np.random.default_rng(0),
        measure=domain_measure,
        codomain_measure=A.codomain.heat_measure(0.5 * b),
        trials=2,
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("remove_rot", [True, False])
def test_altimetry_point_operator(fingerprint_operator, sample_points, remove_rot):
    """
    Tests the adjoint of the geocentric sea level combined with point evaluation.
    """
    response_space = fingerprint_operator.codomain
    state = fingerprint_operator.state

    A = altimetry_point_operator(
        state, response_space, sample_points, remove_rotational_contribution=remove_rot
    )

    domain_measure = fingerprint_operator.response_measure_for_testing()

    check_operator(
        A,
        rng=np.random.default_rng(0),
        measure=domain_measure,
        codomain_measure=inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0),
        trials=2,
        rtol=1e-4,
        atol=1e-4,
    )
