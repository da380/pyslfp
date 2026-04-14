"""
Test suite for satellite altimetry observation models and operators.
"""

import pytest
import numpy as np
import pygeoinf as inf

from pyslfp.state import EarthState
from pyslfp.linear_operators.physics import FingerPrintOperator
from pyslfp.linear_operators.altimetry import (
    ocean_altimetry_points,
    ice_altimetry_points,
    sea_surface_height_operator,
    altimetry_point_operator,
    AltimetryObservationModel,
    JointAltimetryObservationModel,
)


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
        load_parameters=(1.0, 0.1),
        response_parameters=(2.0, 0.1),
        rotational_feedbacks=True,  # Leave True so we can test the centrifugal corrections
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
        testing_state, spacing_degrees=30.0, latitude_min=-60.0, latitude_max=60.0
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
    points = ice_altimetry_points(testing_state, spacing_degrees=30.0)

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
    into a single Sea Surface Height anomaly field.
    """
    response_space = fingerprint_operator.codomain
    state = fingerprint_operator.state
    b = state.model.parameters.mean_sea_floor_radius

    A = sea_surface_height_operator(
        state, response_space, remove_rotational_contribution=remove_rot
    )

    # Domain is the composite SLE response space
    domain_measure = fingerprint_operator.response_measure_for_testing()

    # Codomain is a single field (the SSH field)
    codomain_measure = A.codomain.heat_kernel_gaussian_measure(0.5 * b)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


@pytest.mark.parametrize("remove_rot", [True, False])
def test_altimetry_point_operator(fingerprint_operator, sample_points, remove_rot):
    """
    Tests the adjoint of SSH field calculation combined with point evaluation.
    """
    response_space = fingerprint_operator.codomain
    state = fingerprint_operator.state

    A = altimetry_point_operator(
        state, response_space, sample_points, remove_rotational_contribution=remove_rot
    )

    domain_measure = fingerprint_operator.response_measure_for_testing()
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


# ==================================================================== #
#                  3. Observation Models                               #
# ==================================================================== #


def test_altimetry_observation_model_adjoint(fingerprint_operator, sample_points):
    """
    Rigorously tests the full Ice -> Load -> SLE -> SSH -> Points adjoint chain.
    """
    model = AltimetryObservationModel(
        fingerprint_operator, sample_points, remove_rotational_contribution=True
    )
    A = model.forward_operator

    # Domain is the Ice Parameter space (defaults to the load space of the fingerprint)
    domain_measure = fingerprint_operator.load_measure_for_testing()
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


def test_joint_altimetry_observation_model_adjoint(fingerprint_operator, sample_points):
    """
    Rigorously tests the Joint [Ice, Ocean] -> Load -> SLE -> SSH -> Points chain.
    """
    model = JointAltimetryObservationModel(
        fingerprint_operator, sample_points, remove_rotational_contribution=True
    )
    A = model.forward_operator

    # The domain is a direct sum [Ice, Ocean], so we build a composite measure
    sub_measure = fingerprint_operator.load_measure_for_testing()
    domain_measure = inf.GaussianMeasure.from_direct_sum([sub_measure, sub_measure])

    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


def test_create_forward_problem(fingerprint_operator, sample_points):
    """
    Tests the instantiation of the LinearForwardProblem and noise covariance setups.
    """
    model = AltimetryObservationModel(fingerprint_operator, sample_points)

    prob_scalar = model.create_forward_problem(noise_std=0.1)
    assert isinstance(prob_scalar, inf.LinearForwardProblem)

    prob_array = model.create_forward_problem(noise_std=np.array([0.1, 0.2, 0.3]))
    assert isinstance(prob_array, inf.LinearForwardProblem)

    with pytest.raises(ValueError, match="does not match number of altimetry points"):
        model.create_forward_problem(noise_std=np.array([0.1, 0.2]))
