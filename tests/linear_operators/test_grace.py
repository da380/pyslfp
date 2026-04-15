"""
Test suite for GRACE observation models and the WMB spectral method.
"""

import pytest
import pygeoinf as inf
from pygeoinf.symmetric_space.symmetric_space import InvariantGaussianMeasure

from pyslfp.state import EarthState
from pyslfp.linear_operators.physics import (
    FingerPrintOperator,
    lebesgue_load_space,
    sobolev_load_space,
)
from pyslfp.linear_operators.grace import (
    grace_observation_operator,
    GraceObservationModel,
    WMBMethod,
)


# ==================================================================== #
#                          Fixtures                                    #
# ==================================================================== #


@pytest.fixture(scope="module")
def operator_lmax():
    return 16


@pytest.fixture(scope="module")
def testing_state(operator_lmax):
    state = EarthState.for_testing(operator_lmax)
    return state


@pytest.fixture(scope="module")
def fingerprint_operator(operator_lmax):
    """Provides a base FingerPrintOperator with a Sobolev load space."""
    return FingerPrintOperator.for_testing(
        operator_lmax,
        load_parameters=(1.0, 0.1),
        response_parameters=(2.0, 0.1),
        rotational_feedbacks=False,
    )


# ==================================================================== #
#                  1. Exact Physical Model (SLE)                       #
# ==================================================================== #


def test_grace_observation_operator_adjoint(fingerprint_operator):
    """
    Tests the operator mapping the SLE response space to truncated
    spherical harmonic coefficients of the Gravity Potential Change.
    """
    response_space = fingerprint_operator.codomain
    obs_degree = 10
    min_degree = 2

    A = grace_observation_operator(
        response_space, obs_degree, minimum_degree=min_degree
    )

    # Expected Euclidean dimension: (L+1)^2 - lmin^2
    expected_dim = (obs_degree + 1) ** 2 - min_degree**2
    assert A.codomain.dim == expected_dim

    domain_measure = fingerprint_operator.response_measure_for_testing()
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


def test_grace_observation_model_adjoint(fingerprint_operator):
    """
    Tests the full composite forward operator mapping from Mass Load
    straight to GRACE potential coefficients.
    """
    model = GraceObservationModel(fingerprint_operator, 10, minimum_degree=2)
    A = model.forward_operator

    domain_measure = fingerprint_operator.load_measure_for_testing()
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


# ==================================================================== #
#                  2. Spectral Approximation (WMB Method)              #
# ==================================================================== #


@pytest.mark.parametrize("sobolev", [False, True])
def test_wmb_potential_to_load_operator_adjoint(testing_state, sobolev):
    """
    Tests the adjoint of the fast WMB spectral mapping from potential
    coefficients back to continuous surface loads.
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    wmb = WMBMethod(model, 10, minimum_degree=2)

    if sobolev:
        load_space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        load_space = lebesgue_load_space(model)

    A = wmb.potential_coefficient_to_load_operator(load_space)

    domain_measure = inf.GaussianMeasure.from_standard_deviation(A.domain, 1.0)
    codomain_measure = load_space.heat_kernel_gaussian_measure(0.5 * b)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


def test_wmb_preconditioner_generation(testing_state):
    """
    Verifies the preconditioning workflow correctly pushes measures forward
    and extracts diagonals without crashing.
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    wmb = WMBMethod(model, 10, minimum_degree=2)
    load_space = lebesgue_load_space(model)

    prior_measure = load_space.heat_kernel_gaussian_measure(0.5 * b)
    assert isinstance(prior_measure, InvariantGaussianMeasure)

    obs_measure = wmb.load_measure_to_observation_measure(prior_measure)
    assert isinstance(obs_measure, inf.GaussianMeasure)
    assert obs_measure.domain.dim == wmb.observation_dim

    noise_measure = inf.GaussianMeasure.from_standard_deviation(
        obs_measure.domain, 1e-10
    )

    preconditioner = wmb.bayesian_normal_operator_preconditioner(
        prior_measure, noise_measure, parallel=False
    )

    assert isinstance(preconditioner, inf.DiagonalSparseMatrixLinearOperator)
    assert preconditioner.domain.dim == wmb.observation_dim
