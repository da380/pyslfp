"""
Test suite for GRACE observation models and the WMB spectral method.
"""

import pytest
import numpy as np
import pygeoinf as inf
from pygeoinf.testing import check_operator

from pyslfp.core import EarthState
from pyslfp.linear_operators.physics import (
    FingerPrintOperator,
    lebesgue_load_space,
    sobolev_load_space,
)
from pygeoinf.symmetric_space.sphere import Lebesgue

from pyslfp.linear_operators.spatial import ice_projection_operator
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
    max_obs_degree = 10
    min_degree = 2

    A = grace_observation_operator(
        response_space, max_obs_degree, min_obs_degree=min_degree
    )

    # Expected Euclidean dimension: (L+1)^2 - lmin^2
    expected_dim = (max_obs_degree + 1) ** 2 - min_degree**2
    assert A.codomain.dim == expected_dim

    domain_measure = fingerprint_operator.response_measure_for_testing()

    check_operator(
        A,
        rng=np.random.default_rng(0),
        measure=domain_measure,
        trials=2,
        rtol=1e-4,
        atol=1e-4,
    )


def test_grace_observation_model_adjoint(fingerprint_operator):
    """
    Tests the full composite forward operator mapping from Mass Load
    straight to GRACE potential coefficients.
    """
    model = GraceObservationModel(fingerprint_operator, 10, min_obs_degree=2)
    A = model.forward_operator

    domain_measure = fingerprint_operator.load_measure_for_testing()

    check_operator(
        A,
        rng=np.random.default_rng(0),
        measure=domain_measure,
        trials=2,
        rtol=1e-4,
        atol=1e-4,
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

    wmb = WMBMethod(model, 10, min_obs_degree=2)

    if sobolev:
        load_space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        load_space = lebesgue_load_space(model)

    A = wmb.potential_coefficient_to_load_operator(load_space)

    domain_measure = inf.GaussianMeasure.from_standard_deviation(A.domain, 1.0)

    check_operator(
        A,
        rng=np.random.default_rng(0),
        measure=domain_measure,
        trials=2,
        rtol=1e-4,
        atol=1e-4,
    )


def test_wmb_preconditioner_generation(testing_state):
    """
    Verifies the preconditioning workflow correctly pushes measures forward
    and extracts diagonals without crashing.
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    wmb = WMBMethod(model, 10, min_obs_degree=2)
    load_space = lebesgue_load_space(model)

    prior_measure = load_space.heat_measure(0.5 * b)

    obs_measure = wmb.load_measure_to_observation_measure(prior_measure)
    assert isinstance(obs_measure, inf.GaussianMeasure)
    assert obs_measure.domain.dim == wmb.observation_dim

    noise_measure = inf.GaussianMeasure.from_standard_deviation(
        obs_measure.domain, 1e-10
    )

    preconditioner = wmb.bayesian_normal_operator_preconditioner(
        prior_measure, noise_measure, parallel=False
    )

    assert isinstance(preconditioner, inf.DiagonalLinearOperator)
    assert preconditioner.domain.dim == wmb.observation_dim


# ==================================================================== #
#                  3. Coefficient Convention                           #
# ==================================================================== #


def _observed_labels(min_degree, max_degree):
    """(l, m) for each observed coefficient, in the operators' order."""
    return [
        (l, m)
        for l in range(min_degree, max_degree + 1)
        for m in list(range(0, l + 1)) + list(range(-1, -l - 1, -1))
    ]


def test_grace_coefficients_are_unit_sphere_coefficients(fingerprint_operator):
    """
    The GRACE operator reports the potential change by pyslfp's own
    spherical harmonic coefficients (orthonormal on the unit sphere, no
    Condon-Shortley phase), whatever the radius of the response space.
    The test state's radius is not one, so a dropped factor of the radius
    would show at the 5e-4 level.
    """
    response_space = fingerprint_operator.codomain
    field_space = response_space.subspace(2)
    model = fingerprint_operator.model

    A = grace_observation_operator(response_space, 6, min_obs_degree=2)

    gpc = field_space.random(rng=np.random.default_rng(2))
    response = (field_space.zero(), field_space.zero(), gpc, np.zeros(3))
    coefficients = A(response)

    expected_lm = model.expand_field(gpc).coeffs
    expected = np.array(
        [
            expected_lm[0, l, m] if m >= 0 else expected_lm[1, l, -m]
            for l, m in _observed_labels(2, 6)
        ]
    )
    np.testing.assert_allclose(coefficients, expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("sobolev", [False, True])
def test_wmb_observation_measure_variances(testing_state, sobolev):
    """
    The observation measure, formed from the prior's spectrum, has the
    covariance of the exact push-forward of the prior through the WMB
    operator, whose diagonal is found by applying the composed covariance.
    On a Sobolev space the spectrum is relative to the space's metric, so
    reading it as coefficient variances would fail here.
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    wmb = WMBMethod(model, 8, min_obs_degree=2)
    if sobolev:
        load_space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        load_space = lebesgue_load_space(model)
    prior = load_space.heat_measure(0.3 * b)

    observed = wmb.load_measure_to_observation_measure(prior)
    variances = observed.covariance.diagonals()[0]

    pushed = prior.affine_map(wmb.load_to_potential_coefficient_operator(load_space))
    expected = pushed.covariance.diagonals()[0]

    np.testing.assert_allclose(variances, expected, rtol=1e-10, atol=0.0)


def test_wmb_observation_measure_beyond_truncation(testing_state):
    """
    Observed degrees above the load space's truncation get zero from the
    WMB operator but keep a variance: that of the highest resolved degree,
    scaled by k_l squared.
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius
    top, max_degree = 8, 12

    load_space = Lebesgue(top, radius=b, grid=model.grid, extend=model.extend)
    wmb = WMBMethod(model, max_degree, min_obs_degree=2)
    prior = load_space.heat_measure(0.3 * b)

    variances = wmb.load_measure_to_observation_measure(prior).covariance.diagonals()[0]
    operator = wmb.load_to_potential_coefficient_operator(load_space)
    image = operator(load_space.random(rng=np.random.default_rng(4)))

    beyond = slice((top + 1) ** 2 - 2**2, None)
    k = model.love_numbers.k
    top_variance = variances[top**2 - 2**2] / k[top] ** 2
    expected = np.concatenate(
        [
            np.full(2 * l + 1, k[l] ** 2 * top_variance)
            for l in range(top + 1, max_degree + 1)
        ]
    )

    # zero to the rounding of the transforms that prolong the field
    assert np.max(np.abs(image[beyond])) <= 1e-12 * np.max(np.abs(image))
    np.testing.assert_allclose(variances[beyond], expected, rtol=1e-12, atol=0.0)
    assert np.all(variances > 0.0)


def test_wmb_observation_measure_expectation(testing_state):
    """A prior's expectation is carried through the WMB operator."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    wmb = WMBMethod(model, 8, min_obs_degree=2)
    load_space = lebesgue_load_space(model)
    mean = load_space.random(rng=np.random.default_rng(3))
    prior = load_space.heat_measure(0.3 * b, expectation=mean)

    observed = wmb.load_measure_to_observation_measure(prior)
    expected = wmb.load_to_potential_coefficient_operator(load_space)(mean)

    np.testing.assert_allclose(observed.expectation, expected, rtol=1e-12, atol=0.0)


def test_wmb_observation_measure_refuses_non_invariant(testing_state):
    """A measure whose covariance is not diagonal in the harmonics is refused."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    wmb = WMBMethod(model, 8, min_obs_degree=2)
    load_space = lebesgue_load_space(model)
    restricted = load_space.heat_measure(0.3 * b).affine_map(
        ice_projection_operator(testing_state, load_space)
    )

    with pytest.raises(TypeError, match="invariant"):
        wmb.load_measure_to_observation_measure(restricted)
