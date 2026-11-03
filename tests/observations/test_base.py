"""
Tests for the shape every observation model shares: the forward operator as
the sampling of the observable, and the noise.
"""

import numpy as np
import pytest
from pygeoinf import EuclideanSpace, GaussianMeasure, LinearOperator

from pyslfp.observations import ObservationModel


def _model(noise=None):
    model_space = EuclideanSpace(4)
    observable_space = EuclideanSpace(3)
    data_space = EuclideanSpace(2)
    rng = np.random.default_rng(0)
    observable = LinearOperator.from_matrix(
        model_space, observable_space, rng.normal(size=(3, 4)), form="components"
    )
    sampling = LinearOperator.from_matrix(
        observable_space, data_space, rng.normal(size=(2, 3)), form="components"
    )
    return ObservationModel(observable, sampling, noise=noise), observable, sampling


def test_the_forward_operator_is_the_sampling_of_the_observable():
    model, observable, sampling = _model()
    x = np.arange(4.0)
    np.testing.assert_allclose(model.forward_operator(x), sampling(observable(x)))
    assert model.model_space == EuclideanSpace(4)
    assert model.data_space == EuclideanSpace(2)
    assert model.noise is None


def test_the_forms_of_the_noise():
    """One standard deviation, one per datum, or a measure on the data space;
    a measure on another space is refused."""
    one, _, _ = _model(noise=0.5)
    np.testing.assert_allclose(
        one.noise.covariance.matrix(form="components"), 0.25 * np.eye(2)
    )
    each, _, _ = _model(noise=[0.1, 0.2])
    np.testing.assert_allclose(
        each.noise.covariance.matrix(form="components"), np.diag([0.01, 0.04])
    )
    measure = GaussianMeasure.from_standard_deviation(EuclideanSpace(2), 1.0)
    given, _, _ = _model(noise=measure)
    assert given.noise is measure
    with pytest.raises(ValueError, match="data space"):
        _model(noise=GaussianMeasure.from_standard_deviation(EuclideanSpace(3), 1.0))


def test_with_noise_leaves_the_model_as_it_was():
    model, _, _ = _model()
    noisy = model.with_noise(0.3)
    assert model.noise is None
    np.testing.assert_allclose(
        noisy.noise.covariance.matrix(form="components"), 0.09 * np.eye(2)
    )
    assert noisy.forward_operator is model.forward_operator


def test_a_sampling_of_another_space_is_refused():
    _, observable, _ = _model()
    with pytest.raises(ValueError, match="observable's space"):
        ObservationModel(observable, LinearOperator.identity(EuclideanSpace(2)))
