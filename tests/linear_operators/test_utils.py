"""
Test suite for the linear operator mathematical utilities.
"""

import pytest
import numpy as np

import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.state import EarthState
from pyslfp.linear_operators.utils import (
    underlying_space,
    check_load_space,
    check_response_space,
    l2_products_operator,
    averaging_operator,
    spatial_multiplication_operator,
)
from pyslfp.linear_operators.physics import (
    lebesgue_load_space,
    sobolev_load_space,
    lebesgue_response_space,
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


# ==================================================================== #
#                  1. Space Resolution and Validation                  #
# ==================================================================== #


def test_underlying_space(testing_state):
    """
    Tests that underlying_space recursively strips mass-weighting
    (like Sobolev regularizations) from complex Hilbert spaces.
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    # 1. Base Lebesgue space (should return itself)
    leb = lebesgue_load_space(model)
    assert underlying_space(leb) is leb

    # 2. Sobolev space (should return the underlying Lebesgue space)
    sob = sobolev_load_space(model, 1.0, 0.1 * b)
    resolved_sob = underlying_space(sob)
    assert isinstance(resolved_sob, Lebesgue)
    assert not isinstance(resolved_sob, Sobolev)

    # 3. Direct Sum space
    ds = inf.HilbertSpaceDirectSum([sob, sob])
    resolved_ds = underlying_space(ds)
    assert isinstance(resolved_ds, inf.HilbertSpaceDirectSum)
    assert isinstance(resolved_ds.subspace(0), Lebesgue)
    assert not isinstance(resolved_ds.subspace(0), Sobolev)


def test_check_load_space(testing_state):
    """Tests the validation rules for load spaces."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    leb = lebesgue_load_space(model)
    sob_low = sobolev_load_space(model, 0.5, 0.1 * b)
    sob_high = sobolev_load_space(model, 2.0, 0.1 * b)
    euc = inf.EuclideanSpace(5)

    # Valid spaces
    assert check_load_space(leb) is True
    assert check_load_space(sob_low) is True

    # Invalid space type
    with pytest.raises(ValueError, match="must be a Lebesgue or Sobolev"):
        check_load_space(euc)

    # Point evaluation constraints (requires Sobolev order > 1.0)
    with pytest.raises(ValueError, match="order > 1 for point evaluation"):
        check_load_space(leb, point_values=True)
    with pytest.raises(ValueError, match="order > 1 for point evaluation"):
        check_load_space(sob_low, point_values=True)

    assert check_load_space(sob_high, point_values=True) is True


def test_check_response_space(testing_state):
    """Tests the strict structural validation of the 4-component SLE response space."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    valid_response = lebesgue_response_space(model)

    # Valid
    check_response_space(valid_response)

    # Invalid: Not a direct sum
    with pytest.raises(ValueError, match="must be a HilbertSpaceDirectSum"):
        check_response_space(lebesgue_load_space(model))

    # Invalid: Wrong number of components
    leb = lebesgue_load_space(model)
    wrong_length = inf.HilbertSpaceDirectSum([leb, leb, leb])
    with pytest.raises(ValueError, match="exactly 4 subspaces"):
        check_response_space(wrong_length)

    # Invalid: Subspaces 1 and 2 don't match 0
    sob = sobolev_load_space(model, 1.0, 0.1 * b)
    mismatched = inf.HilbertSpaceDirectSum([leb, sob, leb, inf.EuclideanSpace(2)])
    with pytest.raises(ValueError, match="Subspaces 1 and 2 must match"):
        check_response_space(mismatched)

    # Invalid: Rotation space is wrong
    bad_rot = inf.HilbertSpaceDirectSum([leb, leb, leb, inf.EuclideanSpace(3)])
    with pytest.raises(ValueError, match="2D Euclidean space for rotation"):
        check_response_space(bad_rot)


# ==================================================================== #
#                  2. Utility Operators Adjoint Checks                 #
# ==================================================================== #


@pytest.mark.parametrize("sobolev", [False, True])
def test_l2_products_operator(testing_state, sobolev):
    """Tests the operator mapping fields to a vector of L2 inner products."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        load_space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        load_space = lebesgue_load_space(model)

    # Use ocean and land projections as arbitrary weighting functions
    w1 = testing_state.ocean_projection(value=0.0)
    w2 = testing_state.land_projection(value=0.0)
    weights = [w1, w2]

    A = l2_products_operator(load_space, weights)

    # Measures for testing
    domain_measure = A.domain.heat_kernel_gaussian_measure(0.5 * b)
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-5,
        check_atol=1e-5,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


@pytest.mark.parametrize("sobolev", [False, True])
def test_averaging_operator(testing_state, sobolev):
    """Tests the spatial averaging operator."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        load_space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        load_space = lebesgue_load_space(model)

    w1 = testing_state.ocean_projection(value=0.0)
    w2 = testing_state.land_projection(value=0.0)
    weights = [w1, w2]

    A = averaging_operator(testing_state, load_space, weights)

    # Manual logical check: The average of a constant field over any region is the constant
    constant_grid = model.constant_grid(5.0)
    averages = A(constant_grid)
    assert np.allclose(averages, [5.0, 5.0])

    # Rigorous adjoint check
    domain_measure = A.domain.heat_kernel_gaussian_measure(0.5 * b)
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-5,
        check_atol=1e-5,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


@pytest.mark.parametrize("sobolev", [False, True])
def test_spatial_multiplication_operator(testing_state, sobolev):
    """Tests the spatial multiplication (masking) operator."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        load_space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        load_space = lebesgue_load_space(model)

    mask = testing_state.ocean_projection(value=0.0)

    A = spatial_multiplication_operator(load_space, mask)

    domain_measure = A.domain.heat_kernel_gaussian_measure(0.5 * b)
    codomain_measure = A.codomain.heat_kernel_gaussian_measure(0.5 * b)

    A.check(
        n_checks=2,
        check_rtol=1e-5,
        check_atol=1e-5,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )
