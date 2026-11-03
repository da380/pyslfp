"""
Test suite for the linear operator mathematical utilities.
"""

import pytest
import numpy as np

import pygeoinf as inf
from pygeoinf.testing import check_operator
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.core import EarthState
from pyslfp.core import (
    underlying_space,
    check_load_space,
    check_response_space,
    averaging_operator,
)
from pyslfp.core import (
    lebesgue_load_space,
    sobolev_load_space,
    lebesgue_response_space,
    sobolev_response_space,
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
    Tests that underlying_space resolves Sobolev spaces, and direct sums
    of them, to the Lebesgue spaces on the same grid.
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    # 1. Base Lebesgue space (should return an equal space)
    leb = lebesgue_load_space(model)
    assert underlying_space(leb) == leb

    # 2. Sobolev space (should return the Lebesgue space on its grid)
    sob = sobolev_load_space(model, 1.0, 0.1 * b)
    resolved_sob = underlying_space(sob)
    assert isinstance(resolved_sob, Lebesgue)
    assert not isinstance(resolved_sob, Sobolev)
    assert resolved_sob == leb

    # 3. Direct Sum space
    ds = inf.DirectSum([sob, sob])
    resolved_ds = underlying_space(ds)
    assert isinstance(resolved_ds, inf.DirectSum)
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
    """A response space is a GRDSpace over a sphere space, of Sobolev order
    above 1 when its fields are evaluated at points."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    check_response_space(lebesgue_response_space(model))
    check_response_space(
        sobolev_response_space(model, 2.0, 0.1 * b, horizontal=True),
        point_values=True,
    )
    with pytest.raises(ValueError, match="must be a GRDSpace"):
        check_response_space(lebesgue_load_space(model))
    with pytest.raises(ValueError, match="order > 1"):
        check_response_space(lebesgue_response_space(model), point_values=True)


# ==================================================================== #
#                  2. Utility Operators Adjoint Checks                 #
# ==================================================================== #


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
    domain_measure = A.domain.heat_measure(0.5 * b)

    check_operator(
        A,
        rng=np.random.default_rng(0),
        measure=domain_measure,
        codomain_measure=inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0),
        trials=2,
        rtol=1e-5,
        atol=1e-5,
    )
