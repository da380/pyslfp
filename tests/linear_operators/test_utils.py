"""
Test suite for the general linear operator utilities.

Validates Hilbert space resolution, structural checks, and the adjoint
identities of common spatial operators (averaging, multiplication, filtering).
"""

import pytest

import pygeoinf as inf
from pygeoinf import HilbertSpaceDirectSum, EuclideanSpace
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.core import EarthModel
from pyslfp.state import EarthState
from pyslfp.ice_ng import IceNG, IceModel

from pyslfp.linear_operators.utils import (
    underlying_space,
    check_load_space,
    check_response_space,
    averaging_operator,
    spatial_multiplication_operator,
    remove_ocean_average_operator,
)


@pytest.fixture(scope="module")
def utils_setup():
    """Provides the triad of (State, Sobolev Space, measure) for testing."""
    model = EarthModel(32)
    ice_ng = IceNG(version=IceModel.ICE7G, length_scale=model.parameters.length_scale)
    ice_thickness, sea_level = ice_ng.get_ice_thickness_and_sea_level(0.0, 32)

    state = EarthState(ice_thickness, sea_level, model)

    # Standard Sobolev setup for testing operator 'lifting'
    order, scale = 2.0, 0.25 * model.parameters.mean_sea_floor_radius
    load_space = Sobolev(state.lmax, order, scale, grid=state.grid_type)
    measure = load_space.heat_kernel_gaussian_measure(scale)

    return state, load_space, measure


# ==================================================================== #
#                       Structural Utility Tests                       #
# ==================================================================== #


def test_underlying_space_resolution(utils_setup):
    """Verifies recursive resolution of unweighted L2 spaces."""
    _, load_space, _ = utils_setup

    # 1. Sobolev -> Lebesgue
    assert isinstance(underlying_space(load_space), Lebesgue)

    # 2. Composite -> Composite L2
    direct_sum = HilbertSpaceDirectSum([load_space, load_space, EuclideanSpace(2)])
    resolved = underlying_space(direct_sum)

    assert isinstance(resolved, HilbertSpaceDirectSum)
    assert isinstance(resolved.subspace(0), Lebesgue)
    assert isinstance(resolved.subspace(2), EuclideanSpace)


def test_check_load_space_validation(utils_setup):
    """Tests validation logic for load spaces and point-evaluation requirements."""
    state, _, _ = utils_setup

    # Valid Lebesgue
    l_space = Lebesgue(state.lmax)
    assert check_load_space(l_space) is True

    # Invalid point evaluation (order <= 1)
    s_space_low = Sobolev(state.lmax, 1.0, 1.0)
    with pytest.raises(ValueError, match="order > 1"):
        check_load_space(s_space_low, point_values=True)

    # Valid point evaluation (order > 1)
    s_space_high = Sobolev(state.lmax, 2.0, 1.0)
    assert check_load_space(s_space_high, point_values=True) is True


def test_check_response_space_validation(utils_setup):
    """Tests validation of the 4-component physical response structure."""
    state, load_space, _ = utils_setup

    # 1. Valid structure
    valid_sum = HilbertSpaceDirectSum(
        [load_space, load_space, load_space, EuclideanSpace(2)]
    )
    check_response_space(valid_sum)

    # 2. Mismatched field spaces
    mismatched_sum = HilbertSpaceDirectSum(
        [load_space, Lebesgue(state.lmax), load_space, EuclideanSpace(2)]
    )
    with pytest.raises(ValueError, match="Subspaces 1 and 2 must match"):
        check_response_space(mismatched_sum)


# ==================================================================== #
#                       Functional Operator Tests                      #
# ==================================================================== #


def test_averaging_operator_adjoint_identity(utils_setup):
    """Rigorously tests the L2-averaging adjoint <Au, v> == <u, A*v>."""
    state, load_space, measure = utils_setup

    # Create 5 random weighting functions

    weights = [measure.sample() for _ in range(5)]
    A = averaging_operator(load_space, weights)
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    # Internal pygeoinf assertion
    A.check(n_checks=3, domain_measure=measure, codomain_measure=codomain_measure)


def test_spatial_multiplication_operator_adjoint_identity(utils_setup):
    """Verifies that spatial multiplication is formally self-adjoint."""
    state, load_space, measure = utils_setup

    # Random projection mask
    mask = measure.sample()

    A = spatial_multiplication_operator(mask, load_space)

    A.check(n_checks=3, domain_measure=measure, codomain_measure=measure)


def test_remove_ocean_average_operator_adjoint_identity(utils_setup):
    """Tests the adjoint identity of the ocean-average adjustment operator."""
    state, load_space, measure = utils_setup

    # State provides the integrate() and ocean_function logic
    A = remove_ocean_average_operator(state, load_space)

    A.check(n_checks=3, domain_measure=measure, codomain_measure=measure)
