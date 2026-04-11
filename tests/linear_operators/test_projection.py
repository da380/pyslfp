"""
Test suite for the general linear operator utilities.

Validates Hilbert space resolution, structural checks, and the adjoint
identities of common spatial operators (averaging, multiplication, filtering).
"""

import pytest

from pygeoinf.symmetric_space.sphere import Sobolev

from pyslfp.core import EarthModel
from pyslfp.state import EarthState
from pyslfp.ice_ng import IceNG, IceModel

from pyslfp.linear_operators.projection import (
    ice_projection_operator,
    ocean_projection_operator,
    land_projection_operator,
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


def test_ice_projection_operator(utils_setup):
    state, load_space, measure = utils_setup
    A = ice_projection_operator(state, load_space)
    A.check(n_checks=3, domain_measure=measure, codomain_measure=measure)


def test_ocean_projection_operator(utils_setup):
    state, load_space, measure = utils_setup
    A = ocean_projection_operator(state, load_space)
    A.check(n_checks=3, domain_measure=measure, codomain_measure=measure)


def test_land_projection_operator(utils_setup):
    state, load_space, measure = utils_setup
    A = land_projection_operator(state, load_space)
    A.check(n_checks=3, domain_measure=measure, codomain_measure=measure)
