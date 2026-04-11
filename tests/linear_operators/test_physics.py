"""
Test suite for the pygeoinf physical operator wrappers.
"""

import pytest
import numpy as np
from pyshtools import SHGrid

from pygeoinf import LinearOperator, HilbertSpaceDirectSum, EuclideanSpace
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.core import EarthModel
from pyslfp.state import EarthState
from pyslfp.physics import SeaLevelEquation
from pyslfp.ice_ng import IceNG, IceModel

from pyslfp.linear_operators.physics import (
    lebesgue_load_space,
    lebesgue_response_space,
    sobolev_load_space,
    sobolev_response_space,
    get_lebesgue_linear_operator,
    get_sobolev_linear_operator,
    adjoint_angular_momentum_from_potential,
)

# ==================================================================== #
#                            Fixtures                                  #
# ==================================================================== #


@pytest.fixture(scope="module")
def operator_setup():
    """Provides a basic EarthState and Solver for operator testing."""
    model = EarthModel(lmax=32)  # Lower lmax for extremely fast operator tests
    ice_ng = IceNG(version=IceModel.ICE7G)
    ice_thickness, sea_level = ice_ng.get_ice_thickness_and_sea_level(0.0, 32)

    ice_thickness = ice_thickness / model.parameters.length_scale
    sea_level = sea_level / model.parameters.length_scale

    state = EarthState(ice_thickness, sea_level, model)
    solver = SeaLevelEquation(model)

    return state, solver


def random_grid(state: EarthState) -> SHGrid:
    """Helper to generate a random SHGrid matching the state's properties."""
    grid = SHGrid.from_zeros(
        state.lmax, grid=state.grid, sampling=state.sampling, extend=state.extend
    )
    grid.data += np.random.uniform(-1, 1, size=grid.data.shape)
    return grid


# ==================================================================== #
#                            Space Tests                               #
# ==================================================================== #


def test_lebesgue_spaces(operator_setup):
    """Tests the generation of L2 load and response spaces."""
    state, _ = operator_setup

    domain = lebesgue_load_space(state)
    assert isinstance(domain, Lebesgue)

    codomain = lebesgue_response_space(state)
    assert isinstance(codomain, HilbertSpaceDirectSum)
    assert codomain.number_of_subspaces == 4
    assert isinstance(codomain.subspace(0), Lebesgue)
    assert isinstance(codomain.subspace(3), EuclideanSpace)


def test_sobolev_spaces(operator_setup):
    """Tests the generation of Sobolev load and response spaces."""
    state, _ = operator_setup
    order, scale = 1.0, 1000.0

    domain = sobolev_load_space(state, order, scale)
    assert isinstance(domain, Sobolev)

    codomain = sobolev_response_space(state, order, scale)
    assert isinstance(codomain, HilbertSpaceDirectSum)
    assert codomain.number_of_subspaces == 4
    assert isinstance(codomain.subspace(0), Sobolev)

    # Elliptic regularity: response spaces should have order + 1
    assert codomain.subspace(0).order == order + 1


# ==================================================================== #
#                         Adjoint Helpers Tests                        #
# ==================================================================== #


def test_adjoint_angular_momentum_helper(operator_setup):
    """Tests the angular momentum potential extraction."""
    state, solver = operator_setup
    gpc_load = random_grid(state)

    # Should yield a 2-element numpy array
    adj_avc = adjoint_angular_momentum_from_potential(gpc_load, solver)
    assert isinstance(adj_avc, np.ndarray)
    assert adj_avc.shape == (2,)


# ==================================================================== #
#                         Operator Execution Tests                     #
# ==================================================================== #


def test_lebesgue_linear_operator_forward(operator_setup):
    """Smoke test for the forward execution of the Lebesgue operator."""
    state, solver = operator_setup
    operator = get_lebesgue_linear_operator(solver, state, max_iterations=2)

    assert isinstance(operator, LinearOperator)

    # 1. Provide a single domain element (load)
    test_load = random_grid(state)

    # 2. Map it forward
    response = operator(test_load)

    # 3. Validate Codomain outputs
    assert isinstance(response, list)
    assert len(response) == 4
    assert isinstance(response[0], SHGrid)  # SLC
    assert isinstance(response[1], SHGrid)  # Disp
    assert isinstance(response[2], SHGrid)  # Grav Potential
    assert isinstance(response[3], np.ndarray)  # AVC
    assert response[3].shape == (2,)


def test_lebesgue_linear_operator_adjoint(operator_setup):
    """Smoke test for the adjoint execution of the Lebesgue operator."""
    state, solver = operator_setup
    operator = get_lebesgue_linear_operator(solver, state, max_iterations=2)

    # 1. Create a dummy codomain element (response list)
    dummy_response = [
        random_grid(state),  # Adjoint SLC
        random_grid(state),  # Adjoint Disp
        random_grid(state),  # Adjoint GPC
        np.array([1e-8, 1e-8]),  # Adjoint AVC
    ]

    # 2. Map it backward
    adjoint_load = operator.adjoint(dummy_response)

    # 3. Validate Domain output
    assert isinstance(adjoint_load, SHGrid)
    assert adjoint_load.data.shape == state.sea_level.data.shape


def test_sobolev_linear_operator_instantiation(operator_setup):
    """Ensures the Sobolev wrapper correctly inherits the formal adjoint."""
    state, solver = operator_setup
    sobolev_op = get_sobolev_linear_operator(solver, state, order=1.0, scale=1000.0)

    assert isinstance(sobolev_op, LinearOperator)
    assert isinstance(sobolev_op.domain, Sobolev)
    assert isinstance(sobolev_op.codomain, HilbertSpaceDirectSum)
