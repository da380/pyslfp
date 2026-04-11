"""
Test suite for the pygeoinf physical operator wrappers.

Validates the instantiation of mathematical spaces, operator definitions,
and rigorously tests the adjoint identity using Gaussian measures.
"""

import pytest
import numpy as np
from pyshtools import SHGrid

import pygeoinf as inf
from pygeoinf import HilbertSpaceDirectSum, EuclideanSpace
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
    # We use lmax=64 to balance sufficient harmonic complexity with the speed
    # required for iterative randomized testing in the .check() method.
    model = EarthModel(64)

    ice_ng = IceNG(version=IceModel.ICE7G, length_scale=model.parameters.length_scale)
    ice_thickness, sea_level = ice_ng.get_ice_thickness_and_sea_level(0.0, 64)

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
#                     Rigorous Adjoint Identity Tests                  #
# ==================================================================== #


@pytest.mark.parametrize(
    "sobolev, order, scale_factor, rotational_feedbacks",
    [
        # 1. Lebesgue, no rotational feedbacks
        (False, None, None, False),
        # 2. Lebesgue, with rotational feedbacks
        (False, None, None, True),
        # 3. Sobolev (Order 1), no rotational feedbacks
        (True, 1.0, 0.1, False),
        # 4. Sobolev (Order 2), with rotational feedbacks
        (True, 2.0, 0.2, True),
    ],
    ids=[
        "Lebesgue-NoRotation",
        "Lebesgue-Rotation",
        "Sobolev-O1-NoRotation",
        "Sobolev-O2-Rotation",
    ],
)
def test_linear_operator_adjoint_identity(
    operator_setup,
    sobolev: bool,
    order: float,
    scale_factor: float,
    rotational_feedbacks: bool,
):
    """
    Tests the mathematical adjoint identity <Au, v> == <u, A*v> for the LinearOperator
    using pygeoinf's built-in .check() method with spatially regular Gaussian measures.
    """
    state, solver = operator_setup

    rtol = 1e-9
    check_rtol = 1e-4
    check_atol = 1e-4
    radius = solver.model.parameters.mean_sea_floor_radius

    # 1. Construct the target Linear Operator
    if sobolev:
        A = get_sobolev_linear_operator(
            solver,
            state,
            order,
            scale_factor * radius,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )
    else:
        A = get_lebesgue_linear_operator(
            solver,
            state,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )

    # 2. Domain measure (Load space)
    smoothness_scale = 0.1 * radius
    domain_measure = A.domain.heat_kernel_gaussian_measure(smoothness_scale)

    # 3. Codomain measure (Response space)
    field_space = A.codomain.subspace(0)
    field_measure = field_space.heat_kernel_gaussian_measure(smoothness_scale)

    euclidean_space = A.codomain.subspace(3)
    angular_momentum_std = solver.model.parameters.rotation_frequency * radius**4

    euclidean_measure = inf.GaussianMeasure.from_standard_deviation(
        euclidean_space, angular_momentum_std
    )

    codomain_measure = inf.GaussianMeasure.from_direct_sum(
        [field_measure, field_measure, field_measure, euclidean_measure]
    )

    # 4. Run the comprehensive self-checks
    A.check(
        n_checks=3,
        check_rtol=check_rtol,
        check_atol=check_atol,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )
