"""
Tests for the horizontal displacement of the sea level solver: the
coefficient engine, the potential V of the horizontal displacement, and the
adjoint of the map from a load to V that the fingerprint adjoint gives.
"""

import numpy as np
import pytest
from pyshtools import SHGrid

from pyslfp.core import EarthState, SeaLevelEquation
from pyslfp.linear_operators import FingerPrintOperator, centrifugal_potential_operator
from pyslfp.linear_operators.physics import lebesgue_load_space


@pytest.fixture(scope="module")
def setting():
    state = EarthState.for_testing(16)
    model = state.model
    sle = SeaLevelEquation(model)
    b = model.parameters.mean_sea_floor_radius
    space = lebesgue_load_space(model)
    rng = np.random.default_rng(3)
    load = space.heat_measure(0.3 * b).sample(rng=rng)
    return state, sle, load, rng


def _coefficients(model, grid):
    return model.expand_field(grid).coeffs


def test_sea_level_coefficients_are_those_of_the_grid(setting):
    state, sle, load, _ = setting
    solution = sle._solve_generalised(state, direct_load=load, horizontal=True)
    expected = _coefficients(
        state.model,
        SHGrid.from_array(solution.relative_sea_level_change, grid=state.model.grid),
    )
    np.testing.assert_allclose(
        solution.sea_level_lm,
        expected,
        rtol=0.0,
        atol=1e-12 * np.max(np.abs(expected)),
    )


def test_horizontal_potential_follows_the_tangential_love_numbers(setting):
    """V = l sigma + l^t psi at degree 2, with sigma = zeta + rho_w C xi the
    total load and psi the centrifugal potential, both rebuilt from the
    solver's grid outputs; and V has no degree 0."""
    state, sle, load, _ = setting
    model = state.model
    solution = sle._solve_generalised(state, direct_load=load, horizontal=True)
    sea_level, _, _, omega = sle.solve_generalised_equation(state, direct_load=load)
    rho_w = model.parameters.water_density
    total = _coefficients(model, load + rho_w * state.ocean_function * sea_level)
    psi = _coefficients(model, centrifugal_potential_operator(model)(omega))
    love = model.love_numbers
    expected = love.l[None, :, None] * total
    expected[:, 2, :2] += love.l_t[2] * psi[:, 2, :2]
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(
        solution.horizontal_lm, expected, rtol=0.0, atol=1e-7 * scale
    )
    assert np.all(solution.horizontal_lm[:, 0, 0] == 0.0)
    assert sle._solve_generalised(state, direct_load=load).horizontal_lm is None


@pytest.mark.parametrize("rotation", [True, False])
def test_the_adjoint_of_the_horizontal_potential(setting, rotation):
    """The fingerprint operator's adjoint for a dual on the horizontal
    displacement alone, carried by the extra sea level, angular velocity and
    direct terms, satisfies the adjoint identity."""
    state, _, load, rng = setting
    fingerprint = FingerPrintOperator(
        state, horizontal=True, rotational_feedbacks=rotation, rtol=1e-12
    )
    space = fingerprint.codomain
    dual = space.zero()
    dual.displacement.horizontal = rng.standard_normal(
        dual.displacement.horizontal.size
    )
    lhs = space.inner_product(fingerprint(load), dual)
    rhs = fingerprint.domain.inner_product(load, fingerprint.adjoint(dual))
    assert abs(lhs - rhs) <= 1e-7 * max(abs(lhs), abs(rhs))
