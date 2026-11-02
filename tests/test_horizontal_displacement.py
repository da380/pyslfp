"""
Tests for the horizontal displacement of the sea level solver: the
coefficient engine, the potential V of the horizontal displacement, and the
adjoint of the map from a load to V that the fingerprint adjoint gives.
"""

import numpy as np
import pytest
from pyshtools import SHCoeffs, SHGrid

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
        SHGrid.from_array(solution.sea_level_change, grid=state.model.grid),
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


def test_the_adjoint_of_the_horizontal_potential(setting):
    """For a dual s on the coefficients of V, with d = l s and e the
    synthesis of l^t s at degree 2,

        <V(zeta), s> = (1 / b^2) [ int zeta d + <zeta, F*(rho_w C d, 0, 0, P* e)> ],

    with F the fingerprint operator and P the centrifugal potential operator,
    the coefficient pairing being the surface integral over b^2."""
    state, sle, load, rng = setting
    model = state.model
    b2 = model.parameters.mean_sea_floor_radius**2
    rho_w = model.parameters.water_density
    love = model.love_numbers

    shape = _coefficients(model, load).shape
    s = rng.standard_normal(shape)
    s[1, :, 0] = 0.0
    s[:, np.arange(shape[1])[:, None] < np.arange(shape[2])[None, :]] = 0.0
    s[:, 0, 0] = 0.0

    forward = sle._solve_generalised(state, direct_load=load, horizontal=True)
    lhs = np.sum(forward.horizontal_lm * s)

    def field(coeffs):
        clm = SHCoeffs.from_array(
            coeffs, normalization=model.normalization, csphase=model.csphase
        )
        return model.expand_coefficient(clm)

    d = field(love.l[None, :, None] * s)
    block = np.zeros(shape)
    block[:, 2, :2] = love.l_t[2] * s[:, 2, :2]
    e = field(block)
    fingerprint = FingerPrintOperator(state, rtol=1e-12)
    zero = model.zero_grid()
    rotation = centrifugal_potential_operator(model).adjoint(e)
    pulled = fingerprint.adjoint(
        (rho_w * state.ocean_function * d, zero, zero, rotation)
    )
    rhs = (model.integrate(load * d) + model.integrate(load * pulled)) / b2
    assert abs(lhs - rhs) <= 1e-7 * max(abs(lhs), abs(rhs))
