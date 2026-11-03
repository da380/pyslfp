"""
Tests for the horizontal displacement of the sea level solver: the
coefficient engine, the potential V of the horizontal displacement, the
tangential traction that carries a dual on V, the adjoints, and the response
of the generalised and non-linear solves.
"""

import numpy as np
import pytest
from pyshtools import SHCoeffs, SHGrid

from pyslfp.core import EarthState, SeaLevelEquation
from pyslfp.core.response import DISPLACEMENT, POTENTIAL_CHANGE
from pyslfp.core.response import RELATIVE_SEA_LEVEL_CHANGE
from pyslfp.linear_operators import FingerPrintOperator, centrifugal_potential_operator
from pyslfp.linear_operators.physics import (
    lebesgue_load_space,
    lebesgue_response_space,
)


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


def _random_coefficients(model, rng):
    """A random coefficient array in the model's layout, zero where no
    coefficient exists."""
    c = rng.normal(size=(2, model.lmax + 1, model.lmax + 1))
    for degree in range(model.lmax + 1):
        c[:, degree, degree + 1 :] = 0.0
    c[1, :, 0] = 0.0
    return c


def _shcoeffs(model, c):
    return SHCoeffs.from_array(
        c, normalization=model.normalization, csphase=model.csphase
    )


def _synthesise(model, c):
    return model.expand_coefficient(_shcoeffs(model, c))


def _pair(model, a, b):
    """The surface integral of the product of two grids."""
    return model.integrate(SHGrid.from_array(a.data * b.data, grid=model.grid))


def _relative(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


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


@pytest.mark.parametrize("rotation", [True, "axial", False])
def test_the_adjoint_of_the_horizontal_potential(setting, rotation):
    """The fingerprint operator's adjoint for a dual on the horizontal
    displacement alone, carried by a tangential traction, satisfies the
    adjoint identity."""
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


def test_the_tangential_load_follows_the_tangential_love_numbers(setting):
    """A tangential traction alone is answered through (h_v, l_v, k_v), on top
    of the response to the ocean load it drives: U = h sigma + h_v zeta^v,
    and likewise phi and V, with sigma = rho_w C xi."""
    state, sle, _, rng = setting
    model = state.model
    zeta_v = _synthesise(model, _random_coefficients(model, rng))
    solution = sle._solve_generalised(
        state,
        tangential_load=zeta_v,
        rotational_feedbacks=False,
        horizontal=True,
        rtol=1e-13,
    )
    rho_w = model.parameters.water_density
    sigma = _coefficients(
        model,
        SHGrid.from_array(
            rho_w * state.ocean_function.data * solution.relative_sea_level_change,
            grid=model.grid,
        ),
    )
    zeta = _coefficients(model, zeta_v)
    love = model.love_numbers
    for got, total, tangential in (
        (solution.displacement_lm, love.h, love.h_v),
        (solution.potential_lm, love.k, love.k_v),
        (solution.horizontal_lm, love.l, love.l_v),
    ):
        expected = total[None, :, None] * sigma + tangential[None, :, None] * zeta
        assert _relative(got, expected) < 1e-9


@pytest.mark.parametrize("rotation", [True, "axial", False])
def test_the_adjoint_of_the_engine_with_the_horizontal_part(setting, rotation):
    """<g, G p> = <G^T g, p> with V among the outputs and a ready V among the
    inputs: the horizontal dual carried through the transposed engine, in
    every rotation mode."""
    state, sle, _, rng = setting
    model = state.model
    load = _synthesise(model, _random_coefficients(model, rng))
    r_u, r_phi, r_v = (_random_coefficients(model, rng) for _ in range(3))
    j = float(rng.normal())
    solution = sle._solve_generalised(
        state,
        direct_load=load,
        displacement_response=_shcoeffs(model, r_u),
        gravitational_potential_response=_shcoeffs(model, r_phi),
        horizontal_response=_shcoeffs(model, r_v),
        inertia_moment_change=j,
        rotational_feedbacks=rotation,
        horizontal=True,
        rtol=1e-13,
    )
    xi = SHGrid.from_array(solution.relative_sea_level_change, grid=model.grid)
    u = _synthesise(model, solution.displacement_lm)
    phi = _synthesise(model, solution.potential_lm)
    g_xi, g_u, g_phi = (
        _synthesise(model, _random_coefficients(model, rng)) for _ in range(3)
    )
    g_omega = rng.normal(size=3)
    s_v = _random_coefficients(model, rng)
    forward = (
        _pair(model, g_xi, xi)
        + _pair(model, g_u, u)
        + _pair(model, g_phi, phi)
        + float(g_omega @ solution.angular_velocity_change)
        + float(np.sum(s_v * solution.horizontal_lm))
    )
    f_dual, su_dual, sphi_dual, j_dual = sle.adjoint_generalised_solve(
        state,
        sea_level_dual=g_xi,
        displacement_dual=g_u,
        potential_dual=g_phi,
        angular_velocity_dual=g_omega,
        horizontal_dual=_shcoeffs(model, s_v),
        rotational_feedbacks=rotation,
        rtol=1e-13,
    )
    backward = (
        _pair(model, f_dual, load)
        + float(np.sum(su_dual.coeffs * r_u))
        + float(np.sum(sphi_dual.coeffs * r_phi))
        + float(np.sum(s_v * r_v))
        + j_dual * j
    )
    assert abs(forward - backward) < 1e-9 * abs(forward), (forward, backward)


@pytest.mark.parametrize("rotation", [True, False])
def test_the_response_of_the_generalised_solve(setting, rotation):
    """`solve_response` with the ready responses gives the fields of
    `solve_generalised_equation`, and a ready V alone moves V alone."""
    state, sle, load, rng = setting
    model = state.model
    space = lebesgue_response_space(
        model, horizontal=True, rotation=rotation is not False
    )
    channels = {
        "displacement_response": _shcoeffs(model, _random_coefficients(model, rng)),
        "gravitational_potential_response": _shcoeffs(
            model, 1e-3 * _random_coefficients(model, rng)
        ),
        "inertia_moment_change": 1e-3,
    }
    response = sle.solve_response(
        state, load, space, rotational_feedbacks=rotation, rtol=1e-12, **channels
    )
    xi, u, phi, omega = sle.solve_generalised_equation(
        state, direct_load=load, rotational_feedbacks=rotation, rtol=1e-12, **channels
    )
    for name, grid in (
        (RELATIVE_SEA_LEVEL_CHANGE, xi),
        (DISPLACEMENT, u),
        (POTENTIAL_CHANGE, phi),
    ):
        assert _relative(response.field(name).data, grid.data) < 1e-10, name
    if rotation:
        np.testing.assert_allclose(
            response.angular_velocity_change, omega, rtol=1e-10, atol=0.0
        )

    r_v = _random_coefficients(model, rng)
    alone = sle.solve_response(
        state,
        None,
        space,
        horizontal_response=_shcoeffs(model, r_v),
        rotational_feedbacks=rotation,
    )
    expected = space.from_coefficients(
        np.zeros_like(r_v),
        np.zeros_like(r_v),
        np.zeros_like(r_v),
        horizontal=r_v,
        angular_velocity_change=np.zeros(3) if rotation else None,
    )
    np.testing.assert_array_equal(
        space.to_components(alone), space.to_components(expected)
    )


@pytest.mark.parametrize("rotation", [True, False])
def test_the_nonlinear_response(rotation):
    """`solve_nonlinear_response` gives the fields of
    `solve_nonlinear_equation`, its sea level change band-limited, and V
    follows the final state's load through the tangential Love numbers; a
    ready V joins V and moves nothing else."""
    state = EarthState.for_testing(16)
    model = state.model
    sle = SeaLevelEquation(model)
    space = lebesgue_response_space(
        model, horizontal=True, rotation=rotation is not False
    )
    change = -0.1 * state.ice_thickness
    final_state, response = sle.solve_nonlinear_response(
        state,
        space,
        ice_thickness_change=change,
        rotational_feedbacks=rotation,
        rtol=1e-12,
    )
    same_state, xi, u, phi, omega = sle.solve_nonlinear_equation(
        state, ice_thickness_change=change, rotational_feedbacks=rotation, rtol=1e-12
    )
    np.testing.assert_array_equal(final_state.sea_level.data, same_state.sea_level.data)
    for name, grid in (
        (RELATIVE_SEA_LEVEL_CHANGE, xi),
        (DISPLACEMENT, u),
        (POTENTIAL_CHANGE, phi),
    ):
        assert _relative(response.field(name).data, grid.data) < 1e-10, name

    parameters = model.parameters
    start = state
    total = parameters.ice_density * (
        (1.0 - final_state.ocean_function.data) * final_state.ice_thickness.data
        - (1.0 - start.ocean_function.data) * start.ice_thickness.data
    ) + parameters.water_density * (
        final_state.ocean_function.data * final_state.sea_level.data
        - start.ocean_function.data * start.sea_level.data
    )
    sigma = _coefficients(model, SHGrid.from_array(total, grid=model.grid))
    love = model.love_numbers
    expected = love.l[None, :, None] * sigma
    if rotation:
        psi = _coefficients(model, centrifugal_potential_operator(model)(omega))
        expected[:, 2, :2] += love.l_t[2] * psi[:, 2, :2]
    horizontal = space.member_space(DISPLACEMENT).from_coefficients(
        np.zeros_like(expected), expected
    )
    assert _relative(response.displacement.horizontal, horizontal.horizontal) < 1e-8

    r_v = 1e-3 * _random_coefficients(model, np.random.default_rng(5))
    _, shifted = sle.solve_nonlinear_response(
        state,
        space,
        ice_thickness_change=change,
        horizontal_response=_shcoeffs(model, r_v),
        rotational_feedbacks=rotation,
        rtol=1e-12,
    )
    difference = space.subtract(shifted, response)
    ready = space.member_space(DISPLACEMENT).from_coefficients(np.zeros_like(r_v), r_v)
    np.testing.assert_allclose(
        difference.displacement.horizontal, ready.horizontal, rtol=0.0, atol=1e-12
    )
    assert np.max(np.abs(difference.relative_sea_level_change)) == 0.0
