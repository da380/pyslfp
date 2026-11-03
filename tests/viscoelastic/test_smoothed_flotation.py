"""
Tests for the smoothed flotation rule: the smoothstep and the fields the
derivatives need, the smoothed run tending to the sharp one, mass
conservation with fractional ocean cells, and the derivative of a migrating
run along a direction that crosses flotation, which the smoothed rule makes
exact where the sharp rule's held ocean functions cannot.
"""

from dataclasses import replace

import numpy as np
import pytest
from pyshtools import SHGrid

from pyslfp.core import EarthState
from pyslfp.core.state import flotation_fields, smoothstep
from pyslfp.viscoelastic import Rheology, SeaLevelEvolution, ViscoelasticOperator

from .body import TAU
from .test_evolution import _maxwell_operator
from .test_evolution_operator import _ThickCap
from .test_tangent_linear import _pair, _sea_level, _smooth

DATES = np.linspace(10.0, 0.0, 6)


def _marine(earth, width):
    ice_model = _ThickCap(length_scale=earth.parameters.length_scale)
    ice = ice_model.get_ice_thickness(
        _ThickCap.START, earth.lmax, grid=earth.grid_name, extend=earth.extend
    )
    sea = earth.constant_grid(3000.0 / earth.parameters.length_scale)
    return ice_model, EarthState(ice, sea, earth, flotation_width=width)


def _metres(earth, value):
    return value / earth.parameters.length_scale


# ==================================================================== #
#                             The rule                                 #
# ==================================================================== #


def test_the_smoothstep():
    x = np.linspace(-1.5, 1.5, 61)
    sigma, first, second = smoothstep(x)
    assert np.all(sigma[x <= -1.0] == 0.0) and np.all(sigma[x >= 1.0] == 1.0)
    assert smoothstep(np.array([0.0]))[0] == pytest.approx(0.5)
    step = 1e-5
    inside = (x > -0.99) & (x < 0.99)
    np.testing.assert_allclose(
        first[inside],
        (smoothstep(x + step)[0] - smoothstep(x - step)[0])[inside] / (2 * step),
        rtol=1e-6,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        second[inside],
        (smoothstep(x + step)[1] - smoothstep(x - step)[1])[inside] / (2 * step),
        rtol=1e-6,
        atol=1e-8,
    )
    # twice continuously differentiable: both derivatives vanish at the ends
    ends = smoothstep(np.array([-1.0 + 1e-9, 1.0 - 1e-9]))
    assert np.all(np.abs(ends[1]) < 1e-15) and np.all(np.abs(ends[2]) < 1e-7)


def test_the_fields_are_the_derivatives_of_the_margin_times_the_ocean_function():
    """With F(S) = (rho_w S - rho_i I) C(S), F' = rho_w (C + Lambda) and
    F'' = rho_w K, the fields the first and second derivatives of the loads
    carry; under the sharp rule both extra fields vanish."""
    rng = np.random.default_rng(1)
    rho_w, rho_i, width = 1.0, 0.9, 50.0
    ice = rng.uniform(0.0, 4000.0, size=200)
    sea = rho_i / rho_w * ice + rng.uniform(-80.0, 80.0, size=200)

    def F(s):
        ocean, _, _ = flotation_fields(s, ice, rho_w, rho_i, width=width)
        return (rho_w * s - rho_i * ice) * ocean

    ocean, weight, curvature = flotation_fields(sea, ice, rho_w, rho_i, width=width)
    strip = np.abs(sea - rho_i / rho_w * ice) < width
    assert np.any(strip) and np.all((ocean[strip] > 0.0) & (ocean[strip] < 1.0))
    assert np.all(weight[~strip] == 0.0) and np.any(weight[strip] != 0.0)
    step = 1e-3
    np.testing.assert_allclose(
        rho_w * (ocean + weight),
        (F(sea + step) - F(sea - step)) / (2 * step),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        rho_w * curvature,
        (F(sea + step) - 2 * F(sea) + F(sea - step)) / step**2,
        atol=1e-5,
    )
    sharp = flotation_fields(sea, ice, rho_w, rho_i)
    np.testing.assert_array_equal(
        sharp[0], np.where(sea * rho_w - rho_i * ice > 0, 1.0, 0.0)
    )
    assert np.all(sharp[1] == 0.0) and np.all(sharp[2] == 0.0)
    masked = flotation_fields(sea, ice, rho_w, rho_i, width=width, land_mask=strip)
    assert np.all(masked[0][strip] == 0.0) and np.all(masked[1][strip] == 0.0)


def test_a_state_carries_the_rule(setting):
    """A state whose sea level sits within the width of floating its ice
    over part of the grid."""
    _, earth, _, _, _ = setting
    width = _metres(earth, 500.0)
    rng = np.random.default_rng(3)
    ice = earth.constant_grid(_metres(earth, 2000.0))
    ratio = earth.parameters.ice_density / earth.parameters.water_density
    sea = SHGrid.from_array(
        ratio * ice.data + rng.uniform(-2.0, 2.0, size=ice.data.shape) * width,
        grid=earth.grid,
    )
    state = EarthState(ice, sea, earth, flotation_width=width)
    assert state.flotation_width == width
    C = state.ocean_function.data
    assert np.any((C > 0.0) & (C < 1.0))
    extra = state.effective_ocean_function.data - C
    assert np.any(extra != 0.0) and np.all(extra[(C == 0.0) | (C == 1.0)] == 0.0)
    assert np.any(state.flotation_curvature.data != 0.0)
    sharp = EarthState(ice, sea, earth)
    assert sharp.flotation_width is None
    np.testing.assert_array_equal(
        sharp.effective_ocean_function.data, sharp.ocean_function.data
    )
    assert np.all(sharp.flotation_curvature.data == 0.0)
    with pytest.raises(ValueError, match="positive"):
        EarthState(ice, sea, earth, flotation_width=-1.0)


# ==================================================================== #
#                              The run                                 #
# ==================================================================== #


def test_the_smoothed_run_tends_to_the_sharp_one(setting):
    material, earth, _, _, _ = setting
    operator = _maxwell_operator(material)

    def final(width):
        ice_model, state = _marine(earth, width)
        snapshots, _ = SeaLevelEvolution(operator, state).run_ice_history(
            DATES, ice_model, shorelines="migrating"
        )
        assert snapshots[-1].effective_ocean_function is not None
        return _sea_level(snapshots[-1]).data

    sharp = final(None)
    # the test grid is coarse, so the widths are of the ocean's depth
    gaps = [
        np.abs(final(_metres(earth, w)) - sharp).max() for w in (4000.0, 2000.0, 1000.0)
    ]
    assert gaps[0] > gaps[1] > gaps[2] > 0.0
    assert (
        np.abs(final(_metres(earth, 1.0e-6)) - sharp).max() < 1e-9 * np.abs(sharp).max()
    )


def test_mass_is_conserved_with_fractional_cells(setting):
    material, earth, _, _, _ = setting
    ice_model, state = _marine(earth, _metres(earth, 3000.0))
    snapshots, _ = SeaLevelEvolution(
        _maxwell_operator(material), state
    ).run_ice_history(DATES, ice_model, shorelines="migrating")
    # cells are partly ocean while the cap melts; by the end it is gone
    assert any(
        np.any((snap.ocean_function.data > 0.0) & (snap.ocean_function.data < 1.0))
        for snap in snapshots
    )
    C0 = state.ocean_function.data
    Cf = snapshots[-1].ocean_function.data
    parameters = earth.parameters
    water = parameters.water_density * earth.integrate(
        SHGrid.from_array(
            Cf * (state.sea_level.data + _sea_level(snapshots[-1]).data)
            - C0 * state.sea_level.data,
            grid=earth.grid,
        )
    )
    grounded = parameters.ice_density * earth.integrate(
        SHGrid.from_array(-(1.0 - C0) * state.ice_thickness.data, grid=earth.grid)
    )
    assert abs(water + grounded) < 1e-8 * abs(grounded)


# ==================================================================== #
#                          The derivative                              #
# ==================================================================== #


def test_the_derivative_crosses_flotation(setting):
    """Along a uniform rise of the initial sea level, which moves every
    shoreline, the tangent-linear run of the smoothed rule is the derivative
    of the run: central differences converge to it at second order, where
    the held ocean functions of the sharp rule give a different number; and
    the discrete pair are transposes with the effective ocean functions."""
    material, earth, _, _, _ = setting
    operator = _maxwell_operator(material)
    width = _metres(earth, 3000.0)
    ice_model, state = _marine(earth, width)
    evolution = SeaLevelEvolution(operator, state)
    snapshots, _ = evolution.run_ice_history(
        DATES, ice_model, shorelines="migrating", record=True, rtol=1e-12
    )
    assert any(
        np.any(snap.effective_ocean_function.data != snap.ocean_function.data)
        for snap in snapshots
    )
    g = _smooth(earth, np.random.default_rng(2))
    rise = earth.constant_grid(_metres(earth, 100.0))
    tangent, _ = evolution.tangent_linear_ice_history(
        DATES, snapshots, initial_sea_level=rise, rtol=1e-12
    )
    predicted = _pair(earth, g, _sea_level(tangent[-1]))

    def value(eps):
        moved = EarthState(
            state.ice_thickness,
            SHGrid.from_array(state.sea_level.data + eps * rise.data, grid=earth.grid),
            earth,
            flotation_width=width,
        )
        snaps, _ = evolution._with_state(moved).run_ice_history(
            DATES, ice_model, shorelines="migrating", rtol=1e-12
        )
        return _pair(earth, g, _sea_level(snaps[-1]))

    fd = {eps: (value(eps) - value(-eps)) / (2.0 * eps) for eps in (0.02, 0.01)}
    resid = {eps: abs(f - predicted) for eps, f in fd.items()}
    assert 3.0 < resid[0.02] / resid[0.01] < 5.0
    richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0
    assert abs(richardson - predicted) < 1e-6 * abs(predicted)
    # the held ocean functions alone miss the strip's contribution
    held = evolution.tangent_linear_ice_history(
        DATES,
        [
            replace(snap, effective_ocean_function=snap.ocean_function)
            for snap in snapshots
        ],
        initial_sea_level=rise,
        rtol=1e-12,
    )[0]
    assert abs(_pair(earth, g, _sea_level(held[-1])) - predicted) > 1e-3 * abs(
        predicted
    )
    # the discrete pair are transposes
    kernels = evolution.adjoint_ice_history(
        DATES,
        {0.0: g},
        ocean_functions=[snap.effective_ocean_function for snap in snapshots],
        internal_states=snapshots,
        rtol=1e-12,
    )
    assert abs(_pair(earth, kernels.initial_sea_level, rise) - predicted) < 1e-9 * abs(
        predicted
    )


# ==================================================================== #
#                       The second derivative                          #
# ==================================================================== #


def _pairing(earth, kernels, direction):
    total = _pair(earth, kernels.initial_sea_level, direction["sea_level"])
    if direction["ice"] is not None:
        total += sum(
            _pair(earth, kernels.thickness[n], direction["ice"][n])
            for n in range(len(direction["ice"]))
        )
    return total


def test_the_second_derivative_crosses_flotation(setting):
    """With the smoothed rule the Hessian along a uniform rise, which moves
    every shoreline, matches central differences of the gradient; it is
    symmetric; and the second-order adjoint is the adjoint of the
    second-order run, the shorelines' terms included on both sides."""
    material, earth, _, _, _ = setting
    # a width of the ocean's depth would put the open ocean at the strip's
    # edge, where the rule is only twice differentiable
    width = _metres(earth, 2000.0)
    ice_model, state = _marine(earth, width)
    evolution = SeaLevelEvolution(_maxwell_operator(material), state)
    snapshots, _ = evolution.run_ice_history(
        DATES, ice_model, shorelines="migrating", record=True, rtol=1e-12
    )
    effective = [snap.effective_ocean_function for snap in snapshots]
    curvatures = [snap.flotation_curvature for snap in snapshots]
    rng = np.random.default_rng(4)
    g = _smooth(earth, rng)
    u = {"ice": None, "sea_level": earth.constant_grid(_metres(earth, 100.0))}
    w = {
        "ice": [_smooth(earth, rng, _metres(earth, 20.0)) for _ in DATES],
        "sea_level": _smooth(earth, rng, _metres(earth, 20.0)),
    }
    tangents = {}
    for name, d in (("u", u), ("w", w)):
        tangents[name], _ = evolution.tangent_linear_ice_history(
            DATES,
            snapshots,
            ice_thickness=d["ice"],
            initial_sea_level=d["sea_level"],
            record=True,
            rtol=1e-12,
        )
    derivatives = {}
    for name, d in (("u", u), ("w", w)):
        _, derivatives[name] = evolution.second_order_adjoint_ice_history(
            DATES,
            {0.0: g},
            tangents[name],
            ocean_functions=effective,
            internal_states=snapshots,
            flotation_curvatures=curvatures,
            ice_thickness=d["ice"],
            initial_sea_level=d["sea_level"],
            rtol=1e-12,
        )
    hu_w = _pairing(earth, derivatives["u"], w)
    hw_u = _pairing(earth, derivatives["w"], u)
    assert abs(hu_w - hw_u) < 1e-9 * abs(hu_w)

    def gradient_at(eps):
        moved = EarthState(
            state.ice_thickness,
            SHGrid.from_array(
                state.sea_level.data + eps * u["sea_level"].data, grid=earth.grid
            ),
            earth,
            flotation_width=width,
        )
        evo = evolution._with_state(moved)
        snaps, _ = evo.run_ice_history(
            DATES, ice_model, shorelines="migrating", record=True, rtol=1e-12
        )
        kernels = evo.adjoint_ice_history(
            DATES,
            {0.0: g},
            ocean_functions=[s.effective_ocean_function for s in snaps],
            internal_states=snaps,
            rtol=1e-12,
        )
        return _pairing(earth, kernels, w)

    fd = {
        eps: (gradient_at(eps) - gradient_at(-eps)) / (2.0 * eps)
        for eps in (0.02, 0.01)
    }
    resid = {eps: abs(f - hu_w) for eps, f in fd.items()}
    assert 3.0 < resid[0.02] / resid[0.01] < 5.0
    richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0
    assert abs(richardson - hu_w) < 1e-6 * abs(hu_w)
    # the adjoint of the second-order run
    second, _ = evolution.second_order_tangent_linear_ice_history(
        DATES,
        snapshots,
        tangents["u"],
        tangents["w"],
        ice_u=u["ice"],
        ice_v=w["ice"],
        sea_level_u=u["sea_level"],
        sea_level_v=w["sea_level"],
        rtol=1e-12,
    )
    forward = _pair(earth, g, _sea_level(second[-1]))
    assert abs(forward - hu_w) < 1e-9 * abs(forward)


def test_the_continuous_method_carries_the_shoreline_terms(setting):
    """What the shorelines add to the second-order adjoint, the difference
    between the sweeps with and without the curvature, agrees between the
    methods to the order of the scheme: on a slowly relaxing body the gap
    falls by about four when the dates are halved."""
    material, earth, _, _, _ = setting
    width = _metres(earth, 2000.0)
    ice_model, state = _marine(earth, width)
    operator = ViscoelasticOperator(
        Rheology.from_relaxation_time(material, 10.0 * TAU), earth.lmax
    )
    evolution = SeaLevelEvolution(operator, state)
    rng = np.random.default_rng(5)
    g = _smooth(earth, rng)
    rise = earth.constant_grid(_metres(earth, 100.0))
    w_field = _smooth(earth, rng, _metres(earth, 20.0))
    gaps = []
    for npts in (5, 9):
        dates = np.linspace(10.0, 0.0, npts)
        snapshots, _ = evolution.run_ice_history(
            dates, ice_model, shorelines="migrating", record=True, rtol=1e-12
        )
        effective = [snap.effective_ocean_function for snap in snapshots]
        curvatures = [snap.flotation_curvature for snap in snapshots]
        added = {}
        for method in ("discrete", "continuous"):
            tangent, _ = evolution.tangent_linear_ice_history(
                dates,
                snapshots,
                initial_sea_level=rise,
                method=method,
                record=True,
                rtol=1e-12,
            )
            values = []
            for curvature in (curvatures, None):
                _, derivative = evolution.second_order_adjoint_ice_history(
                    dates,
                    {0.0: g},
                    tangent,
                    ocean_functions=effective,
                    internal_states=snapshots,
                    flotation_curvatures=curvature,
                    initial_sea_level=rise,
                    rtol=1e-12,
                    method=method,
                )
                values.append(_pair(earth, derivative.initial_sea_level, w_field))
            added[method] = values[0] - values[1]
        assert added["discrete"] != 0.0
        gaps.append(
            abs(added["continuous"] - added["discrete"]) / abs(added["discrete"])
        )
    assert gaps[0] < 0.2 and gaps[0] / gaps[1] > 2.5
