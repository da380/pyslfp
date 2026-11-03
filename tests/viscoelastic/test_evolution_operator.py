"""
Tests for the evolution operator: its value is the run, its derivative passes
the Taylor test and the adjoint check, the continuous pair converge to
transposes as the dates are refined, and fixing arguments composes with the
chain rule, giving the linear ice history operator in the frozen case.
"""

import numpy as np
import pytest
from pygeoinf import DirectSum, LinearOperator
from pygeoinf.interval import Sobolev as TimeSobolev
from pygeoinf.testing import check_operator
from planetmodel import RadialMesh
from pyshtools import SHGrid

from pyslfp.core import EarthState, FieldHistorySpace, lebesgue_load_space
from pyslfp.love_numbers import Material
from pyslfp.viscoelastic import (
    IceHistoryOperator,
    LogViscositySpace,
    Rheology,
    SeaLevelEvolution,
    SeaLevelEvolutionOperator,
    ViscoelasticOperator,
)
from pyslfp.viscoelastic.ice_history import _NodalIceModel

from .body import LMAX, TAU
from .test_evolution import _history_state, _MeltingCap
from .test_viscosity import _two_branches

DATES = np.linspace(10.0, 0.0, 6)


class _ThickCap(_MeltingCap):
    """The melting cap, thick enough that its band-limited thickness is
    grounded at the centre of a 3 km ocean and floats as it melts."""

    THICKNESS = 12000.0


def _marine(earth):
    ice_model = _ThickCap(length_scale=earth.parameters.length_scale)
    ice = ice_model.get_ice_thickness(
        _ThickCap.START, LMAX, grid=earth.grid_name, extend=earth.extend
    )
    sea = earth.constant_grid(3000.0 / earth.parameters.length_scale)
    return ice_model, EarthState(ice, sea, earth)


def _spaces(earth, rheology, modes=16):
    history = FieldHistorySpace(
        TimeSobolev(modes, 1.0, 2.0, lower=0.0, upper=10.0), lebesgue_load_space(earth)
    )
    return history, lebesgue_load_space(earth), LogViscositySpace(rheology, 8)


def _history_of(space, dates, ice_model, earth):
    """The history whose values at the dates are the ice model's, the
    least-norm fit over the time modes."""
    field_space = space.field_space
    values = np.stack(
        [
            field_space.to_components(
                ice_model.get_ice_thickness(
                    float(d), LMAX, grid=earth.grid_name, extend=earth.extend
                )
            )
            for d in dates
        ]
    )
    return np.linalg.lstsq(space.basis_matrix(dates), values, rcond=None)[0]


def _point(operator, ice_model, state, earth):
    history_space, _, viscosity_space = operator.domain.subspaces
    return (
        _history_of(history_space, operator.dates, ice_model, earth),
        state.sea_level,
        viscosity_space.log_viscosity(),
    )


def _direction(operator, rng, earth, *, sea_level=None, ice=20.0, chi=0.1):
    history_space, sea_level_space, viscosity_space = operator.domain.subspaces
    scale = earth.parameters.length_scale
    delta_history = history_space.from_components(
        (ice / scale) * rng.standard_normal(history_space.dim)
    )
    if sea_level is None:
        delta_sea_level = sea_level_space.from_components(
            (ice / scale) * rng.standard_normal(sea_level_space.dim)
        )
    else:
        delta_sea_level = sea_level
    delta_chi = viscosity_space.from_components(
        chi * rng.standard_normal(viscosity_space.dim)
    )
    return (delta_history, delta_sea_level, delta_chi)


def _no_shoreline_direction(earth):
    lats = earth.lats()[:, None] * np.ones_like(earth.lons())[None, :]
    lons = np.ones_like(earth.lats())[:, None] * earth.lons()[None, :]
    metre = earth.parameters.length_scale
    return SHGrid.from_array(
        (30.0 / metre) * np.cos(np.radians(lats)) ** 2 * np.sin(np.radians(lons)),
        grid=earth.grid,
    )


def _frozen(setting, **options):
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    rheology = _two_branches(material)
    evolution = SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)
    operator = SeaLevelEvolutionOperator(
        evolution, DATES, *_spaces(earth, rheology), [6.0, 0.0], **options
    )
    return operator, ice_model, state, earth


def _migrating(setting, **options):
    material, earth, _, _, _ = setting
    ice_model, state = _marine(earth)
    rheology = _two_branches(material)
    evolution = SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)
    operator = SeaLevelEvolutionOperator(
        evolution,
        DATES,
        *_spaces(earth, rheology),
        [6.0, 0.0],
        shorelines="migrating",
        **options,
    )
    return operator, ice_model, state, earth


def _taylor_ratio(operator, x, v, codomain):
    linearization = operator.at(x)
    tangent = linearization.derivative(v)
    residuals = {}
    for eps in (0.02, 0.01):
        moved = operator(operator.domain.axpy(eps, v, operator.domain.copy(x)))
        predicted = codomain.axpy(eps, tangent, codomain.copy(linearization.value))
        residuals[eps] = codomain.norm(codomain.subtract(moved, predicted))
    return residuals[0.02] / residuals[0.01], residuals[0.01] / (
        0.01 * codomain.norm(tangent)
    )


# ==================================================================== #
#                            The value                                 #
# ==================================================================== #


def test_the_value_is_the_run(setting):
    operator, ice_model, state, earth = _frozen(setting)
    x = _point(operator, ice_model, state, earth)
    value = operator(x)
    history_space, _, viscosity_space = operator.domain.subspaces
    thickness = operator.thickness_at_dates(x[0])
    evolution = SeaLevelEvolution(
        ViscoelasticOperator(viscosity_space.rheology(x[2]), LMAX), state
    )
    snapshots, _ = evolution.run_ice_history(
        DATES,
        _NodalIceModel(DATES, thickness, length_scale=earth.parameters.length_scale),
    )
    space = operator.response_space
    for got, node in zip(value, (2, 5)):
        np.testing.assert_allclose(
            space.to_components(got), space.to_components(snapshots[node].response)
        )
    assert operator.domain.labels == (
        "ice_history",
        "initial_sea_level",
        "log_viscosity",
    )


# ==================================================================== #
#                          The derivative                              #
# ==================================================================== #


@pytest.mark.parametrize("shorelines", ["frozen", "migrating"])
def test_the_taylor_remainder(setting, shorelines):
    """F(x + eps v) - F(x) - eps F'(x) v falls at second order, in a
    direction moving every argument, about a frozen run and about a
    migrating trajectory, where the sea level direction moves no shoreline
    and the ice direction is small."""
    operator, ice_model, state, earth = (
        _frozen(setting) if shorelines == "frozen" else _migrating(setting)
    )
    rng = np.random.default_rng(1)
    x = _point(operator, ice_model, state, earth)
    v = _direction(
        operator,
        rng,
        earth,
        sea_level=None if shorelines == "frozen" else _no_shoreline_direction(earth),
        ice=20.0 if shorelines == "frozen" else 1.0,
    )
    if shorelines == "migrating":
        _, snapshots = operator._run(x)
        assert not np.array_equal(
            snapshots[0].ocean_function.data, snapshots[-1].ocean_function.data
        )
    ratio, size = _taylor_ratio(operator, x, v, operator.codomain)
    assert 3.0 < ratio < 5.0
    assert size < 0.1


def test_the_derivative_is_linear_in_the_loads(setting):
    """Under frozen shorelines the map is affine in the ice history and
    independent of the initial sea level: the Taylor remainder vanishes
    along those arguments and the derivative in the sea level is zero."""
    operator, ice_model, state, earth = _frozen(setting)
    rng = np.random.default_rng(2)
    x = _point(operator, ice_model, state, earth)
    v = _direction(operator, rng, earth, chi=0.0)
    linearization = operator.at(x)
    codomain = operator.codomain
    moved = operator(operator.domain.axpy(1.0, v, operator.domain.copy(x)))
    predicted = codomain.axpy(
        1.0, linearization.derivative(v), codomain.copy(linearization.value)
    )
    assert codomain.norm(codomain.subtract(moved, predicted)) < 1e-8 * codomain.norm(
        moved
    )
    only_sea_level = (
        operator.domain.subspaces[0].zero(),
        v[1],
        operator.domain.subspaces[2].zero(),
    )
    assert codomain.norm(linearization.derivative(only_sea_level)) == 0.0


@pytest.mark.parametrize("shorelines", ["frozen", "migrating"])
def test_the_discrete_derivative_passes_the_adjoint_check(setting, shorelines):
    operator, ice_model, state, earth = (
        _frozen(setting) if shorelines == "frozen" else _migrating(setting)
    )
    x = _point(operator, ice_model, state, earth)
    check_operator(
        operator.derivative(x), rng=np.random.default_rng(3), trials=2, rtol=1e-8
    )


def test_the_continuous_pair_become_transposes_as_the_dates_are_refined(setting):
    """<F' v, g> - <v, F'* g>, nonzero for the continuous pair, falls at the
    scheme's order as the dates are halved, on a slowly relaxing body, for a
    direction in the viscosity and for a history smooth in time; the
    discrete pair are transposes to rounding at every spacing."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    _, mu = (np.asarray(a).real for a in material.moduli.kappa_mu())
    rheology = Rheology(material, [(mu, np.full(mu.shape, 1.0 / (10.0 * TAU)))])
    evolution = SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)
    gaps = {"continuous": [], "discrete": []}
    for method in gaps:
        for npts in (5, 9):
            dates = np.linspace(10.0, 0.0, npts)
            operator = SeaLevelEvolutionOperator(
                evolution, dates, *_spaces(earth, rheology), [0.0], method=method
            )
            x = _point(operator, ice_model, state, earth)
            derivative = operator.derivative(x)
            history_space, sea_level_space, viscosity_space = operator.domain.subspaces
            rng = np.random.default_rng(4)
            v = _direction(operator, rng, earth)
            profile = history_space.time_space.from_components(
                np.eye(history_space.time_space.dim)[1]
            )
            directions = (
                (
                    history_space.rank_one(profile, v[1]),
                    sea_level_space.zero(),
                    viscosity_space.zero(),
                ),
                (history_space.zero(), sea_level_space.zero(), v[2]),
            )
            g = operator.codomain.from_components(
                rng.standard_normal(operator.codomain.dim)
            )
            back = derivative.adjoint(g)
            gaps[method].append(
                [
                    abs(
                        operator.codomain.inner_product(derivative(w), g)
                        - operator.domain.inner_product(w, back)
                    )
                    / abs(operator.codomain.inner_product(derivative(w), g))
                    for w in directions
                ]
            )
    coarse, fine = (np.array(g) for g in gaps["continuous"])
    assert np.all(coarse < 0.05)
    assert np.all(coarse / fine > 3.0)
    assert np.all(np.array(gaps["discrete"]) < 1e-9)


# ==================================================================== #
#                            Fixing                                    #
# ==================================================================== #


def test_fixing_gives_the_ice_history_operator_when_frozen(setting):
    operator, ice_model, state, earth = _frozen(setting)
    x = _point(operator, ice_model, state, earth)
    linear = operator.fixing(initial_sea_level=x[1], log_viscosity=x[2])
    assert isinstance(linear, IceHistoryOperator)
    assert linear.domain == operator.history_space
    space = operator.response_space
    for a, b in zip(linear(x[0]), operator(x)):
        np.testing.assert_allclose(space.to_components(a), space.to_components(b))
    check_operator(linear, rng=np.random.default_rng(5), trials=2, rtol=1e-8)
    # the continuous method keeps the general form, whose derivative is
    # the same map
    other, _, _, _ = _frozen(setting, method="continuous")
    general = other.fixing(initial_sea_level=x[1], log_viscosity=x[2])
    assert not isinstance(general, LinearOperator)
    for a, b in zip(general(x[0]), linear(x[0])):
        np.testing.assert_allclose(space.to_components(a), space.to_components(b))


def test_fixing_composes_with_the_chain_rule(setting):
    operator, ice_model, state, earth = _frozen(setting)
    x = _point(operator, ice_model, state, earth)
    rng = np.random.default_rng(6)
    v = _direction(operator, rng, earth)
    full = operator.derivative(x)
    space = operator.response_space
    # one argument free: its own space
    viscosity_only = operator.fixing(ice_history=x[0], initial_sea_level=x[1])
    assert viscosity_only.domain == operator.viscosity_space
    partial = viscosity_only.derivative(x[2])(v[2])
    whole = full((operator.history_space.zero(), operator.sea_level_space.zero(), v[2]))
    for a, b in zip(partial, whole):
        np.testing.assert_allclose(space.to_components(a), space.to_components(b))
    # two free: their direct sum, in order
    sea_level_and_viscosity = operator.fixing(ice_history=x[0])
    assert isinstance(sea_level_and_viscosity.domain, DirectSum)
    assert sea_level_and_viscosity.domain.labels == (
        "initial_sea_level",
        "log_viscosity",
    )
    partial = sea_level_and_viscosity.derivative((x[1], x[2]))((v[1], v[2]))
    whole = full((operator.history_space.zero(), v[1], v[2]))
    for a, b in zip(partial, whole):
        np.testing.assert_allclose(space.to_components(a), space.to_components(b))
    assert operator.fixing() is operator
    with pytest.raises(ValueError, match="free"):
        operator.fixing(ice_history=x[0], initial_sea_level=x[1], log_viscosity=x[2])


# ==================================================================== #
#                           Refusals                                   #
# ==================================================================== #


def test_what_is_refused(setting):
    material, earth, _, _, _ = setting
    _, state = _history_state(material, earth)
    rheology = _two_branches(material)
    evolution = SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)
    spaces = _spaces(earth, rheology)
    with pytest.raises(ValueError, match="method"):
        SeaLevelEvolutionOperator(evolution, DATES, *spaces, [0.0], method="exact")
    with pytest.raises(ValueError, match="shorelines"):
        SeaLevelEvolutionOperator(evolution, DATES, *spaces, [0.0], shorelines="fixed")
    with pytest.raises(ValueError, match="not on the run's dates"):
        SeaLevelEvolutionOperator(evolution, DATES, *spaces, [3.0])
    # a space over another rheology on the same material is allowed; one
    # on another material is not
    SeaLevelEvolutionOperator(
        evolution,
        DATES,
        spaces[0],
        spaces[1],
        LogViscositySpace(Rheology.from_relaxation_time(material, TAU), 4),
        [0.0],
    )
    elsewhere = Material(RadialMesh(material.model, ngll=5, lmax=LMAX), material.model)
    other = LogViscositySpace(Rheology.from_relaxation_time(elsewhere, TAU), 4)
    with pytest.raises(ValueError, match="material"):
        SeaLevelEvolutionOperator(evolution, DATES, spaces[0], spaces[1], other, [0.0])


# ==================================================================== #
#                      The second derivative                           #
# ==================================================================== #


@pytest.mark.parametrize("shorelines", ["frozen", "migrating"])
def test_the_second_derivative_passes_the_adjoint_check(setting, shorelines):
    operator, ice_model, state, earth = (
        _frozen(setting) if shorelines == "frozen" else _migrating(setting)
    )
    assert operator.has_second_derivative
    rng = np.random.default_rng(7)
    x = _point(operator, ice_model, state, earth)
    v = _direction(operator, rng, earth, ice=1.0)
    second = operator.second_derivative(x, v)
    check_operator(second, rng=np.random.default_rng(8), trials=2, rtol=1e-8)
    # symmetric in its two directions
    u = _direction(operator, rng, earth, ice=1.0)
    space = operator.response_space
    for a, b in zip(second(u), operator.second_derivative(x, u)(v)):
        np.testing.assert_allclose(
            space.to_components(a), space.to_components(b), rtol=1e-9, atol=1e-16
        )


def test_the_hessian_of_a_misfit_matches_differences_of_its_gradient(setting):
    """For J = |F(x) - d|^2 / 2, H u = F'* F' u + (F''[u, .])* (F(x) - d):
    central differences of the gradient along u converge to it at second
    order and land on it after extrapolation."""
    operator, ice_model, state, earth = _frozen(setting)
    rng = np.random.default_rng(9)
    x = _point(operator, ice_model, state, earth)
    u = _direction(operator, rng, earth)
    domain, codomain = operator.domain, operator.codomain
    data = codomain.from_components(
        codomain.to_components(operator(x))
        + 0.1
        * rng.standard_normal(codomain.dim)
        * np.abs(codomain.to_components(operator(x))).max()
    )

    def gradient(point):
        linearization = operator.at(point)
        residual = codomain.subtract(linearization.value, data)
        return linearization.derivative.adjoint(residual), residual

    base, residual = gradient(x)
    linearization = operator.at(x)
    hessian_u = domain.add(
        linearization.derivative.adjoint(linearization.derivative(u)),
        operator.second_derivative(x, u).adjoint(residual),
    )
    differences = {}
    for eps in (0.02, 0.01):
        plus, _ = gradient(domain.axpy(eps, u, domain.copy(x)))
        minus, _ = gradient(domain.axpy(-eps, u, domain.copy(x)))
        differences[eps] = domain.scale(1.0 / (2.0 * eps), domain.subtract(plus, minus))
    residuals = {
        eps: domain.norm(domain.subtract(d, hessian_u))
        for eps, d in differences.items()
    }
    assert 3.0 < residuals[0.02] / residuals[0.01] < 5.0
    richardson = domain.axpy(
        -1.0 / 3.0, differences[0.02], domain.scale(4.0 / 3.0, differences[0.01])
    )
    assert domain.norm(domain.subtract(richardson, hessian_u)) < 1e-6 * domain.norm(
        hessian_u
    )


def test_the_continuous_second_derivative(setting):
    """By the continuous method the second derivative is symmetric in its
    directions, and its adjoint check holds to the order of the scheme: on a
    slowly relaxing body, along directions smooth in time, the mismatch
    falls by about four when the dates are halved."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    rheology = Rheology.from_relaxation_time(material, 10.0 * TAU)
    evolution = SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)
    gaps = []
    for npts in (5, 9):
        dates = np.linspace(10.0, 0.0, npts)
        operator = SeaLevelEvolutionOperator(
            evolution, dates, *_spaces(earth, rheology), [0.0], method="continuous"
        )
        x = _point(operator, ice_model, state, earth)
        rng = np.random.default_rng(10)
        history_space, sea_level_space, viscosity_space = operator.domain.subspaces
        scale = earth.parameters.length_scale

        def direction(mode, seed):
            profile = history_space.time_space.from_components(
                np.eye(history_space.time_space.dim)[mode]
            )
            field = sea_level_space.from_components(
                (20.0 / scale)
                * np.random.default_rng(seed).standard_normal(sea_level_space.dim)
            )
            chi = viscosity_space.from_components(
                0.1
                * np.random.default_rng(seed + 1).standard_normal(viscosity_space.dim)
            )
            return (history_space.rank_one(profile, field), sea_level_space.zero(), chi)

        u, v = direction(1, 21), direction(2, 31)
        second = operator.second_derivative(x, v)
        space = operator.response_space
        for a, b in zip(second(u), operator.second_derivative(x, u)(v)):
            np.testing.assert_allclose(
                space.to_components(a), space.to_components(b), rtol=1e-9, atol=1e-16
            )
        g = operator.codomain.from_components(
            rng.standard_normal(operator.codomain.dim)
        )
        lhs = operator.codomain.inner_product(second(u), g)
        rhs = operator.domain.inner_product(u, second.adjoint(g))
        gaps.append(abs(lhs - rhs) / abs(lhs))
    assert gaps[0] < 0.1 and gaps[0] / gaps[1] > 3.0
