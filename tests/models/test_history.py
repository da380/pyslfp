"""
Tests for a set-up through time: the product space's axioms, the lifted
operators against the set-up's at a time, the scalar diagnostics as functions
of time, the lifted constraint, the samplers in time, a tide gauge series on
the lifted response, and the composition with the ice history operator.
"""

import numpy as np
import pytest
from pygeoinf import EuclideanSpace, Traits
from pygeoinf.interval import Sobolev as TimeSobolev
from pygeoinf.testing import check_coordinates, check_operator, check_space

from pyslfp.core import (
    EarthState,
    FieldHistorySpace,
    at_time,
    difference_quotient,
    sample_at_times,
    sobolev_load_space,
    time_average,
)
from pyslfp.elastic import FingerPrintOperator
from pyslfp.models import OCEAN_MASS, HistorySetup, ice_and_ocean, ice_ocean_and_density
from pyslfp.observations import TideGaugeObservationModel

LMAX = 8
MODES = 6
SPAN = 10.0  # kiloyears


@pytest.fixture(scope="module")
def state():
    return EarthState.from_defaults(lmax=LMAX)


@pytest.fixture(scope="module")
def scale(state):
    return 0.1 * state.model.parameters.mean_sea_floor_radius


@pytest.fixture(scope="module")
def fingerprint(state, scale):
    return FingerPrintOperator(
        state,
        load_parameters=(0.25, scale),
        response_parameters=(1.25, scale),
        rtol=1e-12,
    )


@pytest.fixture(scope="module")
def time_space(state):
    kiloyear = state.model.parameters.kiloyear
    return TimeSobolev(MODES, 1.0, 2.0 * kiloyear, lower=0.0, upper=SPAN * kiloyear)


@pytest.fixture(scope="module")
def setup(state, scale, fingerprint):
    return ice_ocean_and_density(
        state,
        fingerprint.domain,
        space=sobolev_load_space(state.model, 2.0, scale),
        observable_space=fingerprint.field_space,
    )


@pytest.fixture(scope="module")
def history(setup, time_space):
    return HistorySetup(setup, time_space)


def _random(space, seed):
    return space.from_components(np.random.default_rng(seed).standard_normal(space.dim))


def _profile(time_space, seed):
    return time_space.from_components(
        np.random.default_rng(seed).standard_normal(time_space.dim)
    )


def _times(state, fraction):
    return fraction * SPAN * state.model.parameters.kiloyear


# ==================================================================== #
#                             Structure                                #
# ==================================================================== #


def test_the_space(history, setup, time_space):
    space = history.space
    rng = np.random.default_rng(0)
    check_space(space, rng=rng)
    check_coordinates(space, rng=rng)
    assert space.shape == (time_space.dim, setup.space.dim)
    assert history.names == setup.names
    for name in setup.names:
        operator = history.history(name)
        assert operator.domain == space
        assert operator.codomain == FieldHistorySpace(
            time_space, setup.component(name).space
        )
    with pytest.raises(TypeError):
        HistorySetup(setup, object())


def test_a_lifted_operator_at_a_time_is_the_operator_at_that_time(
    history, setup, state
):
    """Lifting and sampling commute: the load history at a time is the load
    of the set-up's vector at that time, and so for every observable."""
    u = _random(history.space, 1)
    t = _times(state, 0.37)
    value = history.at(t)(u)
    load_space = setup.load_space
    np.testing.assert_allclose(
        load_space.to_components(
            at_time(history.load_space, t)(history.load_operator(u))
        ),
        load_space.to_components(setup.load_operator(value)),
        rtol=1e-10,
        atol=1e-14,
    )
    for name, lifted in history.observables.items():
        observable = setup.observables[name]
        np.testing.assert_allclose(
            observable.codomain.to_components(at_time(lifted.codomain, t)(lifted(u))),
            observable.codomain.to_components(observable(value)),
            rtol=1e-10,
            atol=1e-14,
        )
    check_operator(history.load_operator, rng=np.random.default_rng(2), trials=2)
    check_operator(history.support, rng=np.random.default_rng(3), trials=2)


def test_the_scalar_diagnostics_are_functions_of_time(
    history, setup, time_space, state
):
    """The GMSL history of a rank-one history a(t) f is a(t) times the GMSL
    of f, and the constraint through time keeps its surjectivity."""
    a = _profile(time_space, 4)
    f = _random(setup.space, 5)
    u = history.space.rank_one(a, f)
    total = history.barystatic_gmsl()
    assert total.codomain == time_space
    gmsl_f = setup.barystatic_gmsl()(f)[0]
    np.testing.assert_allclose(
        time_space.to_components(total(u)),
        gmsl_f * time_space.to_components(a),
        rtol=1e-10,
        atol=1e-14,
    )
    check_operator(total, rng=np.random.default_rng(6), trials=2)
    steric = history.steric_gmsl
    assert steric is not None and steric.codomain == time_space
    split = history.gmsl_split
    assert split is not None and split.codomain.shape == (time_space.dim, 2)
    constraint = history.constraints[OCEAN_MASS]
    assert constraint.codomain == time_space
    assert Traits.SURJECTIVE in constraint.traits
    check_operator(constraint, rng=np.random.default_rng(7), trials=2)
    # a history whose every time lies in the constraint's kernel
    projection = setup.constraints[OCEAN_MASS]
    kernel = setup.space.subtract(
        f, projection.adjoint(EuclideanSpace(1).from_components(np.array([1.0])))
    )
    # project f onto the kernel by one Gauss-Newton step on the constraint
    value = projection(f)[0]
    normal = projection(
        projection.adjoint(EuclideanSpace(1).from_components(np.array([1.0])))
    )[0]
    kernel = setup.space.axpy(
        -value / normal,
        projection.adjoint(EuclideanSpace(1).from_components(np.array([1.0]))),
        setup.space.copy(f),
    )
    assert abs(projection(kernel)[0]) < 1e-10 * abs(value)
    assert time_space.norm(constraint(history.space.rank_one(a, kernel))) < 1e-9 * (
        time_space.norm(a) * abs(value)
    )


# ==================================================================== #
#                          Sampling in time                            #
# ==================================================================== #


def test_the_samplers(history, time_space, state):
    """The window average and the difference quotient against the evaluation
    at times, and a series of epochs against the evaluations."""
    space = history.space
    u = _random(space, 8)
    lower, upper = _times(state, 0.2), _times(state, 0.6)
    for operator in (
        at_time(space, lower),
        time_average(space, lower, upper),
        difference_quotient(space, lower, upper),
    ):
        check_operator(operator, rng=np.random.default_rng(9), trials=2)
    # the average by Simpson's rule over the evaluations
    from scipy.integrate import simpson

    grid = np.linspace(lower, upper, 2001)
    values = np.stack(
        [space.factors[1].to_components(at_time(space, t)(u)) for t in grid]
    )
    np.testing.assert_allclose(
        space.factors[1].to_components(time_average(space, lower, upper)(u)),
        simpson(values, x=grid, axis=0) / (upper - lower),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        space.factors[1].to_components(difference_quotient(space, lower, upper)(u)),
        (values[-1] - values[0]) / (upper - lower),
        rtol=1e-10,
    )
    series = sample_at_times(space, [lower, upper])
    for got, want in zip(series(u), (values[0], values[-1])):
        np.testing.assert_allclose(
            space.factors[1].to_components(got), want, rtol=1e-10
        )
    with pytest.raises(ValueError, match="window"):
        time_average(space, upper, lower)
    with pytest.raises(ValueError):
        at_time(space, -1.0)


# ==================================================================== #
#                      With the physics                                #
# ==================================================================== #


def test_a_tide_gauge_series_on_the_lifted_response(history, setup, fingerprint, state):
    """Tide gauges at two epochs of the relative sea level history equal the
    gauges on the set-up's sea level at those epochs."""
    sea_level = history.relative_sea_level_change(fingerprint)
    points = [(45.0, -60.0), (-33.8, 151.2), (51.5, 0.0)]
    epochs = [_times(state, 0.25), _times(state, 0.75)]
    u = _random(history.space, 10)
    series = []
    for t in epochs:
        gauges = TideGaugeObservationModel(
            at_time(sea_level.codomain, t) @ sea_level, points
        )
        series.append(gauges.forward_operator(u))
    single = TideGaugeObservationModel(
        setup.relative_sea_level_change(fingerprint), points
    )
    for t, got in zip(epochs, series):
        np.testing.assert_allclose(
            got, single.forward_operator(history.at(t)(u)), rtol=1e-9
        )
    assert history.response(fingerprint).codomain.factors[1] == fingerprint.codomain
    assert history.gmsl(fingerprint).codomain == history.time_space


def test_the_composition_with_the_ice_history_operator(state, time_space):
    """A set-up's ice history is the argument of the viscoelastic operators."""
    from pyslfp.ice import AnalyticalIceModel
    from pyslfp.viscoelastic import (
        IceHistoryOperator,
        Rheology,
        SeaLevelEvolution,
        ViscoelasticOperator,
    )
    from pyslfp.love_numbers import Material
    from planetmodel import LayeredIsotropicElastic, RadialMesh, units

    earth = state.model
    lmax = earth.lmax
    scales = units.Scales.geophysical(6.371e6)
    model = LayeredIsotropicElastic(
        [0.0, 1.0], rho=[1.0], vp=[3.1], vs=[1.3], scales=scales
    )
    material = Material(RadialMesh(model, ngll=5, lmax=lmax), model)
    from pyslfp.core import EarthModel
    from pyslfp.love_numbers import fluid_tidal_number, love_numbers

    base = EarthModel.from_planet_model(model, lmax, mesh=material.mesh)
    parameters = base.parameters.with_hydrostatic_moments(
        fluid_tidal_number(material, 2)
    )
    earth = EarthModel(
        lmax, parameters=parameters, love_numbers=love_numbers(material, lmax)
    )
    ice_model = AnalyticalIceModel(length_scale=earth.parameters.length_scale)
    start = EarthState.from_ice_model(ice_model, SPAN, earth, land=None)
    evolution = SeaLevelEvolution(
        ViscoelasticOperator(Rheology.from_relaxation_time(material, 6.0e7), lmax),
        start,
    )
    kiloyear = earth.parameters.kiloyear
    time = TimeSobolev(MODES, 1.0, 2.0 * kiloyear, lower=0.0, upper=SPAN * kiloyear)
    load_space = sobolev_load_space(earth, 0.25, 0.1)
    setup = ice_and_ocean(start, load_space, space=sobolev_load_space(earth, 2.0, 0.1))
    history = HistorySetup(setup, time)
    dates = np.linspace(SPAN, 0.0, 5)
    operator = IceHistoryOperator(
        evolution, dates, history.history("ice_sheets").codomain, [0.0], rtol=1e-12
    )
    composed = operator @ history.history("ice_sheets")
    assert composed.domain == history.space
    check_operator(composed, rng=np.random.default_rng(11), trials=2, rtol=1e-8)
