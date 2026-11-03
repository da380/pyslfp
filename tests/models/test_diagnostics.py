"""
Tests for the diagnostics of a set-up through the sea level equation, and for
the ready-made set-ups.
"""

import numpy as np
import pytest
from pygeoinf import EuclideanSpace, GaussianMeasure
from pygeoinf.testing import check_operator

from pyslfp.core import EarthState
from pyslfp.elastic import FingerPrintOperator
from pyslfp.core import sobolev_load_space
from pyslfp.core import averaging_operator
from pyslfp.models import (
    ICE_SURFACE_HEIGHT,
    OCEAN_MASS,
    GEOCENTRIC_SEA_LEVEL,
    STERIC_SEA_LEVEL,
    Role,
    ice_and_firn,
    ice_and_ocean,
    ice_ocean_and_density,
    sea_level_budget,
)

LMAX = 16


@pytest.fixture(scope="module")
def state():
    return EarthState.from_defaults(lmax=LMAX)


@pytest.fixture(scope="module")
def scale(state):
    return 0.1 * state.model.parameters.mean_sea_floor_radius


@pytest.fixture(scope="module")
def fingerprint(state, scale):
    """Loads in H^(1/4), the response in H^(5/4): the regularity chain."""
    return FingerPrintOperator(
        state,
        load_parameters=(0.25, scale),
        response_parameters=(1.25, scale),
        rtol=1e-12,
    )


@pytest.fixture(scope="module")
def budget(state, scale, fingerprint):
    field_space = sobolev_load_space(state.model, 2.0, scale)
    return sea_level_budget(
        state,
        fingerprint.domain,
        space=field_space,
        observable_space=fingerprint.field_space,
    )


def _sample(setup, seed):
    b = setup.state.model.parameters.mean_sea_floor_radius
    measure = GaussianMeasure.from_product(
        [c.space.heat_measure(0.3 * b) for c in setup.components], domain=setup.space
    )
    return measure.sample(rng=np.random.default_rng(seed))


def _hemispheres(state):
    ocean = state.ocean_projection(value=0.0)
    north = state.northern_hemisphere_projection(value=0.0)
    south = state.southern_hemisphere_projection(value=0.0)
    return [ocean * north, ocean * south]


def test_the_global_mean_is_barystatic_plus_steric(budget, fingerprint):
    """Mass conservation: the ocean average of the relative sea level change
    is the barystatic change plus the steric change, the ocean-internal
    loads' own response cancelling the mean of the sterodynamic sea level
    change less its steric part."""
    x = _sample(budget, 1)
    total = budget.gmsl(fingerprint)(x)[0]
    split = budget.barystatic_gmsl()(x)[0] + budget.steric_gmsl(x)[0]
    assert abs(total - split) <= 1e-8 * max(abs(total), abs(split))


def test_the_decomposition_sums_to_the_regional_sea_level(budget, fingerprint, state):
    regions = _hemispheres(state)
    parts = budget.regional_decomposition(fingerprint, regions)
    x = _sample(budget, 2)
    whole = averaging_operator(state, fingerprint.field_space, regions) @ (
        budget.relative_sea_level_change(fingerprint)
    )
    total = (
        parts.steric(x) + parts.dynamic_manometric(x) + parts.barystatic_manometric(x)
    )
    expected = whole(x)
    np.testing.assert_allclose(total, expected, rtol=1e-8, atol=0.0)
    assert parts.steric.codomain == EuclideanSpace(2)


def test_the_diagnostics_pass_the_adjoint_check(budget, fingerprint, state):
    rng = np.random.default_rng(3)
    b = state.model.parameters.mean_sea_floor_radius
    measure = GaussianMeasure.from_product(
        [c.space.heat_measure(0.3 * b) for c in budget.components], domain=budget.space
    )
    parts = budget.regional_decomposition(fingerprint, _hemispheres(state))
    for operator in (budget.gmsl(fingerprint), *parts):
        check_operator(
            operator,
            rng=rng,
            measure=measure,
            codomain_measure=GaussianMeasure.from_standard_deviation(
                operator.codomain, 1.0
            ),
            trials=2,
            rtol=1e-7,
        )
    check_operator(
        budget.relative_sea_level_change(fingerprint),
        rng=rng,
        measure=measure,
        codomain_measure=fingerprint.field_space.heat_measure(0.3 * b),
        trials=2,
        rtol=1e-7,
    )


def test_without_density_there_is_no_steric_part(state, scale, fingerprint):
    field_space = sobolev_load_space(state.model, 2.0, scale)
    setup = ice_and_ocean(
        state,
        fingerprint.domain,
        space=field_space,
        observable_space=fingerprint.field_space,
    )
    parts = setup.regional_decomposition(fingerprint, _hemispheres(state))
    x = _sample(setup, 4)
    assert np.all(parts.steric(x) == 0.0)
    total = setup.gmsl(fingerprint)(x)[0]
    assert abs(total - setup.barystatic_gmsl()(x)[0]) <= 1e-8 * abs(total)


def test_the_spaces_must_match(state, scale, fingerprint, budget):
    other = FingerPrintOperator(state, rtol=1e-6)
    with pytest.raises(ValueError, match="load space"):
        budget.gmsl(other)
    field_space = sobolev_load_space(state.model, 2.0, scale)
    mismatched = ice_and_ocean(state, fingerprint.domain, space=field_space)
    with pytest.raises(ValueError, match="field space"):
        mismatched.relative_sea_level_change(fingerprint)


# ==================================================================== #
#                         Ready-made set-ups                           #
# ==================================================================== #


def test_the_ready_made_set_ups(state, scale, fingerprint):
    load_space = fingerprint.domain
    density = 0.6 * state.model.parameters.ice_density
    expectations = {
        ice_and_ocean: ("ice_sheets", "sterodynamic_sea_level"),
        ice_ocean_and_density: (
            "ice_sheets",
            "sterodynamic_sea_level",
            "ocean_density",
        ),
        sea_level_budget: (
            "ice_sheets",
            "glaciers",
            "land_water",
            "sterodynamic_sea_level",
            "ocean_density",
        ),
    }
    for factory, names in expectations.items():
        setup = factory(state, load_space)
        assert setup.names == names
        assert (OCEAN_MASS in setup.constraints) == ("sterodynamic_sea_level" in names)
    budget = sea_level_budget(state, load_space)
    assert set(budget.observables) == {
        GEOCENTRIC_SEA_LEVEL,
        ICE_SURFACE_HEIGHT,
        STERIC_SEA_LEVEL,
    }
    assert [c.role for c in budget.components].count(Role.MASS_SOURCE) == 3
    firn_setup = ice_and_firn(state, load_space, firn_density=density)
    assert firn_setup.names == ("ice_sheets", "firn")
    assert set(firn_setup.observables) == {ICE_SURFACE_HEIGHT}


def test_the_geocentric_sea_level(budget, fingerprint, state):
    """The geocentric sea level, the observable of altimetry, is the relative
    sea level change plus the vertical displacement of the response to the
    set-up's load, the direct contributions entering both alike; with the
    rotational contribution it passes the adjoint check."""
    x = _sample(budget, 4)
    field_space = fingerprint.field_space
    difference = field_space.subtract(
        budget.geocentric_sea_level(fingerprint, remove_rotational_contribution=False)(
            x
        ),
        budget.relative_sea_level_change(fingerprint)(x),
    )
    displacement = (
        fingerprint.codomain.field_operator("displacement")
        @ fingerprint
        @ budget.load_operator
    )(x)
    scale = np.max(np.abs(displacement.data))
    np.testing.assert_allclose(
        difference.data, displacement.data, rtol=0.0, atol=1e-10 * scale
    )
    b = state.model.parameters.mean_sea_floor_radius
    measure = GaussianMeasure.from_product(
        [c.space.heat_measure(0.3 * b) for c in budget.components], domain=budget.space
    )
    check_operator(
        budget.geocentric_sea_level(fingerprint),
        rng=np.random.default_rng(5),
        measure=measure,
        codomain_measure=field_space.heat_measure(0.3 * b),
        trials=2,
        rtol=1e-7,
    )


def test_the_response(budget, fingerprint):
    """The response to the set-up's load, the observable of the members with
    no direct terms."""
    x = _sample(budget, 6)
    response_space = fingerprint.codomain
    got = budget.response(fingerprint)(x)
    expected = fingerprint(budget.load_operator(x))
    np.testing.assert_allclose(
        response_space.to_components(got), response_space.to_components(expected)
    )
