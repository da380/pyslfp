"""
Tests for the assembler of pyslfp.models.
"""

import numpy as np
import pytest
from pygeoinf import (
    EuclideanSpace,
    GaussianMeasure,
    LinearOperator,
    RowLinearOperator,
    Traits,
)
from pygeoinf.testing import check_operator, check_traits

from pyslfp.core import EarthState
from pyslfp.linear_operators.loads import (
    ice_thickness_change_to_load_operator,
    ocean_density_change_to_load_operator,
    sea_level_change_to_load_operator,
    steric_sea_level_operator,
)
from pyslfp.linear_operators.physics import lebesgue_load_space, sobolev_load_space
from pyslfp.linear_operators.spatial import (
    ice_sheet_projection_operator,
)
from pyslfp.models import (
    ICE_SURFACE_HEIGHT,
    OCEAN_MASS,
    GEOCENTRIC_SEA_LEVEL,
    STERIC_SEA_LEVEL,
    ModelSetup,
    sterodynamic_sea_level,
    firn,
    glaciers,
    ice_sheets,
    land_water_storage,
    ocean_density,
)


@pytest.fixture(scope="module")
def state():
    """A real state: the supports follow the Earth's geography."""
    return EarthState.from_defaults(lmax=16)


@pytest.fixture(scope="module")
def load_space(state):
    """Loads carry indicators, so they lie in H^s only for s < 1/2."""
    b = state.model.parameters.mean_sea_floor_radius
    return sobolev_load_space(state.model, 0.25, 0.1 * b)


@pytest.fixture(scope="module")
def field_space(state):
    """The components' space, where the priors live."""
    b = state.model.parameters.mean_sea_floor_radius
    return sobolev_load_space(state.model, 2.0, 0.1 * b)


@pytest.fixture(scope="module")
def three(state, load_space, field_space):
    """The ice + eta + delta rho space of the Heathcote2026 scripts."""
    return ModelSetup(
        state,
        [
            ice_sheets(state, load_space, space=field_space, name="ice"),
            sterodynamic_sea_level(state, load_space, space=field_space, name="eta"),
            ocean_density(state, load_space, space=field_space, name="rho"),
        ],
    )


@pytest.fixture(scope="module")
def budget(state, load_space, field_space):
    """Every component the vocabulary has, firn included."""
    density = 0.6 * state.model.parameters.ice_density
    on_fields = {"space": field_space}
    return ModelSetup(
        state,
        [
            ice_sheets(state, load_space, name="ice", **on_fields),
            glaciers(state, load_space, name="glaciers", **on_fields),
            firn(state, load_space, density=density, name="firn", **on_fields),
            land_water_storage(state, load_space, name="water", **on_fields),
            sterodynamic_sea_level(state, load_space, name="eta", **on_fields),
            ocean_density(state, load_space, name="rho", **on_fields),
        ],
    )


def _measure(setup):
    b = setup.state.model.parameters.mean_sea_floor_radius
    return GaussianMeasure.from_product(
        [c.space.heat_measure(0.3 * b) for c in setup.components],
        domain=setup.space,
    )


def _random(setup, seed):
    return _measure(setup).sample(rng=np.random.default_rng(seed))


def _assert_same_operator(assembled, reference, setup, seed=0):
    """Two operators on the joint space agree in action and adjoint action."""
    x = _random(setup, seed)
    y = assembled.codomain.random(rng=np.random.default_rng(seed + 1))
    for got, want in (
        (assembled(x), reference(x)),
        (assembled.adjoint(y), reference.adjoint(y)),
    ):
        got_c = got if isinstance(got, np.ndarray) else _components(got)
        want_c = want if isinstance(want, np.ndarray) else _components(want)
        np.testing.assert_allclose(
            got_c, want_c, rtol=0.0, atol=1e-12 * np.max(np.abs(want_c))
        )


def _components(vector):
    """The grid values of a field, or of each field of a tuple, flattened."""
    if isinstance(vector, tuple):
        return np.concatenate([_components(v) for v in vector])
    return np.ravel(vector.data)


# ==================================================================== #
#                     The joint space and its rows                     #
# ==================================================================== #


def test_joint_space_is_labelled(three, load_space, field_space):
    assert three.names == ("ice", "eta", "rho")
    assert three.space.labels == ("ice", "eta", "rho")
    assert three.space.subspace("eta") == field_space
    assert three.space.dim == 3 * field_space.dim
    assert three.load_space == load_space
    assert three.component("rho").name == "rho"
    with pytest.raises(KeyError):
        three.component("glaciers")


def test_load_operator_sums_the_components(budget):
    x = _random(budget, 2)
    total = budget.load_operator(x)
    expected = sum(
        (c.to_load(budget.space.projection(c.name)(x)).data for c in budget.components),
        np.zeros_like(total.data),
    )
    np.testing.assert_allclose(total.data, expected, rtol=0.0, atol=1e-13)


def test_observables_add_up_the_contributions(budget):
    x = _random(budget, 3)
    assert set(budget.observables) == {
        ICE_SURFACE_HEIGHT,
        GEOCENTRIC_SEA_LEVEL,
        STERIC_SEA_LEVEL,
    }
    ice = budget.space.projection("ice")(x).data
    snow = budget.space.projection("firn")(x).data
    height = budget.observables[ICE_SURFACE_HEIGHT](x).data
    np.testing.assert_allclose(height, ice + snow, rtol=0.0, atol=1e-14)
    geocentric = budget.observables[GEOCENTRIC_SEA_LEVEL](x).data
    np.testing.assert_array_equal(geocentric, budget.space.projection("eta")(x).data)


def test_contributions_to_one_observable_must_share_its_space(
    state, load_space, field_space
):
    b = state.model.parameters.mean_sea_floor_radius
    other = sobolev_load_space(state.model, 1.5, 0.1 * b)
    density = 0.6 * state.model.parameters.ice_density
    with pytest.raises(ValueError, match="ice_surface_height"):
        ModelSetup(
            state,
            [
                ice_sheets(
                    state, load_space, space=field_space, observable_space=other
                ),
                firn(state, load_space, density=density, space=field_space),
            ],
        )


@pytest.mark.parametrize("which", ["three", "budget"])
def test_assembled_operators_pass_the_checks(which, request):
    """Every assembled operator passes the adjoint check against smooth
    probes; the support's claimed traits hold, and so does the
    constraint's claimed surjectivity."""
    setup = request.getfixturevalue(which)
    measure = _measure(setup)
    rng = np.random.default_rng(0)
    b = setup.state.model.parameters.mean_sea_floor_radius
    scalar = GaussianMeasure.from_standard_deviation(EuclideanSpace(1), 1.0)
    pair = GaussianMeasure.from_standard_deviation(EuclideanSpace(2), 1.0)

    for operator in (setup.load_operator, *setup.observables.values()):
        check_operator(
            operator,
            rng=rng,
            measure=measure,
            codomain_measure=operator.codomain.heat_measure(0.3 * b),
            rtol=1e-9,
        )
    check_operator(
        setup.support, rng=rng, measure=measure, codomain_measure=measure, rtol=1e-9
    )
    check_traits(setup.support, rng=rng, measure=measure, rtol=1e-9)
    for operator in (
        setup.barystatic_gmsl(),
        setup.steric_gmsl,
        setup.constraints[OCEAN_MASS],
    ):
        check_operator(
            operator, rng=rng, measure=measure, codomain_measure=scalar, rtol=1e-9
        )
    check_operator(
        setup.gmsl_split, rng=rng, measure=measure, codomain_measure=pair, rtol=1e-9
    )
    assert Traits.SURJECTIVE in setup.constraints[OCEAN_MASS].traits


# ==================================================================== #
#              The harvest's hand-built operators, reproduced          #
# ==================================================================== #


def test_three_component_space_reproduces_the_hand_built_operators(
    three, state, load_space, field_space
):
    """The operators the Heathcote2026 scripts build by hand for their
    ice + eta + delta rho space, rebuilt from the same pieces, agree with
    the assembled ones.

    The scripts mask the ice by composing the ice load with the ice
    projection. That composition has the right forward map, but its
    adjoint truncates between the two products and its forward map does
    not, so it is compared in the forward map only; the same row with the
    mask folded into the ice load's multiplier, as a set-up builds it, is
    compared in action and adjoint. The scripts' barystatic functional
    leaves the ice unmasked, relying on a masked prior; for global fields
    its weighting carries the support. Their geocentric sea level takes eta
    unmasked, as a set-up does. The scripts take the ocean averages of the
    mass constraint and of the steric sea level as L2 products with C / A_O,
    which pair the truncations of two rough fields; a set-up integrates the
    fields themselves, which already carry the ocean function, over the
    ocean area, as the sea level equation integrates a load's mass, and the
    comparison is with that form."""
    joint = three.space
    ice = state.ice_sheet_projection(value=0.0)
    ice_to_load = ice_thickness_change_to_load_operator(state, field_space, load_space)
    dyn_to_load = sea_level_change_to_load_operator(state, field_space, load_space)
    rho_to_load = ocean_density_change_to_load_operator(state, field_space, load_space)
    ones = state.model.zero_grid()
    ones.data[:] = 1.0
    average = (1.0 / state.ocean_area) * load_space.l2_products_operator([ones])

    scripts_row = RowLinearOperator(
        [
            ice_to_load @ ice_sheet_projection_operator(state, field_space),
            dyn_to_load,
            rho_to_load,
        ]
    )
    multiplier = (
        state.model.parameters.ice_density * state.one_minus_ocean_function * ice
    )
    l2 = lebesgue_load_space(state.model)
    masked_ice_to_load = LinearOperator.from_formal_adjoint(
        field_space,
        load_space,
        LinearOperator.from_callables(
            l2, l2, lambda u: multiplier * u, adjoint=lambda u: multiplier * u
        ),
    )
    joint_to_load = RowLinearOperator([masked_ice_to_load, dyn_to_load, rho_to_load])
    mass_constraint = RowLinearOperator(
        [
            LinearOperator.zero(field_space, codomain=average.codomain),
            average @ dyn_to_load,
            average @ rho_to_load,
        ]
    )
    steric = (
        average
        @ steric_sea_level_operator(state, field_space, load_space)
        @ joint.projection(2)
    )
    barystatic = field_space.l2_products_operator(
        [state.barystatic_gmsl_weighting * ice]
    ) @ joint.projection(0)
    sea_surface_height = joint.projection(1)

    x = _random(three, 8)
    want = scripts_row(x).data
    np.testing.assert_allclose(
        three.load_operator(x).data,
        want,
        rtol=0.0,
        atol=1e-12 * np.max(np.abs(want)),
    )
    _assert_same_operator(three.load_operator, joint_to_load, three)
    _assert_same_operator(three.constraints[OCEAN_MASS], mass_constraint, three)
    _assert_same_operator(three.steric_gmsl, steric, three)
    _assert_same_operator(three.barystatic_gmsl(), barystatic, three)
    _assert_same_operator(three.barystatic_gmsl("ice"), barystatic, three)
    _assert_same_operator(
        three.observables[GEOCENTRIC_SEA_LEVEL], sea_surface_height, three
    )


# ==================================================================== #
#                         Physical identities                          #
# ==================================================================== #


def test_steric_pair_satisfies_the_mass_constraint(three, state, field_space):
    """A density change with its own steric sea level change, and nothing
    else, carries no water: it lies in the constraint's kernel."""
    rho = three.space.projection("rho")(_random(three, 4))
    eta = steric_sea_level_operator(state, field_space, field_space)(rho)
    x = (field_space.zero(), eta, rho)
    parts = [
        three.constraints[OCEAN_MASS](three.space.inclusion(name)(v))[0]
        for name, v in (("eta", eta), ("rho", rho))
    ]
    total = three.constraints[OCEAN_MASS](x)[0]
    assert abs(total) <= 1e-12 * max(abs(p) for p in parts)


def test_a_global_prior_conditions_exactly_on_the_mass_constraint(budget):
    """A prior on the global fields is positive definite, so pygeoinf
    conditions it on the ocean mass constraint as it stands, and its draws
    then carry no net water."""
    prior = _measure(budget)
    constraint = budget.constraints[OCEAN_MASS]
    posterior = prior.condition(constraint, np.zeros(1))
    rng = np.random.default_rng(9)
    before = abs(constraint(prior.sample(rng=rng))[0])
    after = abs(constraint(posterior.sample(rng=rng))[0])
    assert after <= 1e-8 * before


def test_component_gmsl_sums_to_the_total(budget):
    x = _random(budget, 6)
    parts = sum(
        budget.barystatic_gmsl(name)(x)[0]
        for name in ("ice", "glaciers", "firn", "water")
    )
    np.testing.assert_allclose(budget.barystatic_gmsl()(x)[0], parts, rtol=1e-12)


def test_gmsl_split_is_barystatic_then_steric(budget):
    x = _random(budget, 7)
    split = budget.gmsl_split(x)
    np.testing.assert_allclose(
        split,
        [budget.barystatic_gmsl()(x)[0], budget.steric_gmsl(x)[0]],
        rtol=1e-14,
        atol=0.0,
    )


# ==================================================================== #
#                     What the roles do not admit                      #
# ==================================================================== #


def test_set_up_without_ocean_components(state, load_space):
    setup = ModelSetup(state, [ice_sheets(state, load_space)])
    assert OCEAN_MASS not in setup.constraints
    assert setup.steric_gmsl is None and setup.gmsl_split is None
    assert set(setup.observables) == {ICE_SURFACE_HEIGHT}


def test_set_up_without_mass_sources(state, load_space):
    setup = ModelSetup(state, [sterodynamic_sea_level(state, load_space)])
    with pytest.raises(ValueError, match="no mass source"):
        setup.barystatic_gmsl()
    with pytest.raises(ValueError, match="not a mass source"):
        setup.barystatic_gmsl("sterodynamic_sea_level")
    assert setup.gmsl_split is None


def test_invalid_set_ups_are_refused(state, load_space):
    with pytest.raises(ValueError, match="at least one"):
        ModelSetup(state, [])
    with pytest.raises(ValueError, match="distinct"):
        ModelSetup(
            state,
            [ice_sheets(state, load_space), ice_sheets(state, load_space)],
        )
    other = lebesgue_load_space(state.model)
    with pytest.raises(ValueError, match="same space"):
        ModelSetup(
            state,
            [
                ice_sheets(state, load_space),
                sterodynamic_sea_level(state, other),
            ],
        )
