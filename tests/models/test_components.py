"""
Tests for the load components of pyslfp.models.
"""

import numpy as np
import pytest
from pygeoinf.testing import check_operator, check_traits

from pyslfp.core import EarthState
from pyslfp.core import (
    lebesgue_load_space,
    sobolev_load_space,
)
from pyslfp.models import steric_sea_level_operator
from pyslfp.models import (
    ICE_SURFACE_HEIGHT,
    GEOCENTRIC_SEA_LEVEL,
    STERIC_SEA_LEVEL,
    LoadComponent,
    Role,
    barystatic_gmsl_operator,
    sterodynamic_sea_level,
    firn,
    glaciers,
    ice_sheets,
    land_water_storage,
    ocean_density,
)

FIRN_DENSITY_FRACTION = 0.6


@pytest.fixture(scope="module")
def state():
    """A real state: the supports follow the Earth's geography."""
    return EarthState.from_defaults(lmax=16)


def _firn(state, load_space, **kwargs):
    density = FIRN_DENSITY_FRACTION * state.model.parameters.ice_density
    return firn(state, load_space, density=density, **kwargs)


FACTORIES = {
    "ice_sheets": ice_sheets,
    "glaciers": glaciers,
    "land_water_storage": land_water_storage,
    "firn": _firn,
    "sterodynamic_sea_level": sterodynamic_sea_level,
    "ocean_density": ocean_density,
}

ROLES = {
    "ice_sheets": Role.MASS_SOURCE,
    "glaciers": Role.MASS_SOURCE,
    "land_water_storage": Role.MASS_SOURCE,
    "firn": Role.MASS_SOURCE,
    "sterodynamic_sea_level": Role.OCEAN_INTERNAL,
    "ocean_density": Role.OCEAN_INTERNAL,
}


def _spaces(state, kind):
    """The load space and the component space for a case: both Lebesgue,
    both Sobolev, or, as the regularity of the physics has it, an H^2
    component loading an H^(1/4) space."""
    b = state.model.parameters.mean_sea_floor_radius
    lebesgue = lebesgue_load_space(state.model)
    sobolev = sobolev_load_space(state.model, 2.0, 0.1 * b)
    rough = sobolev_load_space(state.model, 0.25, 0.1 * b)
    return {
        "lebesgue": (lebesgue, lebesgue),
        "sobolev": (sobolev, sobolev),
        "mixed": (rough, sobolev),
    }[kind]


POINTWISE = {
    "ice_sheets": ICE_SURFACE_HEIGHT,
    "firn": ICE_SURFACE_HEIGHT,
    "sterodynamic_sea_level": GEOCENTRIC_SEA_LEVEL,
}


@pytest.mark.parametrize("kind", ["lebesgue", "sobolev", "mixed"])
@pytest.mark.parametrize("factory", list(FACTORIES))
def test_component_operators(state, factory, kind):
    """The load operator, the support and every direct contribution pass
    the adjoint check; the support's claimed traits hold; the role is the
    one the physics gives."""
    load_space, space = _spaces(state, kind)
    component = FACTORIES[factory](state, load_space, space=space)
    b = state.model.parameters.mean_sea_floor_radius
    measure = space.heat_measure(0.3 * b)
    rng = np.random.default_rng(0)

    assert component.role is ROLES[factory]
    assert component.load_space == load_space
    check_operator(component.to_load, rng=rng, measure=measure, rtol=1e-9)
    check_operator(component.support, rng=rng, measure=measure, rtol=1e-9)
    check_traits(component.support, rng=rng, measure=measure, rtol=1e-9)
    for operator in component.observables.values():
        check_operator(operator, rng=rng, measure=measure, rtol=1e-9)


@pytest.mark.parametrize("factory", list(FACTORIES))
def test_support_is_an_indicator(state, factory):
    """The support keeps a field on its support and removes it elsewhere."""
    load_space, space = _spaces(state, "lebesgue")
    component = FACTORIES[factory](state, load_space)
    ones = state.model.zero_grid()
    ones.data[:] = 1.0
    support = component.support(ones).data
    assert set(np.unique(support)) <= {0.0, 1.0}
    assert support.sum() > 0


@pytest.mark.parametrize("factory", list(FACTORIES))
def test_a_field_outside_the_support_loads_nothing(state, factory):
    """The fields are global and the loads carry the supports: a field
    living off a component's support loads nothing and has no steric sea
    level, while a pointwise contribution passes it through unmasked, the
    observation points carrying the support."""
    load_space, _ = _spaces(state, "lebesgue")
    component = FACTORIES[factory](state, load_space)
    field = load_space.random(rng=np.random.default_rng(7))
    outside = field - component.support(field)
    assert np.max(np.abs(outside.data)) > 0
    assert np.max(np.abs(component.to_load(outside).data)) == 0.0
    for observable, operator in component.observables.items():
        expected = outside.data if observable in POINTWISE.values() else 0.0
        np.testing.assert_array_equal(operator(outside).data, expected)


@pytest.mark.parametrize("factory", list(POINTWISE))
def test_pointwise_contributions_land_in_the_observable_space(state, factory):
    """Given an observable space of order above 1, a pointwise contribution
    is the field itself read into it, and passes the adjoint check there."""
    load_space, space = _spaces(state, "mixed")
    b = state.model.parameters.mean_sea_floor_radius
    observable_space = sobolev_load_space(state.model, 1.5, 0.1 * b)
    component = FACTORIES[factory](
        state, load_space, space=space, observable_space=observable_space
    )
    operator = component.observables[POINTWISE[factory]]
    assert operator.codomain == observable_space
    field = space.heat_measure(0.3 * b).sample(rng=np.random.default_rng(4))
    np.testing.assert_array_equal(operator(field).data, field.data)
    check_operator(
        operator,
        rng=np.random.default_rng(5),
        measure=space.heat_measure(0.3 * b),
        codomain_measure=observable_space.heat_measure(0.3 * b),
        rtol=1e-9,
    )


def test_ice_and_firn_load_with_their_own_densities(state):
    """The same thickness change loads in proportion to the densities."""
    load_space, _ = _spaces(state, "lebesgue")
    thickness = load_space.random(rng=np.random.default_rng(1))
    ice_load = ice_sheets(state, load_space).to_load(thickness)
    firn_load = _firn(state, load_space).to_load(thickness)
    np.testing.assert_allclose(
        firn_load.data, FIRN_DENSITY_FRACTION * ice_load.data, rtol=1e-12, atol=0.0
    )
    assert ICE_SURFACE_HEIGHT in ice_sheets(state, load_space).observables
    assert ICE_SURFACE_HEIGHT in _firn(state, load_space).observables


@pytest.mark.parametrize(
    "factory, density, loaded_fraction",
    [
        ("ice_sheets", "ice", "grounded"),
        ("firn", "firn", "grounded"),
        ("glaciers", "ice", "glacier area"),
        ("land_water_storage", "water", "all"),
    ],
)
def test_barystatic_gmsl_is_the_bathtub_answer(
    state, factory, density, loaded_fraction
):
    """A uniform change h everywhere moves the global mean sea level by
    -rho h (loaded area) / (rho_w A_O), the loaded area lying within the
    source's support, which the operator applies itself; computed here
    with the model's own quadrature. A glacier's whole area loads, ocean
    nodes included."""
    load_space, _ = _spaces(state, "lebesgue")
    component = FACTORIES[factory](state, load_space)
    params = state.model.parameters
    rho = {
        "ice": params.ice_density,
        "firn": FIRN_DENSITY_FRACTION * params.ice_density,
        "water": params.water_density,
    }[density]
    h = 0.01
    ones = state.model.zero_grid()
    ones.data[:] = 1.0
    support = component.support(ones)
    uniform = ones * h
    weight = {
        "grounded": state.one_minus_ocean_function,
        "glacier area": state.glacier_area_density(),
        "all": ones,
    }[loaded_fraction]
    loaded_area = state.model.integrate(support * weight)

    expected = -rho * h * loaded_area / (params.water_density * state.ocean_area)
    got = barystatic_gmsl_operator(state, component)(uniform)[0]
    assert loaded_area > 0
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("factory", ["sterodynamic_sea_level", "ocean_density"])
def test_ocean_internal_components_have_no_barystatic_change(state, factory):
    load_space, _ = _spaces(state, "lebesgue")
    with pytest.raises(ValueError, match="not a mass source"):
        barystatic_gmsl_operator(state, FACTORIES[factory](state, load_space))


def test_direct_contributions(state):
    """The sterodynamic sea level change adds itself, unmasked, to the
    geocentric sea level,
    and the density change contributes its steric sea level."""
    load_space, _ = _spaces(state, "lebesgue")
    field = load_space.random(rng=np.random.default_rng(2))

    eta = sterodynamic_sea_level(state, load_space)
    assert set(eta.observables) == {GEOCENTRIC_SEA_LEVEL}
    np.testing.assert_array_equal(
        eta.observables[GEOCENTRIC_SEA_LEVEL](field).data, field.data
    )

    rho = ocean_density(state, load_space)
    assert set(rho.observables) == {STERIC_SEA_LEVEL}
    steric = steric_sea_level_operator(state, load_space, load_space)
    np.testing.assert_allclose(
        rho.observables[STERIC_SEA_LEVEL](field).data,
        steric(field).data,
        rtol=1e-14,
        atol=0.0,
    )

    assert not land_water_storage(state, load_space).observables


def test_component_record_is_frozen(state):
    load_space, _ = _spaces(state, "lebesgue")
    component = ice_sheets(state, load_space)
    with pytest.raises(AttributeError):
        component.name = "other"
    with pytest.raises(TypeError):
        component.observables["new"] = component.support


def test_component_and_observable_spaces_must_share_the_load_grid(state):
    load_space, _ = _spaces(state, "lebesgue")
    coarser = lebesgue_load_space(EarthState.for_testing(8).model)
    with pytest.raises(ValueError, match="grid"):
        ice_sheets(state, load_space, space=coarser)
    with pytest.raises(ValueError, match="grid"):
        sterodynamic_sea_level(state, load_space, observable_space=coarser)


def test_record_checks_its_operators(state):
    load_space, _ = _spaces(state, "lebesgue")
    component = ice_sheets(state, load_space)
    other = lebesgue_load_space(EarthState.for_testing(8).model)
    with pytest.raises(ValueError, match="support"):
        LoadComponent(
            name="broken",
            space=other,
            to_load=None,
            support=component.support,
            role=Role.MASS_SOURCE,
        )
    with pytest.raises(ValueError, match="name"):
        LoadComponent(
            name="",
            space=load_space,
            to_load=component.to_load,
            support=component.support,
            role=Role.MASS_SOURCE,
        )
