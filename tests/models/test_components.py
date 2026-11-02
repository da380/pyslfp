"""
Tests for the load components of pyslfp.models.
"""

import numpy as np
import pytest
from pygeoinf.testing import check_operator, check_traits

from pyslfp.core import EarthState
from pyslfp.linear_operators.physics import lebesgue_load_space, sobolev_load_space
from pyslfp.linear_operators.loads import steric_sea_level_operator
from pyslfp.models import (
    ICE_SURFACE_HEIGHT,
    SEA_SURFACE_HEIGHT,
    STERIC_SEA_LEVEL,
    LoadComponent,
    Role,
    barystatic_gmsl_operator,
    dynamic_topography,
    firn,
    ice_sheets,
    land_water_storage,
    ocean_density,
)

FIRN_DENSITY_FRACTION = 0.6


@pytest.fixture(scope="module")
def state():
    return EarthState.for_testing(16)


def _firn(state, load_space, **kwargs):
    density = FIRN_DENSITY_FRACTION * state.model.parameters.ice_density
    return firn(state, load_space, density=density, **kwargs)


FACTORIES = {
    "ice_sheets": ice_sheets,
    "land_water_storage": land_water_storage,
    "firn": _firn,
    "dynamic_topography": dynamic_topography,
    "ocean_density": ocean_density,
}

ROLES = {
    "ice_sheets": Role.MASS_SOURCE,
    "land_water_storage": Role.MASS_SOURCE,
    "firn": Role.MASS_SOURCE,
    "dynamic_topography": Role.OCEAN_INTERNAL,
    "ocean_density": Role.OCEAN_INTERNAL,
}


def _spaces(state, kind):
    """The load space and the component space for a case: both Lebesgue,
    both Sobolev, or a Sobolev component loading a Lebesgue space."""
    b = state.model.parameters.mean_sea_floor_radius
    lebesgue = lebesgue_load_space(state.model)
    sobolev = sobolev_load_space(state.model, 2.0, 0.1 * b)
    return {
        "lebesgue": (lebesgue, lebesgue),
        "sobolev": (sobolev, sobolev),
        "mixed": (lebesgue, sobolev),
    }[kind]


@pytest.mark.parametrize("kind", ["lebesgue", "sobolev", "mixed"])
@pytest.mark.parametrize("factory", list(FACTORIES))
def test_component_operators(state, factory, kind):
    """The load operator, the support projection and every direct
    contribution pass the adjoint check; the projection's claimed traits
    hold; the role is the one the physics gives."""
    load_space, space = _spaces(state, kind)
    component = FACTORIES[factory](state, load_space, space=space)
    b = state.model.parameters.mean_sea_floor_radius
    measure = space.heat_measure(0.3 * b)
    rng = np.random.default_rng(0)

    assert component.role is ROLES[factory]
    assert component.load_space == load_space
    check_operator(component.to_load, rng=rng, measure=measure, rtol=1e-9)
    check_operator(component.projection, rng=rng, measure=measure, rtol=1e-9)
    check_traits(component.projection, rng=rng, measure=measure, rtol=1e-9)
    for operator in component.observables.values():
        check_operator(operator, rng=rng, measure=measure, rtol=1e-9)


@pytest.mark.parametrize("factory", list(FACTORIES))
def test_projection_selects_the_support(state, factory):
    """The projection keeps a field on its support and removes it elsewhere."""
    load_space, space = _spaces(state, "lebesgue")
    component = FACTORIES[factory](state, load_space)
    ones = state.model.zero_grid()
    ones.data[:] = 1.0
    support = component.projection(ones).data
    assert set(np.unique(support)) <= {0.0, 1.0}
    assert support.sum() > 0


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
        ("land_water_storage", "water", "all"),
    ],
)
def test_barystatic_gmsl_is_the_bathtub_answer(
    state, factory, density, loaded_fraction
):
    """A uniform change h over a mass source's support moves the global
    mean sea level by -rho h (loaded area) / (rho_w A_O), computed here
    with the model's own quadrature."""
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
    support = component.projection(ones)
    uniform = support * h
    weight = state.one_minus_ocean_function if loaded_fraction == "grounded" else ones
    loaded_area = state.model.integrate(support * weight)

    expected = -rho * h * loaded_area / (params.water_density * state.ocean_area)
    got = barystatic_gmsl_operator(state, component)(uniform)[0]
    assert loaded_area > 0
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("factory", ["dynamic_topography", "ocean_density"])
def test_ocean_internal_components_have_no_barystatic_change(state, factory):
    load_space, _ = _spaces(state, "lebesgue")
    with pytest.raises(ValueError, match="not a mass source"):
        barystatic_gmsl_operator(state, FACTORIES[factory](state, load_space))


def test_direct_contributions(state):
    """Dynamic topography adds itself to the sea surface height, and the
    density change contributes its steric sea level."""
    load_space, _ = _spaces(state, "lebesgue")
    field = load_space.random(rng=np.random.default_rng(2))

    eta = dynamic_topography(state, load_space)
    assert set(eta.observables) == {SEA_SURFACE_HEIGHT}
    np.testing.assert_array_equal(
        eta.observables[SEA_SURFACE_HEIGHT](field).data, field.data
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
        component.observables["new"] = component.projection


def test_component_space_must_share_the_load_grid(state):
    load_space, _ = _spaces(state, "lebesgue")
    coarser = lebesgue_load_space(EarthState.for_testing(8).model)
    with pytest.raises(ValueError, match="grid"):
        ice_sheets(state, load_space, space=coarser)


def test_record_checks_its_operators(state):
    load_space, _ = _spaces(state, "lebesgue")
    component = ice_sheets(state, load_space)
    other = lebesgue_load_space(EarthState.for_testing(8).model)
    with pytest.raises(ValueError, match="projection"):
        LoadComponent(
            name="broken",
            space=other,
            to_load=None,
            projection=component.projection,
            role=Role.MASS_SOURCE,
        )
    with pytest.raises(ValueError, match="name"):
        LoadComponent(
            name="",
            space=load_space,
            to_load=component.to_load,
            projection=component.projection,
            role=Role.MASS_SOURCE,
        )
