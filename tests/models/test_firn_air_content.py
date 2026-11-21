"""
Tests for the firn air content, the massless component of pyslfp.models,
and for the ice surface height a set-up builds with it.
"""

import numpy as np
import pytest
from pygeoinf import GaussianMeasure
from pygeoinf.testing import check_operator, check_traits

from pyslfp.core import EarthState, lebesgue_load_space, sobolev_load_space
from pyslfp.elastic import FingerPrintOperator
from pyslfp.models import (
    ICE_SURFACE_HEIGHT,
    OCEAN_MASS,
    LoadComponent,
    ModelSetup,
    Role,
    firn_air_content,
    ice_and_firn,
    ice_and_firn_air,
    ice_sheets,
    land_water_storage,
    sterodynamic_sea_level,
)

LMAX = 16
FIRN_DENSITY_FRACTION = 0.4


@pytest.fixture(scope="module")
def state():
    """A real state: the supports follow the Earth's geography."""
    return EarthState.from_defaults(lmax=LMAX)


@pytest.fixture(scope="module")
def scale(state):
    return 0.1 * state.model.parameters.mean_sea_floor_radius


@pytest.fixture(scope="module")
def fingerprint(state, scale):
    """Loads in H^(1/4), as their indicators allow, and a response one order
    above, where fields have point values."""
    return FingerPrintOperator(
        state, load_parameters=(0.25, scale), response_parameters=(1.25, scale)
    )


@pytest.fixture(scope="module")
def field_space(state, scale):
    """The components' space, where the priors live."""
    return sobolev_load_space(state.model, 2.0, scale)


@pytest.fixture(scope="module")
def setup(state, fingerprint, field_space):
    return ice_and_firn_air(
        state,
        fingerprint.domain,
        space=field_space,
        observable_space=fingerprint.field_space,
    )


def _measure(setup):
    b = setup.state.model.parameters.mean_sea_floor_radius
    return GaussianMeasure.from_product(
        [c.space.heat_measure(0.3 * b) for c in setup.components],
        domain=setup.space,
    )


def _sample(setup, seed):
    return _measure(setup).sample(rng=np.random.default_rng(seed))


def _displacement(setup, fingerprint):
    return (
        fingerprint.codomain.field_operator("displacement")
        @ fingerprint
        @ setup.load_operator
    )


# ==================================================================== #
#                           The component                              #
# ==================================================================== #


@pytest.mark.parametrize("kind", ["lebesgue", "sobolev", "mixed"])
def test_the_component_is_massless(state, scale, kind):
    """The firn air content has no load operator and so no load space; its
    support is the ice sheets', an idempotent mask, and its one direct
    contribution is to the ice surface height, each passing the adjoint
    check."""
    lebesgue = lebesgue_load_space(state.model)
    sobolev = sobolev_load_space(state.model, 2.0, scale)
    rough = sobolev_load_space(state.model, 0.25, scale)
    load_space, space = {
        "lebesgue": (lebesgue, lebesgue),
        "sobolev": (sobolev, sobolev),
        "mixed": (rough, sobolev),
    }[kind]
    component = firn_air_content(state, load_space, space=space)
    b = state.model.parameters.mean_sea_floor_radius
    measure = space.heat_measure(0.3 * b)
    rng = np.random.default_rng(0)

    assert component.name == "firn_air_content"
    assert component.role is Role.MASSLESS
    assert component.to_load is None
    assert component.load_space is None
    assert component.space == space
    assert set(component.observables) == {ICE_SURFACE_HEIGHT}
    check_operator(component.support, rng=rng, measure=measure, rtol=1e-9)
    check_traits(component.support, rng=rng, measure=measure, rtol=1e-9)
    check_operator(
        component.observables[ICE_SURFACE_HEIGHT], rng=rng, measure=measure, rtol=1e-9
    )


def test_the_support_is_the_ice_sheets_and_the_contribution_is_unmasked(state):
    """The support is that of the ice sheets, shelves included; the direct
    contribution passes a field through unmasked, the observation points
    carrying the support."""
    load_space = lebesgue_load_space(state.model)
    component = firn_air_content(state, load_space)
    ones = state.model.zero_grid()
    ones.data[:] = 1.0
    np.testing.assert_array_equal(
        component.support(ones).data, state.ice_sheet_projection(value=0.0).data
    )
    np.testing.assert_array_equal(
        component.support(ones).data, ice_sheets(state, load_space).support(ones).data
    )
    field = load_space.random(rng=np.random.default_rng(3))
    np.testing.assert_array_equal(
        component.observables[ICE_SURFACE_HEIGHT](field).data, field.data
    )


def test_the_contribution_lands_in_the_observable_space(state, scale, field_space):
    """Given an observable space of order above 1, the contribution is the
    field itself read into it."""
    load_space = sobolev_load_space(state.model, 0.25, scale)
    observable_space = sobolev_load_space(state.model, 1.5, scale)
    component = firn_air_content(
        state, load_space, space=field_space, observable_space=observable_space
    )
    operator = component.observables[ICE_SURFACE_HEIGHT]
    assert operator.codomain == observable_space
    b = state.model.parameters.mean_sea_floor_radius
    field = field_space.heat_measure(0.3 * b).sample(rng=np.random.default_rng(4))
    np.testing.assert_array_equal(operator(field).data, field.data)
    check_operator(
        operator,
        rng=np.random.default_rng(5),
        measure=field_space.heat_measure(0.3 * b),
        codomain_measure=observable_space.heat_measure(0.3 * b),
        rtol=1e-9,
    )


def test_a_massless_component_cannot_load(state):
    load_space = lebesgue_load_space(state.model)
    ice = ice_sheets(state, load_space)
    with pytest.raises(ValueError, match="massless"):
        LoadComponent(
            name="broken",
            space=load_space,
            to_load=ice.to_load,
            support=ice.support,
            role=Role.MASSLESS,
        )


def test_the_space_must_share_the_load_grid(state):
    load_space = lebesgue_load_space(state.model)
    coarser = lebesgue_load_space(EarthState.for_testing(8).model)
    with pytest.raises(ValueError, match="grid"):
        firn_air_content(state, load_space, space=coarser)
    with pytest.raises(ValueError, match="grid"):
        firn_air_content(state, load_space, observable_space=coarser)


# ==================================================================== #
#                             In a set-up                              #
# ==================================================================== #


def test_the_ready_made_set_up(setup, fingerprint, field_space):
    assert setup.names == ("ice_sheets", "firn_air_content")
    assert [c.role for c in setup.components] == [Role.MASS_SOURCE, Role.MASSLESS]
    assert setup.load_space == fingerprint.domain
    assert setup.space.subspace("firn_air_content") == field_space
    assert set(setup.observables) == {ICE_SURFACE_HEIGHT}
    assert OCEAN_MASS not in setup.constraints
    assert setup.steric_gmsl is None and setup.gmsl_split is None


def test_the_firn_air_content_carries_no_mass(setup, fingerprint):
    """A change in firn air content alone loads nothing, exactly, so the
    response to it vanishes and it has no barystatic sea level change; the
    set-up's barystatic change is the ice's."""
    h, a = _sample(setup, 1)
    zero = setup.space.subspace("ice_sheets").zero()
    firn_only = (zero, a)
    assert np.max(np.abs(a.data)) > 0
    assert np.max(np.abs(setup.load_operator(firn_only).data)) == 0.0
    response = setup.response(fingerprint)(firn_only)
    assert np.max(np.abs(fingerprint.codomain.to_components(response))) == 0.0
    assert setup.barystatic_gmsl()(firn_only)[0] == 0.0
    with pytest.raises(ValueError, match="not a mass source"):
        setup.barystatic_gmsl("firn_air_content")
    np.testing.assert_allclose(
        setup.barystatic_gmsl()((h, a)),
        setup.barystatic_gmsl("ice_sheets")((h, a)),
        rtol=1e-14,
    )


def test_a_massless_component_stays_out_of_the_ocean_mass_constraint(
    state, fingerprint, field_space
):
    """Beside ocean-internal components, the firn air content enters neither
    the load nor the ocean mass constraint."""
    load_space = fingerprint.domain
    setup = ModelSetup(
        state,
        [
            ice_sheets(state, load_space, space=field_space),
            firn_air_content(state, load_space, space=field_space),
            sterodynamic_sea_level(state, load_space, space=field_space),
        ],
    )
    x = _sample(setup, 2)
    firn_only = (field_space.zero(), x[1], field_space.zero())
    assert setup.constraints[OCEAN_MASS](firn_only)[0] == 0.0
    assert np.max(np.abs(setup.load_operator(firn_only).data)) == 0.0
    check_operator(
        setup.load_operator,
        rng=np.random.default_rng(3),
        measure=_measure(setup),
        codomain_measure=load_space.heat_measure(
            0.3 * state.model.parameters.mean_sea_floor_radius
        ),
        rtol=1e-9,
    )


# ==================================================================== #
#                       The ice surface height                         #
# ==================================================================== #


def test_the_ice_surface_height(setup, fingerprint, state):
    """The ice surface height, the observable of ice altimetry, is the ice
    thickness change plus the firn air content change plus the vertical
    displacement of the response to the set-up's load; it passes the adjoint
    check."""
    x = _sample(setup, 4)
    h, a = x
    displacement = _displacement(setup, fingerprint)(x)
    height = setup.ice_surface_height(fingerprint)(x)
    assert setup.ice_surface_height(fingerprint).codomain == fingerprint.field_space
    scale = np.max(np.abs(height.data))
    np.testing.assert_allclose(
        height.data,
        h.data + a.data + displacement.data,
        rtol=0.0,
        atol=1e-13 * scale,
    )
    b = state.model.parameters.mean_sea_floor_radius
    check_operator(
        setup.ice_surface_height(fingerprint),
        rng=np.random.default_rng(5),
        measure=_measure(setup),
        codomain_measure=fingerprint.field_space.heat_measure(0.3 * b),
        trials=3,
        rtol=1e-8,
    )


def test_the_ice_surface_height_shares_the_frame_of_the_geocentric_sea_level(
    state, fingerprint, field_space
):
    """Altimetry over the ocean and over the ice see one solid surface: with
    no centrifugal height added, the geocentric sea level change less the
    relative sea level change is the displacement the ice surface height
    holds, degree one included."""
    load_space = fingerprint.domain
    on_fields = {"space": field_space, "observable_space": fingerprint.field_space}
    setup = ModelSetup(
        state,
        [
            ice_sheets(state, load_space, **on_fields),
            firn_air_content(state, load_space, **on_fields),
            sterodynamic_sea_level(state, load_space, **on_fields),
        ],
    )
    x = _sample(setup, 6)
    h, a, _ = x
    ocean_side = fingerprint.field_space.subtract(
        setup.geocentric_sea_level(fingerprint, remove_rotational_contribution=False)(
            x
        ),
        setup.relative_sea_level_change(fingerprint)(x),
    )
    ice_side = setup.ice_surface_height(fingerprint)(x)
    ice_side = fingerprint.field_space.subtract(
        ice_side, fingerprint.field_space.add(h, a)
    )
    scale = np.max(np.abs(ocean_side.data))
    np.testing.assert_allclose(
        ice_side.data, ocean_side.data, rtol=0.0, atol=1e-10 * scale
    )
    space = fingerprint.field_space
    degree_one = space.to_components(ice_side)[space.degrees == 1]
    assert np.max(np.abs(degree_one)) > 1e-3 * np.max(
        np.abs(space.to_components(ice_side))
    )


def test_the_contributions_must_lie_in_the_field_space(state, fingerprint, field_space):
    mismatched = ice_and_firn_air(state, fingerprint.domain, space=field_space)
    with pytest.raises(ValueError, match="field space"):
        mismatched.ice_surface_height(fingerprint)
    other = FingerPrintOperator(state)
    with pytest.raises(ValueError, match="load space"):
        mismatched.ice_surface_height(other)


def test_without_ice_the_surface_height_is_the_displacement(
    state, fingerprint, field_space
):
    """A set-up with no contribution to the ice surface height leaves the
    displacement of the solid surface alone."""
    setup = ModelSetup(
        state, [land_water_storage(state, fingerprint.domain, space=field_space)]
    )
    x = _sample(setup, 7)
    np.testing.assert_array_equal(
        setup.ice_surface_height(fingerprint)(x).data,
        _displacement(setup, fingerprint)(x).data,
    )


def test_the_two_descriptions_of_firn_agree(state, fingerprint, field_space):
    """An ice thickness I with a firn thickness F of density rho_f is the
    ice-equivalent thickness h = I + (rho_f / rho_i) F with the firn air
    content a = (1 - rho_f / rho_i) F: the same load, and the same ice
    surface height."""
    load_space = fingerprint.domain
    on_fields = {"space": field_space, "observable_space": fingerprint.field_space}
    fraction = FIRN_DENSITY_FRACTION
    density = fraction * state.model.parameters.ice_density
    with_firn = ice_and_firn(state, load_space, firn_density=density, **on_fields)
    with_air = ice_and_firn_air(state, load_space, **on_fields)
    ice, snow = _sample(with_firn, 8)
    h = field_space.add(ice, field_space.scale(fraction, snow))
    a = field_space.scale(1.0 - fraction, snow)

    load = with_firn.load_operator((ice, snow))
    np.testing.assert_allclose(
        with_air.load_operator((h, a)).data,
        load.data,
        rtol=0.0,
        atol=1e-13 * np.max(np.abs(load.data)),
    )
    height = with_firn.ice_surface_height(fingerprint)((ice, snow))
    np.testing.assert_allclose(
        with_air.ice_surface_height(fingerprint)((h, a)).data,
        height.data,
        rtol=0.0,
        atol=1e-12 * np.max(np.abs(height.data)),
    )
