"""
Tests for the viscoelastic earth models: a Maxwell mantle on a radial
model, PREM's two-layer structure, and the solver objects built from them.
"""

import numpy as np
import pytest
from planetmodel import PREM, units

import pyslfp as sl
from pyslfp.love_numbers import Material, graded_mesh
from pyslfp.viscoelastic import (
    Rheology,
    maxwell_mantle,
    prem_maxwell,
    viscoelastic_earth,
)

LMAX = 16


@pytest.fixture(scope="module")
def prem():
    return PREM(ocean=False).isotropic()


def _viscosity(layer):
    """A layer's viscosity at its midpoint, or None."""
    if "viscosity" not in layer.fields:
        return None
    return float(layer.fields["viscosity"].evaluate(0.5 * sum(layer.interval), 0, 0))


def test_mantle_between_the_core_and_the_lithosphere(prem):
    """The viscosity fills the solid layers between the outer core and the
    lithosphere's base, piecewise constant between the boundaries; a base
    that is not an interface becomes one; the cores and the lithosphere
    stay elastic."""
    lithosphere = 6271.0e3
    model = maxwell_mantle(
        prem,
        (3.0e21, 1.0e21, 4.0e20),
        boundaries=(5701.0e3, 6000.0e3),
        lithosphere=lithosphere,
    )
    assert np.any(np.isclose(model.skeleton.boundaries, lithosphere))
    assert np.any(np.isclose(model.skeleton.boundaries, 6000.0e3))
    for layer in model.layers:
        lo, hi = layer.interval
        eta = _viscosity(layer)
        if hi <= 3480.0e3 or lo >= lithosphere:
            assert eta is None
        elif hi <= 5701.0e3:
            assert eta == 3.0e21
        elif hi <= 6000.0e3:
            assert eta == 1.0e21
        else:
            assert eta == 4.0e20
    assert type(model) is type(prem)


def test_no_lithosphere(prem):
    """A lithosphere at the outer radius leaves every solid layer above the
    core viscous."""
    outer = prem.layers[-1].interval[1]
    model = maxwell_mantle(prem, 1.0e21, lithosphere=outer)
    viscous = [layer for layer in model.layers if _viscosity(layer) is not None]
    assert viscous[0].interval[0] == 3480.0e3
    assert viscous[-1].interval[1] == outer
    assert model.nlayers == prem.nlayers


def test_existing_viscosity_is_replaced(prem):
    first = maxwell_mantle(prem, 1.0e21, lithosphere=6291.0e3)
    second = maxwell_mantle(first, 2.0e21, lithosphere=6151.0e3)
    values = {_viscosity(layer) for layer in second.layers} - {None}
    assert values == {2.0e21}
    assert _viscosity(second.layer("low_velocity_zone")) is None


@pytest.mark.parametrize(
    "viscosity, boundaries, lithosphere, match",
    [
        ((1.0e21, 2.0e21), (), 6291.0e3, "intervals"),
        (-1.0e21, (), 6291.0e3, "positive"),
        ((1.0e21, 2.0e21), (6300.0e3,), 6291.0e3, "increase"),
        ((1.0e21, 2.0e21), (3000.0e3,), 6291.0e3, "increase"),
        (1.0e21, (), 7000.0e3, "increase"),
    ],
)
def test_invalid_structures_are_refused(
    prem, viscosity, boundaries, lithosphere, match
):
    with pytest.raises(ValueError, match=match):
        maxwell_mantle(prem, viscosity, boundaries=boundaries, lithosphere=lithosphere)


def test_prem_maxwell_is_the_two_layer_reference(prem):
    """PREM without Q, an elastic lid, 5e20 Pa s above the 670 km
    discontinuity and 2e21 Pa s below."""
    model = prem_maxwell()
    assert "qmu" not in model.field_names() and "qkappa" not in model.field_names()
    assert model.nlayers == prem.nlayers
    assert _viscosity(model.layer("lower_mantle")) == 2.0e21
    assert _viscosity(model.layer("transition_zone_lower")) == 5.0e20
    assert _viscosity(model.layer("low_velocity_zone")) == 5.0e20
    assert _viscosity(model.layer("lid")) is None
    assert _viscosity(model.layer("outer_core")) is None


def test_viscosity_converts_with_the_model():
    """The viscosity is given in Pa s and reaches the library's frame with the
    model, as the tutorials once converted it by hand."""
    scales = sl.EarthModelParameters.from_defaults().scales
    converted = prem_maxwell().converted(scales)
    expected = 2.0e21 / scales.factor(units.VISCOSITY)
    np.testing.assert_allclose(
        _viscosity(converted.layer("lower_mantle")), expected, rtol=1e-14
    )


def test_viscoelastic_earth_reproduces_the_hand_built_set_up():
    """The Earth model and rheology from `prem_maxwell` are those the
    tutorials built by hand: PREM converted, meshed, a Maxwell body on
    layers 2-4 and 5-8, the elastic Love numbers on the same mesh."""
    scales = sl.EarthModelParameters.from_defaults().scales
    model = PREM(ocean=False).isotropic().converted(scales)
    mesh = graded_mesh(model, LMAX)
    pascal_second = 1.0 / scales.factor(units.VISCOSITY)
    viscosity = {i: 2.0e21 * pascal_second for i in (2, 3, 4)}
    viscosity.update({i: 5.0e20 * pascal_second for i in (5, 6, 7, 8)})
    by_hand = Rheology.maxwell(Material(mesh, model), viscosity)
    earth_by_hand = sl.EarthModel.from_planet_model(model, LMAX, mesh=mesh)

    earth, rheology = viscoelastic_earth(prem_maxwell(), LMAX)
    np.testing.assert_array_equal(rheology.material.mesh.r, mesh.r)
    for got, want in zip(rheology.itau + rheology.mu, by_hand.itau + by_hand.mu):
        np.testing.assert_allclose(got, want, rtol=1e-14, atol=0.0)
    for name in ("h", "k"):
        np.testing.assert_allclose(
            getattr(earth.love_numbers, name),
            getattr(earth_by_hand.love_numbers, name),
            rtol=1e-14,
            atol=0.0,
        )
