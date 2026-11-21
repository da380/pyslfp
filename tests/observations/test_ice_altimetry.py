"""
Test suite for ice altimetry: the grounded ice, the observation points on it,
and the observation of the ice surface height change.
"""

import numpy as np
import pytest
import pygeoinf as inf
from pygeoinf.testing import check_operator

from pyslfp.core import EarthState, sobolev_load_space
from pyslfp.core.response import POTENTIAL_CHANGE, RELATIVE_SEA_LEVEL_CHANGE
from pyslfp.elastic import FingerPrintOperator
from pyslfp.models import ice_and_firn_air
from pyslfp.observations import (
    GraceObservationModel,
    IceAltimetryObservationModel,
    altimetry_point_operator,
    grounded_ice_altimetry_points,
    ice_altimetry_point_operator,
    on_grounded_ice,
)

LMAX = 32

# Places whose ice the default state resolves at this degree.
EAST_ANTARCTICA = (-80.0, 90.0)
WEST_ANTARCTICA = (-80.0, -110.0)
GREENLAND = (72.6, -38.5)
ROSS_ICE_SHELF = (-81.5, -175.0)
RONNE_ICE_SHELF = (-78.0, -60.0)
PACIFIC = (0.0, 180.0)
ELLESMERE_ISLAND = (80.0, -80.0)


# ==================================================================== #
#                          Fixtures                                    #
# ==================================================================== #


@pytest.fixture(scope="module")
def state():
    """A real state: the ice sheets, their shelves and the ocean."""
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
def setup(state, fingerprint, scale):
    return ice_and_firn_air(
        state,
        fingerprint.domain,
        space=sobolev_load_space(state.model, 2.0, scale),
        observable_space=fingerprint.field_space,
    )


@pytest.fixture(scope="module")
def points(state):
    return grounded_ice_altimetry_points(state, spacing=4.0)


@pytest.fixture(scope="module")
def model(setup, fingerprint, state, points):
    return IceAltimetryObservationModel(
        setup.ice_surface_height(fingerprint), state, points, noise=0.01
    )


@pytest.fixture(scope="module")
def grace(setup, fingerprint):
    """GRACE on the same model space, the potential change of the response."""
    observable = fingerprint.codomain.projection(POTENTIAL_CHANGE) @ setup.response(
        fingerprint
    )
    return GraceObservationModel(observable, fingerprint.model, 20)


def _measure(setup):
    b = setup.state.model.parameters.mean_sea_floor_radius
    return inf.GaussianMeasure.from_product(
        [c.space.heat_measure(0.3 * b) for c in setup.components], domain=setup.space
    )


def _field_values(space, field, points):
    return space.point_evaluation_operator(points, route="dense")(field)


# ==================================================================== #
#                  1. The grounded ice and its points                  #
# ==================================================================== #


def test_on_grounded_ice(state):
    """The interiors of the ice sheets are grounded ice, at either longitude
    convention; the floating shelves, the ocean and ice outside the two
    sheets are not."""
    grounded = [EAST_ANTARCTICA, WEST_ANTARCTICA, GREENLAND, (72.6, 321.5)]
    elsewhere = [ROSS_ICE_SHELF, RONNE_ICE_SHELF, PACIFIC, ELLESMERE_ISLAND]
    assert on_grounded_ice(state, grounded).tolist() == [True] * 4
    assert on_grounded_ice(state, elsewhere).tolist() == [False] * 4
    assert on_grounded_ice(
        state, [GREENLAND, EAST_ANTARCTICA], sheets="greenland"
    ).tolist() == [
        True,
        False,
    ]
    with pytest.raises(ValueError, match="latitudes"):
        on_grounded_ice(state, [(91.0, 0.0)])


def test_the_shelves_are_ice_but_not_grounded(state):
    """The points refused on the shelves are on the ice sheets' ice, so it is
    the flotation that refuses them."""
    shelves = [ROSS_ICE_SHELF, RONNE_ICE_SHELF]
    ice = state.ice_sheet_projection(value=0.0)
    values = ice.data[
        [np.argmin(np.abs(ice.lats() - lat)) for lat, _ in shelves],
        [np.argmin(np.abs(ice.lons() - lon % 360.0)) for _, lon in shelves],
    ]
    assert values.tolist() == [1.0, 1.0]
    assert not on_grounded_ice(state, shelves).any()


def test_grounded_ice_altimetry_points(state):
    """The points lie on the grounded ice of both sheets, within the coverage,
    and evenly: none is repeated, none crowds its neighbours towards the
    pole, and their number is the grounded area over the area of a cell."""
    spacing = 2.0
    points = grounded_ice_altimetry_points(state, spacing=spacing)
    assert isinstance(points, list) and isinstance(points[0], tuple)
    assert on_grounded_ice(state, points).all()
    latitudes = np.array([p[0] for p in points])
    assert latitudes.min() >= -88.0 and latitudes.max() <= 88.0
    assert (latitudes < -60.0).any() and (latitudes > 59.0).any()
    assert len(set(points)) == len(points)

    colatitude = np.radians(90.0 - latitudes)
    longitude = np.radians([p[1] for p in points])
    xyz = np.column_stack(
        [
            np.sin(colatitude) * np.cos(longitude),
            np.sin(colatitude) * np.sin(longitude),
            np.cos(colatitude),
        ]
    )
    separation = np.degrees(np.arccos(np.clip(xyz @ xyz.T, -1.0, 1.0)))
    np.fill_diagonal(separation, np.inf)
    assert separation.min() > 0.9 * spacing

    grounded = state.ice_sheet_projection(value=0.0, exclude_ice_shelves=True)
    radius = state.model.parameters.mean_sea_floor_radius
    cells = state.model.integrate(grounded) / (radius * np.radians(spacing)) ** 2
    assert 0.9 * cells < len(points) < 1.1 * cells


def test_points_follow_the_sheets_and_the_coverage(state):
    greenland = grounded_ice_altimetry_points(state, spacing=4.0, sheets="greenland")
    assert greenland and all(lat > 0.0 for lat, _ in greenland)
    north_of_80s = grounded_ice_altimetry_points(
        state, spacing=4.0, sheets="antarctic", latitude_min=-80.0
    )
    assert north_of_80s and min(lat for lat, _ in north_of_80s) >= -80.0
    with pytest.raises(ValueError, match="spacing"):
        grounded_ice_altimetry_points(state, spacing=0.0)


# ==================================================================== #
#                  2. Raw Math Operators                               #
# ==================================================================== #


def test_ice_altimetry_point_operator(fingerprint, state, points):
    """The response's part of the ice surface height is its vertical
    displacement at the points; the operator passes the adjoint check."""
    response_space = fingerprint.codomain
    A = ice_altimetry_point_operator(state, response_space, points)
    assert A.domain == response_space
    assert A.codomain == inf.EuclideanSpace(len(points))

    response = fingerprint.response_measure_for_testing().sample(
        rng=np.random.default_rng(1)
    )
    expected = _field_values(
        fingerprint.field_space, response.field("displacement"), points
    )
    np.testing.assert_allclose(A(response), expected, rtol=1e-12)
    check_operator(
        A,
        rng=np.random.default_rng(0),
        measure=fingerprint.response_measure_for_testing(),
        codomain_measure=inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0),
        trials=3,
        rtol=1e-10,
    )


def test_ice_and_ocean_altimetry_share_a_frame(fingerprint, state, points):
    """Over the ocean altimetry sees the relative sea level change plus the
    vertical displacement, over the ice the thickness changes plus the same
    displacement: in the centre-of-mass frame of the response, with its
    degree one. So the ocean's operator less the relative sea level change is
    the ice's, when no centrifugal height is added to the first."""
    response_space = fingerprint.codomain
    field_space = fingerprint.field_space
    load = fingerprint.load_measure_for_testing().sample(rng=np.random.default_rng(2))
    response = fingerprint(load)

    geocentric = altimetry_point_operator(
        state, response_space, points, remove_rotational_contribution=False
    )(response)
    relative = _field_values(
        field_space, response.field(RELATIVE_SEA_LEVEL_CHANGE), points
    )
    ice = ice_altimetry_point_operator(state, response_space, points)(response)
    np.testing.assert_allclose(
        ice, geocentric - relative, rtol=0.0, atol=1e-10 * np.max(np.abs(ice))
    )

    # The displacement's degree one is part of what both see.
    degree_one = np.where(field_space.degrees == 1, response.displacement.vertical, 0.0)
    part = _field_values(field_space, field_space.from_components(degree_one), points)
    assert np.max(np.abs(part)) > 1e-3 * np.max(np.abs(ice))


# ==================================================================== #
#                  3. The observation model                            #
# ==================================================================== #


def test_the_model_from_a_set_up(model, setup, fingerprint, points):
    """With a set-up's ice surface height as its observable, the model
    observes the ice and its firn together at the points, and passes the
    adjoint check."""
    assert model.model_space == setup.space
    assert model.data_space == inf.EuclideanSpace(len(points))
    assert model.points == points
    assert model.noise.domain == model.data_space
    x = _measure(setup).sample(rng=np.random.default_rng(4))
    expected = _field_values(
        fingerprint.field_space, setup.ice_surface_height(fingerprint)(x), points
    )
    np.testing.assert_allclose(model.forward_operator(x), expected, rtol=1e-12)
    check_operator(
        model.forward_operator,
        rng=np.random.default_rng(5),
        measure=_measure(setup),
        codomain_measure=model.noise,
        trials=3,
        rtol=1e-8,
    )


def test_the_matrix_free_model_agrees(model, setup, fingerprint, state, points):
    matrix_free = IceAltimetryObservationModel(
        setup.ice_surface_height(fingerprint), state, points, matrix_free=True
    )
    x = _measure(setup).sample(rng=np.random.default_rng(6))
    np.testing.assert_allclose(
        matrix_free.forward_operator(x), model.forward_operator(x), rtol=1e-9
    )


@pytest.mark.parametrize(
    "place", [ROSS_ICE_SHELF, RONNE_ICE_SHELF, PACIFIC, ELLESMERE_ISLAND]
)
def test_points_off_the_grounded_ice_are_refused(
    setup, fingerprint, state, points, place
):
    """Floating ice, the ocean and ice outside the ice sheets are refused, by
    the model and by the raw operator alike."""
    observable = setup.ice_surface_height(fingerprint)
    with pytest.raises(ValueError, match="grounded"):
        IceAltimetryObservationModel(observable, state, points[:3] + [place])
    with pytest.raises(ValueError, match="grounded"):
        ice_altimetry_point_operator(state, fingerprint.codomain, [place])


def test_the_sheets_restrict_the_points(setup, fingerprint, state):
    observable = setup.ice_surface_height(fingerprint)
    IceAltimetryObservationModel(observable, state, [GREENLAND], sheets="greenland")
    with pytest.raises(ValueError, match="grounded"):
        IceAltimetryObservationModel(
            observable, state, [EAST_ANTARCTICA], sheets="greenland"
        )
    with pytest.raises(ValueError, match="at least one"):
        IceAltimetryObservationModel(observable, state, [])


# ==================================================================== #
#                  4. The physics                                      #
# ==================================================================== #


def test_a_firn_change_is_seen_by_altimetry_alone(
    model, grace, setup, fingerprint, points
):
    """A change in firn air content raises the surface by itself at the
    points and does nothing else: no load, no sea level change, no
    displacement and no gravity signal, each exactly zero."""
    _, a = _measure(setup).sample(rng=np.random.default_rng(7))
    space = setup.space.subspace("firn_air_content")
    firn_only = (space.zero(), a)

    expected = _field_values(space, a, points)
    assert np.max(np.abs(expected)) > 0
    np.testing.assert_allclose(model.forward_operator(firn_only), expected, rtol=1e-12)
    assert np.max(np.abs(setup.load_operator(firn_only).data)) == 0.0
    response = setup.response(fingerprint)(firn_only)
    assert np.max(np.abs(response.field(RELATIVE_SEA_LEVEL_CHANGE).data)) == 0.0
    assert np.max(np.abs(response.field("displacement").data)) == 0.0
    assert np.max(np.abs(grace.forward_operator(firn_only))) == 0.0


def test_an_ice_change_is_seen_with_the_elastic_displacement(
    model, grace, setup, fingerprint, state, points
):
    """A change in ice thickness is seen as itself plus the vertical
    displacement of the response to its load, and by GRACE. A uniform
    thinning by a metre unloads the Earth, which rises under the ice by a few
    centimetres."""
    space = setup.space.subspace("ice_sheets")
    h, _ = _measure(setup).sample(rng=np.random.default_rng(8))
    ice_only = (h, space.zero())
    response = setup.response(fingerprint)(ice_only)
    displacement = _field_values(
        fingerprint.field_space, response.field("displacement"), points
    )
    thickness = _field_values(space, h, points)
    data = model.forward_operator(ice_only)
    np.testing.assert_allclose(
        data, thickness + displacement, rtol=0.0, atol=1e-12 * np.max(np.abs(data))
    )
    assert np.max(np.abs(displacement)) > 0
    assert np.max(np.abs(grace.forward_operator(ice_only))) > 0

    metre = 1.0 / state.model.parameters.length_scale
    ones = state.model.zero_grid()
    ones.data[:] = 1.0
    thinning = (space.from_components(space.to_components(-metre * ones)), space.zero())
    change = model.forward_operator(thinning)
    uplift = (change + metre) / metre
    assert np.all(uplift > 0.005)
    assert np.all(uplift < 0.15)
