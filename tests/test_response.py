"""
Tests for the response to a load: the surface displacement and its space,
the response and its space, and their meeting with the sea level solver.
"""

import numpy as np
import pytest
from pygeoinf import EuclideanSpace, GaussianMeasure
from pygeoinf.testing import check_coordinates, check_operator, check_space

from pyslfp.core import EarthState, SeaLevelEquation
from pyslfp.core import (
    lebesgue_load_space,
    sobolev_load_space,
)
from pyslfp.core.response import (
    ANGULAR_VELOCITY_CHANGE,
    DISPLACEMENT,
    POTENTIAL_CHANGE,
    RELATIVE_SEA_LEVEL_CHANGE,
    GRDSpace,
    SurfaceDisplacementSpace,
)

LMAX = 16


@pytest.fixture(scope="module")
def state():
    return EarthState.for_testing(LMAX)


def _field_space(state, kind):
    b = state.model.parameters.mean_sea_floor_radius
    if kind == "lebesgue":
        return lebesgue_load_space(state.model)
    return sobolev_load_space(state.model, 2.0, 0.1 * b)


def _smooth(space, seed):
    """A smooth random vector of a coordinate space with a diagonal metric:
    components decaying with degree."""
    rng = np.random.default_rng(seed)
    return space.from_components(rng.standard_normal(space.dim) / space.metric_values)


@pytest.mark.parametrize("kind", ["lebesgue", "sobolev"])
@pytest.mark.parametrize("horizontal", [False, True])
def test_displacement_space_axioms(state, kind, horizontal):
    field_space = _field_space(state, kind)
    space = SurfaceDisplacementSpace(field_space, horizontal=horizontal)
    rng = np.random.default_rng(0)
    check_space(
        space,
        rng=rng,
        rebuild=lambda: SurfaceDisplacementSpace(field_space, horizontal=horizontal),
    )
    check_coordinates(space, rng=rng)
    assert space.dim == field_space.dim + (field_space.dim - 1 if horizontal else 0)


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("rotation", [False, True])
def test_response_space_axioms(state, horizontal, rotation):
    field_space = _field_space(state, "sobolev")
    space = GRDSpace(field_space, horizontal=horizontal, rotation=rotation)
    rng = np.random.default_rng(1)
    check_space(
        space,
        rng=rng,
        rebuild=lambda: GRDSpace(field_space, horizontal=horizontal, rotation=rotation),
    )
    check_coordinates(space, rng=rng)
    expected = (RELATIVE_SEA_LEVEL_CHANGE, DISPLACEMENT, POTENTIAL_CHANGE) + (
        (ANGULAR_VELOCITY_CHANGE,) if rotation else ()
    )
    assert space.names == expected


def test_the_displacement_norm_is_that_of_the_field(state):
    """In L2 the norm is the surface integral of |u|^2 = U^2 + |grad_1 V|^2,
    computed here on the grid from the east and north components."""
    field_space = _field_space(state, "lebesgue")
    space = SurfaceDisplacementSpace(field_space, horizontal=True)
    u = _smooth(space, 2)
    east, north = u.horizontal_fields()
    vertical = u.vertical_field()
    integral = state.model.integrate(vertical * vertical + east * east + north * north)
    np.testing.assert_allclose(space.squared_norm(u), integral, rtol=1e-10)


@pytest.mark.parametrize("horizontal", [False, True])
def test_point_values_are_those_of_the_grids(state, horizontal):
    """At grid nodes away from the poles, the point evaluation gives the
    grids' values; without the horizontal part east and north are zero."""
    field_space = _field_space(state, "lebesgue")
    space = SurfaceDisplacementSpace(field_space, horizontal=horizontal)
    u = _smooth(space, 3)
    lats, lons = state.lats(), state.lons()
    rows = [3, LMAX, 2 * LMAX - 2]
    columns = [0, 5, 2 * LMAX + 1]
    points = [(lats[i], lons[j]) for i in rows for j in columns]
    values = u.at(points)
    east, north = u.horizontal_fields()
    up = u.vertical_field()
    expected = np.array(
        [
            [east.data[i, j], north.data[i, j], up.data[i, j]]
            for i in rows
            for j in columns
        ]
    )
    np.testing.assert_allclose(
        values, expected, rtol=0.0, atol=1e-10 * np.max(np.abs(expected))
    )
    if not horizontal:
        assert np.all(values[:, :2] == 0.0)


@pytest.mark.parametrize("kind", ["lebesgue", "sobolev"])
@pytest.mark.parametrize(
    "directions", [("east", "north", "up"), ("north",), ("up", "east")]
)
def test_point_evaluation_passes_the_adjoint_check(state, kind, directions):
    field_space = _field_space(state, kind)
    space = SurfaceDisplacementSpace(field_space, horizontal=True)
    points = [(10.0, 20.0), (-45.0, 300.0), (70.0, 135.0), (0.0, 0.0)]
    operator = space.point_evaluation_operator(points, directions=directions)
    assert operator.codomain == EuclideanSpace(len(points) * len(directions))
    rng = np.random.default_rng(4)
    check_operator(
        operator,
        rng=rng,
        codomain_measure=GaussianMeasure.from_standard_deviation(
            operator.codomain, 1.0
        ),
        rtol=1e-9,
    )
    with pytest.raises(ValueError, match="poles"):
        space.point_evaluation_operator([(90.0, 0.0)])
    space.point_evaluation_operator([(90.0, 0.0)], directions=("up",))


@pytest.mark.parametrize(
    "name",
    [
        RELATIVE_SEA_LEVEL_CHANGE,
        DISPLACEMENT,
        POTENTIAL_CHANGE,
        ANGULAR_VELOCITY_CHANGE,
    ],
)
def test_projections_and_inclusions_are_adjoint(state, name):
    space = GRDSpace(_field_space(state, "sobolev"), horizontal=True)
    rng = np.random.default_rng(5)
    projection = space.projection(name)
    check_operator(projection, rng=rng, rtol=1e-12)
    x = _smooth(space, 6)
    member = space.member_space(name)
    once = projection(x)
    twice = projection(space.inclusion(name)(once))
    np.testing.assert_array_equal(
        member.to_components(twice), member.to_components(once)
    )


def test_the_solver_coefficients_make_a_response(state):
    """The engine's coefficients, held as a response, give back the solver's
    grids: the sea level change, the vertical displacement and the potential."""
    model = state.model
    sle = SeaLevelEquation(model)
    load = (
        _field_space(state, "lebesgue")
        .heat_measure(0.3 * model.parameters.mean_sea_floor_radius)
        .sample(rng=np.random.default_rng(7))
    )
    solution = sle._solve_generalised(state, direct_load=load, horizontal=True)
    grids = sle.solve_generalised_equation(state, direct_load=load)
    space = GRDSpace(_field_space(state, "lebesgue"), horizontal=True)
    response = space.from_coefficients(
        solution.sea_level_lm,
        solution.displacement_lm,
        solution.potential_lm,
        horizontal=solution.horizontal_lm,
        angular_velocity_change=solution.angular_velocity_change,
    )
    for name, grid in zip(
        (RELATIVE_SEA_LEVEL_CHANGE, DISPLACEMENT, POTENTIAL_CHANGE), grids[:3]
    ):
        got = response.field(name).data
        np.testing.assert_allclose(
            got, grid.data, rtol=0.0, atol=1e-11 * np.max(np.abs(grid.data))
        )
    np.testing.assert_array_equal(response.angular_velocity_change, grids[3])
