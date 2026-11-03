"""
Tests for fields on the sphere through time: the space's axioms and its
product metric, the field at a time, and evaluation at a time, bounded above
order one half in time and free of an end condition at the ends of the
interval.
"""

import numpy as np
import pytest
from pygeoinf.interval import Lebesgue as TimeLebesgue
from pygeoinf.interval import Sobolev as TimeSobolev
from pygeoinf.testing import check_coordinates, check_operator, check_space

from pyslfp.core import EarthState, FieldHistory, FieldHistorySpace
from pyslfp.linear_operators.physics import lebesgue_load_space, sobolev_load_space

LMAX = 8
START, END = 0.0, 26.0
LENGTH = 2.0


@pytest.fixture(scope="module")
def state():
    return EarthState.for_testing(LMAX)


def _time_space(kind, modes=16, **options):
    if kind == "lebesgue":
        return TimeLebesgue(
            modes, lower=START, upper=END, length_scale=LENGTH, **options
        )
    return TimeSobolev(modes, 1.0, LENGTH, lower=START, upper=END, **options)


def _field_space(state, kind):
    b = state.model.parameters.mean_sea_floor_radius
    if kind == "lebesgue":
        return lebesgue_load_space(state.model)
    return sobolev_load_space(state.model, 2.0, 0.3 * b)


def _random(space, seed):
    rng = np.random.default_rng(seed)
    return space.from_components(rng.standard_normal(space.dim))


@pytest.mark.parametrize("time_kind", ["lebesgue", "sobolev"])
@pytest.mark.parametrize("field_kind", ["lebesgue", "sobolev"])
def test_history_space_axioms(state, time_kind, field_kind):
    time_space = _time_space(time_kind)
    field_space = _field_space(state, field_kind)
    space = FieldHistorySpace(time_space, field_space)
    rng = np.random.default_rng(0)
    check_space(
        space, rng=rng, rebuild=lambda: FieldHistorySpace(time_space, field_space)
    )
    check_coordinates(space, rng=rng)
    assert space.shape == (time_space.dim, field_space.dim)
    assert space.dim == time_space.dim * field_space.dim


def test_the_metric_is_the_product(state):
    """For rank-one histories a(t) f(x) and b(t) g(x) the inner product is
    (a, b) (f, g), the norm of the tensor product of the two spaces."""
    time_space = _time_space("sobolev")
    field_space = _field_space(state, "sobolev")
    space = FieldHistorySpace(time_space, field_space)
    a, b = _random(time_space, 1), _random(time_space, 2)
    f, g = _random(field_space, 3), _random(field_space, 4)
    u = space.from_rank_one(a, f)
    v = space.from_rank_one(b, g)
    expected = time_space.inner_product(a, b) * field_space.inner_product(f, g)
    np.testing.assert_allclose(space.inner_product(u, v), expected, rtol=1e-12)


def test_the_field_at_a_time(state):
    """At a node of the time mesh, a rank-one history a(t) f(x) is the
    profile's nodal value times the field."""
    time_space = _time_space("sobolev")
    field_space = _field_space(state, "lebesgue")
    space = FieldHistorySpace(time_space, field_space)
    a = time_space.from_components(np.random.default_rng(5).standard_normal(16))
    f = _random(field_space, 6)
    u = space.from_rank_one(a, f)
    for node in time_space.interior_nodes[[0, 7, -1]]:
        value = a[np.argmin(np.abs(time_space.nodes - node))]
        np.testing.assert_allclose(
            u.at(float(node)).data, value * f.data, rtol=1e-10, atol=1e-14
        )
    assert isinstance(u, FieldHistory)


@pytest.mark.parametrize("time_kind", ["lebesgue", "sobolev"])
def test_the_evaluation_operator(state, time_kind):
    space = FieldHistorySpace(_time_space(time_kind), _field_space(state, "sobolev"))
    for time in (START, 9.3, END):
        check_operator(space.evaluation_operator(time), rng=np.random.default_rng(7))
    with pytest.raises(ValueError):
        space.evaluation_operator(END + 1.0)


def _representer_ratio(space, time, field):
    """|E_t^* y|^2 / |y|^2 for the evaluation at a time: sum phi_k(t)^2 / mu_k,
    whatever the field."""
    operator = space.evaluation_operator(time)
    representer = operator.adjoint(field)
    return space.squared_norm(representer) / space.field_space.squared_norm(field)


def test_evaluation_is_bounded_above_order_one_half(state):
    """With the time order one, the representer of evaluation at a time
    converges as time modes are added, to the Green's function of
    1 - L^2 d^2/dt^2 at zero lag, 1 / (2 L); in L2 it grows with the modes."""
    field_space = _field_space(state, "lebesgue")
    field = _random(field_space, 8)
    sobolev, lebesgue = [], []
    for modes in (32, 64, 128):
        for kind, values in (("sobolev", sobolev), ("lebesgue", lebesgue)):
            space = FieldHistorySpace(_time_space(kind, modes), field_space)
            values.append(_representer_ratio(space, 13.0, field))
    changes = np.diff(sobolev)
    assert changes[1] < 0.6 * changes[0]
    assert abs(sobolev[-1] - 1.0 / (2.0 * LENGTH)) < 0.05 / (2.0 * LENGTH)
    assert lebesgue[-1] > 3.5 * lebesgue[0]


def test_the_padding_keeps_the_ends_free(state):
    """The representer of evaluation at either end of the interval is that
    in its middle, the padding keeping an end condition's mark out of the
    interval; without padding, under the natural condition, an end reflects
    and doubles it."""
    field_space = _field_space(state, "lebesgue")
    field = _random(field_space, 9)
    padded = FieldHistorySpace(_time_space("sobolev", 64), field_space)
    middle = _representer_ratio(padded, 13.0, field)
    for end in (START, END):
        assert abs(_representer_ratio(padded, end, field) / middle - 1.0) < 0.02
    bare = FieldHistorySpace(
        _time_space("sobolev", 64, padding=0.0, boundary=None), field_space
    )
    ratio = _representer_ratio(bare, START, field) / _representer_ratio(
        bare, 13.0, field
    )
    assert ratio > 1.8


def test_a_space_without_a_diagonal_metric_is_refused(state):
    class _Dense:
        has_diagonal_metric = False

    with pytest.raises(ValueError):
        FieldHistorySpace(_Dense(), _field_space(state, "lebesgue"))
