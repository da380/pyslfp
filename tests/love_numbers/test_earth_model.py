"""EarthModel built on Love numbers: the table as the source of the body,
the constructor's ways of taking a table, and the adjoint identity of the
fingerprint operator, which is exact only with a consistent body."""

import numpy as np
import pytest
from planetmodel import PREM, frozen, units

from pyslfp import EarthModel, EarthModelParameters, EarthState
from pyslfp.ice import AnalyticalIceModel
from pyslfp.linear_operators import FingerPrintOperator
from pyslfp.love_numbers import love_numbers

LMAX = 16


@pytest.fixture(scope="module")
def table():
    return love_numbers(PREM(ocean=False), LMAX)


def _state(model: EarthModel) -> EarthState:
    ice = AnalyticalIceModel(length_scale=model.parameters.length_scale)
    h, sl = ice.get_ice_thickness_and_sea_level(
        0.0,
        model.lmax,
        grid=model.grid_name,
        sampling=model.sampling,
        extend=model.extend,
    )
    return EarthState(h, sl, model, exclude_caspian=False)


def _adjoint_mismatch(model: EarthModel, *, seed: int = 0) -> float:
    op = FingerPrintOperator(_state(model), rotational_feedbacks=True, rtol=1e-12)
    mu_x = op.load_measure_for_testing()
    mu_y = op.response_measure_for_testing()
    worst = 0.0
    for _ in range(3):
        x, y = mu_x.sample(), mu_y.sample()
        lhs = op.codomain.inner_product(op(x), y)
        rhs = op.domain.inner_product(x, op.adjoint(y))
        worst = max(worst, abs(lhs - rhs) / abs(lhs))
    return worst


def test_default_model_takes_the_tables_body(table):
    """The default table is the solver's own for PREM, so the default model
    carries PREM's surface gravity and planetmodel's G, and its Love
    numbers are those computed here, to the discretisation error of the
    coarser mesh sized for degree 16 alone."""
    model = EarthModel(LMAX)
    p = model.parameters
    assert np.isclose(p.raw_gravitational_constant, units.G_SI)
    assert np.isclose(p.raw_gravitational_acceleration, table.surface_gravity)
    assert np.isclose(p.raw_mean_sea_floor_radius, table.radius)
    assert np.isclose(
        p.gravitational_constant,
        units.G_SI / p.scales.factor(units.GRAVITATIONAL_CONSTANT),
    )
    assert model.love_numbers.lmax == LMAX
    assert np.isclose(model.love_numbers.G, p.gravitational_constant)
    computed = EarthModel(LMAX, love_numbers=table)
    for name in ("h", "k", "l", "h_t", "k_t", "h_v"):
        assert np.allclose(
            getattr(model.love_numbers, name),
            getattr(computed.love_numbers, name),
            rtol=5e-6,
            atol=1e-13,
        ), name


def test_table_and_file_give_the_same_model(tmp_path, table):
    path = tmp_path / "prem.dat"
    table.write(path)
    a = EarthModel(LMAX, love_numbers=table)
    b = EarthModel(LMAX, love_numbers=path)
    c = EarthModel(LMAX, love_number_file=str(path))
    for other in (b, c):
        assert np.allclose(other.love_numbers.h, a.love_numbers.h, rtol=1e-12)
        assert np.allclose(other.love_numbers.k_t, a.love_numbers.k_t, rtol=1e-12)
        assert other.parameters == a.parameters
    p = a.parameters
    assert np.isclose(p.raw_gravitational_constant, units.G_SI)
    assert np.isclose(p.raw_gravitational_acceleration, table.surface_gravity)
    assert np.isclose(p.raw_mean_sea_floor_radius, table.radius)
    with pytest.raises(ValueError, match="not both"):
        EarthModel(LMAX, love_numbers=table, love_number_file=path)
    with pytest.raises(ValueError, match="exceeds"):
        EarthModel(LMAX + 1, love_numbers=table)


def test_from_planet_model_and_with_degree(table):
    model = EarthModel.from_planet_model(PREM(ocean=False), LMAX)
    assert np.allclose(
        model.love_numbers.h, EarthModel(LMAX, love_numbers=table).love_numbers.h
    )
    smaller = model.with_degree(8)
    assert smaller.lmax == 8 and smaller.love_numbers.lmax == 8
    assert np.allclose(smaller.love_numbers.h, model.love_numbers.h[:9])
    assert smaller.parameters == model.parameters


def test_scales_are_the_users_the_body_is_the_tables(table):
    params = EarthModelParameters(raw_gravitational_acceleration=1.0)  # SI scales
    model = EarthModel(LMAX, parameters=params, love_numbers=table)
    p = model.parameters
    assert p.scales == units.Scales.SI and p.length_scale == 1.0
    assert p.gravitational_acceleration == table.surface_gravity  # overridden
    assert np.allclose(model.love_numbers.h_u, table.h_u[: LMAX + 1])
    assert p.raw_ice_density == params.raw_ice_density


def test_complex_table_is_refused():
    model = PREM(ocean=False)
    cold = love_numbers(frozen(model, 2 * np.pi / 43200.0), 2)
    with pytest.raises(ValueError, match="real"):
        EarthModel(2, love_numbers=cold)


def test_fingerprint_adjoint_is_exact_with_a_consistent_body(table):
    """With the body of the table the identity closes to solver tolerance;
    with another G it drifts in proportion."""
    exact = _adjoint_mismatch(EarthModel(LMAX, love_numbers=table))
    assert exact < 1e-9
    off = EarthModelParameters(raw_gravitational_constant=1.02 * units.G_SI)
    model = EarthModel(LMAX, parameters=off, love_numbers=table)
    assert (
        model.parameters.raw_gravitational_constant == units.G_SI
    )  # adopted from the table
    assert _adjoint_mismatch(model) < 1e-9


def test_axial_feedback_uses_the_tables_degree_zero_numbers(table):
    """
    With the axial numbers a change in spin rate expands or contracts the
    body uniformly, so the degree-0 displacement responds to a zonal
    load; without them (a table from an older file) that response is
    absent, while the adjoint identity closes either way, each treatment
    being self-adjoint.
    """
    from dataclasses import replace

    from pyslfp.physics import LinearSeaLevelEquation

    assert table.has_axial
    without = replace(
        table, h_c=np.nan, k_c=np.nan, m_u=np.nan, m_phi=np.nan, m_c=np.nan
    )
    assert not without.has_axial

    responses = {}
    for name, love in (("with", table), ("without", without)):
        model = EarthModel(LMAX, love_numbers=love)
        state = _state(model)
        load = state.northern_hemisphere_load(fraction=1.0)
        _, disp, _, omega = LinearSeaLevelEquation(state).solve_sea_level_equation(
            load, rtol=1e-12
        )
        responses[name] = (model.expand_field(disp).coeffs[0, 0, 0], omega[2])
        assert _adjoint_mismatch(model) < 1e-9

    u0_with, omega_with = responses["with"]
    u0_without, omega_without = responses["without"]
    assert abs(u0_without) < 1e-6 * abs(u0_with)
    expected = table.converted(EarthModel(LMAX, love_numbers=table).parameters.scales)
    params = EarthModel(LMAX, love_numbers=table).parameters
    assert u0_with == pytest.approx(
        expected.h_c * params.uniform_rotation_factor * omega_with, rel=1e-9
    )
    # the degree-0 feedback on the spin rate itself is small and stabilising
    assert abs(omega_with / omega_without - 1.0) < 1e-3
    assert abs(omega_with) < abs(omega_without)
