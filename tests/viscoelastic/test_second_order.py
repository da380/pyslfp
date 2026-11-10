"""
Tests for the second-order adjoint: the kernels it carries agree with the
first-order sweep's, its derivatives match central differences of the
gradient along the direction, the Hessian it gives is symmetric, and it is
the adjoint of the second-order tangent-linear run.
"""

import numpy as np
import pytest
from pyshtools import SHGrid

from pyslfp.viscoelastic import SeaLevelEvolution, ViscoelasticOperator
from pyslfp.viscoelastic.ice_history import _NodalIceModel

from .body import LMAX, TAU
from .test_evolution import _history_state, _marine_history_state, _maxwell_operator
from .test_tangent_linear import (
    _ditau,
    _pair,
    _rheology_along,
    _sea_level,
    _smooth,
    _thickness_at,
)

DATES = np.linspace(10.0, 0.0, 6)


def _direction(earth, rng, dates, *, sea_level=True, viscosity=True):
    """A perturbation of the ice history, the initial sea level and the
    inverse relaxation times."""
    scale = earth.parameters.length_scale
    return {
        "ice": [_smooth(earth, rng, 20.0 / scale) for _ in dates],
        "sea_level": _smooth(earth, rng, 20.0 / scale) if sea_level else None,
        "itau": [rng.normal(size=earth_mesh_shape) / TAU] if viscosity else None,
    }


earth_mesh_shape = None  # set by the fixture-using helpers below


def _pairing(material, earth, kernels, direction):
    """A kernel object paired with a direction, in every argument: the
    surface fields by int . dS, the relaxation-time kernel by the volume
    pairing."""
    total = sum(
        _pair(earth, kernels.thickness[n], direction["ice"][n])
        for n in range(len(direction["ice"]))
    )
    if direction["sea_level"] is not None:
        total += _pair(earth, kernels.initial_sea_level, direction["sea_level"])
    if direction["itau"] is not None:
        total += material.volume_pairing(kernels.itau[0], direction["itau"][0])
    return total


def _tangent(evolution, dates, snapshots, direction, **options):
    tangent, _ = evolution.tangent_linear_ice_history(
        dates,
        snapshots,
        ice_thickness=direction["ice"],
        initial_sea_level=direction["sea_level"],
        itau=direction["itau"],
        record=True,
        **options,
    )
    return tangent


def _setup(setting, shorelines):
    global earth_mesh_shape
    material, earth, _, _, _ = setting
    earth_mesh_shape = material.mesh.r.shape
    ice_model, state = (
        _history_state(material, earth)
        if shorelines == "frozen"
        else _marine_history_state(material, earth)
    )
    evolution = SeaLevelEvolution(_maxwell_operator(material), state)
    snapshots, _ = evolution.run_ice_history(
        DATES, ice_model, shorelines=shorelines, record=True, rtol=1e-12
    )
    ocean = (
        [snap.ocean_function for snap in snapshots]
        if shorelines == "migrating"
        else None
    )
    return material, earth, evolution, snapshots, ocean


@pytest.mark.parametrize("shorelines", ["frozen", "migrating"])
def test_the_kernels_agree_with_the_first_order_sweep(setting, shorelines):
    material, earth, evolution, snapshots, ocean = _setup(setting, shorelines)
    rng = np.random.default_rng(1)
    measurement = {0.0: _smooth(earth, rng), 6.0: _smooth(earth, rng)}
    v = _direction(earth, rng, DATES)
    tangent = _tangent(evolution, DATES, snapshots, v, rtol=1e-12)
    first = evolution.adjoint_ice_history(
        DATES, measurement, ocean_functions=ocean, internal_states=snapshots, rtol=1e-12
    )
    kernels, derivatives = evolution.second_order_adjoint_ice_history(
        DATES,
        measurement,
        tangent,
        itau=v["itau"],
        ocean_functions=ocean,
        internal_states=snapshots,
        rtol=1e-12,
    )
    for a, b in zip(kernels.thickness, first.thickness):
        np.testing.assert_allclose(a.data, b.data, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(
        kernels.initial_sea_level.data,
        first.initial_sea_level.data,
        rtol=1e-12,
        atol=1e-20,
    )
    np.testing.assert_allclose(kernels.itau[0], first.itau[0], rtol=1e-12)
    assert any(np.any(d.data != 0.0) for d in derivatives.thickness)
    assert np.any(derivatives.itau[0] != 0.0)


def test_the_derivatives_match_differences_of_the_gradient(setting):
    """Along v, central differences of the gradient (the kernels paired
    with u) converge at second order to the second-order adjoint's
    derivatives paired with u, for both schemes."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    rng = np.random.default_rng(2)
    scale = earth.parameters.length_scale
    base = _thickness_at(ice_model, DATES, earth)
    v = {
        "ice": [_smooth(earth, rng, 20.0 / scale) for _ in DATES],
        "sea_level": None,
        "itau": _ditau(material),
    }
    u = {
        "ice": [_smooth(earth, rng, 20.0 / scale) for _ in DATES],
        "sea_level": None,
        "itau": [rng.normal(size=material.mesh.r.shape) / TAU],
    }
    measurement = {0.0: _smooth(earth, rng), 4.0: _smooth(earth, rng)}

    def moved_model(eps):
        return _NodalIceModel(
            DATES,
            [
                SHGrid.from_array(b.data + eps * d.data, grid=earth.grid)
                for b, d in zip(base, v["ice"])
            ],
            length_scale=scale,
        )

    for scheme in ("exponential_trapezoid", "backward_euler"):
        evolution = SeaLevelEvolution(_maxwell_operator(material), state, scheme=scheme)
        snapshots, _ = evolution.run_ice_history(DATES, moved_model(0.0), record=True)
        tangent = _tangent(evolution, DATES, snapshots, v)
        _, derivatives = evolution.second_order_adjoint_ice_history(
            DATES, measurement, tangent, itau=v["itau"], internal_states=snapshots
        )
        predicted = _pairing(material, earth, derivatives, u)

        def gradient_at(eps):
            moved = SeaLevelEvolution(
                ViscoelasticOperator(_rheology_along(material, v["itau"], eps), LMAX),
                state,
                scheme=scheme,
            )
            snaps, _ = moved.run_ice_history(DATES, moved_model(eps), record=True)
            kernels = moved.adjoint_ice_history(
                DATES, measurement, internal_states=snaps
            )
            return _pairing(material, earth, kernels, u)

        fd = {
            eps: (gradient_at(eps) - gradient_at(-eps)) / (2.0 * eps)
            for eps in (0.02, 0.01)
        }
        resid = {eps: abs(f - predicted) for eps, f in fd.items()}
        assert 3.0 < resid[0.02] / resid[0.01] < 5.0, scheme
        richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0
        assert abs(richardson - predicted) < 1e-6 * abs(predicted), scheme


@pytest.mark.parametrize("shorelines", ["frozen", "migrating"])
def test_the_hessian_is_symmetric_and_adjoint_to_the_second_order_run(
    setting, shorelines
):
    """<H u, w> = <H w, u> to rounding, and <g, F''[v, u]> from the
    second-order tangent-linear run equals the second-order adjoint's
    derivatives along v paired with u, about frozen and migrating runs."""
    material, earth, evolution, snapshots, ocean = _setup(setting, shorelines)
    rng = np.random.default_rng(3)
    measurement = {0.0: _smooth(earth, rng), 6.0: _smooth(earth, rng)}
    u = _direction(earth, rng, DATES)
    w = _direction(earth, rng, DATES)
    tangents = {
        name: _tangent(evolution, DATES, snapshots, d, rtol=1e-12)
        for name, d in (("u", u), ("w", w))
    }
    results = {}
    for name, d in (("u", u), ("w", w)):
        _, derivatives = evolution.second_order_adjoint_ice_history(
            DATES,
            measurement,
            tangents[name],
            itau=d["itau"],
            ocean_functions=ocean,
            internal_states=snapshots,
            rtol=1e-12,
        )
        results[name] = derivatives
    hu_w = _pairing(material, earth, results["u"], w)
    hw_u = _pairing(material, earth, results["w"], u)
    assert abs(hu_w - hw_u) < 1e-9 * abs(hu_w)
    # the second-order run along (u, w) paired with the measurement
    second, _ = evolution.second_order_tangent_linear_ice_history(
        DATES,
        snapshots,
        tangents["u"],
        tangents["w"],
        itau_u=u["itau"],
        itau_v=w["itau"],
        rtol=1e-12,
    )
    forward = sum(
        _pair(
            earth, g, _sea_level(second[int(np.flatnonzero(np.isclose(DATES, d))[0])])
        )
        for d, g in measurement.items()
    )
    assert abs(forward - hu_w) < 1e-9 * abs(forward)


# ==================================================================== #
#                      The continuous method                           #
# ==================================================================== #


@pytest.mark.parametrize("shorelines", ["frozen", "migrating"])
def test_the_continuous_kernels_agree_with_the_continuous_sweep(setting, shorelines):
    material, earth, evolution, snapshots, ocean = _setup(setting, shorelines)
    rng = np.random.default_rng(4)
    measurement = {0.0: _smooth(earth, rng), 6.0: _smooth(earth, rng)}
    v = _direction(earth, rng, DATES)
    tangent = _tangent(evolution, DATES, snapshots, v, method="continuous", rtol=1e-12)
    first = evolution.continuous_adjoint_ice_history(
        DATES, measurement, ocean_functions=ocean, internal_states=snapshots, rtol=1e-12
    )
    kernels, derivatives = evolution.second_order_adjoint_ice_history(
        DATES,
        measurement,
        tangent,
        itau=v["itau"],
        ocean_functions=ocean,
        internal_states=snapshots,
        rtol=1e-12,
        method="continuous",
    )
    for a, b in zip(kernels.thickness, first.thickness):
        np.testing.assert_allclose(a.data, b.data, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(
        kernels.initial_sea_level.data,
        first.initial_sea_level.data,
        rtol=1e-12,
        atol=1e-20,
    )
    np.testing.assert_allclose(kernels.itau[0], first.itau[0], rtol=1e-12)
    assert any(np.any(d.data != 0.0) for d in derivatives.thickness)
    assert np.any(derivatives.itau[0] != 0.0)


def test_the_continuous_second_order_adjoint_converges(setting):
    """On a slowly relaxing body, and along directions smooth in time, the
    continuous second-order adjoint's derivatives tend to the discrete ones
    at the scheme's order as the dates are halved, block by block, and so
    does their mismatch with the continuous second-order tangent-linear
    run; the discrete pair match to rounding at every spacing."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    from pyslfp.viscoelastic import Rheology

    rheology = Rheology.from_relaxation_time(material, 10.0 * TAU)
    evolution = SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)
    scale = earth.parameters.length_scale
    r = material.mesh.r
    field = _smooth(earth, np.random.default_rng(11), 20.0 / scale)
    g = _smooth(earth, np.random.default_rng(6))
    ice_gaps, viscosity_gaps, mismatches = [], [], []
    for npts in (5, 9):
        dates = np.linspace(10.0, 0.0, npts)
        snapshots, _ = evolution.run_ice_history(
            dates, ice_model, record=True, rtol=1e-12
        )
        u = {
            "ice": [
                SHGrid.from_array(field.data * (1.0 - d / 10.0), grid=earth.grid)
                for d in dates
            ],
            "sea_level": None,
            "itau": [0.1 * (0.3 + 0.7 * (r / r.max()) ** 2) / TAU],
        }
        v = {
            "ice": [
                SHGrid.from_array(field.data * (d / 10.0) ** 2, grid=earth.grid)
                for d in dates
            ],
            "sea_level": None,
            "itau": [0.1 * (1.0 + r) / TAU],
        }
        results = {}
        for method in ("discrete", "continuous"):
            tangent_v = _tangent(
                evolution, dates, snapshots, v, method=method, rtol=1e-12
            )
            tangent_u = _tangent(
                evolution, dates, snapshots, u, method=method, rtol=1e-12
            )
            _, derivatives = evolution.second_order_adjoint_ice_history(
                dates,
                {0.0: g},
                tangent_v,
                itau=v["itau"],
                internal_states=snapshots,
                rtol=1e-12,
                method=method,
            )
            second, _ = evolution.second_order_tangent_linear_ice_history(
                dates,
                snapshots,
                tangent_v,
                tangent_u,
                itau_u=v["itau"],
                itau_v=u["itau"],
                method=method,
                rtol=1e-12,
            )
            results[method] = (
                sum(
                    _pair(earth, derivatives.thickness[n], u["ice"][n])
                    for n in range(npts)
                ),
                material.volume_pairing(derivatives.itau[0], u["itau"][0]),
                _pair(earth, g, _sea_level(second[-1])),
            )
        discrete, continuous = results["discrete"], results["continuous"]
        ice_gaps.append(abs(discrete[0] - continuous[0]) / abs(discrete[0]))
        viscosity_gaps.append(abs(discrete[1] - continuous[1]) / abs(discrete[1]))
        mismatches.append(
            abs(continuous[0] + continuous[1] - continuous[2]) / abs(continuous[2])
        )
        assert abs(discrete[0] + discrete[1] - discrete[2]) < 1e-9 * abs(discrete[2])
    for coarse, fine in (ice_gaps, viscosity_gaps, mismatches):
        assert coarse < 0.1 and coarse / fine > 3.0
