"""
Tests for the tangent-linear runs: the ice history enters linearly, the
relaxation-time derivative matches finite differences of the forward run,
the discrete tangent-linear run and the discrete adjoint are exact
transposes about frozen and migrating trajectories, the continuous run
converges to the discrete one at the scheme's order, and the initial sea
level's derivative about a migrating trajectory matches finite differences.
"""

import numpy as np
from pyshtools import SHCoeffs, SHGrid

from pyslfp.core import EarthState
from pyslfp.viscoelastic import Rheology, SeaLevelEvolution, ViscoelasticOperator
from pyslfp.viscoelastic.ice_history import _NodalIceModel

from .body import LMAX, TAU
from .test_evolution import _history_state, _marine_history_state, _maxwell_operator


def _smooth(earth, rng, amplitude=1.0):
    """A random band-limited field."""
    c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
    for degree in range(LMAX + 1):
        c[:, degree, degree + 1 :] = 0.0
    c[1, :, 0] = 0.0
    return amplitude * earth.expand_coefficient(
        SHCoeffs.from_array(c, normalization=earth.normalization, csphase=earth.csphase)
    )


def _pair(earth, g, grid):
    return earth.integrate(SHGrid.from_array(g.data * grid.data, grid=earth.grid))


def _sea_level(snapshot):
    return snapshot.response.field("relative_sea_level_change")


def _ditau(material, scale=1.0):
    r = material.mesh.r
    return [scale * (0.3 + 0.7 * (r / r.max()) ** 2) / TAU]


def _rheology_along(material, ditau, eps):
    base = Rheology.from_relaxation_time(material, TAU)
    return Rheology(material, [(base.mu[0], base.itau[0] + eps * ditau[0])])


# ==================================================================== #
#                     The ice history and the methods                  #
# ==================================================================== #


def test_the_ice_history_enters_linearly(setting):
    """Under frozen shorelines the tangent-linear run of a thickness
    perturbation is the run of that perturbation as a history, and both
    methods agree, there being no relaxation-time perturbation."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    evolution = SeaLevelEvolution(_maxwell_operator(material), state)
    dates = np.linspace(10.0, 0.0, 6)
    snapshots, _ = evolution.run_ice_history(dates, ice_model)
    rng = np.random.default_rng(1)
    scale = earth.parameters.length_scale
    deltas = [_smooth(earth, rng, 50.0 / scale) for _ in dates]
    tangent, _ = evolution.tangent_linear_ice_history(
        dates, snapshots, ice_thickness=deltas
    )
    direct, _ = evolution.run_ice_history(
        dates, _NodalIceModel(dates, deltas, length_scale=scale)
    )
    other, _ = evolution.tangent_linear_ice_history(
        dates, snapshots, ice_thickness=deltas, method="continuous"
    )
    space = snapshots[-1].response.space
    for a, b, c in zip(tangent, direct, other):
        np.testing.assert_allclose(
            space.to_components(a.response),
            space.to_components(b.response),
            rtol=1e-8,
            atol=1e-12 * abs(space.to_components(b.response)).max(),
        )
        assert np.array_equal(
            space.to_components(a.response), space.to_components(c.response)
        )
    assert np.all(space.to_components(tangent[0].response) == 0.0)


def test_the_relaxation_time_derivative(setting):
    """The discrete tangent-linear run is the exact derivative of the
    time-stepped map in the inverse relaxation times, for both schemes:
    central differences of a functional of the forward run converge to it
    at second order in the step, and land on it after extrapolation."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    dates = np.linspace(10.0, 0.0, 6)
    g = _smooth(earth, np.random.default_rng(2))
    ditau = _ditau(material)
    for scheme in ("exponential_trapezoid", "backward_euler"):
        evolution = SeaLevelEvolution(_maxwell_operator(material), state, scheme=scheme)
        snapshots, _ = evolution.run_ice_history(dates, ice_model, record=True)
        tangent, _ = evolution.tangent_linear_ice_history(dates, snapshots, itau=ditau)
        predicted = _pair(earth, g, _sea_level(tangent[-1]))

        def functional(eps):
            other = SeaLevelEvolution(
                ViscoelasticOperator(_rheology_along(material, ditau, eps), LMAX),
                state,
                scheme=scheme,
            )
            snaps, _ = other.run_ice_history(dates, ice_model)
            return _pair(earth, g, _sea_level(snaps[-1]))

        fd = {
            eps: (functional(eps) - functional(-eps)) / (2.0 * eps)
            for eps in (0.02, 0.01)
        }
        resid = {eps: abs(f - predicted) for eps, f in fd.items()}
        assert 3.0 < resid[0.02] / resid[0.01] < 5.0, scheme
        richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0
        assert abs(richardson - predicted) < 1e-6 * abs(predicted), scheme


def test_the_continuous_run_converges_to_the_discrete(setting):
    """In the relaxation times the two methods differ by the order of the
    scheme: on a slowly relaxing body their difference falls by about
    four when the dates are halved, and vanishes with checkpoints
    refilled by forward steps all the same."""
    material, earth, _, _, _ = setting
    ice_model, state = _history_state(material, earth)
    g = _smooth(earth, np.random.default_rng(3))
    ditau = _ditau(material, 0.1)
    operator = ViscoelasticOperator(
        Rheology.from_relaxation_time(material, 10.0 * TAU), LMAX
    )
    evolution = SeaLevelEvolution(operator, state)
    gaps = []
    for npts in (5, 9):
        dates = np.linspace(10.0, 0.0, npts)
        snapshots, _ = evolution.run_ice_history(dates, ice_model, record=True)
        values = {}
        for method in ("discrete", "continuous"):
            tangent, _ = evolution.tangent_linear_ice_history(
                dates, snapshots, itau=ditau, method=method
            )
            values[method] = _pair(earth, g, _sea_level(tangent[-1]))
        gaps.append(
            abs(values["discrete"] - values["continuous"]) / abs(values["discrete"])
        )
    assert gaps[0] < 0.1
    assert gaps[0] / gaps[1] > 3.0
    # a checkpointed trajectory, its gaps refilled by forward steps
    dates = np.linspace(10.0, 0.0, 9)
    full, _ = evolution.run_ice_history(dates, ice_model, record=True)
    sparse, _ = evolution.run_ice_history(dates, ice_model, record=3)
    for method in ("discrete", "continuous"):
        a, _ = evolution.tangent_linear_ice_history(
            dates, full, itau=ditau, method=method
        )
        b, _ = evolution.tangent_linear_ice_history(
            dates, sparse, itau=ditau, method=method
        )
        assert abs(
            _pair(earth, g, _sea_level(a[-1])) - _pair(earth, g, _sea_level(b[-1]))
        ) < 1e-10 * abs(_pair(earth, g, _sea_level(a[-1])))


# ==================================================================== #
#                       The discrete pair                              #
# ==================================================================== #


def _check_transposes(earth, evolution, dates, ice_model, shorelines, rng):
    scale = earth.parameters.length_scale
    snapshots, _ = evolution.run_ice_history(
        dates, ice_model, shorelines=shorelines, record=True, rtol=1e-12
    )
    duals = {
        float(dates[-1]): _smooth(earth, rng),
        float(dates[2]): _smooth(earth, rng),
    }
    kernels = evolution.adjoint_ice_history(
        dates,
        duals,
        ocean_functions=[snap.ocean_function for snap in snapshots]
        if shorelines == "migrating"
        else None,
        internal_states=snapshots,
        rtol=1e-12,
    )
    deltas = [_smooth(earth, rng, 20.0 / scale) for _ in dates]
    delta_sea = _smooth(earth, rng, 20.0 / scale)
    ditau = [rng.normal(size=evolution.operator.material.mesh.r.shape) / TAU]
    tangent, _ = evolution.tangent_linear_ice_history(
        dates,
        snapshots,
        ice_thickness=deltas,
        initial_sea_level=delta_sea,
        itau=ditau,
        rtol=1e-12,
    )
    forward = sum(
        _pair(
            earth, g, _sea_level(tangent[int(np.flatnonzero(np.isclose(dates, d))[0])])
        )
        for d, g in duals.items()
    )
    adjoint = (
        sum(_pair(earth, kernels.thickness[n], deltas[n]) for n in range(dates.size))
        + _pair(earth, kernels.initial_sea_level, delta_sea)
        + float(np.sum(kernels.itau[0] * ditau[0]))
    )
    assert abs(forward - adjoint) < 1e-9 * abs(forward)
    return kernels


def test_the_discrete_pair_are_transposes(setting):
    """<g, F' v> = <F'* g, v> for the discrete tangent-linear run and the
    discrete adjoint, in every argument at once, about a frozen run and
    about a migrating trajectory, to the solvers' tolerance."""
    material, earth, _, _, _ = setting
    rng = np.random.default_rng(4)
    dates = np.linspace(10.0, 0.0, 6)
    ice_model, state = _history_state(material, earth)
    frozen = _check_transposes(
        earth,
        SeaLevelEvolution(_maxwell_operator(material), state),
        dates,
        ice_model,
        "frozen",
        rng,
    )
    assert np.all(frozen.initial_sea_level.data == 0.0)
    ice_model, state = _marine_history_state(material, earth)
    migrating = _check_transposes(
        earth,
        SeaLevelEvolution(_maxwell_operator(material), state),
        dates,
        ice_model,
        "migrating",
        rng,
    )
    assert np.any(migrating.initial_sea_level.data != 0.0)


def test_the_initial_sea_level_about_a_migrating_trajectory(setting):
    """Along a perturbation of the initial sea level that moves no
    shoreline, the tangent-linear run matches central differences of the
    migrating run."""
    material, earth, _, _, _ = setting
    ice_model, state = _marine_history_state(material, earth)
    evolution = SeaLevelEvolution(_maxwell_operator(material), state)
    dates = np.linspace(10.0, 0.0, 6)
    snapshots, _ = evolution.run_ice_history(dates, ice_model, shorelines="migrating")
    lats = earth.lats()[:, None] * np.ones_like(earth.lons())[None, :]
    lons = np.ones_like(earth.lats())[:, None] * earth.lons()[None, :]
    metre = earth.parameters.length_scale
    direction = SHGrid.from_array(
        (30.0 / metre) * np.cos(np.radians(lats)) ** 2 * np.sin(np.radians(lons)),
        grid=earth.grid,
    )
    g = _smooth(earth, np.random.default_rng(5))
    tangent, _ = evolution.tangent_linear_ice_history(
        dates, snapshots, initial_sea_level=direction
    )
    predicted = _pair(earth, g, _sea_level(tangent[-1]))
    assert predicted != 0.0

    def functional(eps):
        moved = EarthState(
            state.ice_thickness,
            SHGrid.from_array(
                state.sea_level.data + eps * direction.data, grid=earth.grid
            ),
            earth,
            land=None,
        )
        assert np.array_equal(moved.ocean_function.data, state.ocean_function.data)
        snaps, _ = evolution._with_state(moved).run_ice_history(
            dates, ice_model, shorelines="migrating"
        )
        return _pair(earth, g, _sea_level(snaps[-1]))

    fd = {
        eps: (functional(eps) - functional(-eps)) / (2.0 * eps) for eps in (0.02, 0.01)
    }
    richardson = (4.0 * fd[0.01] - fd[0.02]) / 3.0
    assert abs(richardson - predicted) < 1e-6 * abs(predicted)
