"""The radial viscoelastic core: the elastic and relaxed limits, the
Wu-Peltier correspondence gate, scheme convergence orders, and the
consistency of the slaved solve."""

import numpy as np
import pytest

from planetmodel import LayeredIsotropicElastic, RadialMesh, units
from pyslfp.love_numbers import Material, solve_degree
from pyslfp.viscoelastic import Rheology, ViscoelasticOperator

# the test body lives in the geophysical non-dimensionalisation: unit
# radius, near-reference density, Earth-like moduli, G equal to one
SCALES = units.Scales.geophysical(6.371e6)
RHO, VS = 1.0, 1.3
TAU = 1.0  # the Maxwell time of every test body, in the model's units


def _material(vp: float) -> Material:
    model = LayeredIsotropicElastic(
        [0.0, 1.0], rho=[RHO], vp=[vp], vs=[VS], scales=SCALES
    )
    return Material(RadialMesh(model, ngll=5, lmax=2), model)


def _mu(material) -> np.ndarray:
    """The material's nodal shear modulus, in its own units."""
    return np.asarray(material.moduli.kappa_mu()[1]).real


@pytest.fixture(scope="module")
def material():
    return _material(3.1)


@pytest.fixture(scope="module")
def stiff_material():
    """Near-incompressible (kappa / mu = 1600), the Wu-Peltier regime."""
    return _material(52.0)


def _unit_load(l, t):
    return np.array([1.0])


def test_elastic_limit(material):
    """A branch with an infinite relaxation time never moves: every
    scheme reproduces the elastic response to full precision."""
    shape = material.moduli.shape
    rheo = Rheology(material, [(_mu(material), np.zeros(shape))])
    op = ViscoelasticOperator(rheo, 2)
    for scheme in ("exponential_trapezoid", "backward_euler", "exponential_euler"):
        state = op.zero_state()
        for _ in range(3):
            state, solutions = op.step(state, 1.0, scheme=scheme, load=_unit_load)
        if scheme == "exponential_euler":
            solutions = op.solve_elastic(state, load=_unit_load)
        U = op.surface(solutions, "U")
        phi = op.surface(solutions, "phi")
        for l in (0, 1, 2):
            want = solve_degree(material, l).surface
            assert abs(U[l][0] - want[0]) < 1e-12 * abs(want[0]), (scheme, l)
            if l != 1:
                assert abs(phi[l][0] - want[2]) < 1e-12 * abs(want[2]), (scheme, l)
        for l in (0, 1, 2):
            assert np.all(state.m[l] == 0.0)


def test_relaxed_limit_is_the_long_term_solid(material):
    """A standard linear solid under a held load approaches the
    long-term elastic response; the discrete fixed point of the
    backward Euler and trapezoid updates is exactly m = d, so the
    time-stepped limit matches the long-term solve to solver precision."""
    shape = material.moduli.shape
    rheo = Rheology(material, [(0.5 * _mu(material), np.full(shape, 1.0 / TAU))])
    op = ViscoelasticOperator(rheo, 2, degrees=[2])
    want = solve_degree(rheo.long_term_material(), 2).surface
    for scheme in ("exponential_trapezoid", "backward_euler"):
        state = op.zero_state()
        for _ in range(120):
            state, solutions = op.step(state, 0.5 * TAU, scheme=scheme, load=_unit_load)
        got = op.surface(solutions, "U")[2][0]
        assert abs(got - want[0]) < 1e-9 * abs(want[0]), scheme


def test_wu_peltier_correspondence(stiff_material):
    """The homogeneous, near-incompressible Maxwell sphere under a held
    degree-l load relaxes with the closed-form history (Wu & Peltier
    1982, by correspondence)

        f_l(t) = 1 - A e^(-t / tau_l),   A = mu_l / (1 + mu_l),
        tau_l = (1 + mu_l) tau,          mu_l = (2l^2 + 4l + 3) mu / (l rho g a),

    and the ratio U(t) / U(0+) carries no normalisation convention.
    The tolerance is set by the finite kappa / mu = 1600."""
    l = 2
    rheo = Rheology.from_relaxation_time(stiff_material, TAU)
    op = ViscoelasticOperator(rheo, l, degrees=[l])
    g = stiff_material.surface_gravity
    mu0 = float(np.max(_mu(stiff_material)))
    rho0 = float(np.max(np.asarray(stiff_material.rho).real))
    mu_l = (2.0 * l * l + 4.0 * l + 3.0) * mu0 / (l * rho0 * g * stiff_material.radius)
    tau_l = (1.0 + mu_l) * TAU
    A = mu_l / (1.0 + mu_l)

    state = op.zero_state()
    U0 = op.surface(op.solve_elastic(state, load=_unit_load), "U")[l][0]
    dt = tau_l / 100.0
    worst = 0.0
    for n in range(1, 151):
        state, solutions = op.step(state, dt, load=_unit_load)
        Ut = op.surface(solutions, "U")[l][0]
        t = n * dt
        f_ratio = (1.0 - A * np.exp(-t / tau_l)) / (1.0 - A)
        worst = max(worst, abs((Ut / U0) / f_ratio - 1.0))
    assert worst < 2e-3


def test_scheme_convergence_orders(material):
    """Under a smooth load history the exponential trapezoid is second
    order and backward and exponential Euler first order in dt."""
    rheo = Rheology.from_relaxation_time(material, TAU)
    op = ViscoelasticOperator(rheo, 2, degrees=[2])
    T = 2.0 * TAU

    def load(l, t):
        return np.array([1.0 - np.exp(-t / (0.7 * TAU))])

    def run(scheme, dt):
        state = op.zero_state()
        steps = int(round(T / dt))
        for _ in range(steps):
            state, solutions = op.step(state, dt, scheme=scheme, load=load)
        if scheme == "exponential_euler":
            solutions = op.solve_elastic(state, load=load)
        return op.surface(solutions, "U")[2][0]

    reference = run("exponential_trapezoid", T / 512.0)
    orders = {
        "exponential_trapezoid": 2.0,
        "backward_euler": 1.0,
        "exponential_euler": 1.0,
    }
    for scheme, order in orders.items():
        e1 = abs(run(scheme, T / 8.0) - reference)
        e2 = abs(run(scheme, T / 16.0) - reference)
        measured = np.log2(e1 / e2)
        assert abs(measured - order) < 0.35, (scheme, measured)


def test_slaved_solve_is_consistent_after_a_step(material):
    """After a trapezoid or backward Euler step the effective solve's
    displacement equals the unrelaxed solve at the new state: the
    discrete elimination is exact, so the cached strain matches a
    fresh solve to round-off."""
    rheo = Rheology.from_relaxation_time(material, TAU)
    op = ViscoelasticOperator(rheo, 2, degrees=[2])
    for scheme in ("exponential_trapezoid", "backward_euler"):
        state = op.zero_state()
        for _ in range(3):
            state, _ = op.step(state, 0.4 * TAU, scheme=scheme, load=_unit_load)
        cached = state.cached_strain[2]
        fresh = op.strain(op.solve_elastic(state, load=_unit_load))[2]
        scale = np.max(np.abs(fresh))
        assert np.max(np.abs(cached - fresh)) < 1e-10 * scale, scheme


def test_load_jump_needs_invalidate(material):
    """A load jump at a step boundary changes the start-of-step strain;
    the cached strain must be dropped to see the new left limit."""
    rheo = Rheology.from_relaxation_time(material, TAU)
    op = ViscoelasticOperator(rheo, 2, degrees=[2])

    def jumped(l, t):
        return np.array([2.0])

    state = op.zero_state()
    for _ in range(2):
        state, _ = op.step(state, 0.5 * TAU, load=_unit_load)

    stale, _ = op.step(state, 0.5 * TAU, load=jumped)
    fresh_state = state.copy()  # copy drops the strain cache
    fresh, _ = op.step(fresh_state, 0.5 * TAU, load=jumped)
    diff = max(
        np.max(np.abs(stale.m[2] - fresh.m[2])) / np.max(np.abs(fresh.m[2])), 0.0
    )
    assert diff > 1e-8  # the stale cache gives a different step

    state.invalidate()
    again, _ = op.step(state, 0.5 * TAU, load=jumped)
    assert np.array_equal(again.m[2], fresh.m[2])


def test_columns_are_independent_and_linear(material):
    """Several coefficient columns step together and scale linearly."""
    rheo = Rheology.from_relaxation_time(material, TAU)
    op = ViscoelasticOperator(rheo, 2, degrees=[2])

    def load(l, t):
        return np.array([1.0, 2.0, -3.0])

    state = op.zero_state(ncols=3)
    for _ in range(4):
        state, solutions = op.step(state, 0.3 * TAU, load=load)
    U = op.surface(solutions, "U")[2]
    assert abs(U[1] - 2.0 * U[0]) < 1e-12 * abs(U[0])
    assert abs(U[2] + 3.0 * U[0]) < 1e-12 * abs(U[0])


def test_adaptive_trapezoid(material):
    """The adaptive exponential trapezoid: the embedded ETD1 estimate
    accepts and strides through a Heaviside relaxation, tighter
    tolerances give more steps and smaller errors (the estimate is the
    first-order companion's, hence conservative), quantisation holds
    the step ladder to a handful of flavours, and a purely elastic
    body strides at the growth limit on a zero estimate."""
    rheo = Rheology.from_relaxation_time(material, TAU)
    op = ViscoelasticOperator(rheo, 2, degrees=[2])

    def load(l, t):
        return np.array([1.0])

    T = 20.0 * TAU
    reference_state = op.zero_state()
    for _ in range(2000):
        reference_state, reference = op.step(reference_state, T / 2000, load=load)
    U_ref = op.surface(reference, "U")[2][0]

    results = {}
    for rtol in (1e-3, 1e-5):
        state, solutions, info = op.integrate(
            op.zero_state(), T, 0.05 * TAU, rtol=rtol, load=load
        )
        U = op.surface(solutions, "U")[2][0]
        results[rtol] = (abs(U - U_ref) / abs(U_ref), info)
        assert abs(state.t - T) < 1e-9 * T
    # the steps stride as the transient quietens (judged at the loose
    # tolerance; the final step is clipped to land on t_final)
    strides = results[1e-3][1]["steps"]
    assert max(strides) > 10.0 * strides[0]
    assert results[1e-3][0] < 1e-3 and results[1e-5][0] < 1e-5
    assert results[1e-5][0] < 0.1 * results[1e-3][0]
    assert results[1e-5][1]["accepted"] > 3 * results[1e-3][1]["accepted"]

    _, solutions, info = op.integrate(
        op.zero_state(), T, 0.05 * TAU, rtol=1e-4, load=load, quantise=True
    )
    distinct = len(set(np.round(info["steps"], 12)))
    assert distinct <= 10
    U = op.surface(solutions, "U")[2][0]
    assert abs(U - U_ref) / abs(U_ref) < 1e-4

    shape = material.moduli.shape
    elastic = ViscoelasticOperator(
        Rheology(material, [(_mu(material), np.zeros(shape))]), 2, degrees=[2]
    )
    _, _, info = elastic.integrate(
        elastic.zero_state(), T, 0.05 * TAU, rtol=1e-4, load=load
    )
    assert info["accepted"] <= 8 and info["rejected"] == 0
