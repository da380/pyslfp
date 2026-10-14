"""The radial viscoelastic core: the elastic and relaxed limits, the
Wu-Peltier correspondence gate, scheme convergence orders, and the
consistency of the slaved solve."""

import numpy as np
import pytest

from planetmodel import LayeredIsotropicElastic, RadialMesh
from pyslfp.love_numbers import Material, solve_degree
from pyslfp.gia import Rheology, ViscoelasticOperator

RHO, VS, RADIUS = 5510.0, 5000.0, 6.371e6
MU = RHO * VS * VS
TAU = 1.0  # the Maxwell time of every test body, in seconds


def _material(vp: float) -> Material:
    model = LayeredIsotropicElastic([0.0, RADIUS], rho=[RHO], vp=[vp], vs=[VS])
    return Material(RadialMesh(model, ngll=5, lmax=2), model)


@pytest.fixture(scope="module")
def material():
    return _material(12000.0)


@pytest.fixture(scope="module")
def stiff_material():
    """Near-incompressible (kappa / mu = 1600), the Wu-Peltier regime."""
    return _material(2.0e5)


def _unit_load(l, t):
    return np.array([1.0])


def test_elastic_limit(material):
    """A branch with an infinite relaxation time never moves: every
    scheme reproduces the elastic response to full precision."""
    shape = material.moduli.shape
    rheo = Rheology(material, [(np.full(shape, MU), np.zeros(shape))])
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
    rheo = Rheology(
        material, [(np.full(shape, 0.5 * MU), np.full(shape, 1.0 / TAU))]
    )
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
    rheo = Rheology.maxwell(stiff_material, MU * TAU)
    op = ViscoelasticOperator(rheo, l, degrees=[l])
    g = stiff_material.surface_gravity
    mu_l = (2.0 * l * l + 4.0 * l + 3.0) * MU / (l * RHO * g * RADIUS)
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
    rheo = Rheology.maxwell(material, MU * TAU)
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
    rheo = Rheology.maxwell(material, MU * TAU)
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
    rheo = Rheology.maxwell(material, MU * TAU)
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
    rheo = Rheology.maxwell(material, MU * TAU)
    op = ViscoelasticOperator(rheo, 2, degrees=[2])

    def load(l, t):
        return np.array([1.0, 2.0, -3.0])

    state = op.zero_state(ncols=3)
    for _ in range(4):
        state, solutions = op.step(state, 0.3 * TAU, load=load)
    U = op.surface(solutions, "U")[2]
    assert abs(U[1] - 2.0 * U[0]) < 1e-12 * abs(U[0])
    assert abs(U[2] + 3.0 * U[0]) < 1e-12 * abs(U[0])
