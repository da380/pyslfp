"""The time-domain stepper against the frequency-domain solver.

A Maxwell sphere under a sinusoidal degree-l load reaches a steady
periodic state whose amplitude and phase are the complex Love numbers
of the model frozen at that frequency -- an independent route through
`planetmodel.frozen` and the complex banded solver.  Both sides read
the same `viscosity` field, so the comparison verifies the whole
time-domain path (rheology, memory force, effective systems, stepper)
against machinery that shares none of it.  planetmodel's frozen moduli
follow the e^{i omega t} convention, mu(omega) = mu i omega tau /
(1 + i omega tau)."""

import numpy as np
import pytest

from planetmodel import LayeredIsotropicElastic, RadialMesh, constant_field, frozen
from pyslfp.love_numbers import Material, solve_degree
from pyslfp.gia import Rheology, ViscoelasticOperator

RHO, VS, RADIUS = 5510.0, 5000.0, 6.371e6
MU = RHO * VS * VS
TAU = 1.0
OMEGA = 1.0  # omega tau = 1, squarely viscoelastic


@pytest.fixture(scope="module")
def viscous_model():
    model = LayeredIsotropicElastic([0.0, RADIUS], rho=[RHO], vp=[12000.0], vs=[VS])
    return model.with_field(0, "viscosity", constant_field((0.0, RADIUS), MU * TAU))


def test_steady_periodic_state_matches_the_frozen_solver(viscous_model):
    l = 2
    mesh = RadialMesh(viscous_model, ngll=5, lmax=l)
    material = Material(mesh, viscous_model)
    rheo = Rheology.from_model(material)
    op = ViscoelasticOperator(rheo, l, degrees=[l])

    U_c = solve_degree(
        Material(mesh, frozen(viscous_model, OMEGA)), l, forcing="load"
    ).surface[0]

    def load(_, t):
        return np.array([np.cos(OMEGA * t)])

    period = 2.0 * np.pi / OMEGA
    dt = TAU / 40.0
    settle = int(round(8.0 * period / dt))  # transients decay with tau_l ~ 5 tau
    state = op.zero_state()
    for _ in range(settle):
        state, _ = op.step(state, dt, load=load)
    worst = 0.0
    for _ in range(int(round(period / dt))):
        state, solutions = op.step(state, dt, load=load)
        got = op.surface(solutions, "U")[l][0]
        want = (U_c * np.exp(1j * OMEGA * state.t)).real
        worst = max(worst, abs(got - want))
    assert worst / abs(U_c) < 1e-3
