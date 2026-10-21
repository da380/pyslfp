"""The non-gravitating degree solver and the traction numbers.

On a homogeneous solid sphere the regular non-gravitating solutions at
degree l are the power laws r^(l-1) and r^(l+1), so for l + 1 within the
element order the exact solution lies in the finite-element space and
the answer is mesh-independent to rounding; and at degree 0 a unit
pressure compresses the sphere uniformly, U(a) = -a / (3 kappa), a
closed form.  The gravitating and non-gravitating problems share the
traction forcings exactly, with load(part="force") = g traction(), which
ties the new channels to the released solver.
"""

from __future__ import annotations

import numpy as np
import pytest
from planetmodel import PREM, LayeredIsotropicElastic, RadialMesh

from pyslfp.love_numbers import (
    DegreeSystem,
    Material,
    love_numbers,
    solve_degree,
    traction_numbers,
)

RHO, A = 5500.0, 6371e3
MU = 1.0e11
KAPPA = 2.0e11


@pytest.fixture(scope="module")
def model():
    return LayeredIsotropicElastic.homogeneous(
        A, rho=RHO, vp=np.sqrt((KAPPA + 4 * MU / 3) / RHO), vs=np.sqrt(MU / RHO)
    )


@pytest.fixture(scope="module")
def numbers(model):
    return traction_numbers(model, 8)


def test_degree_zero_uniform_compression(numbers):
    assert numbers.h_u[0] == pytest.approx(-A / (3.0 * KAPPA), rel=1e-12)
    assert numbers.l_u[0] == 0.0 and numbers.h_v[0] == 0.0 and numbers.l_v[0] == 0.0


def test_degree_one_is_excluded(model, numbers):
    assert np.all(
        np.isnan(
            np.array([getattr(numbers, n)[1] for n in ("h_u", "l_u", "h_v", "l_v")])
        )
    )
    material = Material(RadialMesh(model, ngll=5, lmax=4), model)
    with pytest.raises(ValueError, match="degree 1"):
        DegreeSystem(material, 1, gravitating=False)


def test_reciprocity(numbers):
    assert np.max(numbers.reciprocity_residual()) < 1e-10


def test_mesh_independence_of_the_polynomial_degrees(model, numbers):
    """l + 1 <= 4 solutions are powers of r inside the ngll = 5 element
    space: two unrelated meshes agree to rounding, an analytic-strength
    check on a homogeneous body."""
    other = traction_numbers(
        model, 3, mesh=RadialMesh(model, ngll=6, edges=np.linspace(0.0, A, 7))
    )
    for name in ("h_u", "l_u", "h_v", "l_v"):
        a, b = getattr(numbers, name)[:4], getattr(other, name)
        live = ~np.isnan(a)
        assert np.allclose(a[live], b[live], rtol=1e-11, atol=0.0)


def test_traction_channels_tie_to_the_gravitating_solver(model):
    """On the full problem the traction channels are the load-force
    channels divided by g, by construction of the forcings."""
    material = Material(RadialMesh(model, ngll=5, lmax=6), model)
    grav = traction_numbers(material, 6, gravitating=True)
    love = love_numbers(material, 6)
    g = love.surface_gravity
    assert np.allclose(grav.h_u * g, love.h_u, rtol=1e-13)
    assert np.allclose(grav.l_u * g, love.l_u, rtol=1e-13)
    assert np.allclose(grav.h_v * g, love.h_v, rtol=1e-13)
    assert np.allclose(grav.l_v * g, love.l_v, rtol=1e-13)


def test_gravity_matters(model, numbers):
    """The same forcing with and without gravity differs at low degree
    and converges with l (gravity fades against elasticity)."""
    grav = traction_numbers(model, 8, gravitating=True)
    rel = np.abs(grav.h_u[2:] - numbers.h_u[2:]) / np.abs(numbers.h_u[2:])
    assert rel[0] > 0.05 and rel[-1] < rel[0]


def test_refusals(model):
    prem = PREM(ocean=False)
    material = Material(RadialMesh(prem, ngll=5, lmax=4), prem)
    with pytest.raises(ValueError, match="solid"):
        DegreeSystem(material, 2, gravitating=False)
    with pytest.raises(ValueError, match="solid"):
        traction_numbers(material, 4)
    with pytest.raises(ValueError, match="forcing"):
        solve_degree(model, 2, forcing="load", gravitating=False)


def test_solve_degree_traction_profile(model):
    """The radial solution of the non-gravitating degree 2 is the power
    law combination: U is a polynomial, checked by its machine-exact
    interpolation between nodes, and phi is identically zero."""
    solution = solve_degree(model, 2, forcing="traction", gravitating=False)
    r = np.linspace(0.2 * A, A, 7)
    U, V, phi = solution.evaluate(r)
    assert np.all(phi == 0.0)
    assert np.all(np.isfinite(U)) and np.all(np.isfinite(V))
    assert abs(U[-1] - solution.surface[0]) < 1e-14 * abs(solution.surface[0])


def test_si_conversion_roundtrip(model, numbers):
    from planetmodel.units import Scales

    nd = numbers.converted(Scales(length=A, mass=RHO * A**3, time=1000.0))
    assert nd.radius == pytest.approx(1.0)
    back = nd.converted(numbers.scales)
    assert np.allclose(
        back.h_u[~np.isnan(back.h_u)], numbers.h_u[~np.isnan(numbers.h_u)], rtol=1e-14
    )
