"""Love numbers: the archived solver's PREM values, identities, closed
forms, convergence, the frozen viscoelastic path."""

import numpy as np
import pytest

from planetmodel import (
    RadialMesh,
    constant_field,
    frozen,
    LayeredIsotropicElastic,
    PREM,
)
from pyslfp.love_numbers import (
    FORCINGS,
    DegreeSystem,
    Material,
    NodalModuli,
    love_numbers,
    nodal_moduli,
    solve_degree,
)
from planetmodel.units import Scales

# The archived solver on the same mesh (lmax = 20, five GLL nodes), which
# matched DSpecM1D: h_u, k_u, h_phi, k_phi, h_t, k_t per degree.
ARCHIVED = {
    1: (-2.3286097586e-04, 0.0, 0.0, 0.0, -1.0174391584e-01, 0.0),
    2: (
        -1.7326967358e-04,
        6.4504099782e-04,
        6.5628996997e-05,
        -1.3870312749e-03,
        -6.1439423073e-02,
        2.9848702880e-01,
    ),
    3: (
        -1.0385192923e-04,
        2.2002034214e-04,
        2.2385731174e-05,
        -8.3336702235e-04,
        -2.9339363085e-02,
        9.2234042360e-02,
    ),
    10: (
        -3.8788052866e-05,
        1.9449567876e-05,
        1.9788751972e-06,
        -2.5612512635e-04,
        -7.7807069321e-03,
        7.0541835679e-03,
    ),
    20: (
        -2.6739346186e-05,
        7.0204472452e-06,
        7.1428779369e-07,
        -1.3057946214e-04,
        -5.4832549952e-03,
        2.3977651315e-03,
    ),
}
# The older Fortran table that pyslfp ships, for the same degrees.
FORTRAN = {
    1: (-0.23329549e-03, 0.0, 0.0, 0.0, -0.10177450e00, 0.0),
    2: (
        -0.17423503e-03,
        0.64614466e-03,
        0.65761010e-04,
        -0.13866845e-02,
        -0.61581437e-01,
        0.29855114e00,
    ),
    3: (
        -0.10462701e-03,
        0.22043313e-03,
        0.22434458e-04,
        -0.83312478e-03,
        -0.29412035e-01,
        0.92243759e-01,
    ),
    10: (
        -0.39052428e-04,
        0.19341942e-04,
        0.19685153e-05,
        -0.25604606e-03,
        -0.77422930e-02,
        0.70450595e-02,
    ),
    20: (
        -0.26846873e-04,
        0.69422565e-05,
        0.70654427e-06,
        -0.13053883e-03,
        -0.54254373e-02,
        0.23862364e-02,
    ),
}
COLUMNS = ("h_u", "k_u", "h_phi", "k_phi", "h_t", "k_t")


@pytest.fixture(scope="module")
def model():
    return PREM(ocean=False)


@pytest.fixture(scope="module")
def material(model):
    return Material(RadialMesh(model, ngll=5, lmax=20), model)


@pytest.fixture(scope="module")
def love(material):
    return love_numbers(material, 20)


def test_material(model, material):
    mesh = material.mesh
    assert material.rho.shape == material.g.shape == (mesh.nspec, mesh.ngll)
    assert material.fluid.shape == (mesh.nspec,)
    assert set(np.unique(mesh.layer[material.fluid])) == {1}
    assert not material.fluid[-1]
    assert material.radius == 6368e3 and abs(material.surface_gravity - 9.8286) < 1e-3
    assert material.G == model.G and not material.is_complex
    assert material.omega is None
    m = material.moduli
    assert m.shape == (mesh.nspec, mesh.ngll) and not m.is_complex
    fluid = material.fluid
    assert np.all(m.L[fluid] == 0.0) and np.all(m.N[fluid] == 0.0)
    assert np.allclose(m.A[fluid], m.C[fluid]) and np.allclose(m.A[fluid], m.F[fluid])
    assert np.any(m.A[~fluid] != m.C[~fluid])  # PREM's anisotropic mantle


def test_material_refusals(model):
    with pytest.raises(ValueError, match="fluid"):
        Material(RadialMesh(PREM(), ngll=5, lmax=4), PREM())
    with pytest.raises(ValueError, match="truncate the model"):
        Material(RadialMesh(model, ngll=5, drmax=500e3, rmax=6000e3), model)
    with pytest.raises(ValueError, match="different skeleton"):
        Material(RadialMesh(PREM(), ngll=5, lmax=4), model)


def test_nodal_moduli_isotropic_and_voigt():
    model = LayeredIsotropicElastic(
        [0.0, 0.5, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0], vs=[0.0, 1.0]
    )
    mesh = RadialMesh(model, ngll=4, drmax=0.25)
    m = nodal_moduli(mesh, model)
    top = mesh.layer == 1
    assert np.allclose(m.L[top], 1.0) and np.allclose(m.N[top], 1.0)
    assert np.allclose(m.A[top], 4.0) and np.allclose(m.F[top], 2.0)
    assert np.allclose(m.L[~top], 0.0) and np.allclose(m.A[~top], 18.0)
    v = m.voigt()
    for n in ("A", "C", "F", "L", "N"):
        assert np.allclose(getattr(v, n), getattr(m, n))
    kappa, mu = m.kappa_mu()
    assert np.allclose(kappa[top], 4.0 - 4.0 / 3.0) and np.allclose(mu[top], 1.0)
    with pytest.raises(ValueError):
        NodalModuli(m.A, m.C, m.F, m.L, m.N[:1])


def test_prem_matches_the_archived_solver(love):
    for l, vals in ARCHIVED.items():
        for name, want in zip(COLUMNS, vals):
            got = getattr(love, name)[l]
            assert abs(got - want) <= 1e-8 * max(abs(want), 1e-30) + 1e-300, (l, name)


def test_prem_within_two_per_cent_of_the_fortran_table(love):
    for l, vals in FORTRAN.items():
        for name, want in zip(COLUMNS, vals):
            got = getattr(love, name)[l]
            if want == 0.0:
                assert got == 0.0
            else:
                assert abs(got / want - 1.0) < 2e-2, (l, name)


def test_identities(love, material):
    assert love.reciprocity_residual().max() < 1e-12
    a, G, g = material.radius, material.G, material.surface_gravity
    assert abs((love.k_u[0] + love.k_phi[0]) / (-4.0 * np.pi * G * a) - 1.0) < 1e-9
    assert love.h_t[0] == 0.0 and love.l_t[0] == 0.0 and np.all(love.k_t[:2] == 0.0)
    assert love.k_phi[1] == 0.0 and love.k_u[1] == 0.0 and love.l_u[0] == 0.0
    assert love.k_phi[0] != 0.0  # the l = 0 potential dof is live
    assert love.h_t[1] != 0.0
    c = love.conventional()
    assert abs(c["k"][1] + 1.0) < 1e-12 and abs(c["k"][0]) < 1e-9
    assert -1.32 < c["h"][1] < -1.25
    assert -1.01 < c["h"][2] < -0.97 and -0.315 < c["k"][2] < -0.295
    assert 0.018 < c["l"][2] < 0.028
    t = love.tidal()
    assert 0.28 < t["k"][2] < 0.32 and 0.57 < t["h"][2] < 0.65
    assert 0.07 < t["l"][2] < 0.09
    assert np.array_equal(love.h, love.h_u + love.h_phi)
    # the tangential-traction column: absent at degree 0, reciprocal above
    assert love.h_v[0] == 0.0 and love.l_v[0] == 0.0 and love.k_v[0] == 0.0
    k2 = love.degree[1:] * (love.degree[1:] + 1.0)
    assert np.allclose(love.h_v[1:], k2 * love.l_u[1:], rtol=1e-12)
    assert np.allclose(love.k_v[1:], g * k2 * love.l_phi[1:], rtol=1e-12)
    assert np.all(love.l_v[1:] != 0.0)
    tangential = solve_degree(material, 3, forcing="load_tangential").surface
    assert abs(tangential[1] - love.l_v[3]) < 1e-15 * abs(love.l_v[3])


def test_degree_convergence(model):
    m1 = RadialMesh(model, ngll=5, lmax=8)
    m2 = RadialMesh(model, ngll=6, drmax=0.5 * m1.drmax)
    a = solve_degree(Material(m1, model), 4).surface
    b = solve_degree(Material(m2, model), 4).surface
    for x, y in zip(a, b):
        assert abs(x - y) <= 1e-5 * abs(y)


def test_solve_degree_and_the_solution(material, love):
    sol = solve_degree(material, 2, forcing="tide")
    U, V, phi = sol.surface
    assert abs(U - love.h_t[2]) < 1e-15 and abs(V - love.l_t[2]) < 1e-15
    assert abs(phi - love.k_t[2]) < 1e-15
    mesh = material.mesh
    Ue, Ve, Pe = sol.evaluate(mesh.r[:, 1:-1])
    assert np.allclose(Ue, sol.U[:, 1:-1], equal_nan=True)
    assert np.allclose(Pe, sol.phi[:, 1:-1])
    assert np.allclose(sol.evaluate(mesh.r[:, 0])[0], sol.U[:, 0], equal_nan=True)
    deep = sol.evaluate(mesh.left[sol.mesh.start_element(2)] * 0.5)[0]
    assert abs(deep) < 1e-12 * abs(U)
    both = solve_degree(material, 3).surface
    force = solve_degree(material, 3, forcing="load_force").surface
    pot = solve_degree(material, 3, forcing="load_potential").surface
    for x, y, z in zip(both, force, pot):
        assert abs(x - (y + z)) < 1e-12 * abs(x)
    with pytest.raises(ValueError, match="forcing"):
        solve_degree(material, 3, forcing="wind")
    assert FORCINGS == (
        "load",
        "load_force",
        "load_potential",
        "load_tangential",
        "tide",
        "centrifugal",
    )
    with pytest.raises(ValueError):
        solve_degree(material, 3, mesh=material.mesh)
    with pytest.raises(ValueError):
        sol.evaluate(7e6)


def test_displacement_is_undefined_in_a_fluid(material):
    """Above degree zero the fluid displacement is not determined, so U
    and V are NaN through the outer core and finite everywhere else, the
    solid side of each boundary keeping its value; phi is finite
    throughout.  At degree zero the fluid is an elastic medium without
    rigidity and everything is finite."""
    mesh = material.mesh
    fluid = material.fluid
    below, above = np.flatnonzero(fluid)[[0, -1]] + np.array([-1, 1])
    for forcing in ("load", "tide"):
        sol = solve_degree(material, 2, forcing=forcing)
        for a in (sol.U, sol.V):
            assert np.all(np.isnan(a[fluid])) and np.all(np.isfinite(a[~fluid]))
        assert np.all(np.isfinite(sol.phi))
        assert sol.U[below, -1] != 0.0 and sol.U[above, 0] != 0.0
        radii = np.linspace(mesh.left[below + 1], mesh.right[above - 1], 50)
        U, V, phi = sol.evaluate(radii)
        assert np.all(np.isnan(U[:-1])) and np.all(np.isnan(V[:-1]))
        assert np.all(np.isfinite(phi))
        assert U[-1] == sol.U[above, 0] and V[-1] == sol.V[above, 0]
    zero = solve_degree(material, 0)
    for a in (zero.U, zero.V, zero.phi):
        assert np.all(np.isfinite(a))
    assert np.any(zero.U[fluid] != 0.0)


def test_from_a_model_builds_the_mesh(model, love):
    quick = love_numbers(model, 4)
    for name in COLUMNS:
        assert np.all(np.isfinite(getattr(quick, name)))
    c = quick.conventional()
    assert abs(c["h"][2] / love.conventional()["h"][2] - 1.0) < 1e-3


def test_degree_system(material):
    s = DegreeSystem(material, 2)
    assert s.ndof == s.band.shape[1] and s.kd == 3 * (material.mesh.ngll - 1) + 2
    assert s.surface_dof("phi") >= 0 and s.surface_dof("U") >= 0
    s1 = DegreeSystem(material, 1)
    assert s1.surface_dof("phi") == -1
    s0 = DegreeSystem(material, 0)
    assert s0.surface_dof("V") == -1 and s0.dof[0, 0] == -1
    x = s.solve(s.load())
    assert x.shape == (s.ndof,)
    X = s.solve(np.column_stack([s.load(), s.tide()]))
    assert np.allclose(X[:, 0], x)
    U, V, phi = s.expand(x)
    assert U.shape == material.rho.shape and np.all(U[: s.first_element] == 0.0)
    with pytest.raises(ValueError):
        s.load(part="sideways")


def test_ti_differs_from_voigt(model):
    """The isotropic re-description by the Voigt average is read as isotropic."""
    mesh = RadialMesh(model, ngll=5, lmax=8)
    iso = model.isotropic()
    ti = solve_degree(Material(mesh, model), 8).surface[0]
    voigt = solve_degree(Material(mesh, iso), 8).surface[0]
    assert 1e-3 < abs(ti - voigt) / abs(voigt) < 2e-2
    m = nodal_moduli(mesh, iso)
    assert np.allclose(m.A, nodal_moduli(mesh, model).voigt().A)


def test_nondimensional_model_gives_the_same_numbers(model, love):
    nd = model.nondimensionalised()
    mesh = RadialMesh(nd, ngll=5, lmax=20)
    got = love_numbers(Material(mesh, nd), 20)
    assert got.scales == nd.scales and abs(got.G - 1.0) < 1e-12
    for key in ("h", "l", "k"):
        assert np.allclose(
            got.conventional()[key], love.conventional()[key], rtol=1e-9, atol=1e-9
        )
        assert np.allclose(got.tidal()[key], love.tidal()[key], rtol=1e-9)
    si = got.in_si()
    assert si.scales == Scales.SI
    for name in COLUMNS + ("l_u", "l_phi", "l_t"):
        assert np.allclose(getattr(si, name), getattr(love, name), rtol=1e-9)
    assert abs(si.radius - love.radius) < 1e-3 and abs(si.G / love.G - 1.0) < 1e-12
    assert love.converted(Scales.SI) is love


def test_incompressible_homogeneous_sphere():
    """The closed forms of the incompressible homogeneous sphere:
    tidal k = 3/(2(n-1)) / (1 + mu_n), h = (2n+1)/(2(n-1)) / (1 + mu_n),
    l = 3/(2n(n-1)) / (1 + mu_n); load k' = -1 / (1 + mu_n),
    h' = -(2n+1)/3 / (1 + mu_n), with mu_n = (2n^2+4n+3) mu / (n rho g a)."""
    rho, mu, a = 5500.0, 1.0e11, 6371e3
    kappa = 1e6 * mu
    model = LayeredIsotropicElastic.homogeneous(
        a, rho=rho, vp=np.sqrt((kappa + 4 * mu / 3) / rho), vs=np.sqrt(mu / rho)
    )
    love = love_numbers(Material(RadialMesh(model, ngll=6, lmax=8), model), 8)
    g = love.surface_gravity
    n = love.degree[2:].astype(float)
    mun = (2 * n**2 + 4 * n + 3) * mu / (n * rho * g * a)
    t, c = love.tidal(), love.conventional()
    assert np.allclose(t["k"][2:], 3 / (2 * (n - 1)) / (1 + mun), rtol=5e-6)
    assert np.allclose(t["h"][2:], (2 * n + 1) / (2 * (n - 1)) / (1 + mun), rtol=5e-6)
    assert np.allclose(t["l"][2:], 3 / (2 * n * (n - 1)) / (1 + mun), rtol=5e-6)
    assert np.allclose(c["k"][2:], -1 / (1 + mun), rtol=5e-6)
    assert np.allclose(c["h"][2:], -(2 * n + 1) / 3 / (1 + mun), rtol=5e-6)


def test_maxwell_limits_of_a_frozen_model():
    rho, mu, a, eta = 5500.0, 1.0e11, 6371e3, 1e21
    model = LayeredIsotropicElastic.homogeneous(
        a, rho=rho, vp=8000.0, vs=np.sqrt(mu / rho)
    )
    model = model.with_field(
        0, "viscosity", constant_field((0.0, a), eta, name="viscosity")
    )
    mesh = RadialMesh(model, ngll=5, lmax=4)
    elastic = love_numbers(Material(mesh, model), 4)
    tau = eta / mu

    def at(omega):
        mat = Material(mesh, frozen(model, omega))
        assert mat.is_complex and mat.omega == omega
        return love_numbers(mat, 4)

    fast = at(1e6 / tau)
    assert fast.is_complex and fast.omega == 1e6 / tau
    assert abs(fast.tidal()["k"][2] - elastic.tidal()["k"][2]) < 1e-5
    assert abs(fast.tidal()["k"][2].imag) < 1e-5
    slow = at(1e-5 / tau)
    assert abs(slow.tidal()["k"][2] - 1.5) < 1e-3  # a fluid sphere
    # near the fluid limit the system is ill-conditioned and the tangential
    # identities, weighted by l(l + 1), close a decade less tightly
    assert slow.reciprocity_residual().max() < 1e-10
    mid = at(1.0 / tau)
    assert abs(mid.tidal()["k"][2].imag) > 0.1
    sol = solve_degree(Material(mesh, frozen(model, 1.0 / tau)), 2, forcing="tide")
    assert sol.U.dtype.kind == "c"
    assert np.allclose(sol.evaluate(mesh.r[:, 1:-1])[2], sol.phi[:, 1:-1])
    with pytest.raises(ValueError, match="real"):
        mid.write("/dev/null")
    nd = mid.converted(PREM().nondimensionalised().scales)
    assert nd.omega != mid.omega and np.isclose(nd.in_si().omega, mid.omega)


def test_frozen_prem_at_a_tidal_period(material):
    """PREM's own Q gives complex numbers a hair off the elastic ones."""
    omega = 2.0 * np.pi / 43200.0
    cold = Material(material.mesh, frozen(material.model, omega))
    assert cold.is_complex and cold.omega == omega
    love = love_numbers(cold, 4)
    warm = love_numbers(material, 4)
    k2 = love.tidal()["k"][2]
    assert abs(k2.real / warm.tidal()["k"][2] - 1.0) < 2e-2
    assert 0.0 < abs(k2.imag) < 1e-2
    assert love.reciprocity_residual().max() < 1e-12


# ==================================================================== #
#                     The axial numbers of degree 0                    #
# ==================================================================== #


def test_axial_numbers(material):
    """
    The degree-0 response to the centrifugal potential (r/a)^2 and the
    inertia moments: reciprocity of the bilinear form ties the moment of
    the force-channel response to the surface displacement under the
    centrifugal potential, the exterior potential of a spherically
    symmetric deformation vanishes (so k_c and the moment of the
    potential-channel response do too), and a spin-up expands the body.
    """
    from pyslfp.love_numbers import inertia_moment
    from pyslfp.love_numbers.table import AXIAL_NAMES

    love = love_numbers(material, 2)
    assert love.has_axial
    assert love.axial_reciprocity_residual() < 1e-12
    assert abs(love.k_c) < 1e-10 * abs(love.k_t[2])
    assert abs(love.m_phi) < 1e-10 * abs(love.m_u)
    # a negative potential coefficient is a spin-up, an outward body force
    assert love.h_c < 0.0 and love.m_c < 0.0
    # a downward traction compresses the body
    assert love.m_u < 0.0

    # the same numbers from the single-degree solves
    spin = solve_degree(material, 0, forcing="centrifugal")
    assert abs(spin.surface[0] - love.h_c) < 1e-12 * abs(love.h_c)
    assert abs(inertia_moment(material, spin.U) - love.m_c) < 1e-12 * abs(love.m_c)
    # the centrifugal forcing at degree 2 is the harmonic tide
    tide = solve_degree(material, 2, forcing="tide").surface
    same = solve_degree(material, 2, forcing="centrifugal").surface
    for x, y in zip(tide, same):
        assert abs(x - y) < 1e-12 * abs(x)
    # a constant potential is a gauge
    system = DegreeSystem(material, 0)
    assert not np.any(system.tide())

    # units and the file carry them
    nd = love.converted(Scales(length=love.radius, mass=1e24, time=1e3))
    assert nd.has_axial and nd.axial_reciprocity_residual() < 1e-12
    for name in AXIAL_NAMES:
        if name != "k_c":  # dimensionless
            assert getattr(nd, name) != getattr(love, name)
    assert nd.in_si().m_c == pytest.approx(love.m_c, rel=1e-14)
    assert love.truncated(1).has_axial
