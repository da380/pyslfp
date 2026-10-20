"""Rheology construction: the Maxwell and relaxation-time
parameterisations, the refusals, and the effective moduli."""

import numpy as np
import pytest

from planetmodel import (
    PREM,
    AnalyticField,
    LayeredIsotropicElastic,
    RadialMesh,
    constant_field,
)
from pyslfp.love_numbers import Material
from pyslfp.viscoelastic import Rheology

RHO, VS, VP = 5510.0, 5000.0, 12000.0
RADIUS = 6.371e6
MU = RHO * VS * VS
ETA = 1e21


@pytest.fixture(scope="module")
def material():
    model = LayeredIsotropicElastic([0.0, RADIUS], rho=[RHO], vp=[VP], vs=[VS])
    return Material(RadialMesh(model, ngll=5, lmax=2), model)


@pytest.fixture(scope="module")
def layered_material():
    """Solid lower mantle, fluid middle, solid top: layers 0, 1, 2."""
    model = LayeredIsotropicElastic(
        [0.0, 0.4 * RADIUS, 0.7 * RADIUS, RADIUS],
        rho=[2.0 * RHO, 1.5 * RHO, RHO],
        vp=[1.5 * VP, VP, VP],
        vs=[1.2 * VS, 0.0, VS],
    )
    return Material(RadialMesh(model, ngll=5, lmax=2), model)


def test_maxwell_constant_viscosity(material):
    rheo = Rheology.maxwell(material, ETA)
    assert rheo.num_branches == 1
    assert np.allclose(rheo.mu[0], MU, rtol=1e-12)
    assert np.allclose(rheo.itau[0], MU / ETA, rtol=1e-12)
    assert np.allclose(rheo.mu_long_term, 0.0, atol=1e-6 * MU)
    assert rheo.active.all()
    assert abs(rheo.min_relaxation_time() - ETA / MU) < 1e-6 * ETA / MU


def test_relaxation_time_parameterisation(material):
    tau = 250.0 * 365.25 * 86400.0
    rheo = Rheology.from_relaxation_time(material, tau)
    assert np.allclose(rheo.itau[0], 1.0 / tau, rtol=1e-12)
    assert np.allclose(rheo.mu[0], MU, rtol=1e-12)
    # radially varying: tau doubling with radius
    rheo = Rheology.from_relaxation_time(material, lambda r: tau * (1.0 + r / RADIUS))
    mesh = material.mesh
    assert np.allclose(rheo.itau[0], 1.0 / (tau * (1.0 + mesh.r / RADIUS)))


def test_two_layer_viscosity(layered_material):
    mesh = layered_material.mesh
    rheo = Rheology.maxwell(layered_material, {0: 10.0 * ETA, 2: ETA})
    lower = mesh.layer == 0
    top = mesh.layer == 2
    assert np.all(rheo.active[lower].all(axis=-1)) if lower.any() else True
    mu = rheo.mu[0]
    itau = rheo.itau[0]
    assert np.all(mu[lower] > 0.0) and np.all(mu[top] > 0.0)
    assert np.all(mu[mesh.layer == 1] == 0.0)
    assert np.allclose(itau[lower] * 10.0 * ETA, mu[lower], rtol=1e-12)
    assert np.allclose(itau[top] * ETA, mu[top], rtol=1e-12)


def test_refusals(material, layered_material):
    with pytest.raises(ValueError, match="anisotropic"):
        prem = PREM(ocean=False)
        m = Material(RadialMesh(prem, ngll=5, lmax=2), prem)
        Rheology.maxwell(m, ETA)
    with pytest.raises(ValueError, match="fluid"):
        Rheology.maxwell(layered_material, {1: ETA})
    with pytest.raises(ValueError, match="positive"):
        Rheology.maxwell(material, -ETA)
    with pytest.raises(ValueError, match="exceed"):
        mu = np.full(material.moduli.shape, MU)
        itau = np.full_like(mu, MU / ETA)
        Rheology(material, [(mu, itau), (mu, itau)])
    with pytest.raises(ValueError, match="no elements"):
        Rheology.maxwell(material, {7: ETA})


def test_effective_moduli(material):
    rheo = Rheology.maxwell(material, ETA)
    ones = [np.ones_like(rheo.mu[0])]
    eff = rheo.effective_moduli(ones)
    for name in ("A", "C", "F", "L", "N"):
        assert np.allclose(
            getattr(eff, name), getattr(material.moduli, name), rtol=1e-14
        )
    relaxed = rheo.long_term_material().moduli
    kappa = material.moduli.kappa_mu()[0]
    assert np.allclose(relaxed.L, 0.0, atol=1e-6 * MU)
    assert np.allclose(relaxed.A, kappa, rtol=1e-6)


def test_effective_moduli_leave_elastic_layers_alone(layered_material):
    rheo = Rheology.maxwell(layered_material, {0: ETA})
    eff = rheo.effective_moduli([np.full(rheo.mu[0].shape, 0.3)])
    mesh = layered_material.mesh
    outside = mesh.layer != 0
    for name in ("A", "C", "F", "L", "N"):
        a, b = getattr(eff, name), getattr(layered_material.moduli, name)
        assert np.array_equal(a[outside], b[outside])
        assert not np.allclose(a[~outside], b[~outside])


def test_from_model(material):
    model = material.model.with_field(
        0, "viscosity", constant_field((0.0, RADIUS), ETA)
    )
    m = Material(material.mesh, model)
    rheo = Rheology.from_model(m)
    assert np.allclose(rheo.itau[0], MU / ETA, rtol=1e-12)
    with pytest.raises(ValueError, match="viscosity"):
        Rheology.from_model(material)
    lateral = AnalyticField(
        (0.0, RADIUS), lambda r, t, p: ETA * (1.0 + 0.1 * np.cos(t))
    )
    aspherical = Material(
        material.mesh, material.model.with_field(0, "viscosity", lateral)
    )
    with pytest.raises(ValueError, match="spherically_averaged"):
        Rheology.from_model(aspherical)
