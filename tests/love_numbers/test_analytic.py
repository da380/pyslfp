"""Elastic Love numbers against the Kelvin-sphere closed forms.

For the homogeneous incompressible self-gravitating elastic sphere
every degree-l response is a single-mode fraction of its fluid limit,

    R_l(elastic) = R_l(fluid) / (1 + mu_l),
    mu_l = (2 l^2 + 4 l + 3) mu / (l rho g a),

with the fluid limits, in the conventional dimensionless normalisation,

    load:   h'_f = -(2l + 1) / 3,          k'_f = -1,
    tidal:  h_f  = (2l + 1) / (2(l - 1)),  k_f  = 3 / (2(l - 1)),

(Munk & MacDonald 1960; Wu & Peltier 1982).  The solver is
compressible, so the gate runs at kappa / mu = 1600 and the tolerance
is the measured compressibility residual, well below the leading
mu / kappa correction's room."""

import numpy as np
import pytest

from planetmodel import LayeredIsotropicElastic, RadialMesh
from pyslfp.love_numbers import Material, love_numbers

RHO, VS, RADIUS = 5510.0, 5000.0, 6.371e6
MU = RHO * VS * VS
LMAX = 8


@pytest.fixture(scope="module")
def love():
    model = LayeredIsotropicElastic(
        [0.0, RADIUS], rho=[RHO], vp=[2.0e5], vs=[VS]
    )
    material = Material(RadialMesh(model, ngll=5, lmax=LMAX), model)
    return love_numbers(material, LMAX), material.surface_gravity


def test_load_numbers_match_the_kelvin_sphere(love):
    table, g = love
    conventional = table.conventional()
    for l in range(2, LMAX + 1):
        mu_l = (2.0 * l * l + 4.0 * l + 3.0) * MU / (l * RHO * g * RADIUS)
        want_h = -(2.0 * l + 1.0) / (3.0 * (1.0 + mu_l))
        want_k = -1.0 / (1.0 + mu_l)
        assert abs(conventional["h"][l] / want_h - 1.0) < 2e-3, l
        assert abs(conventional["k"][l] / want_k - 1.0) < 2e-3, l


def test_tidal_numbers_match_the_kelvin_sphere(love):
    table, g = love
    tidal = table.tidal()
    for l in range(2, LMAX + 1):
        mu_l = (2.0 * l * l + 4.0 * l + 3.0) * MU / (l * RHO * g * RADIUS)
        want_h = (2.0 * l + 1.0) / (2.0 * (l - 1.0) * (1.0 + mu_l))
        want_k = 3.0 / (2.0 * (l - 1.0) * (1.0 + mu_l))
        assert abs(tidal["h"][l] / want_h - 1.0) < 2e-3, l
        assert abs(tidal["k"][l] / want_k - 1.0) < 2e-3, l


def test_fluid_tidal_number_is_kelvins():
    """The hydrostatic fluid limit depends only on the density profile:
    the homogeneous sphere gives Kelvin's 3 / (2 (l - 1)) to solver
    precision whatever the compressibility, and isotropic PREM the
    k_hyd of the rotational-stability literature."""
    from planetmodel import PREM
    from pyslfp.love_numbers import fluid_tidal_number

    for vp in (2.0e5, 12000.0):
        model = LayeredIsotropicElastic([0.0, RADIUS], rho=[RHO], vp=[vp], vs=[VS])
        material = Material(RadialMesh(model, ngll=5, lmax=3), model)
        for l in (2, 3):
            want = 3.0 / (2.0 * (l - 1.0))
            assert abs(fluid_tidal_number(material, l) / want - 1.0) < 1e-8, (vp, l)
    prem = PREM(ocean=False).isotropic()
    k_hyd = fluid_tidal_number(prem, 2)
    assert 0.90 < k_hyd < 0.96


def test_fluid_limits_as_the_rigidity_vanishes():
    """With mu 10^6 times smaller the numbers sit on the fluid limits."""
    model = LayeredIsotropicElastic(
        [0.0, RADIUS], rho=[RHO], vp=[2.0e5], vs=[VS * 1e-3]
    )
    material = Material(RadialMesh(model, ngll=5, lmax=4), model)
    table = love_numbers(material, 4)
    conventional = table.conventional()
    tidal = table.tidal()
    for l in (2, 3, 4):
        assert abs(conventional["h"][l] / (-(2.0 * l + 1.0) / 3.0) - 1.0) < 1e-3
        assert abs(conventional["k"][l] / (-1.0) - 1.0) < 1e-3
        assert abs(tidal["k"][l] / (3.0 / (2.0 * (l - 1.0))) - 1.0) < 1e-3
