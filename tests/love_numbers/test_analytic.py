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

from planetmodel import LayeredIsotropicElastic, RadialMesh, units
from pyslfp.love_numbers import Material, love_numbers

# the test body lives in the geophysical non-dimensionalisation: unit
# radius, near-reference density, Earth-like rigidity, G equal to one
SCALES = units.Scales.geophysical(6.371e6)
RHO, VS = 1.0, 1.3
LMAX = 8


def _kelvin_mu_l(material, l):
    """The Kelvin modulus ratio mu_l, from the material's own fields
    so the expression holds in any unit system."""
    mu = float(np.max(np.asarray(material.moduli.kappa_mu()[1]).real))
    rho = float(np.max(np.asarray(material.rho).real))
    return (
        (2.0 * l * l + 4.0 * l + 3.0)
        * mu
        / (l * rho * material.surface_gravity * material.radius)
    )


@pytest.fixture(scope="module")
def love():
    model = LayeredIsotropicElastic(
        [0.0, 1.0], rho=[RHO], vp=[52.0], vs=[VS], scales=SCALES
    )
    material = Material(RadialMesh(model, ngll=5, lmax=LMAX), model)
    return love_numbers(material, LMAX), material


def test_load_numbers_match_the_kelvin_sphere(love):
    table, material = love
    conventional = table.conventional()
    for l in range(2, LMAX + 1):
        mu_l = _kelvin_mu_l(material, l)
        want_h = -(2.0 * l + 1.0) / (3.0 * (1.0 + mu_l))
        want_k = -1.0 / (1.0 + mu_l)
        assert abs(conventional["h"][l] / want_h - 1.0) < 2e-3, l
        assert abs(conventional["k"][l] / want_k - 1.0) < 2e-3, l


def test_tidal_numbers_match_the_kelvin_sphere(love):
    table, material = love
    tidal = table.tidal()
    for l in range(2, LMAX + 1):
        mu_l = _kelvin_mu_l(material, l)
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

    for vp in (52.0, 3.1):
        model = LayeredIsotropicElastic(
            [0.0, 1.0], rho=[RHO], vp=[vp], vs=[VS], scales=SCALES
        )
        material = Material(RadialMesh(model, ngll=5, lmax=3), model)
        for l in (2, 3):
            want = 3.0 / (2.0 * (l - 1.0))
            assert abs(fluid_tidal_number(material, l) / want - 1.0) < 1e-8, (vp, l)
    prem = PREM(ocean=False).isotropic().nondimensionalised()
    k_hyd = fluid_tidal_number(prem, 2)
    assert 0.90 < k_hyd < 0.96


def test_fluid_limits_as_the_rigidity_vanishes():
    """With mu 10^6 times smaller the numbers sit on the fluid limits."""
    model = LayeredIsotropicElastic(
        [0.0, 1.0], rho=[RHO], vp=[52.0], vs=[VS * 1e-3], scales=SCALES
    )
    material = Material(RadialMesh(model, ngll=5, lmax=4), model)
    table = love_numbers(material, 4)
    conventional = table.conventional()
    tidal = table.tidal()
    for l in (2, 3, 4):
        assert abs(conventional["h"][l] / (-(2.0 * l + 1.0) / 3.0) - 1.0) < 1e-3
        assert abs(conventional["k"][l] / (-1.0) - 1.0) < 1e-3
        assert abs(tidal["k"][l] / (3.0 / (2.0 * (l - 1.0))) - 1.0) < 1e-3
