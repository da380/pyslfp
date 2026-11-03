"""The homogeneous test body the viscoelastic tests share, in the geophysical
non-dimensionalisation: unit radius, near-reference density, Earth-like
moduli, G equal to one."""

import numpy as np
from planetmodel import LayeredIsotropicElastic, RadialMesh, units
from pyshtools import SHGrid

from pyslfp.core import EarthModel, EarthState
from pyslfp.love_numbers import Material, fluid_tidal_number, love_numbers

SCALES = units.Scales.geophysical(6.371e6)
RHO, VS, VP = 1.0, 1.3, 3.1
LMAX = 4
TAU = 6.0e6  # the Maxwell time, in the material's units (about 310 years)


def homogeneous_setting():
    """The material, the Earth model, a uniform-ocean state, a band-limited
    polar cap load and the Maxwell time in the Earth model's time unit."""
    model = LayeredIsotropicElastic(
        [0.0, 1.0], rho=[RHO], vp=[VP], vs=[VS], scales=SCALES
    )
    mesh = RadialMesh(model, ngll=5, lmax=LMAX)
    material = Material(mesh, model)
    # the homogeneous body sustains far less oblateness than Earth's
    # moments imply (its fluid tidal number is Kelvin's 3/2), so the
    # transverse feedback needs the hydrostatic-moment construction
    base = EarthModel.from_planet_model(model, LMAX, mesh=mesh)
    parameters = base.parameters.with_hydrostatic_moments(
        fluid_tidal_number(material, 2)
    )
    earth = EarthModel(
        LMAX, parameters=parameters, love_numbers=love_numbers(material, LMAX)
    )
    ice = earth.zero_grid()
    sea = earth.constant_grid(3000.0 / earth.parameters.length_scale)
    state = EarthState(ice, sea, earth)
    lats = state.model.lats()[:, None] * np.ones_like(state.model.lons())[None, :]
    thickness = 100.0 / earth.parameters.length_scale
    cap = np.where(lats > 45.0, earth.parameters.ice_density * thickness, 0.0)
    load = SHGrid.from_array(cap * np.ones_like(ice.data), grid=earth.grid, copy=False)
    # band-limit the sharp cap so every solver sees the same field
    load = state.model.expand_coefficient(state.model.expand_field(load))
    # the Maxwell time in the earth model's time unit: a pure ratio of
    # the two non-dimensionalisations
    tau_nd = (
        TAU
        * material.scales.factor(units.TIME)
        / earth.parameters.scales.factor(units.TIME)
    )
    return material, earth, state, load, tau_nd
