"""The two-frame covariance gate for the Earth model's own scales.

The library's outputs are physical, so running the same body, history
and (SI-anchored) measurements under two different non-dimensional
schemes for `EarthModelParameters` must give the same SI-converted
answers.  This guards the scales refactor (docs/planning/
scales_plan.md): any formula that silently dropped a factor because
the default scheme makes it one (g, G, a density ratio) breaks
covariance here before it breaks a physical result.

The material's frame is held fixed (planetmodel's geophysical scales);
only the sea-level side's scheme varies, which is exactly the boundary
the conversion factors in `SeaLevelEvolution` cross.
"""

import numpy as np
import pytest
from planetmodel import LayeredIsotropicElastic, RadialMesh, units
from pyshtools import SHCoeffs, SHGrid

from pyslfp.core import EarthModel, EarthState
from pyslfp.core.model import (
    GRAVITATIONAL_ACCELERATION,
    MASS,
    MEAN_RADIUS,
    EarthModelParameters,
)
from pyslfp.ice import AnalyticalIceModel
from pyslfp.love_numbers import Material, fluid_tidal_number, love_numbers
from pyslfp.viscoelastic import (
    Measurement,
    Rheology,
    SeaLevelEvolution,
    ViscoelasticOperator,
)

LMAX = 4
TAU = 6.0e6  # the Maxwell time in the material's (geophysical) units
SURFACE_DENSITY = units.DENSITY * units.LENGTH
POTENTIAL = units.LENGTH**2 * units.TIME**-2
PHI_ANCHOR = 1.0e7  # m^2 s^-2
OMEGA_ANCHOR = 1.0e-4  # s^-1

SCHEMES = {
    # g = 1: the scheme of the pre-refactor defaults
    "classic": dict(
        length_scale=MEAN_RADIUS,
        density_scale=3.0 * MASS / (4.0 * np.pi * MEAN_RADIUS**3),
        time_scale=np.sqrt(MEAN_RADIUS / GRAVITATIONAL_ACCELERATION),
    ),
    # G = 1: planetmodel's documented geophysical scales
    "geophysical": dict(
        length_scale=units.Scales.geophysical(MEAN_RADIUS).factor(units.LENGTH),
        density_scale=units.Scales.geophysical(MEAN_RADIUS).factor(units.DENSITY),
        time_scale=units.Scales.geophysical(MEAN_RADIUS).factor(units.TIME),
    ),
}


@pytest.fixture(scope="module")
def body():
    """The homogeneous test body, authored in the geophysical frame;
    each scheme ingests its own converted copy."""
    scales = units.Scales.geophysical(MEAN_RADIUS)
    return LayeredIsotropicElastic(
        [0.0, 1.0], rho=[1.0], vp=[3.1], vs=[1.3], scales=scales
    )


def _setting(body, scheme):
    """The body converted into the scheme's frame, with its material,
    Love numbers and earth model all living there."""
    s = SCHEMES[scheme]
    scales = units.Scales(
        length=s["length_scale"],
        mass=s["density_scale"] * s["length_scale"] ** 3,
        time=s["time_scale"],
    )
    model = body.converted(scales)
    mesh = RadialMesh(model, ngll=5, lmax=LMAX)
    material = Material(mesh, model)
    table = love_numbers(material, LMAX)
    params = EarthModelParameters(**SCHEMES[scheme])
    base = EarthModel(LMAX, parameters=params, love_numbers=table)
    parameters = base.parameters.with_hydrostatic_moments(
        fluid_tidal_number(material, 2)
    )
    return material, EarthModel(LMAX, parameters=parameters, love_numbers=table)


def _si_outputs(body, scheme):
    """The forward run and both kernel methods under one scheme, with
    every output converted to SI."""
    material, earth = _setting(body, scheme)
    scales = earth.parameters.scales
    ice_model = AnalyticalIceModel(length_scale=earth.parameters.length_scale)
    state = EarthState.from_ice_model(ice_model, 10.0, earth, exclude_caspian=False)
    # the Maxwell time, authored in the geophysical frame, in this
    # scheme's units
    tau = (
        10.0
        * TAU
        * units.Scales.geophysical(MEAN_RADIUS).factor(units.TIME)
        / scales.factor(units.TIME)
    )
    op = ViscoelasticOperator(Rheology.from_relaxation_time(material, tau), LMAX)
    evo = SeaLevelEvolution(op, state)

    # the measurement duals, SI-anchored so every scheme poses the same
    # physical functional (the potential dual scales as T^2 and the
    # angular-velocity dual as 1/T, the trap of open-issues issue 2)
    rng = np.random.default_rng(7)

    def dual():
        c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
        for degree in range(LMAX + 1):
            c[:, degree, degree + 1 :] = 0.0
        c[1, :, 0] = 0.0
        return earth.expand_coefficient(
            SHCoeffs.from_array(
                c, normalization=earth.normalization, csphase=earth.csphase
            )
        )

    pot_conv = scales.factor(POTENTIAL) / PHI_ANCHOR
    om_conv = (1.0 / scales.factor(units.TIME)) / OMEGA_ANCHOR
    g_xi, g_u, g_phi = dual(), dual(), dual()
    dates = np.linspace(10.0, 0.0, 6)
    measurement = {
        float(dates[2]): g_xi,
        0.0: Measurement(
            displacement=g_u,
            gravity_potential=SHGrid.from_array(
                pot_conv * g_phi.data, grid=earth.grid_name, copy=False
            ),
            angular_velocity=om_conv * np.array([0.2, -0.4, 0.8]),
        ),
    }

    snapshots, _ = evo.run_ice_history(dates, ice_model, record=True)
    states = [snap.internal_state for snap in snapshots]
    out = {
        "xi": snapshots[-1].sea_level_change.data * scales.factor(units.LENGTH)
    }
    load_si = 1.0 / (scales.factor(SURFACE_DENSITY) * scales.factor(units.LENGTH) ** 2)
    for name, method in (
        ("discrete", evo.adjoint_ice_history),
        ("continuous", evo.continuous_adjoint_ice_history),
    ):
        kernels = method(dates, measurement, internal_states=states, rtol=1e-11)
        out[f"{name} load duals"] = (
            np.stack([k.data for k in kernels.load_duals]) * load_si
        )
        # J is dimensionless by construction; the kernel pairs with
        # d(itau) in the scheme's inverse time
        out[f"{name} itau"] = kernels.itau[0] * scales.factor(units.TIME)
    return out


def test_scheme_covariance(body):
    results = {scheme: _si_outputs(body, scheme) for scheme in SCHEMES}
    a, b = (results[s] for s in ("classic", "geophysical"))
    for name in a:
        resid = np.linalg.norm(a[name] - b[name]) / np.linalg.norm(a[name])
        assert resid < 1e-8, f"{name}: {resid:.3e}"
