# %% [markdown]
# # 9. The decay time of postglacial emergence, and what it knows of mantle viscosity
#
# Close to the centre of a former ice sheet, and once the melting has
# stopped, relative sea level falls along a curve that is well described
# by a decaying exponential; its decay time is a classic datum of GIA,
# measured around Hudson Bay and the Gulf of Bothnia, because it is
# insensitive to the poorly known ice thickness and directly shaped by
# mantle viscosity. This script computes that datum in a forward model
# and then its sensitivity kernel with respect to viscosity at every
# depth, in three steps.
#
# First a forward run: a Laurentide-scale cap melts over ten thousand
# years on a viscoelastic PREM and the following five thousand years of
# emergence are recorded at a site under the cap's centre. The run also
# records its internal states on a checkpoint stride: between
# checkpoints the adjoint reconstructs the trajectory by exact backward
# steps, so the full volumetric history is never stored -- dissipation
# amplifies roundoff by exp(dt / tau) per reconstructed step, and the
# stride is chosen to spend a few of the sixteen digits available.
#
# Second, the fit. The decay time is not a field value but the
# parameter of a subsidiary least-squares problem, so its derivative
# with respect to the sea-level series comes from differentiating the
# fit's stationarity conditions. Following Al-Attar & Tromp (2014,
# Appendix C), the model times join the fit as free parameters penalised
# towards the nominal ones, which accounts for the timing uncertainty of
# real sea-level index points; the two uncertainties below are a
# modelling choice, made explicitly, and only their ratio matters.
#
# Third, the kernels: the fit's weights become a multi-date measurement,
# one reversed sweep returns the exact derivative of the decay time with
# respect to the inverse relaxation times at every node of the radial
# mesh, and the chain rule turns that into a viscosity kernel. The
# figure shows the sensitivity of the decay time to a uniform fractional
# change of viscosity in each element of the mesh. For a cap of
# Laurentide scale the kernel reaches deep -- the load's low harmonics
# feel the lower mantle -- and the two regions pull with opposite signs,
# the trade-off that viscosity inversions from such data must resolve.
#
# The run takes about two minutes at degree 64.

# %%
import matplotlib.pyplot as plt
import numpy as np
from planetmodel import PREM, units
from pyshtools import SHGrid

import pyslfp as sl
from pyslfp.ice import BaseIceModel, IceNG
from pyslfp.love_numbers import Material, graded_mesh
from pyslfp.viscoelastic import (
    Rheology,
    SeaLevelEvolution,
    ViscoelasticOperator,
    decay_time_measurement,
)

LMAX = 64

# %% [markdown]
# ## The earth model and rheology
#
# Those of the forward tutorial, converted into the library's frame, with
# the viscosities converted once.

# %%
scales = sl.EarthModelParameters.from_defaults().scales
model = PREM(ocean=False).isotropic().converted(scales)
mesh = graded_mesh(model, LMAX)
material = Material(mesh, model)

pascal_second = 1.0 / scales.factor(units.VISCOSITY)
viscosity = {i: 2.0e21 * pascal_second for i in (2, 3, 4)}
viscosity.update({i: 5.0e20 * pascal_second for i in (5, 6, 7, 8)})
rheology = Rheology.maxwell(material, viscosity)
operator = ViscoelasticOperator(rheology, LMAX)

earth_model = sl.EarthModel.from_planet_model(model, LMAX, mesh=mesh)
background = IceNG(length_scale=earth_model.parameters.length_scale)
state = sl.EarthState.from_ice_model(background, 0.0, earth_model)
evolution = SeaLevelEvolution(operator, state)

parameters = earth_model.parameters
metre = parameters.length_scale

# %% [markdown]
# ## The ice history
#
# A tapered cap of 1500 m over Hudson Bay melting linearly between
# fifteen and five thousand years before present and gone thereafter,
# so the last five thousand years are load-free and the site's curve
# decays.

# %%
cap = state.disk_load(10.0, 60.0, -85.0, 1500.0 / metre)
taper = np.exp(-((np.arange(LMAX + 1) / 32.0) ** 4))
cap_lm = earth_model.expand_field(cap)
cap_lm.coeffs *= taper[None, :, None]
cap = earth_model.expand_coefficient(cap_lm)


class CapHistory(BaseIceModel):
    """The tapered cap, melting from 15 to 5 ka and gone after."""

    def get_ice_thickness(self, date, lmax, /, *, grid="DH", sampling=1, extend=True):
        fraction = np.clip((date - 5.0) / 10.0, 0.0, 1.0)
        return SHGrid.from_array(
            fraction * cap.data, grid=self.resolve_grid(grid, sampling)[0]
        )


ice_history = CapHistory(length_scale=metre)
dates = np.linspace(15.0, 0.0, 31)

snapshots, _ = evolution.run_ice_history(dates, ice_history, record=5)

# %% [markdown]
# ## The site series and the fit
#
# The site is a five-degree disc under the cap's centre; its relative
# sea level falls as the land rebounds, and the fall over the post-melt
# window is close to exponential. The value uncertainty is two per cent
# of the emergence and the timing uncertainty a tenth of a kiloyear --
# the explicit choices of the estimator.

# %%
patch = state.disk_load(5.0, 60.0, -85.0, 1.0)
patch_lm = earth_model.expand_field(patch)
patch_lm.coeffs *= taper[None, :, None]
patch = earth_model.expand_coefficient(patch_lm)
site_dual = (metre / earth_model.integrate(patch)) * patch

window = dates[dates <= 4.0 + 1e-9]
series = np.array(
    [
        earth_model.integrate(site_dual * snap.sea_level_change)
        for snap in snapshots
        if snap.date <= 4.0 + 1e-9
    ]
)

fit, measurement = decay_time_measurement(
    window,
    series,
    site_dual,
    time_uncertainty=0.1,
    value_uncertainty=0.02 * np.ptp(series),
)
print(f"decay time of the site's emergence: {fit.decay_time:.2f} kyr")

elapsed = window[0] - window
curve = fit.amplitude * np.exp(-elapsed / fit.decay_time) + fit.offset

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(window, series, "o", label="model sea level at the site")
ax.plot(window, curve, "-", label=f"fit, decay time {fit.decay_time:.2f} kyr")
ax.set_xlabel("Kiloyears before present")
ax.set_ylabel("Relative sea level change (m)")
ax.invert_xaxis()
ax.legend()
ax.set_title("Post-melt emergence at the cap's centre")

# %% [markdown]
# ## The kernels
#
# The fit's weights turn the decay time into a multi-date measurement,
# and the reversed sweep -- pairing its duals with the recorded forward
# trajectory -- returns the exact derivative with respect to the inverse
# relaxation time at every node. The nodal values carry the mesh's
# quadrature, so dividing it out leaves the kernel as a continuous
# function of radius: the density $k(r)$ with
#
# $$\delta\tau_d = \int k(r)\, \delta(\ln\eta)(r)\, dr,$$
#
# sampled at the GLL nodes and plotted against depth. Its integrals over
# the two mantle regions, quoted below, are the responses to uniform
# fractional viscosity changes there.

# %%
kernels = evolution.adjoint_ice_history(dates, measurement, internal_states=snapshots)

log_eta_nodal = -rheology.itau[0] * kernels.itau[0]
measure = mesh.jac[:, None] * mesh.w[None, :]
km = parameters.length_scale / 1e3
density_r = log_eta_nodal / measure / km  # kyr per d(ln eta) per km

node_depth = (mesh.r.max() - mesh.r) * km
depth_km = node_depth.ravel()
order = np.argsort(depth_km)

fig, ax = plt.subplots(figsize=(10, 6))
mantle = depth_km[order] < 2900.0
ax.plot(depth_km[order][mantle], density_r.ravel()[order][mantle])
ax.axhline(0.0, color="black", linewidth=0.5)
ax.set_xlabel("Depth (km)")
ax.set_ylabel("d(decay time) / d(ln eta) (kyr per km)")
ax.set_title("Viscosity sensitivity of the decay time")

upper = float(np.sum(log_eta_nodal[(node_depth > 77.0) & (node_depth < 670.0)]))
lower = float(np.sum(log_eta_nodal[(node_depth >= 670.0) & (node_depth < 2900.0)]))
print(f"a 1% rise of upper-mantle viscosity changes it by {0.01 * upper * 1e3:+.2f} yr")
print(f"the same in the lower mantle changes it by {0.01 * lower * 1e3:+.2f} yr")

# %% [markdown]
# ## The resolution diagnostic
#
# As for the ice kernels: the continuous adjoint's viscosity kernel
# differs from the exact discrete one by the commute error of the time
# discretisation.

# %%
continuous = evolution.continuous_adjoint_ice_history(
    dates, measurement, internal_states=snapshots
)
disagreement = np.linalg.norm(continuous.itau[0] - kernels.itau[0]) / np.linalg.norm(
    kernels.itau[0]
)
print(f"continuous vs discrete viscosity kernel: {100.0 * disagreement:.2f}%")

plt.show()
