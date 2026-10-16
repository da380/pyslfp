"""Ice-history sensitivity kernels.

Where, and when, would an observation have felt the ice?  The adjoint
of the viscoelastic sea level equation answers exactly that: for an
objective functional of the solution -- here the present-day sea level
averaged over a patch of the central Pacific -- it returns the
derivative of the objective with respect to the ice thickness at every
point and every time of a glacial history, in one reversed sweep
costing about as much as a forward run.

Under frozen shorelines the equation is linear in the ice history, so
the discrete kernels need no forward run at all: the sweep below starts
from the measurement and works backwards through fifteen thousand
years on a viscoelastic PREM.  Two views of the kernel are plotted.
The delta amplitude at the observation time is the instantaneous
(elastic) fingerprint adjoint: ice lost anywhere at the moment of
measurement perturbs sea level through self-gravity alone.  The smooth
density at earlier times is the viscous memory: the solid earth is
still responding to that ice, so the sensitivity spreads and grows
with time before present.

The same kernels are then computed by the continuous adjoint, which
never sees the time stepper's internals.  The two converge as the run
resolves the rheology, so their disagreement is a resolution
diagnostic for the time step -- it is printed at the end.

The run takes a couple of minutes at degree 64.
"""

import matplotlib.pyplot as plt
import numpy as np
from planetmodel import PREM, units

import pyslfp as sl
from pyslfp.ice import IceNG
from pyslfp.love_numbers import Material, graded_mesh
from pyslfp.viscoelastic import Rheology, SeaLevelEvolution, ViscoelasticOperator

LMAX = 64

# ----------------------------------------------------------------------
# The earth model and its rheology, as in the forward tutorial: PREM
# without its ocean, collapsed to isotropy, with a two-layer Maxwell
# mantle under an elastic lithosphere.  The solvers work in the model's
# own geophysical units -- unit radius, G equal to one -- so the model
# is non-dimensionalised at construction and the viscosities, quoted in
# Pa s, convert once at the boundary.
# ----------------------------------------------------------------------

model = PREM(ocean=False).isotropic().nondimensionalised()
mesh = graded_mesh(model, LMAX)
material = Material(mesh, model)

eta_unit = model.scales.factor(units.MODULUS) * model.scales.factor(units.TIME)
viscosity = {i: 2.0e21 / eta_unit for i in (2, 3, 4)}
viscosity.update({i: 5.0e20 / eta_unit for i in (5, 6, 7, 8)})
rheology = Rheology.maxwell(material, viscosity)
operator = ViscoelasticOperator(rheology, LMAX)

earth_model = sl.EarthModel.from_planet_model(model, LMAX, mesh=mesh)
ice_model = IceNG(length_scale=earth_model.parameters.length_scale)
state = sl.EarthState.from_ice_model(ice_model, 0.0, earth_model)
evolution = SeaLevelEvolution(operator, state)

parameters = earth_model.parameters
metre = parameters.length_scale

# ----------------------------------------------------------------------
# The objective: the sea level change at the present day, averaged over
# a ten-degree patch of the central Pacific, in metres.  Its dual is
# the normalised patch itself, and the measurement dictionary keys it
# by its date.  The date grid spans fifteen thousand years at half-
# kiloyear steps; the kernels live on it.
# ----------------------------------------------------------------------

patch = state.disk_load(10.0, 0.0, 180.0, 1.0)
# taper the patch in degree space, as for any sharp disc at a finite
# truncation: an untapered dual rings through every relaxation mode
# and the ringing would dominate the kernel's small smooth values
taper = np.exp(-((np.arange(LMAX + 1) / 32.0) ** 4))
patch_lm = earth_model.expand_field(patch)
patch_lm.coeffs *= taper[None, :, None]
patch = earth_model.expand_coefficient(patch_lm)
site_dual = (metre / earth_model.integrate(patch)) * patch
dates = np.linspace(15.0, 0.0, 31)
measurement = {0.0: site_dual}

kernels = evolution.adjoint_ice_history(dates, measurement)

# ----------------------------------------------------------------------
# The instantaneous view.  The delta amplitude at the observation time
# is the generalised elastic fingerprint adjoint of the measurement:
# removing ice AT the moment of observation changes the site's sea
# level only through the instantaneous redistribution.  Far from the
# site it is nearly uniform -- the eustatic part -- and near the site
# the lost attraction pulls the local ocean down.
# ----------------------------------------------------------------------

lump = kernels.instantaneous[0.0] / metre

fig, ax = sl.create_map_figure(figsize=(12, 8))
sl.plot(lump, ax=ax, colorbar_kwargs={"label": "Site metres per metre of ice"})
ax.set_title("Instantaneous kernel: ice lost at the moment of measurement")

# ----------------------------------------------------------------------
# The viscous memory.  The density view of the kernel (per kiloyear) at
# ten thousand years before present: ice melted then still perturbs the
# present-day measurement through the mantle's ongoing flow, so the
# sensitivity has spread far beyond the elastic pattern and carries the
# signature of the relaxing low harmonics.
# ----------------------------------------------------------------------

mid = int(np.argmin(np.abs(dates - 10.0)))
density = kernels.density[mid] / metre

fig, ax = sl.create_map_figure(figsize=(12, 8))
sl.plot(
    density,
    ax=ax,
    colorbar_kwargs={"label": "Site metres per metre of ice per kiloyear"},
)
ax.set_title("Kernel density ten thousand years before present")

# ----------------------------------------------------------------------
# The kernel through time at two land points -- the thickness kernel
# vanishes over the ocean, where no grounded ice can sit, so the
# points must be continental: the Canadian interior for the near field
# of the vanished Laurentide loads, and central Australia, the land
# nearest the Pacific site.  The quelled (rate) view is the cumulative
# kernel that pairs with thickness RATES, bounded and piecewise
# continuous.  Both curves share the eustatic floor at early times --
# ice lost long ago has fully relaxed, leaving only the water it added
# -- and they part towards the present: under the former ice the
# sensitivity deepens, because recent melting still has its whole
# viscous response to deliver to the measurement, while on the land
# nearest the site the lost attraction pulls the other way and bends
# the curve back.
# ----------------------------------------------------------------------

lat_canada = int(np.argmin(np.abs(earth_model.lats() - 55.0)))
lon_canada = int(np.argmin(np.abs(earth_model.lons() - (360.0 - 100.0))))
lat_australia = int(np.argmin(np.abs(earth_model.lats() + 25.0)))
lon_australia = int(np.argmin(np.abs(earth_model.lons() - 135.0)))

mid_dates = 0.5 * (dates[:-1] + dates[1:])
near = [r.data[lat_canada, lon_canada] / metre for r in kernels.rate]
far = [r.data[lat_australia, lon_australia] / metre for r in kernels.rate]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mid_dates, near, label="Canadian interior (55N, 100W)")
ax.plot(mid_dates, far, label="central Australia (25S, 135E)")
ax.set_xlabel("Kiloyears before present")
ax.set_ylabel("Site metres per metre-per-kiloyear of ice change")
ax.invert_xaxis()
ax.legend()
ax.set_title("The quelled kernel: sensitivity to the rate of ice change")

# ----------------------------------------------------------------------
# The resolution diagnostic.  The continuous adjoint never sees the
# stepper's internals, so its kernels differ from the exact discrete
# ones by the discretise/transpose commute error, which vanishes as the
# time step resolves the rheology.  The disagreement printed below is
# therefore a statement about the forward discretisation itself: if it
# is more than a few per cent, the half-kiloyear step is too coarse for
# the viscosity model, not the adjoint wrong.
# ----------------------------------------------------------------------

continuous = evolution.continuous_adjoint_ice_history(dates, measurement)
num = sum(
    np.linalg.norm(a.data - b.data)
    for a, b in zip(continuous.thickness, kernels.thickness)
)
den = sum(np.linalg.norm(b.data) for b in kernels.thickness)
print(f"continuous vs discrete kernels: {100.0 * num / den:.2f}% disagreement")

plt.show()
