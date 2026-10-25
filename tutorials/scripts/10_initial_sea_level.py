# %% [markdown]
# # 10. Calibrating the initial sea level
#
# An ice history fixes the thickness of the ice through time, but a GIA
# run also needs the sea level at its first date: the palaeotopography
# from which the oceans and the ice start. The ICE-NG products carry
# their own, but a history of thickness alone does not, and the standard
# device is to choose the initial sea level so that the run reproduces
# the sea level observed today. This script does that for the thickness
# of ICE-7G from 26 ka, on the Maxwell PREM of tutorial 7, with migrating
# shorelines, by the two methods the library offers:
#
# - the classical update, which replaces the initial sea level by the
#   observed present-day sea level minus the modelled change since the
#   first date, and is Newton's method with the identity standing in for
#   the Jacobian;
# - steepest descent with the gradient from the adjoint (Yu et al. 2025),
#   which under frozen shorelines is the same update, and which in
#   general minimises a misfit that may be weighted.
#
# It then shows the palaeotopography found, compares it with ICE-7G's own
# reconstruction, and measures what freezing the shorelines would have
# cost.
#
# The steps are a kiloyear long, coarser than the half-kiloyear slices of
# ICE-7G, to keep the run to about five minutes at degree 64.

# %%
import matplotlib.pyplot as plt
import numpy as np
from planetmodel import PREM, units

import pyslfp as sl
from pyslfp.ice import IceNG
from pyslfp.love_numbers import Material, graded_mesh
from pyslfp.viscoelastic import Rheology, SeaLevelEvolution, ViscoelasticOperator

LMAX = 64

# %% [markdown]
# ## The earth model
#
# As in tutorial 7: PREM collapsed to isotropy and converted into the
# library's frame, 5 x 10^20 Pa s above the 670 km discontinuity and
# 2 x 10^21 Pa s below, the top 77 km elastic.

# %%
scales = sl.EarthModelParameters.from_defaults().scales
model = PREM(ocean=False).isotropic().converted(scales)
mesh = graded_mesh(model, LMAX)
material = Material(mesh, model)

pascal_second = 1.0 / scales.factor(units.VISCOSITY)
viscosity = {i: 2.0e21 * pascal_second for i in (2, 3, 4)}
viscosity.update({i: 5.0e20 * pascal_second for i in (5, 6, 7, 8)})
operator = ViscoelasticOperator(Rheology.maxwell(material, viscosity), LMAX)
earth_model = sl.EarthModel.from_planet_model(model, LMAX, mesh=mesh)
metre = earth_model.parameters.length_scale

# %% [markdown]
# ## A thickness-only history and its target
#
# The ice model is ICE-7G, but the runs read only its thickness. The
# target is the sea level ICE-7G records for today, the present-day
# topography and bathymetry. The starting guess, as in the classical
# scheme, is that present-day sea level itself under the ice of 26 ka:
# the evolution's background state supplies the guess, and the
# calibration replaces its sea level.

# %%
ice_model = IceNG(length_scale=metre)
grid = dict(grid=earth_model.grid_name, extend=earth_model.extend)
present = ice_model.get_sea_level(0.0, LMAX, **grid)
guess = sl.EarthState(
    ice_model.get_ice_thickness(26.0, LMAX, **grid), present, earth_model
)
dates = np.arange(26.0, -0.0001, -1.0)

# %% [markdown]
# ## The classical update
#
# Each iteration is one forward run with migrating shorelines. The
# residual of the guess is the whole sea level change of the
# deglaciation, about 150 m in root-mean-square; the update removes it in
# one step wherever the shorelines stay put, and what remains comes from
# the shorelines that move, whose water load depends on the initial sea
# level itself. Anderson mixing accelerates the iteration.

# %%
classical = SeaLevelEvolution(operator, guess).calibrate_initial_sea_level(
    dates, ice_model, present, shorelines="migrating"
)
print(
    "classical update, RMS misfit by run (m):",
    ", ".join(f"{m * metre:.3g}" for m in classical.misfits),
)

# %% [markdown]
# ## The adjoint gradient
#
# The same calibration by steepest descent on the misfit
# $\frac12\int (S_0 + \xi(t_f) - S_f)^2\,dS$, from the same guess. Its
# gradient is the residual plus the initial sea level kernel of a
# sea level measurement of that residual today, which one adjoint sweep
# of the run returns. Unit steps down the gradient contract more slowly
# than the classical update, which approximately inverts the Jacobian
# where the gradient applies its transpose; the acceleration narrows the
# gap. The two methods arrive at the same initial sea level.

# %%
gradient = SeaLevelEvolution(operator, guess).calibrate_initial_sea_level(
    dates, ice_model, present, shorelines="migrating", method="gradient"
)
print(
    "adjoint gradient, RMS misfit by run (m):",
    ", ".join(f"{m * metre:.3g}" for m in gradient.misfits),
)

difference = np.max(
    np.abs(classical.state.sea_level.data - gradient.state.sea_level.data)
)
print(
    f"the two calibrated initial sea levels differ by at most {difference * metre:.2f} m"
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(np.array(classical.misfits) * metre, "o-", label="classical update")
ax.semilogy(np.array(gradient.misfits) * metre, "s-", label="adjoint gradient")
ax.set_xlabel("Forward run")
ax.set_ylabel("RMS misfit to the present-day sea level (m)")
ax.legend()
ax.set_title("Convergence of the calibration")

# %% [markdown]
# ## The palaeotopography
#
# The calibrated sea level at 26 ka, relative to today's. Across the
# oceans it stands about 120 m lower, the lowstand of the Last Glacial
# Maximum; under the ice sheets it is far higher, because sea level is
# measured from a solid surface depressed by the ice, and the peripheral
# bulges around them carry the opposite sign. The colour scale saturates
# so that the far field stays visible.

# %%
initial = classical.state.sea_level
ocean_today = sl.EarthState(
    ice_model.get_ice_thickness(0.0, LMAX, **grid), present, earth_model
).ocean_function
lowstand = earth_model.integrate(
    (initial - present) * ocean_today
) / earth_model.integrate(ocean_today)
print(
    f"sea level at 26 ka relative to today, averaged over today's oceans: {lowstand * metre:.1f} m"
)

fig, ax = sl.create_map_figure(figsize=(12, 8))
sl.plot(
    (initial - present) * metre,
    ax=ax,
    vmin=-200.0,
    vmax=200.0,
    colorbar_kwargs={"label": "Sea level at 26 ka minus today's (m)"},
)
ax.set_title("The calibrated palaeotopography, relative to today")

# %% [markdown]
# ## Against ICE-7G's own reconstruction
#
# ICE-7G carries its own sea level at 26 ka, from an inversion with its
# own earth model and calibration. The difference from the one found here
# is therefore the signature of the two earth models: near the former
# ice sheets the viscosity profile sets how much of the depression
# remains today, and so how deep the crust stood at 26 ka.

# %%
own = ice_model.get_sea_level(26.0, LMAX, **grid)
fig, ax = sl.create_map_figure(figsize=(12, 8))
sl.plot(
    (initial - own) * metre,
    ax=ax,
    vmin=-100.0,
    vmax=100.0,
    colorbar_kwargs={"label": "This calibration minus ICE-7G's own (m)"},
)
ax.set_title("Initial sea level at 26 ka: this calibration against ICE-7G's")

# %% [markdown]
# ## What frozen shorelines would have given
#
# Under frozen shorelines the problem is linear. The shorelines are those
# of the starting guess, today's sea level under the ice of 26 ka, held
# fixed through the run, so the change of sea level does not depend on
# the initial sea level at all, and one run gives the answer exactly: the
# present-day sea level minus the modelled change. The calibrated state
# keeps those shorelines as a prescribed ocean function. Its answer
# differs from the migrating one where the shorelines moved: on the
# continental shelves exposed at the lowstand, which the frozen equation
# covers with water throughout, and at the marine margins of the ice
# sheets; and by a few metres everywhere else, since in the migrating run
# part of the meltwater goes into flooding the shelves rather than
# raising the open ocean.

# %%
frozen = SeaLevelEvolution(operator, guess).calibrate_initial_sea_level(
    dates, ice_model, present
)
print(
    "frozen shorelines, RMS misfit before and after the single run (m):",
    ", ".join(f"{m * metre:.3g}" for m in frozen.misfits),
)

fig, ax = sl.create_map_figure(figsize=(12, 8))
sl.plot(
    (frozen.state.sea_level - initial) * metre,
    ax=ax,
    vmin=-20.0,
    vmax=20.0,
    colorbar_kwargs={"label": "Frozen minus migrating (m)"},
)
ax.set_title("Initial sea level: the cost of frozen shorelines")

plt.show()
