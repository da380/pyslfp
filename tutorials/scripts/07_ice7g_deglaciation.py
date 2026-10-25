# %% [markdown]
# # 7. The last deglaciation: an ICE-7G run from 26 ka to the present
#
# The forward GIA solver is driven by a real ice history: ICE-7G from
# the Last Glacial Maximum to today, on isotropic PREM with a two-layer
# Maxwell mantle and an elastic lithosphere, with the sea level equation
# and the full rotational feedback closed at every half-kiloyear step
# under the 26 ka shorelines. Three figures result: the map of
# present-day crustal uplift rates, whose peak sits over southeast
# Hudson Bay at the observed level of about 13 mm per year; the
# ocean-mean sea level curve against ICE-7G's own inverted record; and
# the emergence curve at Richmond Gulf, the classic Hudson Bay record.
#
# Two caveats worth knowing. The run cold-starts at 26 ka with no
# internal memory, as if the ice had sat on an undisturbed elastic
# earth, so the first few kiloyears carry a spin-up bias; production
# calculations start a glacial cycle earlier. And the shorelines are
# frozen at their 26 ka positions (the linear equation);
# `run_ice_history(..., shorelines="migrating")` solves the nonlinear
# equation with moving shorelines instead, at greater cost.
#
# The run takes a couple of minutes at degree 128, and about ten at
# degree 256. The ICE-7G files are downloaded on first use.

# %%
import matplotlib.pyplot as plt
import numpy as np
from planetmodel import PREM, units

import pyslfp as sl
from pyslfp.viscoelastic import Rheology, SeaLevelEvolution, ViscoelasticOperator
from pyslfp.ice import IceNG
from pyslfp.love_numbers import Material, graded_mesh

LMAX = 128

# %% [markdown]
# ## The earth model and rheology
#
# As in the previous tutorial: PREM collapsed to isotropy and converted
# into the library's frame, 5 x 10^20 Pa s above the 670 km
# discontinuity and 2 x 10^21 Pa s below, the top 77 km elastic. PREM
# with Earth's observed moments of inertia is a consistent pairing for
# the rotational feedback: the oblateness the moments sustain exceeds
# the model's hydrostatic fluid limit, so the transverse (polar wander)
# feedback is stable. For idealised bodies that is not automatic; see
# `EarthModelParameters.with_hydrostatic_moments`.

# %%
scales = sl.EarthModelParameters.from_defaults().scales
model = PREM(ocean=False).isotropic().converted(scales)
mesh = graded_mesh(model, LMAX)
material = Material(mesh, model)

pascal_second = 1.0 / scales.factor(units.VISCOSITY)
viscosity = {i: 2.0e21 * pascal_second for i in (2, 3, 4)}
viscosity.update({i: 5.0e20 * pascal_second for i in (5, 6, 7, 8)})
operator = ViscoelasticOperator(Rheology.maxwell(material, viscosity), LMAX)

# %% [markdown]
# ## The background state
#
# The background state is what ICE-7G records at 26 ka: being the
# product of an inversion, it carries a sea level history alongside the
# ice, so the state can be read straight off it. A thickness-only ice
# model would instead need its initial sea level found by iteration
# against the present day (`pyslfp.viscoelastic.calibrate_initial_sea_level`).

# %%
earth_model = sl.EarthModel.from_planet_model(model, LMAX, mesh=mesh)
ice_model = IceNG(length_scale=earth_model.parameters.length_scale)
state = sl.EarthState.from_ice_model(ice_model, 26.0, earth_model)

# %% [markdown]
# ## The run
#
# Dates are in kiloyears before present, decreasing, on the
# half-kiloyear grid of the ICE-7G files, so the history's breakpoints
# fall on step boundaries.

# %%
dates = np.arange(26.0, -0.0001, -0.5)
evolution = SeaLevelEvolution(operator, state)
snapshots, _ = evolution.run_ice_history(dates, ice_model)

# %% [markdown]
# ## Present-day uplift rates
#
# From the last half-kiloyear of the run. The twin peaks of postglacial
# rebound stand over the former Laurentide and Fennoscandian ice sheets,
# ringed by the subsiding peripheral bulges.

# %%
metre = earth_model.parameters.length_scale
rate = (
    (snapshots[-1].displacement - snapshots[-2].displacement) * metre * 1000.0 / 500.0
)  # mm per year

fig, ax = sl.create_map_figure(figsize=(12, 8))
sl.plot(
    rate,
    ax=ax,
    vmin=-14.0,
    vmax=14.0,
    colorbar_kwargs={"label": "Uplift rate (mm/yr)"},
)
ax.set_title("Present-day crustal uplift rate")

# %% [markdown]
# ## The ocean-mean sea level curve
#
# Compared with the record ICE-7G itself carries. The two are
# independent in method -- a forward viscoelastic run against an
# inversion built with its own earth model and migrating shorelines --
# and agree to a few per cent.

# %%
ocean = state.ocean_function
area = state.ocean_area


def ocean_mean(field):
    return metre * earth_model.integrate(field * ocean) / area


run_curve = np.array([ocean_mean(snap.sea_level_change) for snap in snapshots])
sl26 = ice_model.get_sea_level(
    26.0, LMAX, grid=earth_model.grid_name, extend=earth_model.extend
)
own_curve = np.array(
    [
        ocean_mean(
            ice_model.get_sea_level(
                date, LMAX, grid=earth_model.grid_name, extend=earth_model.extend
            )
            - sl26
        )
        for date in dates
    ]
)

# %% [markdown]
# ## Relative sea level at Richmond Gulf
#
# The emergence curve of southeast Hudson Bay. Relative to the present,
# sea level there stood hundreds of metres higher at deglaciation and
# has fallen ever since as the crust rebounds -- the raised beaches of
# the classic record.

# %%
lats, lons = earth_model.lats(), earth_model.lons()
i = int(np.argmin(np.abs(lats - 56.5)))
j = int(np.argmin(np.abs(lons - (360.0 - 76.2))))
site = metre * np.array(
    [
        snap.sea_level_change.data[i, j] - snapshots[-1].sea_level_change.data[i, j]
        for snap in snapshots
    ]
)

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
top.plot(dates, run_curve, label="this run")
top.plot(dates, own_curve, linestyle="--", label="ICE-7G's own record")
top.set_ylabel("Ocean-mean sea level change (m)")
top.legend()
bottom.plot(dates, site)
bottom.set_ylabel("Relative sea level at Richmond Gulf (m)")
bottom.set_xlabel("Date (ka before present)")
bottom.invert_xaxis()
fig.tight_layout()

print(
    f"ocean-mean rise 26 ka to present: {run_curve[-1]:.1f} m "
    f"(ICE-7G's own record: {own_curve[-1]:.1f} m)"
)
print(f"peak present-day uplift rate: {rate.data.max():.1f} mm/yr")

plt.show()
