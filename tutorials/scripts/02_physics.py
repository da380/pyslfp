# %% [markdown]
# # 2. A closer look at the physics
#
# The first script took every default. This one builds each piece by
# hand: the non-dimensionalisation, the Earth model, an initial state
# from a simple analytic ice model, a load of one's own, the linear
# solution with all of its fields, and the non-linear solution with its
# migrating shorelines.

# %%
import matplotlib.pyplot as plt
from cartopy import crs as ccrs

import pyslfp as sl

LMAX = 256

# %% [markdown]
# ## The Earth model and the initial state
#
# `EarthModelParameters` holds the physical constants and the scales that
# make the calculation non-dimensional. Any scales will do; the results
# come back in the same units they went in. The Earth model then adds
# the spherical harmonic truncation and, by default, the precomputed
# Love numbers for PREM, whose radius, surface gravity and gravitational
# constant it takes on. The initial state is an ice thickness and a sea
# level on the model's grid, here from an analytic model with an ice cap
# at each pole.

# %%
parameters = sl.EarthModelParameters(
    length_scale=1000.0e3,  # a thousand kilometres
    density_scale=1000.0,  # the density of water
    time_scale=3600.0,  # an hour
)
earth_model = sl.EarthModel(LMAX, parameters=parameters)
print("g in these units:", earth_model.parameters.gravitational_acceleration)

ice_model = sl.ice.AnalyticalIceModel(length_scale=parameters.length_scale)
ice_thickness, sea_level = ice_model.get_ice_thickness_and_sea_level(0, LMAX)
initial_state = sl.EarthState(
    ice_thickness, sea_level, earth_model, exclude_caspian=False
)

metre = parameters.length_scale

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    subplot_kw={"projection": ccrs.Miller()},
    layout="constrained",
)
sl.plot(
    ice_thickness * metre,
    ax=ax1,
    coasts=False,
    colorbar_kwargs={"label": "Ice thickness (m)"},
)
sl.plot(
    sea_level * metre,
    ax=ax2,
    coasts=False,
    symmetric=True,
    colorbar_kwargs={"label": "Sea level (m)"},
)
# the coastline of this state is the zero set of rho_w SL - rho_i I
initial_state.plot_coastline(ax1)
initial_state.plot_coastline(ax2)

# %% [markdown]
# ## A load of one's own
#
# A load is any field on the model's grid. Here all of the ice in the
# western hemisphere melts: a mask on longitude, multiplied by the initial
# thickness, gives the change in ice thickness, and the state converts
# that into a surface mass load.

# %%
ice_thickness_change = earth_model.zero_grid()
lons = ice_thickness_change.lons()
west = (lons > 180) | (lons < 0)
ice_thickness_change.data[:, west] = -1.0
ice_thickness_change *= ice_thickness

ax, im = sl.plot(
    ice_thickness_change * initial_state.ice_projection() * metre,
    coasts=False,
    symmetric=True,
    colorbar_kwargs={"label": "Ice thickness change (m)"},
)
initial_state.plot_coastline(ax)

direct_load = initial_state.direct_load_from_ice_thickness_change(ice_thickness_change)

# %% [markdown]
# ## The linear equation and its fields
#
# `SeaLevelEquation` solves the linearised equation for a given state and
# load, returning the sea level change, the vertical displacement, the
# potential change and the angular velocity change. The geoid anomaly is
# minus the potential change over gravity.

# %%
sle = sl.SeaLevelEquation(earth_model)
slc, disp, potential, omega = sle.solve_sea_level_equation(initial_state, direct_load)
g = parameters.gravitational_acceleration

fig, axes = plt.subplot_mosaic(
    [["slc", "disp"], ["geoid", "."]],
    figsize=(12, 8),
    subplot_kw={"projection": ccrs.Robinson()},
    layout="constrained",
)
for key, field, label in (
    ("slc", slc * metre, "Sea level change (m)"),
    ("disp", disp * metre, "Vertical displacement (m)"),
    ("geoid", potential * (-metre / g), "Geoid anomaly (m)"),
):
    sl.plot(
        field,
        ax=axes[key],
        coasts=False,
        symmetric=True,
        colorbar_kwargs={"label": label},
    )
    initial_state.plot_coastline(axes[key])

# %% [markdown]
# ## The non-linear equation and shoreline migration
#
# The same solver handles the non-linear problem, in which the ocean
# function is updated as sea level changes and the shorelines move. It
# takes the ice thickness change itself (and, optionally, changes in
# sediment thickness and in dynamic sea level) and returns the new state
# along with the fields. The last panel shows the difference from the
# linear solution, with the old coastline in black and the new one in
# red.

# %%
new_state, slc_nonlinear, disp, potential, omega = sle.solve_nonlinear_equation(
    initial_state, ice_thickness_change=ice_thickness_change
)

fig, axes = plt.subplot_mosaic(
    [["slc", "disp"], ["geoid", "diff"]],
    figsize=(14, 10),
    subplot_kw={"projection": ccrs.Robinson()},
    layout="constrained",
)
for key, field, label in (
    ("slc", slc_nonlinear * metre, "Sea level change (m)"),
    ("disp", disp * metre, "Vertical displacement (m)"),
    ("geoid", potential * (-metre / g), "Geoid anomaly (m)"),
    ("diff", (slc_nonlinear - slc) * metre, "Non-linear minus linear (m)"),
):
    sl.plot(
        field,
        ax=axes[key],
        coasts=False,
        symmetric=True,
        colorbar_kwargs={"label": label},
    )
initial_state.plot_coastline(axes["slc"])
initial_state.plot_coastline(axes["disp"])
initial_state.plot_coastline(axes["geoid"])
initial_state.plot_coastline(axes["diff"], linestyles="-")
new_state.plot_coastline(axes["diff"], color="red", linestyles="-.")

plt.show()
