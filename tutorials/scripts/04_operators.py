# %% [markdown]
# # 4. Operators, composition and sensitivity kernels
#
# The sea level equation is linear in the load, so the fingerprint is a
# linear operator from loads to responses. `pyslfp.linear_operators`
# expresses it as one, on Hilbert spaces from `pygeoinf`, together with
# the projections, averages and observation models that compose with it.
# Composition gives derived quantities in a line, and the adjoint of any
# composition is its sensitivity kernel: the field that says how much
# each unit of mass, wherever it is placed, contributes to the quantity.

# %%
import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs

from pygeoinf.testing import check_operator

import pyslfp as sl
from pyslfp.linear_operators import (
    FingerPrintOperator,
    TideGaugeObservationModel,
    averaging_operator,
    ice_sheet_basis_operator,
    ice_thickness_change_to_load_operator,
    lebesgue_load_space,
    ocean_average_operator,
    remove_ocean_average_operator,
)

LMAX = 128

# %% [markdown]
# ## The fingerprint operator
#
# The operator carries its domain, the space of loads, and its codomain,
# the response space. A response holds the sea level change, the surface
# displacement, the gravitational potential change and the angular
# velocity change, each by name; the fields are held as spherical harmonic
# coefficients, all of them band-limited, and `field` gives one as a grid.
# Applying the operator solves the sea level equation.

# %%
fingerprint = FingerPrintOperator.from_defaults(lmax=LMAX)
state = fingerprint.state
params = state.model.parameters
metre = params.length_scale

response_space = fingerprint.codomain
field_space = response_space.field_space
print(
    "domain dimension",
    fingerprint.domain.dim,
    "; codomain dimension",
    response_space.dim,
)

load = state.west_antarctic_load(fraction=0.1)
response = fingerprint(load)
print("members of the response:", response_space.names)
relative_sea_level_change = response.field("relative_sea_level_change")

fig, ax = sl.create_map_figure(figsize=(10, 5))
sl.plot(
    relative_sea_level_change * state.ocean_projection() * metre,
    ax=ax,
    vmin=-1.0,
    vmax=1.0,
    colorbar_kwargs={"label": "Sea level change (m)"},
)

# %% [markdown]
# ## Composition
#
# `field_operator` picks the sea level out of the response, an ocean average
# turns it into the global mean, and removing that average leaves the
# departure from it. Each is an operator, and `@` composes them.

# %%
sea_level = response_space.field_operator("relative_sea_level_change")
mean_sea_level = ocean_average_operator(state, field_space) @ sea_level @ fingerprint
print(f"global mean sea level change: {mean_sea_level(load)[0] * metre * 1000:.1f} mm")

anomaly = remove_ocean_average_operator(state, field_space) @ sea_level @ fingerprint
anomaly_field = anomaly(load)

fig, ax = sl.create_map_figure(figsize=(10, 5))
sl.plot(
    anomaly_field * state.ocean_projection() * metre,
    ax=ax,
    vmin=-0.5,
    vmax=0.5,
    colorbar_kwargs={"label": "Sea level change relative to the global mean (m)"},
)

# a single site: the average over a small cap
SITE = (40.7, -74.0)  # New York
cap = state.disk_load(4.0, SITE[0], SITE[1], 1.0)
local_anomaly = (
    averaging_operator(state, field_space, [cap])
    @ remove_ocean_average_operator(state, field_space)
    @ sea_level
    @ fingerprint
)
print(
    f"local sea level relative to the global mean: "
    f"{local_anomaly(load)[0] * metre * 1000:.1f} mm"
)

# %% [markdown]
# The input variable can be changed as easily: an operator from a few
# basin amplitudes to ice thickness change, then to load, then through
# the fingerprint to the global mean.

# %%
ice_space = lebesgue_load_space(state.model)
basins = ice_sheet_basis_operator(state, ice_space, groupings="macro_regions")
ice_to_load = ice_thickness_change_to_load_operator(
    state, ice_space, fingerprint.domain
)
basin_to_mean = mean_sea_level @ ice_to_load @ basins

labels = ["West Antarctica", "East Antarctica", "Antarctic Peninsula", "Greenland"]
identity = np.eye(basin_to_mean.domain.dim)
for i, label in enumerate(labels):
    print(
        f"{label:22s} {basin_to_mean(identity[i])[0] * 1000:8.2f} mm of global mean "
        "sea level per metre of thickening"
    )

# %% [markdown]
# ## Adjoints and kernels
#
# The adjoint of a composition is available without further work, and
# `check_operator` from `pygeoinf.testing` verifies the adjoint identity
# on random inputs, raising an error if it fails. The adjoint of
# the global mean applied to one is its kernel, which is a constant: every
# unit of mass raises the global mean by the same amount. The kernel of
# the local anomaly has structure, and is shown in millimetres of local
# sea level per thousand gigatonnes of mass added.

# %%
report = check_operator(
    fingerprint,
    rng=np.random.default_rng(0),
    measure=fingerprint.load_measure_for_testing(),
    codomain_measure=fingerprint.response_measure_for_testing(),
    trials=2,
    rtol=1e-4,
)
print(report)

kernel = mean_sea_level.adjoint(np.array([1.0]))
print(f"kernel range: {kernel.data.min():.12e} to {kernel.data.max():.12e}")
print(f"predicted:    {-1.0 / (params.water_density * state.ocean_area):.12e}")

gigatonne = 1.0e12 / (params.density_scale * params.length_scale**3)


def per_thousand_gigatonnes(k):
    return k * (1000 * gigatonne) * metre * 1000


local_kernel = per_thousand_gigatonnes(local_anomaly.adjoint(np.array([1.0])))
fig, ax = sl.create_map_figure(figsize=(10, 5))
sl.plot(
    local_kernel,
    ax=ax,
    vmin=-5.0,
    vmax=5.0,
    colorbar_kwargs={"label": "mm of local sea level per 1000 Gt"},
)

# %% [markdown]
# ## Polar wander and the length of day
#
# The angular velocity change is the response's last member.
# Its two components orthogonal to the rotation axis give the shift of the
# pole, and their kernels are spherical harmonics of degree two and order
# one. The component along the axis gives the change in the length of day,
# and its kernel is the zonal harmonic of degree two: mass moved from the
# poles towards the equator slows the Earth.

# %%
rotation = response_space.projection("angular_velocity_change") @ fingerprint
omega = rotation(load)
pole_factor = params.mean_sea_floor_radius / params.rotation_frequency
shift = omega[:2] * pole_factor * metre
print(f"pole displacement: {np.linalg.norm(shift):.1f} m")

second = params.time_scale
day = 2 * np.pi / params.rotation_frequency * second
lod_factor = -day / params.rotation_frequency
print(f"length of day change: {omega[2] * lod_factor * 1000:.3f} ms")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(19, 4),
    subplot_kw={"projection": ccrs.Robinson()},
    layout="constrained",
)
for ax, i, label in zip(axes, [0, 1], ["x axis", "y axis"]):
    unit = np.zeros(3)
    unit[i] = 1.0
    sl.plot(
        per_thousand_gigatonnes(rotation.adjoint(unit) * pole_factor),
        ax=ax,
        symmetric=True,
        colorbar_kwargs={"label": f"Pole shift towards the {label} (mm per 1000 Gt)"},
    )
lod_kernel = rotation.adjoint(np.array([0.0, 0.0, 1.0])) * lod_factor
sl.plot(
    lod_kernel * (1000 * gigatonne) * 1000,
    ax=axes[2],
    symmetric=True,
    colorbar_kwargs={"label": "Length of day change (ms per 1000 Gt)"},
)

# %% [markdown]
# ## Horizontal displacement
#
# A load moves the surface sideways as well as up and down. The
# displacement is one field, u = U r + grad V, where V is the potential
# of the horizontal part; a surface load drives no rotation of the
# surface about the vertical, so V is all the horizontal part there is.
# Built with `horizontal=True`, the operator's response holds V as well,
# and the displacement gives east, north and up at points and the
# horizontal part on the grid. `plot_displacement` draws both parts: the
# vertical in colour and the horizontal as arrows, here outward from the
# uplift where the ice was lost.

# %%
horizontal_fingerprint = FingerPrintOperator.from_defaults(lmax=LMAX, horizontal=True)
displacement = horizontal_fingerprint(load).displacement
sites = [(-75.0, 250.0), (-80.0, 200.0), (-70.0, 290.0)]
for (lat, lon), (east, north, up) in zip(sites, displacement.at(sites) * metre * 1000):
    print(
        f"{lat:6.1f}N {lon:6.1f}E: east {east:7.2f} mm, north {north:7.2f} mm, "
        f"up {up:7.2f} mm"
    )

fig, ax = sl.create_map_figure(figsize=(7, 7), projection=ccrs.SouthPolarStereo())
sl.plot_displacement(
    displacement,
    ax=ax,
    scale=metre * 1000,
    units="mm",
    arrows=25,
    map_extent=[-180, 180, -90, -60],
    colorbar_kwargs={"label": "Vertical displacement (mm)"},
)

# %% [markdown]
# ## Sobolev spaces and point observations
#
# A tide gauge reads sea level at a point, and point evaluation is only
# bounded on a space of functions with some smoothness. Built over
# Sobolev spaces, the fingerprint composes with point evaluation into a
# tide gauge observation model, whose adjoint gives the representer of
# each station.

# %%
sobolev_fingerprint = FingerPrintOperator.from_defaults(
    lmax=LMAX, load_parameters=(2.0, 0.05), response_parameters=(2.0, 0.05)
)
gauges = TideGaugeObservationModel.from_gloss_network(sobolev_fingerprint)
print(f"{len(gauges.points)} stations")

data = gauges.forward_operator(load) * metre * 1000
fig, ax = sl.create_map_figure(figsize=(10, 5))
sl.plot_points(
    gauges.points,
    data=data,
    ax=ax,
    s=25,
    symmetric=True,
    colorbar=True,
    colorbar_kwargs={"label": "Sea level change (mm)"},
)

distances = [(lat - SITE[0]) ** 2 + (lon - SITE[1]) ** 2 for lat, lon in gauges.points]
index = int(np.argmin(distances))
unit = np.zeros(len(gauges.points))
unit[index] = 1.0
representer = gauges.forward_operator.adjoint(unit)
fig, ax = sl.create_map_figure(figsize=(10, 5))
sl.plot(
    representer,
    ax=ax,
    symmetric=0.3,
    colorbar_kwargs={"label": f"Representer of {gauges.names[index]}"},
)

plt.show()
