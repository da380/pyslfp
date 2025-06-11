import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs
from pygeoinf import (
    LinearOperator,
    LinearForwardProblem,
    LinearLeastSquaresInversion,
    CholeskySolver,
    CGSolver,
)
from pygeoinf.homogeneous_space.sphere import Sobolev
from pyslfp import EarthModelParamters, FingerPrint

# Set up the fingerprint.
fingerprint = FingerPrint(
    lmax=128,
    earth_model_parameters=EarthModelParamters.from_standard_non_dimensionalisation(),
)
fingerprint.set_state_from_ice_ng()

# Set the model space.
model_space = Sobolev(
    fingerprint.lmax, 2, 0.1, radius=fingerprint.mean_sea_floor_radius
)


# Set the ice projection operator.
ice_projection = LinearOperator.formally_self_adjoint(
    model_space, lambda direct_load: fingerprint.ice_projection(0) * direct_load
)

# Set a prior distribution.
model_prior_measure = model_space.sobolev_gaussian_measure(2, 0.05, 1)


# Set the sea level operator.
sea_level_operator = LinearOperator.formally_self_adjoint(
    model_space,
    lambda direct_load: fingerprint(direct_load=direct_load, rtol=1e-6)[0],
)

sea_level_operator = sea_level_operator @ ice_projection

# Set the observation operator.
n = 30
lats = np.random.uniform(-60, 60, n)
lons = np.random.uniform(-180, 180, n)
observation_operator = model_space.point_evaluation_operator(lats, lons)


forward_operator = observation_operator @ sea_level_operator @ ice_projection


# Set the forward problem with no data errors.
forward_problem = LinearForwardProblem(forward_operator)

# Set up the inversion
inversion = LinearLeastSquaresInversion(forward_problem)

# Generate synthetic data.
model, data = forward_problem.synthetic_model_and_data(model_prior_measure)
sea_level_change = sea_level_operator(model)


"""

# Plot the direct load input.
fig, ax, im = fingerprint.plot(model, ice_projection=True)
model_clim = im.get_clim()
fig.colorbar(im, ax=ax, orientation="horizontal", label="direct load")
ax.plot(
    lons,
    lats,
    "o",
    color="k",
    markersize=4,
    transform=ccrs.PlateCarree(),
)

# Plot the resulting sea level
fig, ax, im = fingerprint.plot(sea_level_change, ocean_projection=True)
sea_clim = im.get_clim()
fig.colorbar(im, ax=ax, orientation="horizontal", label="sea level change")
ax.plot(
    lons,
    lats,
    "o",
    color="k",
    markersize=4,
    transform=ccrs.PlateCarree(),
)

"""

# Invert the data.
damping = 0.001
inverted_model = inversion.least_squares_operator(damping, CGSolver())(data)
inverted_sea_level_change = sea_level_operator(inverted_model)

"""

# Plot the inverted direct load input.
fig, ax, im = fingerprint.plot(inverted_model, ice_projection=True)
im.set_clim(model_clim)
fig.colorbar(im, ax=ax, orientation="horizontal", label="inverted direct load")
ax.plot(
    lons,
    lats,
    "o",
    color="k",
    markersize=4,
    transform=ccrs.PlateCarree(),
)

# Plot the resulting sea level
fig, ax, im = fingerprint.plot(inverted_sea_level_change, ocean_projection=True)
im.set_clim(sea_clim)
fig.colorbar(im, ax=ax, orientation="horizontal", label="inverted sea level change")
ax.plot(
    lons,
    lats,
    "o",
    color="k",
    markersize=4,
    transform=ccrs.PlateCarree(),
)

"""

print(fingerprint.solver_counter)
print(fingerprint.ocean_average(sea_level_change))
print(fingerprint.ocean_average(inverted_sea_level_change))


# plt.show()
