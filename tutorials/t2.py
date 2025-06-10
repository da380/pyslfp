"""
Example of explore some of the options within the FingerPrint class. 
"""

# Import necessary modules.
import numpy as np
import matplotlib.pyplot as plt
from pyshtools import SHGrid
from pyslfp import FingerPrint, EarthModelParamters

# Set up the earth model parameters. Only non-dimensionalisation terms are set with
# all physical parameters taking their default values.
parameters = EarthModelParamters(length_scale=1e3, mass_scale=1e12, time_scale=1e7)

# Set up fingerprint based on a Gauss-Legendre quadrature grid
fingerprint = FingerPrint(
    lmax=128, earth_model_parameters=parameters, grid="GLQ", extend=False
)

# Print some values.
print(f"Density scale {fingerprint.density_scale} in kg/m^3")
print(f"Radius of the Earth {fingerprint.mean_radius} in non-dimensional units")
print(
    f"Polar moment of inertia {fingerprint.polar_moment_of_inertia} in non-dimensional units"
)
print(f"density of ice {fingerprint.ice_density} in non-dimensional units")


# Generate a zero-field
zero = fingerprint.zero_grid()

# Print out its parameters.
print(zero, "\n")


# We can access the latitudes and longitudes in the grid.
lats = fingerprint.lats()
print(f"First five latitudes:\n{lats[:5]}")

lons = fingerprint.lons()
print(f"First five longitudes:\n:{lons[:5]} ")

# We can't access the sea level or ice thickness (or related methods) until they have been set.
try:
    sea_level = fingerprint.sea_level
except NotImplementedError as e:
    print(e)

# Set values using ice-5g at LGM.
fingerprint.set_state_from_ice_ng(version=5, date=21)

fig, ax, im = fingerprint.plot(fingerprint.sea_level)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, label="sea level at LGM")

fig, ax, im = fingerprint.plot(fingerprint.ice_thickness, ice_projection=True)
fig.colorbar(
    im, ax=ax, orientation="horizontal", shrink=0.7, label="ice thickness at LGM"
)

fig, ax, im = fingerprint.plot(fingerprint.ocean_function)
fig.colorbar(
    im, ax=ax, orientation="horizontal", shrink=0.7, label="ocean function at LGM"
)


plt.show()


# We can also set the equilibrium state directly.

# Generate a numpy array of all latitudes
lats, _ = np.meshgrid(
    fingerprint.lats(),
    fingerprint.lons(),
    indexing="ij",
)

# Construct the sea level values
sea_level_amplitude = 3000 / fingerprint.length_scale
sea_level = SHGrid.from_array(
    np.where(lats < 0, sea_level_amplitude, -sea_level_amplitude), grid=fingerprint.grid
)


# Now set the ice sheet thickness using a disk geometry.
disk_angular_radius = 20
disk_centre_lat = 0
disk_centre_lon = 0
ice_thickness_amplitude = 2000 / fingerprint.length_scale
ice_thickness = fingerprint.disk_load(
    disk_angular_radius, disk_centre_lat, disk_centre_lon, ice_thickness_amplitude
)

# Set the values and plot the fields.
fingerprint.sea_level = sea_level
fingerprint.ice_thickness = ice_thickness

# Note coasts are not plotted in these cases.
fig, ax, im = fingerprint.plot(fingerprint.sea_level, coasts=False)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, label="sea level")

fig, ax, im = fingerprint.plot(fingerprint.ice_thickness, coasts=False)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, label="ice thickness")

plt.show()


# Set the change in ice_thickness.
ice_thickness_change = -0.5 * fingerprint.ice_thickness

# Convert to the direct load:
direct_load = fingerprint.direct_load_from_ice_thickness_change(ice_thickness_change)

# Note that the above method is equivalent to setting:
# direct_load = fingerprint.ice_density * fingerprint.one_minus_ocean_function * ice_thickness_change

fig, ax, im = fingerprint.plot(direct_load, coasts=False)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, label="direct load")

plt.show()


# Solve the sea level equation and display all outputs.
(sea_level_change, displacement, gravity_potential_change, angular_velocity_change) = (
    fingerprint(direct_load=direct_load, rtol=1.0e-9)
)


# Note that the results are dimensionalised before plotting.
fig, ax, im = fingerprint.plot(
    sea_level_change * fingerprint.length_scale, coasts=False
)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, label="sea level change")

fig, ax, im = fingerprint.plot(displacement * fingerprint.length_scale, coasts=False)
fig.colorbar(
    im, ax=ax, orientation="horizontal", shrink=0.7, label="vertical displacement"
)

fig, ax, im = fingerprint.plot(
    gravity_potential_change * fingerprint.gravitational_potential_scale, coasts=False
)
fig.colorbar(
    im, ax=ax, orientation="horizontal", shrink=0.7, label="gravity potential change"
)

print(
    f"relative angular velocity change {angular_velocity_change / fingerprint.rotation_frequency}"
)

plt.show()


# We can recover the gravitational potential change by subtracting centrifugal contribution.
gravitational_potential_change = (
    fingerprint.gravity_potential_change_to_gravitational_potential_change(
        gravity_potential_change, angular_velocity_change
    )
)


fig, ax, im = fingerprint.plot(
    gravitational_potential_change * fingerprint.gravitational_potential_scale,
    coasts=False,
)
fig.colorbar(
    im,
    ax=ax,
    orientation="horizontal",
    shrink=0.7,
    label="gravitational potential change",
)

# Get the centrifigual potential change
centrifugal_potential_change = gravity_potential_change - gravitational_potential_change

# The above calculation can also be done directly by
# centrifugal_potential_change = fingerprint.centrifugal_potential_change(angular_velocity_change)

fig, ax, im = fingerprint.plot(
    centrifugal_potential_change * fingerprint.gravitational_potential_scale,
    coasts=False,
)
fig.colorbar(
    im,
    ax=ax,
    orientation="horizontal",
    shrink=0.7,
    label="centrifugal potential change",
)

plt.show()
