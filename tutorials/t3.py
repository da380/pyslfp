"""
This example checks the sea level reciprocity theorem and then uses it to compute
sensitivity kernels for point measurements of sea level change. 
"""

# Import necessary modules.
import numpy as np
import matplotlib.pyplot as plt
from pyslfp import FingerPrint


# ======================================================#
#       Do a first check of reciprocity relation        #
# ======================================================#

# Set up the FingerPrint instance.
fingerprint = FingerPrint()

# Set the initial sea level and ice thickness.
fingerprint.set_state_from_ice_ng()

# Generate the first pair by melting all northern hemisphere ice.
direct_load_1 = fingerprint.northern_hemisphere_load()
sea_level_change_1, _, _, _ = fingerprint(direct_load=direct_load_1)

# Generate the second pair by melting all southern hemisphere ice.
direct_load_2 = fingerprint.southern_hemisphere_load()
sea_level_change_2, _, _, _ = fingerprint(direct_load=direct_load_2)

# Compute the two integrals
lhs = fingerprint.integrate(direct_load_2 * sea_level_change_1)
rhs = fingerprint.integrate(direct_load_1 * sea_level_change_2)

# Print the values for comparison
print(f"lhs value is {lhs}")
print(f"rhs value is {rhs}")

# Work out the relative difference. This should be of the same order of
# rtol=1e-6 used in solving the sea level equation.
print(f"relative difference is is {np.abs(rhs-lhs) / np.abs(rhs)}")


# ======================================================#
#              Do a second check to be sure             #
# ======================================================#

# Set up the first pair using a disk load
angular_radius_1 = 10
centre_lat_1 = 10
centre_lon_1 = 50
amplitude_1 = 10
direct_load_1 = fingerprint.disk_load(
    angular_radius_1, centre_lat_1, centre_lon_1, amplitude_1
)
sea_level_change_1, _, _, _ = fingerprint(direct_load=direct_load_1)

# Generate the second pair by melting all southern hemisphere ice.
angular_radius_2 = 30
centre_lat_2 = -50
centre_lon_2 = -40
amplitude_2 = -4
direct_load_2 = fingerprint.disk_load(
    angular_radius_2, centre_lat_2, centre_lon_2, amplitude_2
)
sea_level_change_2, _, _, _ = fingerprint(direct_load=direct_load_2)

# Compute the two integrals
lhs = fingerprint.integrate(direct_load_2 * sea_level_change_1)
rhs = fingerprint.integrate(direct_load_1 * sea_level_change_2)

# Print the values for comparison
print(f"lhs value is {lhs}")
print(f"rhs value is {rhs}")

# Work out the relative difference. This should be of the same order of
# rtol=1e-6 used in solving the sea level equation.
print(f"relative difference is is {np.abs(rhs-lhs) / np.abs(rhs)}")


# =======================================================#
#        Compute a sea level senitivity kernel          #
# =======================================================#

# Set the observation point to Boston, MA.
lat = 42.3555
lon = -71.0565

# Generate data from an ice load.
ice_thickness_change = (
    -1 * fingerprint.northern_hemisphere_projection(0) * fingerprint.ice_thickness
)
direct_load = fingerprint.direct_load_from_ice_thickness_change(ice_thickness_change)
sea_level_change, _, _, _ = fingerprint(direct_load=direct_load, rtol=1e-9)

# Evaluate the sea level change at the observation point
sea_level_change_at_location = fingerprint.point_evaulation(sea_level_change, lat, lon)
print(
    f"sea level change at the observation point is {sea_level_change_at_location:.8f}m"
)

# Set up the point load and compute the sensitivity kernel.
point_load = fingerprint.point_load(lat, lon)
adjoint_sea_level_change, _, _, _ = fingerprint(direct_load=point_load, rtol=1e-9)

# Integrate the adjoint sea level change against the direct load:
print(
    f"integral of direct load against adjoint sea level {fingerprint.integrate(adjoint_sea_level_change * direct_load):.8f}m"
)

# Form the sensitivity kernel wrt ice thickness change
kernel = (
    fingerprint.ice_density
    * fingerprint.one_minus_ocean_function
    * adjoint_sea_level_change
)
print(
    f"Integral of ice thickness change against sensitivity kernel {fingerprint.integrate(kernel * ice_thickness_change):.8f}m"
)

# Make a plot of the sensitivity kernel, projecting values onto the ice sheets.
fig, ax, im = fingerprint.plot(kernel, ice_projection=True)
fig.colorbar(
    im,
    ax=ax,
    orientation="horizontal",
    shrink=0.7,
    label="ice thickness kernel for Boston",
)

plt.show()


# Set up the point load and compute the sensitivity kernel now using smoothing over 2 degrees.
point_load = fingerprint.point_load(lat, lon, smoothing_angle=2)
adjoint_sea_level_change, _, _, _ = fingerprint(direct_load=point_load, rtol=1e-9)

# Form the sensitivity kernel wrt ice thickness change
kernel = (
    fingerprint.ice_density
    * fingerprint.one_minus_ocean_function
    * adjoint_sea_level_change
)


# Repeat the calculation but with a smoothed point load to remove non-physical oscillations.

print(
    f"average of the sea level about the observation point {fingerprint.integrate(point_load*sea_level_change):.8f}m"
)
print(
    f"integral of sensitivity kernel again ice thickness change {fingerprint.integrate(kernel*ice_thickness_change):.8f}m"
)

# Make a plot of the sensitivity kernel, projecting values onto the ice sheets.
fig, ax, im = fingerprint.plot(kernel, ice_projection=True)
fig.colorbar(
    im,
    ax=ax,
    orientation="horizontal",
    shrink=0.7,
    label="smoothed ice thickness kernel for Boston",
)

plt.show()
