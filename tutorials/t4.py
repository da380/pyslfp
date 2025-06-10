"""
In this example we look at the generalised reciprocity theorem and use it to compute 
sensitivity kernels for  a range of observables. 
"""

# Import necessary modules.
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from pyslfp import FingerPrint


# Set up the fingerprint instance.
fingerprint = FingerPrint()
fingerprint.set_state_from_ice_ng()


def random_disk_load():
    """
    Function that returns a randomised load for testing.
    """
    delta = np.random.uniform(10, 30)
    lat = np.random.uniform(-90, 90)
    lon = np.random.uniform(-180, 180)
    amp = np.random.randn()
    return fingerprint.disk_load(delta, lat, lon, amp)


def random_angular_momentum():
    """
    Function that returns a randomised angular momentum jump
    whose magnitude is comparable that of the loads.
    """
    b = fingerprint.mean_sea_floor_radius
    omega = fingerprint.rotation_frequency
    load = random_disk_load()
    load_lm = load.expand(lmax_calc=2)
    return omega * b**4 * load_lm.coeffs[:, 2, 1]


############################################################
#                  Reciprocity theorem test                #
############################################################

print("Checking reciprocity theorem")

direct_load_1 = random_disk_load()
direct_load_2 = random_disk_load()
displacement_load_1 = random_disk_load()
displacement_load_2 = random_disk_load()
gravitational_potential_load_1 = random_disk_load()
gravitational_potential_load_2 = random_disk_load()
angular_momentum_change_1 = random_angular_momentum()
angular_momentum_change_2 = random_angular_momentum()

(
    sea_level_change_1,
    displacement_1,
    gravity_potential_change_1,
    angular_velocity_change_1,
) = fingerprint(
    direct_load=direct_load_1,
    displacement_load=displacement_load_1,
    gravitational_potential_load=gravitational_potential_load_1,
    angular_momentum_change=angular_momentum_change_1,
    rtol=1e-9,
)

(
    sea_level_change_2,
    displacement_2,
    gravity_potential_change_2,
    angular_velocity_change_2,
) = fingerprint(
    direct_load=direct_load_2,
    displacement_load=displacement_load_2,
    gravitational_potential_load=gravitational_potential_load_2,
    angular_momentum_change=angular_momentum_change_2,
    rtol=1e-9,
)

g = fingerprint.gravitational_acceleration

lhs_integrand = direct_load_2 * sea_level_change_1 - (1 / g) * (
    g * displacement_load_2 * displacement_1
    + gravitational_potential_load_2 * gravity_potential_change_1
)
lhs = (
    fingerprint.integrate(lhs_integrand)
    - np.dot(angular_momentum_change_2, angular_velocity_change_1) / g
)

rhs_integrand = direct_load_1 * sea_level_change_2 - (1 / g) * (
    g * displacement_load_1 * displacement_2
    + gravitational_potential_load_1 * gravity_potential_change_2
)
rhs = (
    fingerprint.integrate(rhs_integrand)
    - np.dot(angular_momentum_change_1, angular_velocity_change_2) / g
)


print(f"lhs of identity =  {lhs}")
print(f"rhs of identity =  {rhs}")
print(f"relative difference = {np.abs(rhs-lhs)/np.abs(rhs)}\n")


# ===========================================================================#
#                      Displacement sensitivity kernel                      #
# ===========================================================================#

print("Displacement sensitivity kernel test\n")

# Set the observation point for SENU GPS station in Greenland
lat = 61.0696
lon = -47.1413


# Set the direct load
direct_load = fingerprint.northern_hemisphere_load()

# Solve the forward problem
_, displacement, _, _ = fingerprint(direct_load=direct_load, rtol=1e-9)


# Get the displacement at the observation point.
displacement_observed = fingerprint.point_evaulation(displacement, lat, lon)

# Set the adjoint displacement load. In this case we use a method that
# does the calculation for us. Other adjoing force terms are returned as
#  None and here ignored.
adjoint_displacement_load = -1 * fingerprint.point_load(lat, lon)

# Solve the adjoint problem.
adjoint_sea_level_change, _, _, _ = fingerprint(
    displacement_load=adjoint_displacement_load, rtol=1e-9
)


# Print the observed value and prediction via sensitivity kernel
print(f"Displacement at observation point =  {displacement_observed:.8f}m")
displacement_prediction = fingerprint.integrate(adjoint_sea_level_change * direct_load)
print(
    f"Predicted displacement using sensitivity kernel = {displacement_prediction:.8f}m"
)
print(
    f"Relative difference = {np.abs(displacement_observed-displacement_prediction)/np.abs(displacement_observed)}\n"
)

# Recalculate a smoothed kernel for plotting. In this case we use a method that does the
# calculation for us. Other adjoint force terms are returned and None and here ignored.
_, adjoint_displacement_load, _, _ = (
    fingerprint.adjoint_loads_for_displacement_point_measurement(
        lat, lon, smoothing_angle=1.0
    )
)
adjoint_sea_level_change, _, _, _ = fingerprint(
    displacement_load=adjoint_displacement_load, rtol=1e-9
)
kernel = fingerprint.direct_load_from_ice_thickness_change(adjoint_sea_level_change)

fig, ax, im = fingerprint.plot(
    kernel,
    projection=ccrs.Orthographic(central_longitude=lon, central_latitude=lat + 8),
    map_extent=[lon - 16, lon + 16, lat - 4, lat + 12],
    lat_interval=4,
    lon_interval=4,
)
plt.show()


#######################################################################
#                    Gravitaitonal potential kernel.                  #
#######################################################################


print("Gravitational potential sensitivity kernel test\n")

# Set the direct load
direct_load = fingerprint.southern_hemisphere_load()

# Solve the forward problem
_, _, gravity_potential_change, angular_velocity_change = fingerprint(
    direct_load=direct_load, rtol=1e-9
)

# Get the gravitational potential change.
gravitational_potential_change = (
    fingerprint.gravity_potential_change_to_gravitational_potential_change(
        gravity_potential_change, angular_velocity_change
    )
)

# set the observation degree and order
l = 15
m = 4


# Print the coefficient.
observed_coefficient = fingerprint.coefficient_evaluation(
    gravitational_potential_change, l, m
)
print(f"The degree {l} and order {m} coefficient = {observed_coefficient}")


# Get the adjoint load.
_, _, adjoint_gravitational_potential_load, adjoint_angular_momentum_change = (
    fingerprint.adjoint_loads_for_gravitational_potential_coefficient(l, m)
)

# Solve the adjoint problem.
adjoint_sea_level_change, _, _, _ = fingerprint(
    gravitational_potential_load=adjoint_gravitational_potential_load,
    angular_momentum_change=adjoint_angular_momentum_change,
    rtol=1e-9,
)

predicted_coefficient = fingerprint.integrate(adjoint_sea_level_change * direct_load)
print(f"Integral of direct load against adjoint sea level {predicted_coefficient}")

print(
    f"Relative difference = {np.abs(observed_coefficient-predicted_coefficient)/ np.abs(observed_coefficient)}\n"
)

# Plot the adjoint sea level.
fingerprint.plot(adjoint_sea_level_change)

# Plot the corresponding ice thickness kernel
kernel = fingerprint.direct_load_from_ice_thickness_change(adjoint_sea_level_change)
fingerprint.plot(kernel, ice_projection=True)

plt.show()


######################################################################
#                      Angular velocity kernel                       #
######################################################################

print("Angular velocity kernel teset\n")

# Set the measurement direction.
direction = np.array([1, 0], dtype=float)
direction /= np.linalg.norm(direction)

# Set the direct load
direct_load = fingerprint.northern_hemisphere_load()

# Solve the forward problem
_, _, _, angular_velocity_change = fingerprint(direct_load=direct_load, rtol=1e-9)


# Observed value
observed_value = np.dot(direction, angular_velocity_change)
print(f"observed angular velocity change = {observed_value}")

# Set the adjoint force
adjoint_angular_momentum_change = -g * direction

# Solve the adjoint problem
adjoint_sea_level_change, _, _, _ = fingerprint(
    angular_momentum_change=adjoint_angular_momentum_change, rtol=1e-9
)


# Integrate direct load against the kernel.
predicted_value = fingerprint.integrate(adjoint_sea_level_change * direct_load)

print(f"Integral of adjoint sea level against direct load = {predicted_value}")


print(
    f"Relative difference = {np.abs(observed_value-predicted_value)/ np.abs(observed_value)}\n"
)


# Plot the adjoint sea level.
fingerprint.plot(adjoint_sea_level_change)

# Plot the corresponding ice thickness kernel
kernel = fingerprint.direct_load_from_ice_thickness_change(adjoint_sea_level_change)
fingerprint.plot(kernel, ice_projection=True)

plt.show()
