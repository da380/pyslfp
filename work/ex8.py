import matplotlib.pyplot as plt
import numpy as np
from pyslfp import FingerPrint, plot, IceModel


# 1. Initialise the fingerprint model
# lmax sets the spherical harmonic resolution.
fp = FingerPrint()

# 2. Set the background state (ice and sea level) to the present day
# This uses the built-in ICE-7G model loader.
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)


# 3. Define a surface mass load
# This function calculates the load corresponding to melting 10% of
# the Northern Hemisphere's ice mass.


greenland_fraction = np.random.randn()
west_antarctic_fraction = np.random.randn()
east_antarctic_fraction = np.random.randn()

latitude_min = -66
latitude_max = 66

direct_load = (
    greenland_fraction * fp.greenland_load()
    + west_antarctic_fraction * fp.west_antarctic_load()
    + east_antarctic_fraction * fp.east_antarctic_load()
)


# 4. Solve the sea level equation for the given load
# This returns the sea level change, surface displacement, gravity change,
# and angular velocity change. In this instance, only the first of the
# returned fields is used.
(
    sea_level_change,
    displacement,
    gravitational_potential_change,
    angular_velocity_change,
) = fp(direct_load=direct_load)


sea_surface_height_change = fp.sea_surface_height_change(
    sea_level_change, displacement, angular_velocity_change
)

mean_sea_level_change = fp.mean_sea_level_change(direct_load)

altimetry_projection = fp.altimetry_projection(
    latitude_min=latitude_min, latitude_max=latitude_max, value=0
)

altimetry_projection_integral = fp.integrate(altimetry_projection)

altimetry_weighting_function = altimetry_projection / altimetry_projection_integral

mean_sea_level_change_estimate = fp.integrate(
    altimetry_weighting_function * sea_surface_height_change
)

print(f"True mean sea level change = {mean_sea_level_change}m")
print(f"Estimated mean sea level change = {mean_sea_level_change_estimate}m")
print(
    f"Relative error in estimate {100* np.abs(mean_sea_level_change_estimate-mean_sea_level_change)/np.abs(mean_sea_level_change)}%"
)

# 5. Plot the resulting sea level fingerprint,
# showing the result only over the oceans.
# fig, ax, im = plot(
#    sea_surface_height_change * fp.ocean_projection(),
# )


# plt.show()
