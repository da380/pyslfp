"""
A first example of calculating a sea level fingerprint. 
"""

# Import necessary modules.
import matplotlib.pyplot as plt
from pyslfp import FingerPrint

# Construct the FingerPrint object using default parameters.
fingerprint = FingerPrint()

# Set the background state using ice-7g present-day values.
fingerprint.set_state_from_ice_ng()

# Plot the sea level.
fig, ax, im = fingerprint.plot(fingerprint.sea_level)
fig.colorbar(
    im, ax=ax, orientation="horizontal", shrink=0.7, label="Present-day sea level"
)

# Plot the ice sheet thickness.
fig, ax, im = fingerprint.plot(fingerprint.ice_thickness, ice_projection=True)
fig.colorbar(
    im, ax=ax, orientation="horizontal", shrink=0.7, label="Present-day ice thickness"
)

# Set the direct load.
direct_load = fingerprint.northern_hemisphere_load()

# Plot the direct load.
fig, ax, im = fingerprint.plot(direct_load, ice_projection=True)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, label="direct load")


# Solve the sea level equation storing just the sea level change.
sea_level_change, _, _, _ = fingerprint(direct_load=direct_load, verbose=True)

# Plot the solution globally
fig, ax, im = fingerprint.plot(sea_level_change)
fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, label="sea level change")


# Plot the solution in the oceans only, and normalise by the mean sea level change.
mean_sea_level_change = fingerprint.mean_sea_level_change(direct_load)
fig, ax, im = fingerprint.plot(
    sea_level_change / mean_sea_level_change, ocean_projection=True, vmin=-1.5, vmax=1.5
)
fig.colorbar(
    im, ax=ax, orientation="horizontal", shrink=0.7, label="normalised sea level change"
)

plt.show()
