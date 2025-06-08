import matplotlib.pyplot as plt
from pyslfp import FingerPrint


# Set up the FingerPrint instance.
finger_print = FingerPrint()

# Set the intial sea level and ice thickness to present-day
# values from ice-7g
finger_print.set_state_from_ice_ng()

# Set the direct load.
direct_load = finger_print.northern_hemisphere_load()

# Compute the sea level change.
sea_level_change, _, _, _ = finger_print(direct_load=direct_load, verbose=True)

# Normalise by the mean sea level change.
mean_sea_level_change = finger_print.mean_sea_level_change(direct_load)
sea_level_change /= mean_sea_level_change

# Plot the results.
fig, ax, im = finger_print.plot(
    sea_level_change, ocean_projection=True, vmin=-1.5, vmax=1.5
)
cbar = fig.colorbar(
    im, ax=ax, orientation="horizontal", shrink=0.7, label="Normalised sea level change"
)
ax.set_title("My first sea level fingerprint", y=1.1)
plt.show()
