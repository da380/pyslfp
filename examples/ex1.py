import matplotlib.pyplot as plt
import numpy as np
from pyslfp import FingerPrint, plot, IceModel


# 1. Initialise the fingerprint model
# lmax sets the spherical harmonic resolution.
fp = FingerPrint(lmax=512)

# 2. Set the background state (ice and sea level) to the present day
# This uses the built-in ICE-7G model loader.
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)


# 3. Define a surface mass load
# This function calculates the load corresponding to melting 10% of
# the Northern Hemisphere's ice mass.
# direct_load = fp.west_antarctic_load(fraction=1)
direct_load = fp.greenland_load(fraction=1)


# 4. Solve the sea level equation for the given load
# This returns the sea level change, surface displacement, gravity change,
# and angular velocity change. In this instance, only the first of the
# returned fields is used.
sea_level_change, _, _, _ = fp(direct_load=direct_load)

mean_sea_level_change = fp.mean_sea_level_change(direct_load)
sea_level_change /= mean_sea_level_change


# 5. Plot the resulting sea level fingerprint,
# showing the result only over the oceans.
levels = np.linspace(-8, 2, 40)
fig, ax, im = plot(
    sea_level_change * fp.ocean_projection(),
    contour=True,
    levels=levels,
)

# Customize the plot
# ax.set_title("Sea Level Fingerprint for the West Antarctic ice sheet", y=1.1)
cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("Normalised Sea Level Change")
cbar.set_ticks([-8, -6, -4, -2, 0, 2])

plt.show()
