import matplotlib.pyplot as plt
import numpy as np
from pyslfp import FingerPrint, plot, IceModel


# 1. Initialise the fingerprint model
# lmax sets the spherical harmonic resolution.
fp = FingerPrint(lmax=128)

# 2. Set the background state (ice and sea level) to the present day
# This uses the built-in ICE-7G model loader.
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)


# 3. Define a surface mass load
# This function calculates the load corresponding to melting 10% of
# the Northern Hemisphere's ice mass.
# direct_load = fp.west_antarctic_load(fraction=1)
# direct_load = fp.greenland_load(fraction=1)

X = fp.lebesgue_load_space()
mu = X.heat_kernel_gaussian_measure(0.1 * fp.mean_sea_floor_radius)
fields = mu.samples(2)


for field in fields:

    direct_load = field * fp.ice_projection(value=0)

    fig1, ax1, im1 = plot(
        direct_load * fp.ice_projection(),
    )

    sea_level_change, _, _, _ = fp(direct_load=direct_load)

    fig2, ax2, im2 = plot(
        sea_level_change * fp.ocean_projection(),
    )


plt.show()
