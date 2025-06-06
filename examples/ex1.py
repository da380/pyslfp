import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyslfp import FingerPrint


finger_print = FingerPrint()


finger_print.set_background_state_from_ice_ng()

direct_load = finger_print.northern_hemisphere_load()


sea_level_change, _, _, _ = finger_print(direct_load=direct_load, verbose=True)

mean_sea_level_change = finger_print.mean_sea_level_change(direct_load)
sea_level_change /= mean_sea_level_change


fig = finger_print.plot(
    sea_level_change,
    cmap="RdBu",
    borders=True,
    gridlines=True,
    colorbar=True,
    ocean_projection=True,
    vmin=-1.5,
    vmax=1.5,
    cbar_label="Normalised sea level change",
    figsize=(25, 15),
)

plt.show()
