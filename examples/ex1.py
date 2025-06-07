import matplotlib.pyplot as plt
from pyslfp import FingerPrint


finger_print = FingerPrint(lmax=256)


finger_print.set_state_from_ice_ng()

direct_load = finger_print.northern_hemisphere_load()
finger_print.plot(direct_load, ice_projection=True, colorbar=True)

sea_level_change, _, _, _ = finger_print(direct_load=direct_load, verbose=True)

mean_sea_level_change = finger_print.mean_sea_level_change(direct_load)
sea_level_change /= mean_sea_level_change


fig, _ = finger_print.plot(
    sea_level_change, ocean_projection=True, colorbar=True, vmin=-1.5, vmax=1.5
)

plt.show()
