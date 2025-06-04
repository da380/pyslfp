import matplotlib.pyplot as plt
from pyslfp import FingerPrint, plot_SHGrid

finger_print = FingerPrint(lmax=128)


finger_print.set_background_state_from_ice_ng()

direct_load = finger_print.northern_hemisphere_load()


(
    sea_level_change,
    vertical_displacement,
    gravity_potential_change,
    angular_velocity_change,
) = finger_print.generalised_solver(
    direct_load, verbose=True, rotational_feedbacks=False
)


ax = finger_print.plot(sea_level_change * finger_print.ocean_mask(), cmap="RdBu")
plt.colorbar()
plt.show()
