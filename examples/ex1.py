import matplotlib.pyplot as plt
from pyslfp import FingerPrint, plot_SHGrid

finger_print = FingerPrint()


finger_print.set_background_state_from_ice_ng()

plot_SHGrid(finger_print.sea_level)
plt.show()
