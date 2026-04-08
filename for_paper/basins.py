import pyslfp as sl
import matplotlib.pyplot as plt
from cartopy import crs as ccrs

fp = sl.FingerPrint(lmax=512)
fp.set_state_from_ice_ng()

load_space = fp.lebesgue_load_space()

B = sl.ice_sheet_averaging_operator(load_space, fp)


v = B.codomain.random()

u = B.adjoint(v)

sl.plot(u * fp.ice_projection())

plt.show()
