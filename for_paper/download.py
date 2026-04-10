import pyslfp as sl
import matplotlib.pyplot as plt

fp = sl.FingerPrint()
fp.set_state_from_ice_ng(version=sl.IceModel.ICE5G)

sl.plot(fp.sea_level, symmetric=True)
plt.show()
