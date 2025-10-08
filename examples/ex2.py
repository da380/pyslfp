import matplotlib.pyplot as plt
import pyslfp as sl

fp = sl.FingerPrint(
    lmax=256,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

fig, ax = fp.plot_greens_functions_split(n_points=1000)

plt.show()
