import matplotlib.pyplot as plt
import numpy as np
import random
import pyslfp as sl


fp = sl.FingerPrint()
fp.set_state_from_ice_ng()

X = fp.sobolev_load_space(2, 0.1 * fp.mean_sea_floor_radius)

mu = X.point_value_scaled_heat_kernel_gaussian_measure(0.1 * fp.mean_sea_floor_radius)

f1 = fp.greenland_projection(value=0) * fp.ice_projection(value=0)
f2 = fp.southern_hemisphere_projection(value=0) * fp.ice_projection(value=0)

alpha1 = 2
alpha2 = 1

f = alpha1 * f1 + alpha2 * f2

A = sl.spatial_mutliplication_operator(f, X)

nu = mu.affine_mapping(operator=A)

u = nu.sample()
std = nu.sample_pointwise_std(100)

sl.plot(u)
sl.plot(std)
plt.show()
