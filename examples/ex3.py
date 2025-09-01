import numpy as np
import matplotlib.pyplot as plt
from pyslfp import FingerPrint, plot, IceModel, EarthModelParameters
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pygeoinf import LinearOperator, EuclideanSpace, GaussianMeasure
from pyshtools import SHGrid, SHCoeffs
from pyslfp.utils import SHVectorConverter

import cartopy.crs as ccrs

from pyslfp.operators import (
    grace_operator,
    field_to_sh_coefficient_operator,
    sh_coefficient_to_field_operator,
    WMBMethod,
)


fp = FingerPrint(
    lmax=256,
    earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

A = fp.as_lebesgue_linear_operator()
# A = fp.as_sobolev_linear_operator(2, 0.1)

load_space = A.domain
potential_space = A.codomain.subspace(2)

observation_degree = 256

B = grace_operator(A.codomain, observation_degree)


def projection_mapping(u: SHGrid) -> SHGrid:
    ulm = load_space.to_coefficient(u)
    ulm.coeffs[:, :1, :] = 0
    return load_space.from_coefficient(ulm)


P = inf.LinearOperator.self_adjoint(load_space, projection_mapping)
mu = load_space.sobolev_kernel_gaussian_measure(2, 0.3)
nu = mu.affine_mapping(operator=P)


print("computing...")
u = nu.sample()


v = A(u)
w = B(v)


wmb_method = WMBMethod.from_finger_print(observation_degree, fp)

D = wmb_method.potential_coefficient_to_load_operator(load_space)
y = D(w)

print("plotting...")

fig1, ax1, im1 = plot(u)
fig1.colorbar(im1, ax=ax1, orientation="horizontal")


fig2, ax2, im2 = plot(y)
im2.set_clim(im1.get_clim())
fig2.colorbar(im2, ax=ax2, orientation="horizontal")

norm = np.max(np.abs(u.data))
fig3, ax3, im3 = plot((u - y) / norm)
# im3.set_clim(im1.get_clim())
fig3.colorbar(im3, ax=ax3, orientation="horizontal")


plt.show()
