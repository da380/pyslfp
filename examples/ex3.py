import numpy as np
import matplotlib.pyplot as plt
from pyslfp import FingerPrint, plot, IceModel, EarthModelParameters
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pygeoinf import LinearOperator, EuclideanSpace, GaussianMeasure
from pyshtools import SHGrid, SHCoeffs
from pyslfp.utils import SHVectorConverter


# Import the cartopy coordinate reference system for plotting
import cartopy.crs as ccrs

from pyslfp.operators import (
    grace_operator,
    field_to_sh_coefficient_operator,
    wahr_operator,
)


fp = FingerPrint(
    lmax=256,
    earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

A = fp.as_lebesgue_linear_operator()

load_space = A.domain
potential_space = A.codomain.subspace(2)

observation_degree = 60

B = grace_operator(A.codomain, observation_degree)

mu = load_space.heat_kernel_gaussian_measure(0.1)

print("computing...")
u = mu.sample()

# u = fp.southern_hemisphere_load()
v = A(u)
w = B(v)


P = field_to_sh_coefficient_operator(potential_space, lmax=observation_degree, lmin=2)
x = P.adjoint(w)

Q = wahr_operator(
    fp.love_numbers,
    potential_space,
    load_space,
    lmax=observation_degree,
)
y = Q(x)

print("plotting...")

fig1, ax1, im1 = plot(u)
fig1.colorbar(im1, ax=ax1, orientation="horizontal")


fig2, ax2, im2 = plot(y)
im2.set_clim(im1.get_clim())
fig2.colorbar(im2, ax=ax2, orientation="horizontal")

norm = np.max(np.abs(u.data))
fig3, ax3, im3 = plot(100 * (u - y) / norm)
fig3.colorbar(im3, ax=ax3, orientation="horizontal")


plt.show()
