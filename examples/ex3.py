import numpy as np
import matplotlib.pyplot as plt
from pyslfp import FingerPrint, plot, IceModel, EarthModelParameters
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pygeoinf import LinearOperator, EuclideanSpace
from pyshtools import SHGrid, SHCoeffs
from pyslfp.utils import SHVectorConverter


# Import the cartopy coordinate reference system for plotting
import cartopy.crs as ccrs

from pyslfp.operators import grace_operator


fp = FingerPrint(
    lmax=128,
    earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

A = fp.as_lebesgue_linear_operator()

B = grace_operator(A.codomain, 4)


u = B.domain.zero
v = B.codomain.random()

phi = u[2]
phi_lm = fp.expand_field(phi)
phi_lm.coeffs[0, 2, 0] = 1
phi = fp.expand_coefficient(phi_lm)
u[2] = phi


lhs = B.codomain.inner_product(v, B(u))
rhs = B.domain.inner_product(B.adjoint(v), u)

print(lhs, rhs, np.abs(lhs - rhs) / np.abs(rhs))
print(rhs / lhs, fp.mean_sea_floor_radius**2)
