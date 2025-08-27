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

from pyslfp.operators import grace_operator, sh_coefficient_operator, wahr_operator


fp = FingerPrint(
    lmax=128,
    earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

ln = fp.love_numbers

X = fp.sobolev_load_space(2, 0.5)
A = wahr_operator(fp.love_numbers, X, X)

u = X.random()
v = X.random()

lhs = X.inner_product(v, A(u))
rhs = X.inner_product(A.adjoint(v), u)

print(lhs, rhs, np.abs(lhs - rhs) / np.abs(rhs))
