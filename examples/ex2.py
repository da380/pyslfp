import matplotlib.pyplot as plt
from pyslfp import FingerPrint, plot, IceModel
import pygeoinf as inf

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng()


A = fp.as_sobolev_linear_operator(2, 0.2 * fp.mean_sea_floor_radius)


u1 = fp.northern_hemisphere_load()
u2 = fp.southern_hemisphere_load()

v1 = A(u1)
v2 = A(u2)


X = A.domain
Y = A.codomain

lhs = Y.inner_product(v2, A(u1))
rhs = X.inner_product(A.adjoint(v2), u1)

print(lhs, rhs)
