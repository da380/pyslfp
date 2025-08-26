import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
import pyslfp as sl


fp = sl.FingerPrint(lmax=64)
fp.set_state_from_ice_ng()

order = 3
scale = 0.2

load_space = fp.load_space(order, scale)

w = fp.constant_grid(1)
A = inf.LinearOperator(
    load_space,
    inf.EuclideanSpace(1),
    lambda u: fp.integrate(u * w),
    formal_adjoint_mapping=lambda v: v * w,
)

u = fp.constant_grid(1)
v = np.random.randn()

lhs = A.codomain.inner_product(v, A(u))
rhs = A.domain.inner_product(A.adjoint(v), u)

print(lhs, rhs)


"""
mu = load_space.heat_gaussian_measure(0.1, 1)
weighting_functions = [fp.constant_grid(1)]


P = sl.LoadAveragingOperator(
    fingerprint=fp,
    order=order,
    scale=scale,
    weighting_functions=weighting_functions,
)

u = fp.constant_grid(1)
v = P.codomain.random()


lhs = P.codomain.inner_product(v, P(u))
rhs = P.domain.inner_product(P.adjoint(v), u)

print(lhs, rhs)
"""
