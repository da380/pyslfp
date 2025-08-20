import pyslfp as sl
import numpy as np
import matplotlib.pyplot as plt

fp = sl.FingerPrint(lmax=64)
fp.set_state_from_ice_ng()

order = 1
scale = 0.2

load_space = fp.load_space(order, scale)
mu = load_space.heat_gaussian_measure(0.1, 1)
weighting_functions = mu.samples(4)


P = sl.LoadAveragingOperator(
    fingerprint=fp,
    order=order,
    scale=scale,
    weighting_functions=weighting_functions,
)

u = mu.sample()
v = P.codomain.random()

lhs = P.codomain.inner_product(v, P(u))
rhs = P.domain.inner_product(P.adjoint(v), u)

print(lhs, rhs * fp.mean_sea_floor_radius**2)
