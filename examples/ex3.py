import pyslfp as sl
import numpy as np
import matplotlib.pyplot as plt

fp = sl.FingerPrint()
fp.set_state_from_ice_ng()

order = 2
scale = 0.2


P = sl.TideGaugeObservationOperator(fp, 2, 0.2, [(0, 0)])

u = P.domain.random()
v = P.codomain.random()

lhs = P.codomain.inner_product(v, P(u))
rhs = P.domain.inner_product(P.adjoint(v), u)

print(lhs, rhs)
