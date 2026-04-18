import matplotlib.pyplot as plt

import pygeoinf as inf
import pyslfp as sl

A = sl.linear_operators.FingerPrintOperator.for_testing(128)
P1 = A.codomain.subspace_projection(0)


B = P1 @ A

measure = inf.white_noise_measure(B.domain)

C = inf.LowRankSVD.from_randomized(B, 100, method="fixed", measure=measure, power=3)


zeta = A.state.northern_hemisphere_load()

sl1 = B(zeta)
sl2 = C(zeta)

ax1, im1 = sl.plot(
    sl1 * A.parameters.length_scale * A.state.ocean_projection(),
    coasts=False,
    symmetric=True,
)
ax2, im2 = sl.plot(
    sl2 * A.parameters.length_scale * A.state.ocean_projection(),
    coasts=False,
    symmetric=True,
)
A.state.plot_coastline(ax1)
A.state.plot_coastline(ax2)

plt.show()
