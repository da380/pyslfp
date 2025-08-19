import matplotlib.pyplot as plt
from pyslfp import FingerPrint, plot, IceModel
import pygeoinf as inf

fp = FingerPrint(lmax=32)
fp.set_state_from_ice_ng()


A = fp.as_linear_operator(2, 0.5)

P = inf.LinearOperator.formally_self_adjoint(
    A.domain, lambda u: u * fp.northern_hemisphere_projection(0) * fp.ice_projection(0)
)

A = A @ P

direct_load = fp.northern_hemisphere_load(fraction=0.1)

sea_level_change1, _, _, _ = A(direct_load)
fig, ax, im = plot(
    sea_level_change1 * fp.ocean_projection(),
)
ax.set_title("Sea Level Fingerprint of Northern Hemisphere Ice Melt", y=1.1)
cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("Sea Level Change (meters)")


L, D, R = A.random_svd(50, power=2)

B = L @ D @ R

sea_level_change2, _, _, _ = B(direct_load)
fig, ax, im = plot(
    sea_level_change2 * fp.ocean_projection(),
)
ax.set_title("Sea Level Fingerprint of Northern Hemisphere Ice Melt", y=1.1)
cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("Sea Level Change (meters)")

fig, ax, im = plot(
    (sea_level_change2 - sea_level_change1) * fp.ocean_projection(),
)
ax.set_title("Sea Level Fingerprint of Northern Hemisphere Ice Melt", y=1.1)
cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("Sea Level Change (meters)")


plt.show()
