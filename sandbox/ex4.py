import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import pyslfp as sl

from cartopy import crs as ccrs


def generate_symmetric_levels(
    max_val: float, num_positive_levels: int = 30
) -> np.ndarray:
    """
    Generates a non-linear, symmetric set of contour levels around zero.

    The levels are spaced with a power-law distribution (denser near zero).

    Args:
        max_val: The absolute maximum value (defines the range [-max_val, max_val]).
        num_positive_levels: The number of levels to generate between 0 and max_val.

    Returns:
        A NumPy array of sorted contour levels.
    """
    # Generate positive levels with a power-law spacing (e.g., quadratic)
    positive_levels = max_val * np.linspace(0, 1, num_positive_levels) ** 2

    # Create the full set of levels
    levels = np.concatenate([-positive_levels[:0:-1], positive_levels])

    # Ensure levels are unique to avoid plotting issues
    return np.unique(levels)


fp = sl.FingerPrint(
    lmax=512,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

A = fp.as_sobolev_linear_operator(1.5, 0.1)
load_space = A.domain

P = A.codomain.subspace_projection(0)

B = P @ A
sea_level_space = B.codomain


C = sl.spatial_mutliplication_operator(
    fp.water_density * fp.ocean_function, sea_level_space
)

D = C @ B

I = inf.LinearOperator.from_formal_adjoint(
    sea_level_space, load_space, load_space.underlying_space.identity_operator()
)

E = load_space.identity_operator() + I @ D

mu = A.domain.point_value_scaled_sobolev_kernel_gaussian_measure(1.1, 0.1, 1)

u = mu.sample()

v = D(u)

w = E(u)

u_max = np.max(np.abs(u.data))
v_max = np.max(np.abs(v.data))
uv_max = max(u_max, v_max)

fac = 0.10


fig1, ax1, im1 = sl.plot(
    u,
    vmin=-uv_max,
    vmax=uv_max,
)
ax1.set_title("Direct load", y=1.1)
cbar1 = fig1.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.05, shrink=0.7)
cbar1.set_label("Direct load")

fig2, ax2, im2 = sl.plot(
    v,
    vmin=-fac * uv_max,
    vmax=fac * uv_max,
)
ax2.set_title("Induced load", y=1.1)
cbar2 = fig2.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.05, shrink=0.7)
cbar2.set_label("Induced load")

fig3, ax3, im3 = sl.plot(
    w,
    vmin=-uv_max,
    vmax=uv_max,
)
ax3.set_title("Total load", y=1.1)
cbar3 = fig3.colorbar(im3, ax=ax3, orientation="horizontal", pad=0.05, shrink=0.7)
cbar3.set_label("Total load")


nu = mu.affine_mapping(operator=E)

point = (-15.96, -5.7)
dirac = load_space.dirac_representation(point)

mu_covariance = mu.covariance(dirac)
nu_covariance = nu.covariance(dirac)


mu_covariance_max = np.max(np.abs(mu_covariance.data))
nu_covariance_max = np.max(np.abs(nu_covariance.data))
corr_max = np.max([mu_covariance_max, nu_covariance_max])


fig4, ax4, im4 = sl.plot(
    mu_covariance,
    vmin=-corr_max,
    vmax=corr_max,
)
ax4.set_title("Direct load two-point covariance", y=1.1)
cbar4 = fig4.colorbar(im4, ax=ax4, orientation="horizontal", pad=0.05, shrink=0.7)
cbar4.set_label("covariance")
ax4.plot([point[1]], [point[0]], "m^", markersize=5, transform=ccrs.PlateCarree())


fig5, ax5, im5 = sl.plot(
    nu_covariance,
    vmin=-corr_max,
    vmax=corr_max,
)
ax5.set_title("Total load two-point covariance", y=1.1)
cbar5 = fig5.colorbar(im5, ax=ax5, orientation="horizontal", pad=0.05, shrink=0.7)
cbar5.set_label("covariance")
ax5.plot([point[1]], [point[0]], "m^", markersize=5, transform=ccrs.PlateCarree())


fig6, ax6, im6 = sl.plot(
    nu_covariance - mu_covariance,
    vmin=-fac * corr_max,
    vmax=fac * corr_max,
)
ax6.set_title("Difference in two-point covariance", y=1.1)
cbar6 = fig6.colorbar(im6, ax=ax6, orientation="horizontal", pad=0.05, shrink=0.7)
cbar6.set_label("covariance")
ax6.plot([point[1]], [point[0]], "m^", markersize=5, transform=ccrs.PlateCarree())


point = (-70, -65)
dirac = load_space.dirac_representation(point)

mu_covariance = mu.covariance(dirac)
nu_covariance = nu.covariance(dirac)


mu_covariance_max = np.max(np.abs(mu_covariance.data))
nu_covariance_max = np.max(np.abs(nu_covariance.data))
corr_max = np.max([mu_covariance_max, nu_covariance_max])

fig7, ax7, im7 = sl.plot(
    mu_covariance,
    vmin=-corr_max,
    vmax=corr_max,
)
ax7.set_title("Direct load two-point covariance", y=1.1)
cbar7 = fig7.colorbar(im7, ax=ax7, orientation="horizontal", pad=0.05, shrink=0.7)
cbar7.set_label("covariance")
ax7.plot([point[1]], [point[0]], "m^", markersize=5, transform=ccrs.PlateCarree())


fig8, ax8, im8 = sl.plot(
    nu_covariance,
    vmin=-corr_max,
    vmax=corr_max,
)
ax8.set_title("Total load two-point covariance", y=1.1)
cbar8 = fig8.colorbar(im8, ax=ax8, orientation="horizontal", pad=0.05, shrink=0.7)
cbar8.set_label("covariance")
ax8.plot([point[1]], [point[0]], "m^", markersize=5, transform=ccrs.PlateCarree())


fig9, ax9, im9 = sl.plot(
    nu_covariance - mu_covariance,
    vmin=-fac * corr_max,
    vmax=fac * corr_max,
)
ax9.set_title("Difference in two-point covariance", y=1.1)
cbar9 = fig9.colorbar(im9, ax=ax9, orientation="horizontal", pad=0.05, shrink=0.7)
cbar9.set_label("covariance")
ax9.plot([point[1]], [point[0]], "m^", markersize=5, transform=ccrs.PlateCarree())


plt.show()
