import pyslfp as sl

fp = sl.FingerPrint(
    lmax=16,
    # earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation()
)
fp.set_state_from_ice_ng()

A = fp.as_sobolev_linear_operator(2, 0.1 * fp.mean_sea_floor_radius)

load_space = A.domain
response_space = A.codomain

mu = load_space.heat_kernel_gaussian_measure(0.1 * fp.mean_sea_floor_radius)

direct_load = fp.direct_load_from_ice_thickness_change(
    mu.sample() * fp.ice_projection(value=0)
)

# response = A(direct_load)

B = sl.sea_surface_height_operator(
    fp, response_space, remove_rotational_contribution=False
)
B.check()

# sea_surface_height_change = B(response)


# fig, ax, im = sl.plot(sea_surface_height_change, symmetric=True)
# fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
# plt.show()
