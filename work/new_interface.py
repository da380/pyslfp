

import pyslfp.linear_operators as op


A = op.FingerPrintOperator.for_sobolev_testing(64, 2, 0.1)

A.check(check_rtol=1e-6)

"""
direct_load = A.state.northern_hemisphere_load()

sea_level_change, _, _, _ = A(direct_load)


sl.plot(
    direct_load
    * A.state.ice_projection(exclude_glaciers=False)
    * A.state.model.parameters.length_scale
    / A.state.model.parameters.ice_density,
    coasts=False,
    symmetric=True,
)
sl.plot(
    sea_level_change
    * A.state.ocean_projection()
    * A.state.model.parameters.length_scale,
    coasts=False,
    symmetric=True,
)

plt.show()
"""

"""
#state = A.state
#parameters = A.parameters

direct_load = state.greenland_load()

slc, _, _, _ = A(direct_load)

sl.plot(slc * parameters.length_scale)

plt.show()


parameters = A.parameters

load_space = A.domain
response_space = A.codomain
field_space = response_space.subspace(0)


load_measure = load_space.heat_kernel_gaussian_measure(0.1 * parameters.mean_radius)
field_measure = field_space.heat_kernel_gaussian_measure(0.1 * parameters.mean_radius)
avc_measure = inf.GaussianMeasure.from_standard_deviation(
    response_space.subspace(3),
    parameters.rotation_frequency * parameters.mean_sea_floor_radius**4,
)
response_measure = inf.GaussianMeasure.from_direct_sum(
    [field_measure, field_measure, field_measure, avc_measure]
)

A.check(domain_measure=load_measure, codomain_measure=response_measure, check_rtol=1e-6)
"""
