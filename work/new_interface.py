import matplotlib.pyplot as plt
import pygeoinf as inf
import pyslfp as sl

from pyslfp.core import EarthModelParameters, EarthModel
from pyslfp.ice_ng import IceNG
from pyslfp.physics import SeaLevelEquation
from pyslfp.state import EarthState

import pyslfp.linear_operators as op

lmax = 64

parameters = EarthModelParameters.from_standard_non_dimensionalisation()

earth_model = EarthModel(lmax, parameters=parameters)

ice_model = IceNG(length_scale=parameters.length_scale)

ice_thickness, sea_level = ice_model.get_ice_thickness_and_sea_level(0, lmax)

state = EarthState(ice_thickness, sea_level, earth_model)

sle = SeaLevelEquation(earth_model)

A = op.get_lebesgue_linear_operator(sle, state, rtol=1e-9, verbose=True)

load_space = A.domain
response_space = A.codomain
field_space = response_space.subspace(0)


load_measure = load_space.heat_kernel_gaussian_measure(0.1 * parameters.mean_radius)
field_measure = field_space.heat_kernel_gaussian_measure(0.1 * parameters.mean_radius)
avc_measure = inf.GaussianMeasure.from_standard_deviation(
    response_space.subspace(3), 0.01
)
response_measure = inf.GaussianMeasure.from_direct_sum(
    [field_measure, field_measure, field_measure, avc_measure]
)

A.check(domain_measure=load_measure, codomain_measure=response_measure, check_rtol=1e-4)
