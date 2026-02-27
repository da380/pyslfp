import pygeoinf as inf
import pyslfp as sl
import numpy as np
import matplotlib.pyplot as plt


lmax = 64
fp = sl.FingerPrint(lmax=64)
fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(2, 0.1)

load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain

prior1 = load_space.point_value_scaled_heat_kernel_gaussian_measure(
    0.5 * fp.mean_sea_floor_radius
)
prior2 = load_space.point_value_scaled_heat_kernel_gaussian_measure(
    0.1 * fp.mean_radius
)

ice_projection_operator = sl.ice_projection_operator(fp, load_space)
ocean_projection_operator = sl.ocean_projection_operator(fp, load_space)

ice_prior = prior1.affine_mapping(operator=ice_projection_operator)
ocean_prior = prior2.affine_mapping(operator=ocean_projection_operator)

model_space = inf.HilbertSpaceDirectSum([load_space, load_space])
model_prior = inf.GaussianMeasure.from_direct_sum([ice_prior, ocean_prior])

ice_to_load = sl.ice_thickness_change_to_load_operator(fp, load_space)
ocean_to_load = sl.sea_level_change_to_load_operator(fp, load_space)

model_to_load = inf.RowLinearOperator([ice_to_load, ocean_to_load])


# Load the full list of GLOSS tide gauge stations
lats, lons = sl.read_gloss_tide_gauge_data()

# --- Configuration for data selection ---
use_all_stations = True
if not use_all_stations:
    number_of_stations_to_sample = 100


# -----------------------------------------
tide_gauge_points = list(zip(lats, lons))

points = sl.read_gloss_tide_gauge_data()
tide_gauge_operator = sl.tide_gauge_operator(response_space, tide_gauge_points)

response_to_ssh_operator = sl.sea_surface_height_operator(fp, response_space)
points = response_to_ssh_operator.codomain.random_points(10)
ssh_to_data = response_to_ssh_operator.codomain.point_evaluation_operator(points)
ssh_operator = ssh_to_data @ response_to_ssh_operator

forward_operator_part1 = (
    inf.ColumnLinearOperator([tide_gauge_operator, ssh_operator])
    @ fingerprint_operator
    @ model_to_load
)

data_space = forward_operator_part1.codomain

forward_operator_part2 = (
    data_space.subspace_inclusion(1) @ ssh_to_data @ model_space.subspace_projection(1)
)

forward_operator = forward_operator_part1 + forward_operator_part2
