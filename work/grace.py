import numpy as np
import matplotlib.pyplot as plt
import pyslfp as sl
import pygeoinf as inf


# Truncation degree for simulations.
lmax = 256

# Observation degree for GRACE data
lmax_obs = 100


# Set  up the fingerprint instance.
fp = sl.FingerPrint(
    lmax=lmax,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

# For the associated operator.
load_order = 2
load_scale = 0.05 * fp.mean_sea_floor_radius
fingerprint_operator = fp.as_sobolev_linear_operator(load_order, load_scale)

# Get the load and response space.
load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain


# Set up the prior for the load
load_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(load_scale)


# Set up the wmb and grace operators.
wmb = sl.WMBMethod.from_finger_print(fp, lmax_obs)
grace_operator = sl.grace_operator(response_space, lmax_obs)


# Set up the data errors using a noise-load
noise_load_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
    load_scale / 4, std=0.1
)
data_error_measure = wmb.load_measure_to_observation_measure(noise_load_measure)


# Set up the forward problem.
forward_operator = grace_operator @ fingerprint_operator
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# Generate synthetic data.
load, data = forward_problem.synthetic_model_and_data(load_prior)


# Build the preconditioner.
normal_preconditioner = wmb.bayesian_normal_operator_preconditioner(
    load_prior, data_error_measure
)

# Set up and solve the inverse problem.
inverse_problem = inf.LinearBayesianInversion(forward_problem, load_prior)

count1 = fp.solver_counter

load_posterior = inverse_problem.model_posterior_measure(
    data, inf.CGMatrixSolver(), preconditioner=normal_preconditioner
)

count2 = fp.solver_counter

print(f"number of sea level solutions = {count2-count1}")

# Compare the true load and the posterior expectation.
load_max = np.max(np.abs(load.data))
fig1, ax1, im1 = sl.plot(load, vmin=-load_max, vmax=load_max)
fig2, ax2, im2 = sl.plot(load_posterior.expectation, vmin=-load_max, vmax=load_max)
fig3, ax3, im3 = sl.plot(
    100 * (load_posterior.expectation - load) / load_max, symmetric=True
)
plt.show()
