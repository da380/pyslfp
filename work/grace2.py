"""
Tutorial Script: A Bayesian Inverse Problem - Inferring Ice Melt from GRACE Data

This script demonstrates how to infer global ice thickness changes from synthetic
GRACE (Gravity Recovery and Climate Experiment) data. It leverages the WMBMethod
to generate realistic observational noise and to construct a highly efficient,
purely spectral preconditioner for the Bayesian normal equations.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyslfp as sl
import pygeoinf as inf
from pygeoinf import plot_corner_distributions

# ==========================================
# Setup and Initialization
# ==========================================
print("Initializing FingerPrint model...")
lmax = 128
lmax_obs = 100

fp = sl.FingerPrint(
    lmax=lmax,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

# ==========================================
# Defining the Model Space
# ==========================================
# The unknown is the global pattern of ice thickness change.
order = 2.0
scale_km = 250.0
scale = scale_km * 1000 / fp.length_scale

model_space = fp.sobolev_load_space(order, scale)


# ==========================================
# Defining the Forward Operator
# ==========================================
# A = Grace Observation @ Fingerprint @ Thickness-to-Load @ Ice Projection
print("Building the forward operator chain...")
op1 = sl.ice_projection_operator(fp, model_space)
op2 = sl.ice_thickness_change_to_load_operator(fp, model_space)
op3 = fp.as_sobolev_linear_operator(order, scale, verbose=True)
response_space = op3.codomain
op4 = sl.grace_operator(response_space, lmax_obs)

A = op4 @ op3 @ op2 @ op1
data_space = A.codomain

# ==========================================
# Prior Model
# ==========================================
print("Generating synthetic ground truth and data...")
pointwise_std_m = 0.1  # 10 cm typical ice thickness change
pointwise_std = pointwise_std_m / fp.length_scale

# Base invariant measure, then projected to exist only over current ice sheets
initial_model_prior_measure = (
    model_space.point_value_scaled_heat_kernel_gaussian_measure(
        scale, std=pointwise_std
    )
)
model_prior_measure = initial_model_prior_measure.affine_mapping(operator=op1)

# ==========================================
# Data Error Model (GRACE Noise)
# ==========================================
# We use the WMBMethod to generate a realistic noise model. We start with an
# invariant load measure that is rougher (scale/4) than our expected signal.
print("Setting up observational noise model...")
wmb = sl.WMBMethod.from_finger_print(fp, lmax_obs)

noise_std_fac = 0.1 * fp.water_density
noise_scale_fac = 0.25
noise_load_measure = model_space.point_value_scaled_heat_kernel_gaussian_measure(
    noise_scale_fac * scale, std=noise_std_fac * pointwise_std
)

# Map this load noise directly into the observation space covariance
data_error_measure = wmb.load_measure_to_observation_measure(noise_load_measure)


# ==========================================
# Generate synthetic data
# ==========================================
forward_problem = inf.LinearForwardProblem(A, data_error_measure=data_error_measure)
model_true, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# ==========================================
# WMB Preconditioner
# ==========================================
# The preconditioner needs an *invariant* load measure to stay diagonal.
# We approximate our spatially-masked ice thickness prior by simply multiplying
# the unmasked invariant measure by the ice density.
print("Constructing WMB spectral preconditioner...")
precon_load_measure = initial_model_prior_measure * fp.ice_density

normal_preconditioner = wmb.bayesian_normal_operator_preconditioner(
    precon_load_measure, data_error_measure
)

# ==========================================
# Solving the Bayesian Inverse Problem
# ==========================================
print("Solving the Bayesian inverse problem...")
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior_measure)

count1 = fp.solver_counter

# The CG solver will use our ultra-fast diagonal preconditioner
model_posterior_measure = inverse_problem.model_posterior_measure(
    data, inf.CGMatrixSolver(), preconditioner=normal_preconditioner
)

count2 = fp.solver_counter
print(f"Number of full Sea Level Equation solutions required: {count2 - count1}")

model_posterior_expectation = model_posterior_measure.expectation

# ==========================================
# Visualizing the Inferred Ice Melt
# ==========================================
print("Plotting ice thickness results...")
max_abs_ice_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [model_true.data.flatten(), model_posterior_expectation.data.flatten()]
            )
        )
    )
    * 1000
    * fp.length_scale
)

fig1, ax1, im1 = sl.plot(
    1000 * model_true * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax1.set_title("a) True Ice Thickness Change")

fig2, ax2, im2 = sl.plot(
    1000 * model_posterior_expectation * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax2.set_title("b) Posterior Expectation (from GRACE)")
plt.show()


# ==========================================
# Mapping to GMSL and Regional Contributions
# ==========================================
print("Mapping posterior to GMSL contributions...")

GLI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.greenland_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
WAI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.west_antarctic_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
EAI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.east_antarctic_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

C = sl.averaging_operator(
    model_space,
    [GLI_weighting_function, WAI_weighting_function, EAI_weighting_function],
)

property_true = C(model_true)
property_posterior_measure = model_posterior_measure.affine_mapping(operator=C)

plot_corner_distributions(
    property_posterior_measure,
    true_values=property_true,
    labels=["Greenland (mm)", "West Antarctica (mm)", "East Antarctica (mm)"],
    title="Joint Posterior Distributions of GMSL Contributions",
)
plt.show()

print("Done!")
