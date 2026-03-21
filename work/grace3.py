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
order = 2.0
scale_km = 250.0
scale = scale_km * 1000 / fp.length_scale


# ==========================================
# Defining the Forward Operator
# ==========================================
print("Building the forward operator chain...")
finger_print_operator = fp.as_sobolev_linear_operator(order, scale)

model_space = finger_print_operator.domain
response_space = finger_print_operator.codomain
grace_operator = sl.grace_operator(response_space, lmax_obs)

forward_operator = grace_operator @ finger_print_operator
data_space = forward_operator.codomain

# ==========================================
# Prior Model
# ==========================================
print("Generating synthetic ground truth and data...")
pointwise_std = fp.water_density * 0.01 / fp.length_scale

# Base invariant measure, then projected to exist only over current ice sheets
model_prior_measure = model_space.point_value_scaled_heat_kernel_gaussian_measure(
    scale, std=pointwise_std
)


# ==========================================
# Data Error Model (GRACE Noise)
# ==========================================
print("Setting up observational noise model...")
wmb = sl.WMBMethod.from_finger_print(fp, lmax_obs)

noise_std_fac = 0.1
noise_scale_fac = 0.1
noise_load_measure = model_space.point_value_scaled_heat_kernel_gaussian_measure(
    noise_scale_fac * scale, std=noise_std_fac * pointwise_std
)

# Map this load noise directly into the observation space covariance
data_error_measure = wmb.load_measure_to_observation_measure(noise_load_measure)


# ==========================================
# Generate synthetic data
# ==========================================
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)
model_true, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# ==========================================
# WMB Preconditioner
# ==========================================
print("Constructing WMB spectral preconditioner...")
normal_preconditioner = wmb.bayesian_normal_operator_preconditioner(
    model_prior_measure, data_error_measure
)

# ==========================================
# Solving the Bayesian Inverse Problem
# ==========================================
print("Solving the Bayesian inverse problem...")
inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior_measure)


# The CG solver will use our ultra-fast diagonal preconditioner
model_posterior_measure = inverse_problem.model_posterior_measure(
    data, inf.CGMatrixSolver(), preconditioner=normal_preconditioner
)


model_posterior_expectation = model_posterior_measure.expectation

# ==========================================
# Visualizing the Inferred Ice Melt
# ==========================================
to_mm = 1000 * fp.length_scale / fp.water_density

norm = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [model_true.data.flatten(), model_posterior_expectation.data.flatten()]
            )
        )
    )
    * to_mm
)

fig1, ax1, im1 = sl.plot(
    model_true * to_mm,
    coasts=True,
    cmap="seismic",
    vmin=-norm,
    vmax=norm,
    colorbar_label="Equivalent water thickness (mm)",
)
ax1.set_title("a) True model")

fig2, ax2, im2 = sl.plot(
    model_posterior_expectation * to_mm,
    coasts=True,
    cmap="seismic",
    vmin=-norm,
    vmax=norm,
    colorbar_label="Equivalent water thickness (mm)",
)
ax2.set_title("b) Posterior Expectation ")


# ==========================================
# Estimate pointwise std
# ==========================================
print("Sampling from the posterior...")
n_sample = 100
pointwise_std = model_posterior_measure.sample_pointwise_std(
    n_sample, parallel=True, n_jobs=10
)

fig3, ax3, im3 = sl.plot(
    pointwise_std * to_mm,
    coasts=True,
    colorbar_label="Equivalent water thickness (mm)",
    cmap="Reds",
)
ax3.set_title(f"Pointwise std from {n_sample} samples")

plt.show()


print("Done!")
