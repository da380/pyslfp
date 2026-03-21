import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import cartopy.crs as ccrs

import pyslfp as sl


# Initialize the core fingerprint model
fp = sl.FingerPrint(
    lmax=64,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

# Define the model space for the ocean dynamic topography
order = 2.0
scale_km = 250.0
scale = scale_km * 1000 / fp.length_scale

model_space = inf.symmetric_space.sphere.Sobolev(
    fp.lmax, order, scale, radius=fp.mean_sea_floor_radius
)


# Set the altimetry locations
altimetry_points = fp.ocean_altimetry_points(spacing_degrees=4)
lats = [p[0] for p in altimetry_points]
lons = [p[1] for p in altimetry_points]
print(f"Using {len(altimetry_points)} altimetry points for the inversion.")


# Maps scalar-fields on sphere to be non-zero only over the oceans.
ocean_projection = sl.ocean_projection_operator(fp, model_space)

# Maps an ice thickness change to the corresponding load.
ocean_thickness_to_load = sl.ocean_thickness_change_to_load_operator(fp, model_space)

# Maps a direct load to the full response.
finger_print_operator = fp.as_sobolev_linear_operator(order, scale)
response_space = finger_print_operator.codomain


# Maps response to sea surface height
response_to_ssh = sl.sea_surface_height_operator(fp, response_space)


identity = model_space.identity_operator()
inclusion = response_to_ssh.codomain.inclusion_operator(order)

op1 = inf.ColumnLinearOperator([ocean_thickness_to_load, identity])

op2 = inf.BlockDiagonalLinearOperator(
    [
        response_to_ssh @ finger_print_operator,
        identity,
    ]
)

op3 = inf.RowLinearOperator([inclusion, identity])

op4 = model_space.point_evaluation_operator(altimetry_points)


A_shh = op3 @ op2 @ op1
A = op4 @ A_shh

data_space = A.codomain


# Define the data error statistics
altimetry_std_dev_m = 0.001
altimetry_std_dev = altimetry_std_dev_m / fp.length_scale
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    data_space, altimetry_std_dev
)

# Bundle everything into a forward problem object
forward_problem = inf.LinearForwardProblem(A, data_error_measure=data_error_measure)


# Set the initial model prior measure
pointwise_std_m = 0.1
pointwise_std = pointwise_std_m / fp.length_scale
initial_model_prior_measure = (
    model_space.point_value_scaled_heat_kernel_gaussian_measure(
        scale, std=pointwise_std
    )
)

# Transform so that ice thickness change non-zero only over current ice sheets.
model_prior_measure = initial_model_prior_measure.affine_mapping(
    operator=ocean_projection
)


# --- Generate the synthetic ground truth and noisy data ---
model_true, data = forward_problem.synthetic_model_and_data(model_prior_measure)


# Set up the Bayesian inversion method
bayesian_inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)

# Set up the diagonal preconditioner
blocks = sl.partition_points_by_grid(altimetry_points, 10)
print(f"Forming the preconditioner (blocks = {len(blocks)})")
preconditioner = bayesian_inversion.diagonal_normal_preconditioner(
    blocks=blocks, parallel=True, n_jobs=6
)

# Solve for the posterior distribution
print("Solving the linear system...")
solver = inf.CGMatrixSolver()
model_posterior_measure = bayesian_inversion.model_posterior_measure(
    data, solver, preconditioner=preconditioner
)

print(f"Number of iterations = {solver.iterations}")

# Get the posterior expectation
model_posterior_expectation = model_posterior_measure.expectation


print("Visualisation the expected values...")
# --- Calculate a shared, symmetric color scale for the ocean thickness plots ---
max_abs_ocean_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    (model_true).data.flatten(),
                    (model_posterior_expectation).data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)

# --- Plot 1: The "Ground Truth" Model ---
fig1, ax1, im1 = sl.plot(
    1000 * model_true * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ocean_change,
    vmax=max_abs_ocean_change,
    colorbar_label="ocean Thickness Change (mm)",
)
ax1.plot(
    lons,
    lats,
    "m^",
    markersize=5,
    transform=ccrs.PlateCarree(),
    markercolor="k",
    alpha=0.1,
)
ax1.set_title("a) True ocean Thickness Change")


# --- Plot 2: The Posterior Expectation ---
fig2, ax2, im2 = sl.plot(
    1000 * model_posterior_expectation * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ocean_change,
    vmax=max_abs_ocean_change,
    colorbar_label="ocean Thickness Change (mm)",
)
ax2.plot(
    lons,
    lats,
    "m^",
    markersize=5,
    transform=ccrs.PlateCarree(),
    markercolor="k",
    alpha=0.1,
)
ax2.set_title("b) Posterior Expectation")


plt.show()
