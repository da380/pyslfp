import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import cartopy.crs as ccrs
from pygeoinf import plot_1d_distributions, plot_corner_distributions

import pyslfp as sl


# Initialize the core fingerprint model
fp = sl.FingerPrint(
    lmax=128,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

# Define the model space for the unknown ice thickness change
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


# Maps scalar-fields on sphere to be non-zero only over background ice sheets.
op1 = sl.ice_projection_operator(fp, model_space)

# Maps an ice thickness change to the corresponding load.
op2 = sl.ice_thickness_change_to_load_operator(fp, model_space)

# Maps a direct load to the full response.
op3 = fp.as_sobolev_linear_operator(order, scale)

# Maps response to sea surface height at locations
op4 = sl.ocean_altimetry_operator(fp, op3.codomain, altimetry_points)

# Form the forward operator by composition.
A = op4 @ op3 @ op2 @ op1
data_space = A.codomain

# Form also a mapping from the model space to the sea surface height field for convenience.
response_to_ssh = sl.sea_surface_height_operator(fp, op3.codomain)
A_ssh = response_to_ssh @ op3 @ op2 @ op1


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
model_prior_measure = initial_model_prior_measure.affine_mapping(operator=op1)

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
# --- Calculate a shared, symmetric color scale for the ice thickness plots ---
max_abs_ice_change = (
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
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax1.plot(lons, lats, "m^", markersize=5, transform=ccrs.PlateCarree())
ax1.set_title("a) True Ice Thickness Change")


# --- Plot 2: The Posterior Expectation ---
fig2, ax2, im2 = sl.plot(
    1000 * model_posterior_expectation * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax2.plot(lons, lats, "m^", markersize=5, transform=ccrs.PlateCarree())
ax2.set_title("b) Posterior Expectation")


# Calculate the predicted sea surface height field from the posterior expectation model
ssh_posterior = A_ssh(model_posterior_expectation)

# Calculate the true SSH change.
ssh_true = A_ssh(model_true)

# --- Calculate a shared, symmetric color scale for the SSH plots ---
ocean_mask = fp.ocean_projection()
max_abs_sl_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    (ssh_true * ocean_mask).data.flatten(),
                    (ssh_posterior * ocean_mask).data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)


# --- Plot 3: The "True" SSH Field ---
fig3, ax3, im3 = sl.plot(
    1000 * ssh_true * ocean_mask * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="SSH Change (mm)",
)
ax3.set_title("a) True Sea surface height change")
ax3.plot(lons, lats, "m^", markersize=5, transform=ccrs.PlateCarree())


# --- Plot 4: The SSH Field Predicted by the Inversion ---
fig4, ax4, im4 = sl.plot(
    1000 * ssh_posterior * fp.ocean_projection() * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="SSH Change (mm)",
)
ax4.set_title("b) Predicted Sea surface height Fingerprint")
ax4.plot(lons, lats, "m^", markersize=5, transform=ccrs.PlateCarree())


# Set the weighting function for GMSL estimates  - Note that length scale factor to dimensionalise the result into mm
print("PDF for GMSL change...")
GMSL_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

# Form the mapping to GSML.
B = sl.averaging_operator(model_space, [GMSL_weighting_function])

# Get the true GMSL
GMSL_true = B(model_true)

# Push forward the posterior to the GMSL space.
GMSL_prior_measure = model_prior_measure.affine_mapping(operator=B)
GMSL_posterior_measure = model_posterior_measure.affine_mapping(operator=B)

# Plot the PDFs
fig5, ax5 = plot_1d_distributions(
    GMSL_posterior_measure,
    true_value=GMSL_true[0],
    xlabel="GMSL Change (mm)",
    title="Global Mean Sea Level Change Inference from GRACE Data",
    show_plot=False,
)


print("Joint PDFs for Ice sheet contributions...")
# --- Plot 6: Corner Plot for GMSL Contributions (mm) ---
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


# Visualise the distribution using a corner plot
fig6, axes = plot_corner_distributions(
    property_posterior_measure,
    true_values=property_true,
    labels=[
        "Greenland Contribution (mm)",
        "West Antarctica Contribution (mm)",
        "East Antarctica Contribution (mm)",
    ],
    title="Joint Posterior Distributions of GMSL Contributions from Major Ice Sheets",
    show_plot=False,
)

plt.show()
