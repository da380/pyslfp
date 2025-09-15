import matplotlib.pyplot as plt
import numpy as np
import random
import pyslfp as sl
import pygeoinf as inf
import cartopy.crs as ccrs
import scipy.stats as stats
import matplotlib.colors as colors

# Initialize the core fingerprint model
fp = sl.FingerPrint(
    lmax=256,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()


# Load the full list of GLOSS tide gauge stations
lats, lons = sl.read_gloss_tide_gauge_data()

# --- Configuration for data selection ---
use_all_stations = True


# -----------------------------------------
if use_all_stations:
    tide_gauge_points = list(zip(lats, lons))
else:
    number_of_stations_to_sample = 100
    random.seed(123)
    indices = random.sample(range(len(lats)), number_of_stations_to_sample)
    sampled_lats = [lats[int(i)] for i in indices]
    sampled_lons = [lons[int(i)] for i in indices]
    tide_gauge_points = list(zip(sampled_lats, sampled_lons))

print(f"Using {len(tide_gauge_points)} tide gauge stations for the inversion.")

# Define the model space for the unknown ice thickness change
order = 1.2
scale_km = 250.0
scale = scale_km * 1000 / fp.length_scale

model_space = inf.symmetric_space.sphere.Sobolev(
    fp.lmax, order, scale, radius=fp.mean_sea_floor_radius
)

# Define operators
op1 = sl.ice_projection_operator(fp, model_space)
op2 = sl.ice_thickness_change_to_load_operator(fp, model_space)
op3 = fp.as_sobolev_linear_operator(order, scale, rtol=1e-9)
op4 = sl.operators.tide_gauge_operator(op3.codomain, tide_gauge_points)

# Form the forward operator
A = op4 @ op3 @ op2 @ op1
data_space = A.codomain

P = op3.codomain.subspace_projection(0)
A_sl = P @ op3 @ op2 @ op1

# Define the data error statistics
tide_gauge_std_dev_m = 0.001
tide_gauge_std_dev = tide_gauge_std_dev_m / fp.length_scale
data_error_measure = inf.GaussianMeasure.from_standard_deviation(
    data_space, tide_gauge_std_dev
)

# Bundle into a forward problem object
forward_problem = inf.LinearForwardProblem(A, data_error_measure=data_error_measure)

# Set the initial model prior measure
pointwise_std_m = 0.1
pointwise_std = pointwise_std_m / fp.length_scale
initial_model_prior_measure = (
    model_space.point_value_scaled_heat_kernel_gaussian_measure(scale, pointwise_std)
)
model_prior_measure = initial_model_prior_measure.affine_mapping(operator=op1)

# --- Generate the synthetic ground truth and noisy data ---
model_true, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# Set up and solve the Bayesian inversion
bayesian_inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)
print("Solving the linear system...")
model_posterior_measure = bayesian_inversion.model_posterior_measure(
    data,
    inf.EigenSolver(parallel=True),
)
model_posterior_expectation = model_posterior_measure.expectation

# ==============================================================================
# PLOTTING SECTION (ALL UNITS IN MM)
# ==============================================================================

# --- Calculate a shared, symmetric color scale for the ice thickness plots ---
max_abs_ice_change_m = (
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
    * fp.length_scale
)
# CONVERT TO MM
max_abs_ice_change_mm = max_abs_ice_change_m * 1000

# --- Plot 1: The "Ground Truth" Model (mm) ---
fig1, ax1, im1 = sl.plot(
    model_true * fp.length_scale * 1000,  # CONVERT TO MM
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change_mm,
    vmax=max_abs_ice_change_mm,
)
ax1.set_title("a) True Ice Thickness Change")
fig1.colorbar(
    im1,
    ax=ax1,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="Ice Thickness Change (mm)",  # UPDATED LABEL
)

# --- Plot 2: The Posterior Expectation (Our Best Estimate) (mm) ---
fig2, ax2, im2 = sl.plot(
    model_posterior_expectation * fp.length_scale * 1000,  # CONVERT TO MM
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change_mm,
    vmax=max_abs_ice_change_mm,
)
ax2.set_title("b) Posterior Expectation (Inferred from Data)")
fig2.colorbar(
    im2,
    ax=ax2,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="Ice Thickness Change (mm)",  # UPDATED LABEL
)

# Calculate the predicted sea-level fields
sea_level_posterior = A_sl(model_posterior_expectation)
sea_level_true = A_sl(model_true)

# --- Calculate a shared, symmetric color scale for the sea-level plots ---
ocean_mask = fp.ocean_projection()
max_abs_sl_change_m = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    (sea_level_true * ocean_mask).data.flatten(),
                    (sea_level_posterior * ocean_mask).data.flatten(),
                ]
            )
        )
    )
    * fp.length_scale
)
# CONVERT TO MM
max_abs_sl_change_mm = max_abs_sl_change_m * 1000

# --- Plot 3: The "True" Sea-Level Field (mm) ---
fig1, ax1, im1 = sl.plot(
    sea_level_true * ocean_mask * fp.length_scale * 1000,  # CONVERT TO MM
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change_mm,
    vmax=max_abs_sl_change_mm,
)
ax1.set_title("a) True Sea-Level Fingerprint")
ax1.plot(lons, lats, "m^", markersize=5, transform=ccrs.PlateCarree())
fig1.colorbar(
    im1,
    ax=ax1,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="Sea Level Change (mm)",  # UPDATED LABEL
)

# --- Plot 4: The Sea-Level Field Predicted by the Inversion (mm) ---
fig2, ax2, im2 = sl.plot(
    sea_level_posterior
    * fp.ocean_projection()
    * fp.length_scale
    * 1000,  # CONVERT TO MM
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change_mm,
    vmax=max_abs_sl_change_mm,
)
ax2.set_title("b) Predicted Sea-Level Fingerprint")
ax2.plot(lons, lats, "m^", markersize=5, transform=ccrs.PlateCarree())
fig2.colorbar(
    im2,
    ax=ax2,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="Sea Level Change (mm)",  # UPDATED LABEL
)

# --- Plot 5: 1D GMSL Distributions (mm) ---
GMSL_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
B = sl.averaging_operator(model_space, [GMSL_weighting_function])
GMSL_true_m = B(model_true)

GMSL_prior_measure = model_prior_measure.affine_mapping(operator=B)
GMSL_posterior_measure = model_posterior_measure.affine_mapping(operator=B)

# Get stats in meters first
gmsl_posterior_mean_m = GMSL_posterior_measure.expectation[0]
gmsl_posterior_var_m2 = GMSL_posterior_measure.covariance.matrix(dense=True)[0, 0]
gmsl_posterior_std_m = np.sqrt(gmsl_posterior_var_m2)
gmsl_prior_mean_m = GMSL_prior_measure.expectation[0]
gmsl_prior_var_m2 = GMSL_prior_measure.covariance.matrix(dense=True)[0, 0]
gmsl_prior_std_m = np.sqrt(gmsl_prior_var_m2)

# CONVERT TO MM for plotting
gmsl_posterior_mean_mm = gmsl_posterior_mean_m * 1000
gmsl_posterior_std_mm = gmsl_posterior_std_m * 1000
gmsl_prior_mean_mm = gmsl_prior_mean_m * 1000
gmsl_prior_std_mm = gmsl_prior_std_m * 1000
GMSL_true_mm = GMSL_true_m[0] * 1000

# Define an x-axis in mm
x_min = min(
    gmsl_prior_mean_mm - 6 * gmsl_prior_std_mm,
    gmsl_posterior_mean_mm - 6 * gmsl_posterior_std_mm,
)
x_max = max(
    gmsl_prior_mean_mm + 6 * gmsl_prior_std_mm,
    gmsl_posterior_mean_mm + 6 * gmsl_posterior_std_mm,
)
x_axis = np.linspace(x_min, x_max, 1000)

prior_pdf_values = stats.norm.pdf(
    x_axis, loc=gmsl_prior_mean_mm, scale=gmsl_prior_std_mm
)
posterior_pdf_values = stats.norm.pdf(
    x_axis, loc=gmsl_posterior_mean_mm, scale=gmsl_posterior_std_mm
)

fig, ax1 = plt.subplots(figsize=(12, 7))
color1 = "green"
ax1.set_xlabel("Total GMSL Change (mm)")  # UPDATED LABEL
ax1.set_ylabel("Prior Probability Density", color=color1)
ax1.plot(x_axis, prior_pdf_values, color=color1, lw=2, linestyle=":", label="Prior PDF")
ax1.fill_between(x_axis, prior_pdf_values, color=color1, alpha=0.15)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.grid(True, linestyle="--")

ax2 = ax1.twinx()
color2 = "blue"
ax2.set_ylabel("Posterior Probability Density", color=color2)
ax2.plot(
    x_axis, posterior_pdf_values, color=color2, lw=2, label="Posterior PDF (Estimated)"
)
ax2.fill_between(x_axis, posterior_pdf_values, color=color2, alpha=0.2)
ax2.tick_params(axis="y", labelcolor=color2)

# Plot vertical lines with updated labels
line1 = ax1.axvline(
    gmsl_posterior_mean_mm,
    color="red",
    linestyle="--",
    lw=2,
    label=f"Posterior Mean: {gmsl_posterior_mean_mm:.2f} mm",
)
line2 = ax1.axvline(
    GMSL_true_mm,
    color="black",
    linestyle="-",
    lw=2,
    label=f"True Value: {GMSL_true_mm:.2f} mm",
)

handles, labels_leg = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
all_handles = handles + handles2
all_labels = [h.get_label() for h in all_handles]
fig.legend(all_handles, all_labels, loc="upper right", bbox_to_anchor=(0.9, 0.9))
fig.suptitle("Prior and Posterior Probability Distributions of GMSL", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])


# --- Plot 6: Corner Plot for GMSL Contributions (mm) ---
GLI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.greenland_projection(value=0)
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
WAI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.west_antarctic_projection(value=0)
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
EAI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.east_antarctic_projection(value=0)
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

C = sl.averaging_operator(
    model_space,
    [GLI_weighting_function, WAI_weighting_function, EAI_weighting_function],
)

property_true = C(model_true)
property_posterior_measure = model_posterior_measure.affine_mapping(operator=C)

# Extract stats
labels = ["Greenland (mm)", "West Antarctica (mm)", "East Antarctica (mm)"]
mean_posterior = property_posterior_measure.expectation
cov_posterior = property_posterior_measure.covariance.matrix(dense=True)
true_values = property_true

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Joint Posterior Distribution of GMSL Contributions", fontsize=16)

for i in range(3):
    for j in range(3):
        ax = axes[i, j]

        if i == j:  # Diagonal plots (1D)
            # CONVERT TO MM
            mu = mean_posterior[i] * 1000
            sigma = np.sqrt(cov_posterior[i, i]) * 1000
            true_val_mm = true_values[i] * 1000
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            pdf = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, pdf, "darkblue", label="Posterior PDF")
            ax.fill_between(x, pdf, color="lightblue", alpha=0.6)
            # Use mm values in labels
            ax.axvline(mu, color="red", linestyle="--", label=f"Mean: {mu:.2f} mm")
            ax.axvline(
                true_val_mm,
                color="black",
                linestyle="-",
                label=f"True: {true_val_mm:.2f} mm",
            )
            ax.set_xlabel(labels[i])
            ax.set_yticklabels([])
            ax.set_ylabel("Density" if i == 0 else "")

        elif i > j:  # Off-diagonal plots (2D)
            # CONVERT TO MM
            mean_2d = np.array([mean_posterior[j], mean_posterior[i]]) * 1000
            cov_2d = (
                np.array(
                    [
                        [cov_posterior[j, j], cov_posterior[j, i]],
                        [cov_posterior[i, j], cov_posterior[i, i]],
                    ]
                )
                * 1000**2
            )
            x_range = np.linspace(
                mean_2d[0] - 4 * np.sqrt(cov_2d[0, 0]),
                mean_2d[0] + 4 * np.sqrt(cov_2d[0, 0]),
                100,
            )
            y_range = np.linspace(
                mean_2d[1] - 4 * np.sqrt(cov_2d[1, 1]),
                mean_2d[1] + 4 * np.sqrt(cov_2d[1, 1]),
                100,
            )
            X, Y = np.meshgrid(x_range, y_range)
            pos = np.dstack((X, Y))
            rv = stats.multivariate_normal(mean_2d, cov_2d)
            Z = rv.pdf(pos)
            pcm = ax.pcolormesh(
                X,
                Y,
                Z,
                shading="auto",
                cmap="Blues",
                norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
            )
            ax.contour(X, Y, Z, colors="black", linewidths=0.5, alpha=0.6)
            if i == 1 and j == 0:
                cbar = fig.colorbar(pcm, ax=ax, orientation="vertical", shrink=0.8)
                cbar.set_label("Probability Density")
            # Plot points using mm values
            ax.plot(
                mean_posterior[j] * 1000,
                mean_posterior[i] * 1000,
                "r+",
                markersize=10,
                mew=2,
                label="Posterior Mean",
            )
            ax.plot(
                true_values[j] * 1000,
                true_values[i] * 1000,
                "kx",
                markersize=10,
                mew=2,
                label="True Value",
            )
            ax.set_xlabel(labels[j])
            ax.set_ylabel(labels[i])

        else:  # Hide upper triangle
            ax.axis("off")

# Final adjustments for corner plot legend
handles, labels_leg = axes[0, 0].get_legend_handles_labels()
handles.extend(axes[1, 0].get_legend_handles_labels())
fig.legend(
    handles,
    [l.split(":")[0] for l in labels_leg],
    loc="upper right",
    bbox_to_anchor=(0.9, 0.95),
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show all plots at the end
plt.show()
