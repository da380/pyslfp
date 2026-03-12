import numpy as np
import pyslfp as sl

# Initialize the core fingerprint model - lower lmax to reduce calculation times for this tutorial.
fp = sl.FingerPrint(
    lmax=128,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()


# Define the model space for the unknown ice thickness change
order = 2.0
scale_km = 250.0
scale = scale_km * 1000 / fp.length_scale


# Set up the forward operator
fingerprint_operator = fp.as_sobolev_linear_operator(order, scale)

model_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain

ice_projection_operator = sl.ice_projection_operator(fp, model_space)
ice_thickness_change_to_load_operator = sl.ice_thickness_change_to_load_operator(
    fp, model_space
)

observation_degree = 16
grace_operator = sl.grace_operator(response_space, observation_degree)

forward_operator = (
    grace_operator
    @ fingerprint_operator
    @ ice_thickness_change_to_load_operator
    @ ice_projection_operator
)
data_space = forward_operator.codomain

# Set up the prior model
pointwise_std_m = 0.1
pointwise_std = pointwise_std_m / fp.length_scale


initial_model_prior_measure = model_space.heat_kernel_gaussian_measure(scale)

dirac = model_space.dirac_representation((0, 0))
pointwise_var = model_space.inner_product(
    initial_model_prior_measure.covariance(dirac), dirac
)


model_prior_measure = (
    initial_model_prior_measure.affine_mapping(operator=ice_projection_operator)
    * pointwise_std
    / np.sqrt(pointwise_var)
)

model = model_prior_measure.sample()


prior_diagonal = model_space._degree_dependent_scaling_values()


"""

# Set random field for generation of the noise model
noise_scale = 0.5 * scale
noise_std = 0.05 * pointwise_std * fp.water_density
noise_field = model_space.point_value_scaled_heat_kernel_gaussian_measure(
    noise_scale, noise_std
)

# Set the data error measure
data_error_measure = noise_field.affine_mapping(
    operator=grace_operator @ fingerprint_operator
)
error_samples = data_error_measure.samples(10)
data_error_measure = inf.GaussianMeasure.from_samples(data_space, error_samples)




# Set the forward problem
forward_problem = inf.LinearForwardProblem(
    forward_operator, data_error_measure=data_error_measure
)

# Generate synthetic data
model_true, data = forward_problem.synthetic_model_and_data(model_prior_measure)

# Set up the inverse problem
bayesian_inversion = inf.LinearBayesianInversion(forward_problem, model_prior_measure)

"""

"""

# Solve for the posterior measure
model_posterior_measure = bayesian_inversion.model_posterior_measure(
    data,
    inf.CGSolver,
)
model_posterior_expectation = model_posterior_measure.expectation


fig1, ax1, im1 = sl.plot(model)

fig2, ax2, im2 = sl.plot(model_posterior_expectation)

plt.show()

"""

"""

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
ax1.set_title("a) True Ice Thickness Change")

# --- Plot 2: The Posterior Expectation (Our Best Estimate) ---
fig2, ax2, im2 = sl.plot(
    1000 * model_posterior_expectation * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax2.set_title("b) Posterior Expectation (Inferred from Data)")

plt.show()











# Solve for the posterior distribution




# Calculate the predicted sea-level field from the posterior expectation model
sea_level_posterior = A_sl(model_posterior_expectation)

# Calculate the true sea level change.
sea_level_true = A_sl(model_true)

# --- Calculate a shared, symmetric color scale for the sea-level plots ---
ocean_mask = fp.ocean_projection()
max_abs_sl_change = (
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
    * 1000
    * fp.length_scale
)


# --- Plot 3: The "True" Sea-Level Field ---
fig1, ax1, im1 = sl.plot(
    1000 * sea_level_true * ocean_mask * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="Sea Level Change (mm)",
)
ax1.set_title("a) True Sea-Level Fingerprint")
ax1.plot(lons, lats, "m^", markersize=5, transform=ccrs.PlateCarree())


# --- Plot 4: The Sea-Level Field Predicted by the Inversion ---
fig2, ax2, im2 = sl.plot(
    1000 * sea_level_posterior * fp.ocean_projection() * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="Sea Level Change (mm)",
)
ax2.set_title("b) Predicted Sea-Level Fingerprint")
ax2.plot(lons, lats, "m^", markersize=5, transform=ccrs.PlateCarree())
plt.show()


# Estimate pointwise variance using 5 random samples
pointwise_variance = model_posterior_measure.sample_pointwise_variance(10)

# Convert to pointwise standard deviation
pointwise_standard_deviation = pointwise_variance.copy()
pointwise_standard_deviation.data[:, :] = np.sqrt(pointwise_variance.data[:, :])


# Plot the results
fig, ax, im = sl.plot(
    1000 * pointwise_standard_deviation * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change standard deviation (mm)",
)
ax.set_title("Posterior pointwise standard deviation")
plt.show()


# Set the weighting function for GMSL estimates  - Note that length scale factor to dimensionalise the result into mm
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
fig, ax = plot_1d_distributions(
    GMSL_posterior_measure,
    prior_measures=GMSL_prior_measure,
    true_value=GMSL_true[0],
    xlabel="GMSL Change (mm)",
    title="Global Mean Sea Level Change Inference from GRACE Data",
)

plt.show()

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


property_true = C(model_true)
property_posterior_measure = model_posterior_measure.affine_mapping(operator=C)

# Visualise the distribution using a corner plot
plot_corner_distributions(
    property_posterior_measure,
    true_values=property_true,
    labels=[
        "Greenland Contribution (mm)",
        "West Antarctica Contribution (mm)",
        "East Antarctica Contribution (mm)",
    ],
    title="Joint Posterior Distributions of GMSL Contributions from Major Ice Sheets",
)

plt.show()
"""
