"""
Prior Sensitivity Analysis for Bayesian GRACE Inversion
=====================================================

This script systematically tests the sensitivity of the Bayesian GRACE inversion
to misspecifications in the prior distribution and assumed noise levels.
It performs five distinct parameter sweeps.

For each sweep, two figures are generated:
  1. Maps (3x2 grid): True Load + 5 Posterior Expectation maps.
  2. PDFs (3x2 grid): 5 Tested Priors (aggregated) + 5 Individual Posterior PDFs
     for the Gulf of Mexico regional average.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pygeoinf as inf
import grace_utils as utils
import pyslfp as sl


matplotlib.use("Agg")


def main():
    # --- 1. CONFIGURATION & TRUTH SETUP ---
    output_dir = "output_plots_grace_prior_sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    inf.configure_threading(n_threads=1)

    lmax = 256
    obs_degree = 100
    load_order = 2.0
    load_scale_km = 500.0

    print("Building exact physical operators (Truth)...")
    state, load_space, response_space, fp_op, ewt_scale = (
        utils.build_physics_components(lmax, load_order, load_scale_km)
    )

    total_load_op = utils.build_total_load_operator(
        state, response_space, load_space, fp_op
    )
    grace_obs_op = sl.linear_operators.grace_observation_operator(
        response_space, obs_degree
    )
    exact_forward_op = grace_obs_op @ fp_op @ total_load_op
    wmb_method = sl.linear_operators.WMBMethod(state.model, obs_degree)

    # Baseline Parameters (The "Truth")
    base_scale = 250.0
    base_std = 0.01
    base_noise_scale_fac = 0.25
    base_noise_std_fac = 0.1

    print("Generating True Load and Synthetic Data...")
    _, true_prior, true_noise, _ = utils.build_measures(
        state,
        load_space,
        base_scale,
        base_std,
        base_noise_scale_fac,
        base_noise_std_fac,
    )

    true_data_error_measure = wmb_method.load_measure_to_observation_measure(true_noise)
    forward_problem = inf.LinearForwardProblem(
        exact_forward_op, data_error_measure=true_data_error_measure
    )

    # Draw the single true model and resulting data to be used across ALL sweeps
    true_model, synthetic_data = forward_problem.synthetic_model_and_data(true_prior)

    # Setup Regional Averaging for the Map Overlays and the PDF plots
    region_names, avg_op, _, regions_dict = utils.get_regional_averaging(
        state, load_space
    )

    # Build the 1D extraction operator for the Gulf of Mexico
    tot_avg_op = ewt_scale * avg_op @ total_load_op
    gom_idx = region_names.index("Gulf of Mexico")
    gom_proj = tot_avg_op.codomain.subspace_projection(gom_idx)
    gom_op = gom_proj @ tot_avg_op

    true_gom_val = gom_op(true_model)[0]

    # Calculate the standard deviation of the true prior specifically for the GoM region
    true_gom_prior_1d = true_prior.affine_mapping(
        operator=gom_op
    ).with_dense_covariance()
    gom_prior_std = np.sqrt(true_gom_prior_1d.covariance.matrix(dense=True)[0, 0])

    # Generate a fixed shape for the mean shift experiments
    # We loop until we draw a shape with a non-trivial GoM signal (> 0.2 sigma)
    # to ensure normalizing it doesn't cause the rest of the globe to blow up.
    while True:
        fixed_shift_shape = true_prior.sample()
        shift_gom_val = gom_op(fixed_shift_shape)[0]
        if np.abs(shift_gom_val) > 0.2 * gom_prior_std:
            break

    # Normalize the shape so that a shift_mag of 1.0 shifts the GoM average by exactly +1 prior standard deviation
    fixed_shift_shape = fixed_shift_shape * (gom_prior_std / shift_gom_val)

    # Global color scale limits based on the true model
    vmax = np.max(np.abs(true_model.data * ewt_scale)) * 1.2

    # --- 2. DEFINE THE EXPERIMENTAL SWEEPS (5 variations per sweep) ---
    sweeps = {
        "Length_Scale": [
            {
                "scale": 100.0,
                "std": base_std,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Very Short (100 km)",
            },
            {
                "scale": 150.0,
                "std": base_std,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Short (150 km)",
            },
            {
                "scale": 250.0,
                "std": base_std,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Baseline (250 km)",
            },
            {
                "scale": 350.0,
                "std": base_std,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Long (350 km)",
            },
            {
                "scale": 500.0,
                "std": base_std,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Very Long (500 km)",
            },
        ],
        "Prior_Amplitude": [
            {
                "scale": base_scale,
                "std": 0.002,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Overconfident (0.002 m)",
            },
            {
                "scale": base_scale,
                "std": 0.005,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Tight (0.005 m)",
            },
            {
                "scale": base_scale,
                "std": 0.01,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Baseline Std (0.01 m)",
            },
            {
                "scale": base_scale,
                "std": 0.02,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Loose (0.02 m)",
            },
            {
                "scale": base_scale,
                "std": 0.05,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Underconfident (0.05 m)",
            },
        ],
        "Mean_Shift": [
            {
                "scale": base_scale,
                "std": base_std,
                "shift_mag": 0.0,
                "noise_fac": base_noise_std_fac,
                "title": "Baseline (No Shift)",
            },
            {
                "scale": base_scale,
                "std": base_std,
                "shift_mag": 0.5,
                "noise_fac": base_noise_std_fac,
                "title": "Small Shift (0.5x)",
            },
            {
                "scale": base_scale,
                "std": base_std,
                "shift_mag": 1.0,
                "noise_fac": base_noise_std_fac,
                "title": "Medium Shift (1.0x)",
            },
            {
                "scale": base_scale,
                "std": base_std,
                "shift_mag": 1.5,
                "noise_fac": base_noise_std_fac,
                "title": "Large Shift (1.5x)",
            },
            {
                "scale": base_scale,
                "std": base_std,
                "shift_mag": 2.0,
                "noise_fac": base_noise_std_fac,
                "title": "V. Large Shift (2.0x)",
            },
        ],
    }

    # --- 3. EXECUTE SWEEPS ---
    solver = inf.CGSolver(rtol=0.01, callback=inf.ProgressCallback())

    for sweep_name, configs in sweeps.items():
        print(f"\n--- Running Sweep: {sweep_name} ---")

        # Create two distinct 3x2 figures
        fig_maps, axes_maps = sl.subplots(3, 2, figsize=(18, 16))
        axes_maps = axes_maps.flatten()

        fig_pdfs, axes_pdfs = plt.subplots(3, 2, figsize=(18, 16), layout="constrained")
        axes_pdfs = axes_pdfs.flatten()

        # 1. Plot True Model on the first Map Panel
        sl.plot(
            true_model * ewt_scale,
            ax=axes_maps[0],
            colorbar=True,
            vmin=-vmax,
            vmax=vmax,
            cmap="seismic",
            colorbar_kwargs={"label": "Load (mm EWT)"},
        )
        axes_maps[0].set_title("True Load", fontsize=14, fontweight="bold")
        utils.draw_region_boundaries(state, axes_maps[0], regions_dict)

        # Storage for 1D distributions
        gom_priors = []
        gom_posteriors = []
        plot_labels = []

        # 2. Iterate through 5 configs
        for i, config in enumerate(configs):
            panel_idx = i + 1
            print(f"  Solving config {panel_idx}/5: {config['title']}...")

            # Build perturbed measures
            initial_prior_base, test_prior_base, test_noise, _ = utils.build_measures(
                state,
                load_space,
                config["scale"],
                config["std"],
                base_noise_scale_fac,
                base_noise_std_fac,
            )

            # Apply Shift
            if config["shift_mag"] != 0.0:
                test_prior = test_prior_base.affine_mapping(
                    translation=fixed_shift_shape * config["shift_mag"]
                )
            else:
                test_prior = test_prior_base

            # Invert
            test_data_err_measure = wmb_method.load_measure_to_observation_measure(
                test_noise
            )
            test_forward_prob = inf.LinearForwardProblem(
                exact_forward_op, data_error_measure=test_data_err_measure
            )
            preconditioner = wmb_method.bayesian_normal_operator_preconditioner(
                initial_prior_base, test_data_err_measure
            )
            test_inverse = inf.LinearBayesianInversion(test_forward_prob, test_prior)

            posterior = test_inverse.model_posterior_measure(
                synthetic_data, solver, preconditioner=preconditioner
            )

            # Map the Prior and Posterior to the 1D Gulf of Mexico space
            gom_prior_1d = test_prior.affine_mapping(
                operator=gom_op
            ).with_dense_covariance()
            gom_post_1d = posterior.affine_mapping(
                operator=gom_op
            ).with_dense_covariance()

            gom_priors.append(gom_prior_1d)
            gom_posteriors.append(gom_post_1d)

            short_label = config["title"].split(" (")[0]
            plot_labels.append(short_label)

            # Plot Posterior Map
            post_exp = posterior.expectation * ewt_scale
            sl.plot(
                post_exp,
                ax=axes_maps[panel_idx],
                colorbar=True,
                vmin=-vmax,
                vmax=vmax,
                cmap="seismic",
                colorbar_kwargs={"label": "Load (mm EWT)"},
            )
            axes_maps[panel_idx].set_title(config["title"], fontsize=14)
            utils.draw_region_boundaries(state, axes_maps[panel_idx], regions_dict)

        # 3. Plot the PDFs
        print("  Generating Gulf of Mexico PDFs...")

        # PDF Panel 0: All 5 Tested Priors
        # We pass them as the first argument so they draw on the primary axis cleanly.
        inf.plot_1d_distributions(
            gom_priors,
            true_value=true_gom_val,
            ax=axes_pdfs[0],
            title="Tested Priors (Gulf of Mexico)",
            posterior_labels=[f"Prior: {label}" for label in plot_labels],
            xlabel="Mass Anomaly (mm EWT)",
            fill_density=False,
        )

        # PDF Panels 1-5: Individual Posteriors
        for i in range(5):
            panel_idx = i + 1
            inf.plot_1d_distributions(
                [gom_posteriors[i]],
                true_value=true_gom_val,
                ax=axes_pdfs[panel_idx],
                title=f"Posterior: {plot_labels[i]}",
                posterior_labels=["Posterior PDF"],
                xlabel="Mass Anomaly (mm EWT)",
                fill_density=False,
            )

        # Save both Grid figures
        map_filepath = os.path.join(
            output_dir, f"prior_sensitivity_maps_{sweep_name}.png"
        )
        fig_maps.savefig(map_filepath, dpi=300, bbox_inches="tight")
        print(f"Saved Maps: {map_filepath}")

        pdf_filepath = os.path.join(
            output_dir, f"prior_sensitivity_pdfs_{sweep_name}.png"
        )
        fig_pdfs.savefig(pdf_filepath, dpi=300, bbox_inches="tight")
        print(f"Saved PDFs: {pdf_filepath}")

        plt.close(fig_maps)
        plt.close(fig_pdfs)

    print("\nAll sensitivity sweeps completed successfully.")


if __name__ == "__main__":
    main()
