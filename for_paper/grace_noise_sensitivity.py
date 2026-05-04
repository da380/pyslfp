"""
Noise Sensitivity Analysis for Bayesian GRACE Inversion
=====================================================

This script systematically tests the sensitivity of the Bayesian GRACE inversion
to misspecifications in the assumed instrument/data noise model.
It performs two distinct parameter sweeps: Assumed Noise Level and Assumed Noise Scale.

The "True" Prior and "True" Data are held strictly constant across all tests.
Only the noise covariance matrix passed to the inverse problem is altered.

For each sweep, two figures are generated:
  1. Maps (3x2 grid): True Load + 5 Posterior Expectation maps.
  2. PDFs (3x2 grid): True Prior + 5 Individual Posterior PDFs
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
    output_dir = "output_plots_grace_noise_sensitivity"
    os.makedirs(output_dir, exist_ok=True)
    inf.configure_threading(n_threads=1)

    lmax = 128
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
    invariant_prior, true_prior, true_noise, _ = utils.build_measures(
        state,
        load_space,
        base_scale,
        base_std,
        base_noise_scale_fac,
        base_noise_std_fac,
    )

    true_data_error_measure = wmb_method.load_measure_to_observation_measure(true_noise)
    true_forward_problem = inf.LinearForwardProblem(
        exact_forward_op, data_error_measure=true_data_error_measure
    )

    # Draw the single true model and resulting data to be used across ALL sweeps
    true_model, synthetic_data = true_forward_problem.synthetic_model_and_data(
        true_prior
    )

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

    # Global color scale limits based on the true model
    vmax = np.max(np.abs(true_model.data * ewt_scale)) * 1.2

    # --- 2. DEFINE THE EXPERIMENTAL SWEEPS (5 variations per sweep) ---
    sweeps = {
        "Assumed_Noise_Level": [
            {
                "noise_fac": 0.02,
                "noise_scale_fac": base_noise_scale_fac,
                "title": "Assumed Low Noise (x0.02)",
            },
            {
                "noise_fac": 0.05,
                "noise_scale_fac": base_noise_scale_fac,
                "title": "Assumed Med-Low (x0.05)",
            },
            {
                "noise_fac": 0.1,
                "noise_scale_fac": base_noise_scale_fac,
                "title": "Baseline Noise (x0.1)",
            },
            {
                "noise_fac": 0.2,
                "noise_scale_fac": base_noise_scale_fac,
                "title": "Assumed High Noise (x0.2)",
            },
            {
                "noise_fac": 0.5,
                "noise_scale_fac": base_noise_scale_fac,
                "title": "Assumed V. High (x0.5)",
            },
        ],
        "Assumed_Noise_Scale": [
            {
                "noise_fac": base_noise_std_fac,
                "noise_scale_fac": 0.1,
                "title": "Short Noise (x0.1)",
            },
            {
                "noise_fac": base_noise_std_fac,
                "noise_scale_fac": 0.25,
                "title": "Baseline Noise (x0.25)",
            },
            {
                "noise_fac": base_noise_std_fac,
                "noise_scale_fac": 0.5,
                "title": "Med Noise (x0.5)",
            },
            {
                "noise_fac": base_noise_std_fac,
                "noise_scale_fac": 0.75,
                "title": "Long Noise (x0.75)",
            },
            {
                "noise_fac": base_noise_std_fac,
                "noise_scale_fac": 1.0,
                "title": "Matched Scale (x1.0)",
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
        gom_posteriors = []
        plot_labels = []

        # Map the Fixed True Prior to the 1D Gulf of Mexico space
        gom_prior_1d = true_prior.affine_mapping(
            operator=gom_op
        ).with_dense_covariance()

        # 2. Iterate through 5 configs
        for i, config in enumerate(configs):
            panel_idx = i + 1
            print(f"  Solving config {panel_idx}/5: {config['title']}...")

            # Build ONLY the perturbed noise measure. The prior remains strictly TRUE.
            _, _, assumed_noise, _ = utils.build_measures(
                state,
                load_space,
                base_scale,
                base_std,
                config["noise_scale_fac"],
                config["noise_fac"],
            )

            # Setup the misspecified inversion
            assumed_data_err_measure = wmb_method.load_measure_to_observation_measure(
                assumed_noise
            )
            assumed_forward_prob = inf.LinearForwardProblem(
                exact_forward_op, data_error_measure=assumed_data_err_measure
            )
            preconditioner = wmb_method.bayesian_normal_operator_preconditioner(
                invariant_prior, assumed_data_err_measure
            )

            # Invert using the TRUE prior, the ASSUMED noise, and the TRUE synthetic data
            test_inverse = inf.LinearBayesianInversion(assumed_forward_prob, true_prior)
            posterior = test_inverse.model_posterior_measure(
                synthetic_data, solver, preconditioner=preconditioner
            )

            # Extract 1D Posterior
            gom_post_1d = posterior.affine_mapping(
                operator=gom_op
            ).with_dense_covariance()
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

        # PDF Panel 0: The fixed True Prior
        inf.plot_1d_distributions(
            [gom_prior_1d],
            true_value=true_gom_val,
            ax=axes_pdfs[0],
            title="True Prior (Gulf of Mexico)",
            posterior_labels=["True Prior PDF"],
            xlabel="Mass Anomaly (mm EWT)",
            fill_density=True,
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
            output_dir, f"noise_sensitivity_maps_{sweep_name}.png"
        )
        fig_maps.savefig(map_filepath, dpi=300, bbox_inches="tight")
        print(f"Saved Maps: {map_filepath}")

        pdf_filepath = os.path.join(
            output_dir, f"noise_sensitivity_pdfs_{sweep_name}.png"
        )
        fig_pdfs.savefig(pdf_filepath, dpi=300, bbox_inches="tight")
        print(f"Saved PDFs: {pdf_filepath}")

        plt.close(fig_maps)
        plt.close(fig_pdfs)

    print("\nAll noise sensitivity sweeps completed successfully.")


if __name__ == "__main__":
    main()
