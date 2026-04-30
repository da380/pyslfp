"""
Bayesian GRACE Inversion
========================

This script performs a Bayesian inversion of synthetic GRACE gravimetry data
to estimate the causative surface mass load. It highlights the differences
between the exact elastic Sea Level Equation response and the purely spectral
WMB approximation, particularly focusing on how these differences impact
inversion results and preconditioning.
"""

import argparse
import os
import numpy as np
import matplotlib

# Force headless backend to avoid Wayland/Qt display errors

import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import pygeoinf as inf
import pyslfp as sl


import grace_utils as utils

matplotlib.use("Agg")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of GRACE data for total mass load."
    )
    # --- Output Options ---
    parser.add_argument(
        "--all", action="store_true", help="Enable all plotting options."
    )
    parser.add_argument("--plot-maps", action="store_true", help="Plot spatial maps.")
    parser.add_argument(
        "--plot-regions", action="store_true", help="Plot regional PDFs."
    )

    # --- Resolution & Physics Settings ---
    parser.add_argument(
        "--lmax", type=int, default=128, help="Exact model max SH degree."
    )
    parser.add_argument(
        "--surrogate-degree", type=int, default=32, help="Preconditioner max SH degree."
    )
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=100,
        help="Max SH degree of GRACE observations.",
    )
    parser.add_argument(
        "--load-order", type=float, default=2.0, help="Sobolev space order."
    )
    parser.add_argument(
        "--load-scale-km", type=float, default=500.0, help="Sobolev length scale."
    )

    # --- Prior Settings ---
    parser.add_argument(
        "--direct-scale-km",
        type=float,
        default=250.0,
        help="Prior correlation scale (km).",
    )
    parser.add_argument(
        "--direct-std-m", type=float, default=0.01, help="Prior std dev (m EWT)."
    )
    parser.add_argument(
        "--prior-shift", type=float, default=1.0, help="Prior mean shift factor."
    )
    parser.add_argument(
        "--remove-degree-1", action="store_true", help="Remove degree 1 from prior."
    )

    # --- Noise Settings ---
    parser.add_argument(
        "--noise-scale-factor",
        type=float,
        default=0.25,
        help="Noise correlation scale factor.",
    )
    parser.add_argument(
        "--noise-std-factor", type=float, default=0.1, help="Noise std factor."
    )

    # --- Preconditioner Options ---
    parser.add_argument(
        "--no-precond", action="store_true", help="Disable the preconditioner entirely."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.all:
        args.plot_maps = args.plot_regions = True

    output_dir = "output_plots_grace"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

    inf.configure_threading(n_threads=1)

    # ------------------ 1. EXACT MODEL SETUP ------------------
    print(f"\nBuilding EXACT physical operators (lmax={args.lmax})...")
    state, load_space, response_space, fp_op, ewt_scale = (
        utils.build_physics_components(args.lmax, args.load_order, args.load_scale_km)
    )

    initial_prior, model_prior, noise_spatial = utils.build_measures(
        state,
        load_space,
        args.direct_scale_km,
        args.direct_std_m,
        args.noise_scale_factor,
        args.noise_std_factor,
        remove_degree_1=args.remove_degree_1,
        prior_shift=args.prior_shift,
    )

    # Construct the exact forward observation operator
    total_load_op = utils.build_total_load_operator(
        state, response_space, load_space, fp_op
    )
    grace_obs_op = sl.linear_operators.grace_observation_operator(
        response_space, args.obs_degree
    )
    exact_forward_op = grace_obs_op @ fp_op @ total_load_op

    # Build the exact data noise measure using WMB spectral mapping
    wmb_method = sl.linear_operators.WMBMethod(state.model, args.obs_degree)
    data_error_measure = wmb_method.load_measure_to_observation_measure(noise_spatial)

    print("\nDrawing synthetic model and dataset...")
    forward_problem = inf.LinearForwardProblem(
        exact_forward_op, data_error_measure=data_error_measure
    )
    true_model, synthetic_data = forward_problem.synthetic_model_and_data(model_prior)
    inverse_problem = inf.LinearBayesianInversion(forward_problem, model_prior)

    # ------------------ 2. PRECONDITIONER SETUP ------------------
    preconditioner = None
    if not args.no_precond:
        print(
            f"\nBuilding SURROGATE operators (lmax={args.surrogate_degree}) for WMB preconditioning..."
        )

        preconditioner = wmb_method.bayesian_normal_operator_preconditioner(
            initial_prior, data_error_measure
        )

    # ------------------ 3. POSTERIOR SOLVE ------------------
    print("\nSolving for posterior expectation...")
    callback = inf.ProgressCallback()
    solver = inf.CGSolver(callback=callback, rtol=1e-4)

    model_posterior = inverse_problem.model_posterior_measure(
        synthetic_data, solver, preconditioner=preconditioner
    )
    print(f"\nSolution reached in {solver.iterations} iterations.")

    # ------------------ 4. MAPPING ------------------
    if args.plot_maps:
        print("Generating spatial maps...")
        cmap = "seismic"

        true_load = true_model
        post_load = model_posterior.expectation

        # Calculate the WMB direct data inversion (Spectral Only, No SLE)
        wmb_inv_op = wmb_method.potential_coefficient_to_load_operator(load_space)
        wmb_estimate = wmb_inv_op(synthetic_data)

        vmax = max(
            np.max(np.abs(true_load.data * ewt_scale)),
            np.max(np.abs(post_load.data * ewt_scale)),
            np.max(np.abs(wmb_estimate.data * ewt_scale)),
        )

        fig_maps, axes = plt.subplots(
            1,
            3,
            figsize=(18, 5),
            subplot_kw={"projection": ccrs.Robinson()},
            layout="constrained",
        )

        sl.plot(
            true_load * ewt_scale,
            ax=axes[0],
            colorbar=True,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            colorbar_kwargs={"label": "Load (mm EWT)"},
        )
        axes[0].set_title("True Mass Load")

        sl.plot(
            wmb_estimate * ewt_scale,
            ax=axes[1],
            colorbar=True,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            colorbar_kwargs={"label": "Load (mm EWT)"},
        )
        axes[1].set_title("WMB Spectral Estimate\n(No SLE / Mass Conserv.)")

        sl.plot(
            post_load * ewt_scale,
            ax=axes[2],
            colorbar=True,
            vmin=-vmax,
            vmax=vmax,
            cmap=cmap,
            colorbar_kwargs={"label": "Load (mm EWT)"},
        )
        axes[2].set_title("Exact Bayesian Posterior\n(Full SLE Constraints)")

        figures_to_save["grace_posterior_maps"] = fig_maps

    # ------------------ 5. REGIONAL ANALYSIS ------------------
    if args.plot_regions:
        print("\nDecomposing Regional Signals...")
        region_names, avg_op, _, _ = utils.get_regional_averaging(
            state, load_space, smoothing_scale_km=args.load_scale_km
        )

        # Extract true regional averages
        true_avgs_mm = avg_op(true_load) * ewt_scale

        # WMB Estimator Distribution (Data Space -> Regional Average)
        wmb_avg_op = avg_op @ wmb_method.potential_coefficient_to_load_operator(
            load_space
        )
        # Because we need the full distribution of the WMB estimator, we map the Master Joint Measure
        # The joint measure is a direct sum of [Prior, Noise]. We map it using: [WMB @ Forward, WMB]
        op_wmb_err = inf.RowLinearOperator([wmb_avg_op @ exact_forward_op, wmb_avg_op])

        # We need the joint measure defined over [model_space, data_space]
        joint_prior = inverse_problem.joint_prior_measure
        wmb_meas = joint_prior.affine_mapping(operator=op_wmb_err)

        # Exact Bayesian Estimator Distribution (Model Posterior -> Regional Average)
        post_meas = model_posterior.affine_mapping(operator=avg_op)

        results_dict = {
            "WMB Spectral": {
                "means": wmb_meas.expectation * ewt_scale,
                "stds": np.sqrt(np.diag(wmb_meas.covariance.matrix(dense=True)))
                * ewt_scale,
            },
            "Exact Bayesian": {
                "means": post_meas.expectation * ewt_scale,
                "stds": np.sqrt(np.diag(post_meas.covariance.matrix(dense=True)))
                * ewt_scale,
            },
        }

        utils.plot_regional_pdfs(results_dict, region_names, true_avgs_mm)
        figures_to_save["grace_regional_pdfs"] = plt.gcf()

    # ------------------ SAVE ALL FIGURES ------------------
    if figures_to_save:
        print(f"\nSaving {len(figures_to_save)} plots to '{output_dir}/'...")
        for name, fig in figures_to_save.items():
            filepath = os.path.join(output_dir, f"{name}.png")
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {filepath}")
            plt.close(fig)


if __name__ == "__main__":
    main()
