"""
Bayesian Inversion Testing Suite (Convergence & Sensitivity)
==========================================================

This unified script performs extensive testing on the Bayesian inversion of
GRACE gravimetry data.

Available Modes:
1. Convergence (--convergence): Tests the effect of increasing the Earth
   model's spherical harmonic truncation degree (Lmax) on the posterior.
2. Sensitivity (--sensitivity): Tests the effect of changing the prior's
   length scale, amplitude, and mean expectation against fixed data.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

import pygeoinf as inf
import pyslfp as sl
import grace_utils as utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Unified testing suite for Bayesian GRACE inversions."
    )
    # --- Run Modes ---
    parser.add_argument(
        "--all", action="store_true", help="Run both convergence and sensitivity tests."
    )
    parser.add_argument(
        "--convergence", action="store_true", help="Run Lmax convergence testing."
    )
    parser.add_argument(
        "--sensitivity", action="store_true", help="Run prior sensitivity testing."
    )

    # --- Resolution Parameters (Lowered for rapid testing) ---
    parser.add_argument(
        "--lmax-truth",
        type=int,
        default=256,
        help="Lmax for generating the synthetic truth.",
    )
    parser.add_argument(
        "--lmax-test",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Lmax values for convergence testing.",
    )
    parser.add_argument(
        "--lmax-base",
        type=int,
        default=128,
        help="Fixed Lmax used for the sensitivity tests.",
    )
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=10,
        help="Max SH degree of GRACE observations.",
    )

    # --- Baseline Physics & Prior Parameters (Scaled up for low-degree resolution) ---
    parser.add_argument("--load-order", type=float, default=2.0)
    parser.add_argument("--load-scale-km", type=float, default=1000.0)
    parser.add_argument("--base-direct-scale-km", type=float, default=1000.0)
    parser.add_argument("--base-direct-std-m", type=float, default=0.01)
    parser.add_argument("--noise-scale-factor", type=float, default=0.25)
    parser.add_argument("--noise-std-factor", type=float, default=0.1)

    return parser.parse_args()


def plot_pdfs(results_dict, title, region_names, true_averages_mm):
    """Helper function to plot a 2x2 grid of regional PDFs."""

    def gaussian_pdf(x, mean, std):
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

    ncols = 2
    nrows = int(np.ceil(len(region_names) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), layout="constrained"
    )
    axes_flat = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_dict)))

    for i, region in enumerate(region_names):
        ax = axes_flat[i]
        true_val = true_averages_mm[i]

        all_means = [res["means"][i] for res in results_dict.values()] + [true_val]
        max_std = max([res["stds"][i] for res in results_dict.values()])
        plot_min = min(all_means) - 3.5 * max_std
        plot_max = max(all_means) + 3.5 * max_std
        x_vals = np.linspace(plot_min, plot_max, 400)

        for c_idx, (label, res) in enumerate(results_dict.items()):
            mean = res["means"][i]
            std = res["stds"][i]
            y_vals = gaussian_pdf(x_vals, mean, std)
            ax.plot(
                x_vals,
                y_vals,
                color=colors[c_idx],
                linewidth=2.5,
                label=rf"{label} ($\mu$={mean:.2f}, $\sigma$={std:.3f})",
            )
            ax.fill_between(x_vals, 0, y_vals, color=colors[c_idx], alpha=0.1)

        ax.axvline(
            true_val,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"True Value: {true_val:.2f}",
        )
        ax.set_title(f"{region}", fontsize=14)
        ax.set_xlabel("Regional Average Mass (mm EWT)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=18, fontweight="bold")


def main():
    args = parse_arguments()

    if args.all:
        args.convergence = True
        args.sensitivity = True

    if not (args.convergence or args.sensitivity):
        print("No tests selected. Run with --convergence, --sensitivity, or --all.")
        return

    # =========================================================================
    # 1. Generate Master High-Resolution "True" Data (Used for all tests)
    # =========================================================================
    print(f"Generating Master Synthetic Truth at Lmax = {args.lmax_truth}...")
    fp_true, load_space_true, response_space_true, fp_op_true, scale_mm = (
        utils.build_physics_components(
            args.lmax_truth, args.load_order, args.load_scale_km
        )
    )

    _, prior_true, noise_true = utils.build_measures(
        fp_true,
        load_space_true,
        args.base_direct_scale_km,
        args.base_direct_std_m,
        args.noise_scale_factor,
        args.noise_std_factor,
    )

    wmb_true = sl.WMBMethod.from_finger_print(fp_true, args.obs_degree)
    data_error_measure_true = wmb_true.load_measure_to_observation_measure(noise_true)

    grace_op_true = sl.grace_operator(response_space_true, args.obs_degree)
    forward_op_true = grace_op_true @ fp_op_true

    forward_problem_true = inf.LinearForwardProblem(
        forward_op_true, data_error_measure=data_error_measure_true
    )

    true_direct_load, master_synthetic_data = (
        forward_problem_true.synthetic_model_and_data(prior_true)
    )

    tot_op_true = utils.build_total_load_operator(
        fp_true, response_space_true, load_space_true, fp_op_true
    )
    region_names, avg_op_true = utils.get_regional_averaging(fp_true, load_space_true)
    true_averages_mm = (avg_op_true @ tot_op_true)(true_direct_load) * scale_mm

    # =========================================================================
    # 2. CONVERGENCE TESTS
    # =========================================================================
    if args.convergence:
        print("\n" + "=" * 50)
        print("STARTING CONVERGENCE TESTS")
        print("=" * 50)
        results_conv = {}

        for lmax in sorted(args.lmax_test):
            if lmax < args.obs_degree:
                print(
                    f"[WARNING] Skipping Lmax={lmax}: Must be >= obs_degree ({args.obs_degree})."
                )
                continue

            print(f"\n--- Running Inversion at Lmax = {lmax} ---")
            fp, load_space, response_space, fp_op, _ = utils.build_physics_components(
                lmax, args.load_order, args.load_scale_km
            )

            initial_prior, prior, noise = utils.build_measures(
                fp,
                load_space,
                args.base_direct_scale_km,
                args.base_direct_std_m,
                args.noise_scale_factor,
                args.noise_std_factor,
            )

            wmb = sl.WMBMethod.from_finger_print(fp, args.obs_degree)
            forward_op = sl.grace_operator(response_space, args.obs_degree) @ fp_op

            forward_problem = inf.LinearForwardProblem(
                forward_op, data_error_measure=data_error_measure_true
            )
            inverse_problem = inf.LinearBayesianInversion(forward_problem, prior)
            preconditioner = wmb.bayesian_normal_operator_preconditioner(
                initial_prior, data_error_measure_true
            )
            solver = inf.CGMatrixSolver()

            load_posterior = inverse_problem.model_posterior_measure(
                master_synthetic_data, solver, preconditioner=preconditioner
            )
            print(f"Solution in {solver.iterations} iterations")

            tot_avg_op = utils.get_regional_averaging(fp, load_space)[
                1
            ] @ utils.build_total_load_operator(fp, response_space, load_space, fp_op)
            post_avg_measure = load_posterior.affine_mapping(operator=tot_avg_op)

            results_conv[f"$L_{{max}}$={lmax}"] = {
                "means": post_avg_measure.expectation * scale_mm,
                "stds": np.sqrt(np.diag(post_avg_measure.covariance.matrix(dense=True)))
                * scale_mm,
            }

        if results_conv:
            plot_pdfs(
                results_conv,
                r"Bayesian Posterior Convergence with Increasing $L_{max}$",
                region_names,
                true_averages_mm,
            )

    # =========================================================================
    # 3. SENSITIVITY TESTS
    # =========================================================================
    if args.sensitivity:
        print("\n" + "=" * 50)
        print(f"STARTING SENSITIVITY TESTS (Using Baseline Lmax = {args.lmax_base})")
        print("=" * 50)

        if args.lmax_base < args.obs_degree:
            raise ValueError(
                f"lmax-base ({args.lmax_base}) must be >= obs_degree ({args.obs_degree})"
            )

        fp, load_space, response_space, fp_op, _ = utils.build_physics_components(
            args.lmax_base, args.load_order, args.load_scale_km
        )
        wmb = sl.WMBMethod.from_finger_print(fp, args.obs_degree)
        forward_op = sl.grace_operator(response_space, args.obs_degree) @ fp_op
        forward_problem = inf.LinearForwardProblem(
            forward_op, data_error_measure=data_error_measure_true
        )
        tot_avg_op = utils.get_regional_averaging(fp, load_space)[
            1
        ] @ utils.build_total_load_operator(fp, response_space, load_space, fp_op)

        def run_inversion(test_cond_prior, test_init_prior):
            inv_prob = inf.LinearBayesianInversion(forward_problem, test_cond_prior)
            precond = wmb.bayesian_normal_operator_preconditioner(
                test_init_prior, data_error_measure_true
            )
            solver = inf.CGMatrixSolver()
            post_measure = inv_prob.model_posterior_measure(
                master_synthetic_data, solver, preconditioner=precond
            )
            post_avg_measure = post_measure.affine_mapping(operator=tot_avg_op)
            return {
                "means": post_avg_measure.expectation * scale_mm,
                "stds": np.sqrt(np.diag(post_avg_measure.covariance.matrix(dense=True)))
                * scale_mm,
            }

        # --- A. Sensitivity to Length Scale ---
        print("\n--- Sensitivity: Prior Length Scale ---")
        # Scaled up testing bounds for low-degree tests
        scales_to_test = [500.0, 1000.0, 2000.0]
        results_scale = {}
        for s in scales_to_test:
            print(f"  Inverting with scale = {s} km...")
            init_p, cond_p, _ = utils.build_measures(
                fp,
                load_space,
                s,
                args.base_direct_std_m,
                args.noise_scale_factor,
                args.noise_std_factor,
            )
            results_scale[f"Scale {s}km"] = run_inversion(cond_p, init_p)
        plot_pdfs(
            results_scale,
            "Sensitivity to Prior Length Scale",
            region_names,
            true_averages_mm,
        )

        # --- B. Sensitivity to Amplitude (Std) ---
        print("\n--- Sensitivity: Prior Amplitude ---")
        stds_to_test = [0.005, 0.01, 0.05]
        results_std = {}
        for std in stds_to_test:
            print(f"  Inverting with std = {std} m...")
            init_p, cond_p, _ = utils.build_measures(
                fp,
                load_space,
                args.base_direct_scale_km,
                std,
                args.noise_scale_factor,
                args.noise_std_factor,
            )
            results_std[f"Std {std}m"] = run_inversion(cond_p, init_p)
        plot_pdfs(
            results_std,
            "Sensitivity to Prior Amplitude (Std)",
            region_names,
            true_averages_mm,
        )

        # --- C. Sensitivity to Expectation Offset ---
        print("\n--- Sensitivity: Prior Expectation (Mean Offset) ---")
        base_init_p, base_cond_p, _ = utils.build_measures(
            fp,
            load_space,
            args.base_direct_scale_km,
            args.base_direct_std_m,
            args.noise_scale_factor,
            args.noise_std_factor,
        )
        offset_shape = base_cond_p.sample()  # Draw a random realistic physical offset
        multipliers = [0.0, 1.0, 2.0]
        results_exp = {}
        for mult in multipliers:
            print(f"  Inverting with offset multiplier = {mult}x...")
            shifted_prior = base_cond_p.affine_mapping(translation=offset_shape * mult)
            results_exp[f"Offset {mult}x"] = run_inversion(shifted_prior, base_init_p)
        plot_pdfs(
            results_exp,
            "Sensitivity to Prior Expectation (Mean Offset)",
            region_names,
            true_averages_mm,
        )

    # =========================================================================
    # Show All Rendered Plots
    # =========================================================================
    print("\nRendering plots...")
    plt.show()


if __name__ == "__main__":
    main()
