"""
Bayesian Inversion Testing Suite (Convergence & Sensitivity)
==========================================================

This unified script performs extensive testing on the Bayesian inversion of
GRACE gravimetry data.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
import pyslfp as sl
import grace_utils as utils


def parse_arguments():
    """Parses command-line arguments to toggle testing suites and physics parameters."""
    parser = argparse.ArgumentParser(
        description="Unified testing suite for Bayesian GRACE inversions."
    )
    parser.add_argument(
        "--all", action="store_true", help="Run both convergence and sensitivity tests."
    )
    parser.add_argument(
        "--convergence", action="store_true", help="Run Lmax convergence testing."
    )
    parser.add_argument(
        "--sensitivity", action="store_true", help="Run prior sensitivity testing."
    )

    parser.add_argument(
        "--lmax-truth",
        type=int,
        default=128,
        help="High resolution Lmax for generating the synthetic truth.",
    )
    parser.add_argument(
        "--lmax-test",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="List of Lmax values to test in the convergence inversion.",
    )
    parser.add_argument(
        "--lmax-base",
        type=int,
        default=64,
        help="Fixed Lmax used for the sensitivity tests.",
    )
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=32,
        help="Max SH degree of GRACE observations.",
    )

    parser.add_argument(
        "--load-order",
        type=float,
        default=2.0,
        help="Sobolev space order for the load.",
    )
    parser.add_argument(
        "--load-scale-km",
        type=float,
        default=1000.0,
        help="Length scale (in km) defining the load space.",
    )
    parser.add_argument(
        "--smoothing-scale-km",
        type=float,
        default=None,
        help="Scale (in km) for spatial smoothing applied to regional averages. Defaults to --load-scale-km.",
    )

    parser.add_argument(
        "--base-direct-scale-km",
        type=float,
        default=1000.0,
        help="Baseline correlation length scale (in km) for the prior measure.",
    )
    parser.add_argument(
        "--base-direct-std-m",
        type=float,
        default=0.01,
        help="Baseline pointwise standard deviation (in m EWT) for the prior.",
    )
    parser.add_argument(
        "--noise-scale-factor",
        type=float,
        default=0.25,
        help="Factor scaling the noise correlation length relative to the prior.",
    )
    parser.add_argument(
        "--noise-std-factor",
        type=float,
        default=0.1,
        help="Factor scaling the noise standard deviation relative to the prior.",
    )
    parser.add_argument(
        "--remove-deg-1",
        action="store_true",
        help="Remove degree 1 components from the prior measure.",
    )

    parser.add_argument(
        "--scale-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Multipliers applied to the baseline prior length scale.",
    )
    parser.add_argument(
        "--std-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 5.0],
        help="Multipliers applied to the baseline prior amplitude (std).",
    )
    parser.add_argument(
        "--offset-multipliers",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0],
        help="Multipliers defining the mean offset shift in the prior expectation.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.smoothing_scale_km is None:
        args.smoothing_scale_km = args.load_scale_km

    if args.all:
        args.convergence = args.sensitivity = True
    if not (args.convergence or args.sensitivity):
        print("No tests selected. Run with --convergence, --sensitivity, or --all.")
        return

    print(f"Generating Master Truth at Lmax = {args.lmax_truth}...")
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
        remove_deg_1=args.remove_deg_1,
    )

    wmb_true = sl.WMBMethod.from_finger_print(fp_true, args.obs_degree)
    data_err_meas_true = wmb_true.load_measure_to_observation_measure(noise_true)
    fwd_prob_true = inf.LinearForwardProblem(
        sl.grace_operator(response_space_true, args.obs_degree) @ fp_op_true,
        data_error_measure=data_err_meas_true,
    )

    true_direct_load, master_data = fwd_prob_true.synthetic_model_and_data(prior_true)

    tot_op_true = utils.build_total_load_operator(
        fp_true, response_space_true, load_space_true, fp_op_true
    )
    region_names, avg_op_true, _ = utils.get_regional_averaging(
        fp_true, load_space_true, args.smoothing_scale_km
    )
    true_averages_mm = (avg_op_true @ tot_op_true)(true_direct_load) * scale_mm

    # ------------------ CONVERGENCE TESTS ------------------
    if args.convergence:
        print("\n--- STARTING CONVERGENCE TESTS ---")
        results_conv = {}
        for lmax in sorted(args.lmax_test):
            if lmax < args.obs_degree:
                print(
                    f"[WARNING] Skipping Lmax={lmax}: Must be >= obs_degree ({args.obs_degree})."
                )
                continue

            print(f"  Inverting at Lmax = {lmax}...")
            fp, load_space, response_space, fp_op, _ = utils.build_physics_components(
                lmax, args.load_order, args.load_scale_km
            )
            init_prior, prior, _ = utils.build_measures(
                fp,
                load_space,
                args.base_direct_scale_km,
                args.base_direct_std_m,
                args.noise_scale_factor,
                args.noise_std_factor,
                remove_deg_1=args.remove_deg_1,
            )

            wmb = sl.WMBMethod.from_finger_print(fp, args.obs_degree)
            inv_prob = inf.LinearBayesianInversion(
                inf.LinearForwardProblem(
                    sl.grace_operator(response_space, args.obs_degree) @ fp_op,
                    data_error_measure=data_err_meas_true,
                ),
                prior,
            )
            solver = inf.CGMatrixSolver()
            post = inv_prob.model_posterior_measure(
                master_data,
                solver,
                preconditioner=wmb.bayesian_normal_operator_preconditioner(
                    init_prior, data_err_meas_true
                ),
            )

            tot_avg_op = utils.get_regional_averaging(
                fp, load_space, args.smoothing_scale_km
            )[1] @ utils.build_total_load_operator(
                fp, response_space, load_space, fp_op
            )
            post_avg_meas = post.affine_mapping(operator=tot_avg_op)

            results_conv[f"$L_{{max}}$={lmax}"] = {
                "means": post_avg_meas.expectation * scale_mm,
                "stds": np.sqrt(np.diag(post_avg_meas.covariance.matrix(dense=True)))
                * scale_mm,
            }

        utils.plot_regional_pdfs(
            results_conv,
            r"Posterior Convergence with Increasing $L_{max}$",
            region_names,
            true_averages_mm,
        )

    # ------------------ SENSITIVITY TESTS ------------------
    if args.sensitivity:
        print(
            f"\n--- STARTING SENSITIVITY TESTS (Baseline Lmax = {args.lmax_base}) ---"
        )
        fp, load_space, response_space, fp_op, _ = utils.build_physics_components(
            args.lmax_base, args.load_order, args.load_scale_km
        )
        wmb = sl.WMBMethod.from_finger_print(fp, args.obs_degree)
        fwd_prob = inf.LinearForwardProblem(
            sl.grace_operator(response_space, args.obs_degree) @ fp_op,
            data_error_measure=data_err_meas_true,
        )
        tot_avg_op = utils.get_regional_averaging(
            fp, load_space, args.smoothing_scale_km
        )[1] @ utils.build_total_load_operator(fp, response_space, load_space, fp_op)

        def run_inv(test_cond, test_init):
            solver = inf.CGMatrixSolver()
            post = inf.LinearBayesianInversion(
                fwd_prob, test_cond
            ).model_posterior_measure(
                master_data,
                solver,
                preconditioner=wmb.bayesian_normal_operator_preconditioner(
                    test_init, data_err_meas_true
                ),
            )
            meas = post.affine_mapping(operator=tot_avg_op)
            return {
                "means": meas.expectation * scale_mm,
                "stds": np.sqrt(np.diag(meas.covariance.matrix(dense=True))) * scale_mm,
            }

        base_init, base_cond, _ = utils.build_measures(
            fp,
            load_space,
            args.base_direct_scale_km,
            args.base_direct_std_m,
            args.noise_scale_factor,
            args.noise_std_factor,
            remove_deg_1=args.remove_deg_1,
        )

        print("\n  Running Length Scale Sweep...")
        res_scale = {}
        for m in args.scale_multipliers:
            val = args.base_direct_scale_km * m
            init_p, cond_p, _ = utils.build_measures(
                fp,
                load_space,
                val,
                args.base_direct_std_m,
                args.noise_scale_factor,
                args.noise_std_factor,
                remove_deg_1=args.remove_deg_1,
            )
            res_scale[f"Scale {val}km"] = run_inv(cond_p, init_p)
        utils.plot_regional_pdfs(
            res_scale,
            "Sensitivity to Prior Length Scale",
            region_names,
            true_averages_mm,
        )

        print("\n  Running Amplitude (Std) Sweep...")
        res_std = {}
        for m in args.std_multipliers:
            val = args.base_direct_std_m * m
            init_p, cond_p, _ = utils.build_measures(
                fp,
                load_space,
                args.base_direct_scale_km,
                val,
                args.noise_scale_factor,
                args.noise_std_factor,
                remove_deg_1=args.remove_deg_1,
            )
            res_std[f"Std {val}m"] = run_inv(cond_p, init_p)
        utils.plot_regional_pdfs(
            res_std,
            "Sensitivity to Prior Amplitude (Std)",
            region_names,
            true_averages_mm,
        )

        print("\n  Running Mean Offset Sweep...")
        offset_shape = base_cond.sample()
        res_exp = {}
        for m in args.offset_multipliers:
            res_exp[f"Offset {m}x"] = run_inv(
                base_cond.affine_mapping(translation=offset_shape * m), base_init
            )
        utils.plot_regional_pdfs(
            res_exp,
            "Sensitivity to Prior Expectation (Mean Offset)",
            region_names,
            true_averages_mm,
        )

    plt.show()


if __name__ == "__main__":
    main()
