"""
Bayesian Inversion KL Divergence Convergence Testing
====================================================

This script evaluates the convergence of the Bayesian inversion of GRACE
data as a function of the spherical harmonic truncation degree (Lmax)
using the Kullback-Leibler (KL) divergence metric.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
import pyslfp as sl
import grace_utils as utils


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="KL divergence convergence testing for Bayesian GRACE inversions."
    )
    parser.add_argument(
        "--lmax-truth",
        type=int,
        default=128,
        help="High resolution Lmax used to generate the reference 'true' posterior.",
    )
    parser.add_argument(
        "--lmax-test",
        type=int,
        nargs="+",
        default=[32, 48, 64, 128],
        help="List of Lmax values to test in the convergence sweep.",
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
        "--direct-scale-km",
        type=float,
        default=1000.0,
        help="Correlation length scale (in km) for the prior measure.",
    )
    parser.add_argument(
        "--direct-std-m",
        type=float,
        default=0.01,
        help="Pointwise standard deviation (in m EWT) for the prior.",
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
        "--parallel",
        action="store_true",
        help="Compute dense covariance matrices in parallel.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Ensure lmax-truth is at least as large as the maximum test value
    args.lmax_truth = max(args.lmax_truth, max(args.lmax_test))

    print(f"--- Generating Reference Truth at Lmax = {args.lmax_truth} ---")
    fp_true, load_space_true, response_space_true, fp_op_true, scale_mm = (
        utils.build_physics_components(
            args.lmax_truth, args.load_order, args.load_scale_km
        )
    )

    init_prior_true, cond_prior_true, noise_true = utils.build_measures(
        fp_true,
        load_space_true,
        args.direct_scale_km,
        args.direct_std_m,
        args.noise_scale_factor,
        args.noise_std_factor,
    )

    wmb_true = sl.WMBMethod.from_finger_print(fp_true, args.obs_degree)
    data_err_meas_true = wmb_true.load_measure_to_observation_measure(noise_true)
    fwd_prob_true = inf.LinearForwardProblem(
        sl.grace_operator(response_space_true, args.obs_degree) @ fp_op_true,
        data_error_measure=data_err_meas_true,
    )

    true_direct_load, master_data = fwd_prob_true.synthetic_model_and_data(
        cond_prior_true
    )

    # Solve the "True" Inverse Problem
    inv_prob_true = inf.LinearBayesianInversion(fwd_prob_true, cond_prior_true)
    precond_true = wmb_true.bayesian_normal_operator_preconditioner(
        init_prior_true, data_err_meas_true
    )
    solver = inf.CGMatrixSolver()

    print("Solving reference inversion...")
    post_true = inv_prob_true.model_posterior_measure(
        master_data, solver, preconditioner=precond_true
    )
    print(f"Solution done in {solver.iterations} iterations")

    # Extract Regional Averages for the Reference Posterior
    tot_op_true = utils.build_total_load_operator(
        fp_true, response_space_true, load_space_true, fp_op_true
    )
    region_names, avg_op_true, _ = utils.get_regional_averaging(
        fp_true, load_space_true, args.load_scale_km
    )

    tot_avg_op_true = avg_op_true @ tot_op_true

    print("Extracting dense reference posterior (this may take a moment)...")
    post_avg_meas_true = post_true.affine_mapping(
        operator=tot_avg_op_true
    ).with_dense_covariance(parallel=args.parallel)

    true_averages_mm = tot_avg_op_true(true_direct_load) * scale_mm

    # ------------------ CONVERGENCE TESTS ------------------
    print("\n--- STARTING CONVERGENCE SWEEP ---")

    kl_divergences = []
    results_for_plotting = {}

    for lmax in sorted(args.lmax_test):
        if lmax < args.obs_degree:
            print(
                f"[WARNING] Skipping Lmax={lmax}: Must be >= obs_degree ({args.obs_degree})."
            )
            continue

        print(f"  Evaluating Lmax = {lmax}...")

        # 1. Build Physics
        fp, load_space, response_space, fp_op, _ = utils.build_physics_components(
            lmax, args.load_order, args.load_scale_km
        )

        # 2. Build Measures
        init_prior, cond_prior, _ = utils.build_measures(
            fp,
            load_space,
            args.direct_scale_km,
            args.direct_std_m,
            args.noise_scale_factor,
            args.noise_std_factor,
        )

        # 3. Formulate Forward & Inverse Problem using the MASTER DATA and TRUE NOISE MODEL
        wmb = sl.WMBMethod.from_finger_print(fp, args.obs_degree)
        fwd_prob = inf.LinearForwardProblem(
            sl.grace_operator(response_space, args.obs_degree) @ fp_op,
            data_error_measure=data_err_meas_true,
        )

        inv_prob = inf.LinearBayesianInversion(fwd_prob, cond_prior)
        precond = wmb.bayesian_normal_operator_preconditioner(
            init_prior, data_err_meas_true
        )

        # 4. Solve
        post = inv_prob.model_posterior_measure(
            master_data, solver, preconditioner=precond
        )

        # 5. Extract Averages and Cache Dense Covariance
        tot_op = utils.build_total_load_operator(fp, response_space, load_space, fp_op)
        _, avg_op, _ = utils.get_regional_averaging(fp, load_space, args.load_scale_km)

        tot_avg_op = avg_op @ tot_op
        post_avg_meas = post.affine_mapping(operator=tot_avg_op).with_dense_covariance(
            parallel=args.parallel
        )

        # 6. Compute KL Divergence against the Reference Truth
        # Note: We compute D_KL(Test || Truth).
        kl_div = post_avg_meas.kl_divergence(post_avg_meas_true)
        kl_divergences.append(kl_div)
        print(f"    KL Divergence: {kl_div:.4e}")

        # Store for PDF plotting
        results_for_plotting[f"$L_{{max}}$={lmax}"] = {
            "means": post_avg_meas.expectation * scale_mm,
            "stds": np.sqrt(np.diag(post_avg_meas.covariance.matrix(dense=True)))
            * scale_mm,
        }

    # Add the reference truth to the plotting dictionary
    results_for_plotting[f"Truth ($L_{{max}}$={args.lmax_truth})"] = {
        "means": post_avg_meas_true.expectation * scale_mm,
        "stds": np.sqrt(np.diag(post_avg_meas_true.covariance.matrix(dense=True)))
        * scale_mm,
    }

    # ------------------ PLOTTING ------------------
    print("\nGenerating plots...")

    # 1. Plot KL Divergence Convergence
    valid_lmax_test = [l for l in sorted(args.lmax_test) if l >= args.obs_degree]

    plt.figure(figsize=(8, 5))
    plt.semilogy(
        valid_lmax_test,
        kl_divergences,
        "o-",
        color="darkblue",
        linewidth=2,
        markersize=8,
    )
    plt.xlabel(r"Spherical Harmonic Truncation Degree ($L_{max}$)", fontsize=12)
    plt.ylabel(r"KL Divergence $D_{KL}(P_{test} || P_{truth})$", fontsize=12)
    plt.title(
        f"Convergence of Posterior Distributions (Ref $L_{{max}}$={args.lmax_truth})",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, which="both", ls=":", alpha=0.7)
    plt.tight_layout()

    # 2. Plot Regional PDFs
    utils.plot_regional_pdfs(
        results_for_plotting,
        r"Posterior Distributions across $L_{max}$",
        region_names,
        true_averages_mm,
    )

    plt.show()


if __name__ == "__main__":
    main()
