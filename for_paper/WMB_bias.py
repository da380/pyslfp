"""
WMB Method Bias Evaluation
==========================

This script quantifies the systematic bias introduced by the Wahr, Molenaar, & Bryan (1998)
(WMB) method when estimating regional surface mass changes from satellite gravimetry
(e.g., GRACE / GRACE-FO).

The WMB method isolates surface mass anomalies by applying a purely spectral scaling
(via load Love numbers) to observed gravitational potential coefficients. However, this
approach inherently neglects gravitational self-attraction and loading (SAL) effects—most
notably the induced redistribution of water mass across the global ocean (governed by the
sea-level equation).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import regionmask
import pygeoinf as inf
import pyslfp as sl
import grace_utils as utils


def parse_arguments():
    """Parses command-line arguments to configure the WMB bias evaluation."""
    parser = argparse.ArgumentParser(
        description="Calculate WMB method bias using analytical Gaussian measures."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of Monte Carlo samples to draw for validating analytical error distributions.",
    )
    parser.add_argument(
        "--plot-loads",
        action="store_true",
        help="Plot an example of the direct load and the induced water load with region boxes.",
    )

    parser.add_argument(
        "--lmax",
        type=int,
        default=128,
        help="Maximum spherical harmonic degree for the Earth model.",
    )
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=100,
        help="Maximum spherical harmonic degree of the GRACE observations.",
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
        default=100.0,
        help="Length scale (in km) defining the load space.",
    )
    parser.add_argument(
        "--smoothing-scale-km",
        type=float,
        default=None,
        help="Scale (in km) for spatial smoothing. Defaults to --load-scale-km.",
    )

    parser.add_argument(
        "--direct-scale-km",
        type=float,
        default=100.0,
        help="Correlation length scale (in km) for the prior.",
    )
    parser.add_argument(
        "--direct-std-m",
        type=float,
        default=0.01,
        help="Pointwise standard deviation (in m EWT) for the prior.",
    )
    parser.add_argument(
        "--prior-shift",
        type=float,
        default=0.0,
        help="Shift the prior expectation by drawing a sample and multiplying by this factor.",
    )
    parser.add_argument(
        "--noise-scale-factor",
        type=float,
        default=0.25,
        help="Factor scaling the noise correlation length.",
    )
    parser.add_argument(
        "--noise-std-factor",
        type=float,
        default=0.1,
        help="Factor scaling the noise standard deviation.",
    )
    parser.add_argument(
        "--remove-degree-1",
        action="store_true",
        help="Remove degree 1 components from the prior measure.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.smoothing_scale_km is None:
        args.smoothing_scale_km = args.load_scale_km

    fp, load_space, response_space, fp_op, scale_mm = utils.build_physics_components(
        args.lmax, args.load_order, args.load_scale_km
    )

    _, cond_prior, noise = utils.build_measures(
        fp,
        load_space,
        args.direct_scale_km,
        args.direct_std_m,
        args.noise_scale_factor,
        args.noise_std_factor,
        remove_degree_1=args.remove_degree_1,
    )

    if args.prior_shift != 0.0:
        print(
            f"Applying a {args.prior_shift}x mean shift to the underlying mass distribution..."
        )
        offset_shape = cond_prior.sample()
        cond_prior = cond_prior.affine_mapping(
            translation=offset_shape * args.prior_shift
        )

    wmb = sl.WMBMethod.from_finger_print(fp, args.obs_degree)
    region_names, avg_op, weighting_functions = utils.get_regional_averaging(
        fp, load_space, args.smoothing_scale_km
    )
    wmb_avg_op = avg_op @ wmb.potential_coefficient_to_load_operator(load_space)

    data_err = wmb.load_measure_to_observation_measure(noise)
    data_space = data_err.domain

    sea_level_proj = response_space.subspace_projection(0)
    sle_to_load = sl.sea_level_change_to_load_operator(
        fp, sea_level_proj.codomain, load_space
    )

    # ------------------ OPTION: PLOT EXAMPLE LOADS ------------------
    if args.plot_loads:
        print("Plotting example direct and induced loads...")
        sample_direct = cond_prior.sample()
        sample_induced = (sle_to_load @ sea_level_proj @ fp_op)(sample_direct)

        ax1, im1 = sl.plot(
            sample_direct * scale_mm,
            colorbar_label="EWT (mm)",
            symmetric=True,
        )
        ax1.set_title("Example Direct Load Sample")

        ax2, im2 = sl.plot(
            sample_induced * scale_mm,
            colorbar_label="EWT (mm)",
            symmetric=True,
        )
        ax2.set_title("Resulting Induced Water Load")

    # ------------------ CORE BIAS EVALUATION ------------------
    op1 = inf.BlockLinearOperator(
        [
            [load_space.identity_operator(), data_space.zero_operator(load_space)],
            [fp_op, data_space.zero_operator(response_space)],
            [load_space.zero_operator(data_space), data_space.identity_operator()],
        ]
    )
    op2 = inf.BlockLinearOperator(
        [
            [
                load_space.identity_operator(),
                sle_to_load @ sea_level_proj,
                data_space.zero_operator(load_space),
            ],
            [
                load_space.zero_operator(data_space),
                sl.grace_operator(response_space, args.obs_degree),
                data_space.identity_operator(),
            ],
        ]
    )

    avgs_space = avg_op.codomain
    true_op = (
        inf.RowLinearOperator([avg_op, data_space.zero_operator(avgs_space)])
        @ op2
        @ op1
    )
    err_op = inf.RowLinearOperator([avg_op, -1 * wmb_avg_op]) @ op2 @ op1

    joint_meas = inf.GaussianMeasure.from_direct_sum([cond_prior, data_err])
    true_meas = joint_meas.affine_mapping(operator=true_op)
    err_meas = joint_meas.affine_mapping(operator=err_op)

    true_stds = np.sqrt(np.diag(true_meas.covariance.matrix(dense=True))) * scale_mm
    err_stds = np.sqrt(np.diag(err_meas.covariance.matrix(dense=True))) * scale_mm
    err_means = err_meas.expectation * scale_mm

    wmb_noise_meas = data_err.affine_mapping(operator=wmb_avg_op)
    wmb_stds = np.sqrt(np.diag(wmb_noise_meas.covariance.matrix(dense=True))) * scale_mm

    if args.samples > 0:
        print(f"Drawing {args.samples} MC samples...")
        joint_samples_list = joint_meas.samples(args.samples)
        err_samples = np.zeros((args.samples, len(region_names)))
        for idx, sample in enumerate(joint_samples_list):
            err_samples[idx, :] = err_op(sample) * scale_mm

    print("Plotting Bias PDFs...")

    def gaussian_pdf(x, mean, std):
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

    def make_forward(std_val):
        return lambda x: x / std_val

    def make_inverse(std_val):
        return lambda x: x * std_val

    nrows = int(np.ceil(len(region_names) / 2))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=2, figsize=(14, 5 * nrows), layout="constrained"
    )
    for i, region in enumerate(region_names):

        ax = axes.flatten()[i]
        mu, std = err_means[i], err_stds[i]

        norm_std = true_stds[i]
        norm_label = r"Error Standardized by True Signal $\sigma$"

        if args.samples > 0:
            ax.hist(
                err_samples[:, i],
                bins=50,
                alpha=0.3,
                color="red",
                density=True,
                label="MC Samples",
            )

        plot_min = mu - 4 * std
        plot_max = mu + 4 * std
        x_vals = np.linspace(plot_min, plot_max, 300)

        ax.plot(
            x_vals,
            gaussian_pdf(x_vals, mu, std),
            "r-",
            linewidth=2,
            label=rf"Actual Bias ($\mu$={mu:.3f}, $\sigma$={std:.3f})",
        )

        ax.plot(
            x_vals,
            gaussian_pdf(x_vals, 0, wmb_stds[i]),
            "b-",
            linewidth=2,
            label=rf"Theoretical Bias ($\mu$={0:.3f}, $\sigma$={wmb_stds[i]:.3f})",
        )

        ax.set_title(region, fontsize=14)
        ax.set_xlabel("Error (mm)", fontsize=12)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=9)

        sec_ax = ax.secondary_xaxis(
            "top", functions=(make_forward(norm_std), make_inverse(norm_std))
        )
        sec_ax.set_xlabel(norm_label, fontsize=10, color="darkgreen")
        sec_ax.tick_params(axis="x", colors="darkgreen")

    for j in range(i + 1, len(axes.flatten())):
        axes.flatten()[j].set_visible(False)

    summed_weights = sum(weighting_functions)
    vmax_w = np.max(np.abs(summed_weights.data))

    gs = axes[1, 1].get_subplotspec()
    axes[1, 1].remove()

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=gs,
        width_ratios=[0.1, 0.8, 0.1],
        height_ratios=[0.1, 0.8, 0.1],
    )

    ax_map = fig.add_subplot(inner_gs[1, 1], projection=ccrs.Robinson())

    _, im0 = sl.plot(
        summed_weights,
        colorbar_label="Weight",
        cmap="Reds",
        vmin=0,
        vmax=vmax_w,
        symmetric=False,
        ax=ax_map,
    )

    ar6 = regionmask.defined_regions.ar6.all
    idxs = [ar6.map_keys(r) for r in region_names]
    ar6[idxs].plot(
        ax=ax_map,
        add_label=True,
        label="abbrev",
        line_kws=dict(color="black", linewidth=2.5, linestyle="-"),
        text_kws={
            "color": "black",
            "fontweight": "bold",
            "fontsize": 8,
            "bbox": dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        },
    )

    plt.show()


if __name__ == "__main__":
    main()
