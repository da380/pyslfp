"""
Altimetry Bias Evaluation
=========================
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
import pyslfp as sl

import altimetry_utils as utils

from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points, altimetry_averaging_operator


def parse_arguments():
    """Parses command-line arguments to configure the altimetry bias evaluation."""
    parser = argparse.ArgumentParser(
        description="Calculate Altimetry method bias using analytical Gaussian measures."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of Monte Carlo samples to draw for validating analytical error distributions.",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot an example of the SSH sample and altimetry points.",
    )

    parser.add_argument(
        "--lmax",
        type=int,
        default=128,
        help="Maximum spherical harmonic degree for the Earth model.",
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
        default=500.0,
        help="Length scale (in km) defining the load space.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=4.0,
        help="Spacing in degrees for the altimetry observation points.",
    )
    parser.add_argument(
        "--ice-scale-factor",
        type=float,
        default=1.0,
        help="Relative correlation length scale for the ice thickness prior.",
    )
    parser.add_argument(
        "--ice-std-mm",
        type=float,
        default=10.0,
        help="Pointwise standard deviation (in mm) for the ice thickness prior.",
    )
    parser.add_argument(
        "--ocean-scale-factor",
        type=float,
        default=0.2,
        help="Relative correlation length scale for the ocean dynamic thickness prior.",
    )
    parser.add_argument(
        "--ocean-std-factor",
        type=float,
        default=10.0,
        help="Ocean dynamic thickness noise standard deviation as a factor of the expected GMSL std.",
    )
    parser.add_argument(
        "--noise-std-factor",
        type=float,
        default=2.0,
        help="Instrument noise standard deviation per point as a factor of the expected GMSL std.",
    )
    parser.add_argument(
        "--noise-scale-factor",
        type=float,
        default=00.0,
        help="Relative correlation length scale for the noise field.",
    )
    parser.add_argument(
        "--prior-shift",
        type=float,
        default=1.0,
        help="Shift the prior expectation by drawing a sample and multiplying by this factor.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Initializing Earth State and Fingerprint Operators...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)

    (state, load_space, _, continuous_ssh_op, model_to_ssh_op, scale_mm) = (
        utils.build_physics_components(
            args.lmax, args.load_order, args.load_scale_km, points, is_surrogate=False
        )
    )

    model_prior, noise_meas, _ = utils.build_measures(
        state,
        load_space,
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_scale_factor,
        args.ocean_std_factor,
        args.noise_scale_factor,
        args.noise_std_factor,
        points,
        scale_mm,
        prior_shift=args.prior_shift,
    )

    joint_meas = inf.GaussianMeasure.from_direct_sum([model_prior, noise_meas])
    data_space = noise_meas.domain

    true_gmsl_op = utils.true_gmsl_operator(state, load_space, continuous_ssh_op)
    alt_avg_op = altimetry_averaging_operator(points)
    est_gmsl_op = alt_avg_op @ model_to_ssh_op
    err_gmsl_op = true_gmsl_op - est_gmsl_op

    op_true = inf.RowLinearOperator(
        [true_gmsl_op, data_space.zero_operator(inf.EuclideanSpace(1))]
    )
    op_err = inf.RowLinearOperator([err_gmsl_op, -1.0 * alt_avg_op])

    print("Calculating analytical moments...")
    true_meas = joint_meas.affine_mapping(operator=op_true)
    err_meas = joint_meas.affine_mapping(operator=op_err)
    alt_noise_meas = noise_meas.affine_mapping(operator=alt_avg_op)

    true_std = np.sqrt(true_meas.covariance.matrix(dense=True)[0, 0]) * scale_mm
    err_std = np.sqrt(err_meas.covariance.matrix(dense=True)[0, 0]) * scale_mm
    err_mean = err_meas.expectation[0] * scale_mm
    alt_noise_std = (
        np.sqrt(alt_noise_meas.covariance.matrix(dense=True)[0, 0]) * scale_mm
    )

    # -- Plotting --
    if args.plot_maps:
        model_sample = model_prior.sample()
        ssh_sample = continuous_ssh_op(model_sample)

        ice_thickness, ocean_thickness = model_sample
        ocean_mask = scale_mm * state.ocean_projection(value=0.0)
        ice_mask = scale_mm * state.ice_projection(value=0.0)

        _, ax1 = sl.create_map_figure(figsize=(12, 6))
        sl.plot(
            ice_thickness * ice_mask,
            ax=ax1,
            colorbar_kwargs={"label": "Ice Thickness (mm)"},
            symmetric=True,
        )

        _, ax2 = sl.create_map_figure(figsize=(12, 6))
        sl.plot(
            ocean_thickness * ocean_mask,
            ax=ax2,
            colorbar_kwargs={"label": "Ocean Dynamic (mm)"},
            symmetric=True,
        )

        ssh_grid_mm = ssh_sample * ocean_mask
        observed_data_mm = (
            model_to_ssh_op(model_sample) + noise_meas.sample()
        ) * scale_mm
        shared_vmax = max(
            np.max(np.abs(ssh_grid_mm.data)), np.max(np.abs(observed_data_mm))
        )

        _, ax3 = sl.create_map_figure(figsize=(12, 6))
        sl.plot(
            ssh_grid_mm,
            ax=ax3,
            cmap="RdBu",
            vmin=-shared_vmax,
            vmax=shared_vmax,
            colorbar_kwargs={"label": "SSH (mm)"},
        )

        _, ax4 = sl.create_map_figure(figsize=(12, 6))
        ax4.set_global()
        sl.plot_points(
            points,
            data=observed_data_mm,
            ax=ax4,
            cmap="RdBu",
            vmin=-shared_vmax,
            vmax=shared_vmax,
            s=10,
            edgecolors="none",
            colorbar=True,
            colorbar_kwargs={"label": "SSH (mm)"},
            zorder=5,
        )

    err_samples = None
    if args.samples > 0:
        joint_samples_list = joint_meas.samples(args.samples)
        err_samples = np.array(
            [op_err(sample)[0] * scale_mm for sample in joint_samples_list]
        )

    print("Plotting Bias PDFs...")

    def gaussian_pdf(x, mean, std):
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

    _, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    x_vals = np.linspace(err_mean - 4 * err_std, err_mean + 4 * err_std, 300)

    if err_samples is not None:
        ax.hist(
            err_samples,
            bins=50,
            alpha=0.3,
            color="red",
            density=True,
            label="MC Samples",
        )

    ax.plot(
        x_vals,
        gaussian_pdf(x_vals, err_mean, err_std),
        "r-",
        linewidth=2.5,
        label=rf"Actual error ($\mu$={err_mean:.3f}, $\sigma$={err_std:.3f})",
    )
    ax.plot(
        x_vals,
        gaussian_pdf(x_vals, 0, alt_noise_std),
        "b",
        linewidth=2,
        label=rf"Theoretical error ($\mu$=0.000, $\sigma$={alt_noise_std:.3f})",
    )

    ax.set_xlabel("Error in GMSL estimate (mm)", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=11)

    sec_ax = ax.secondary_xaxis(
        "top", functions=(lambda x: x / true_std, lambda x: x * true_std)
    )
    sec_ax.set_xlabel(
        r"Error Standardized by True Signal $\sigma$", fontsize=12, color="darkgreen"
    )
    sec_ax.tick_params(axis="x", colors="darkgreen")

    if any([args.plot_maps, args.samples > 0, True]):
        plt.show()


if __name__ == "__main__":
    main()
