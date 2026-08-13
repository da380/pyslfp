"""
Extended Altimetry Bias Evaluation (3-Component Model)
======================================================

This script calculates the analytical method bias and error distribution for
satellite altimetry using a 3-component physical model (Ice Thickness,
Sterodynamic SSH, and Ocean Density). It strictly evaluates the difference
between true Global Mean Sea Level (water column thickness change) and the
standard SSH-based area-averaging estimator.
"""

import argparse
import os
import numpy as np
import matplotlib

# Force headless backend to avoid Wayland/Qt display errors
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pygeoinf as inf
import pyslfp as sl
import altimetry_utils as utils

from pyslfp.state import EarthState
from pyslfp.linear_operators import ocean_altimetry_points, altimetry_averaging_operator


def parse_arguments():
    """Parses command-line arguments to configure the extended altimetry bias evaluation."""
    parser = argparse.ArgumentParser(
        description="Calculate 3-Component Altimetry method bias using analytical Gaussian measures."
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
        help="Plot an example of the sampled fields, SSH, and altimetry points.",
    )

    # --- Resolution Settings ---
    parser.add_argument(
        "--lmax", type=int, default=256, help="Exact model max SH degree."
    )
    parser.add_argument(
        "--load-order", type=float, default=2.0, help="Sobolev space order."
    )
    parser.add_argument(
        "--load-scale-km", type=float, default=500.0, help="Sobolev length scale."
    )
    parser.add_argument(
        "--spacing", type=float, default=1.0, help="Altimetry observation spacing."
    )

    # --- Prior Settings ---
    parser.add_argument(
        "--ice-scale-factor", type=float, default=1.0, help="Ice correlation scale."
    )
    parser.add_argument(
        "--ice-std-mm", type=float, default=10.0, help="Ice std dev (mm)."
    )

    parser.add_argument(
        "--ocean-dyn-scale-factor",
        type=float,
        default=0.2,
        help="Ocean dynamic correlation scale.",
    )
    parser.add_argument(
        "--ocean-dyn-std-factor",
        type=float,
        default=2.0,
        help="Ocean dynamic std as factor of GMSLR std.",
    )

    parser.add_argument(
        "--ocean-rho-scale-factor",
        type=float,
        default=1.0,
        help="Ocean density correlation scale.",
    )
    parser.add_argument(
        "--ocean-rho-std-factor",
        type=float,
        default=0.25,
        help="Effective steric SL std as a fraction of the Sterodynamic SL change std.",
    )

    parser.add_argument(
        "--noise-std-factor",
        type=float,
        default=1.0,
        help="Local (uncorrelated) altimetry noise std as factor of GMSLR std.",
    )
    parser.add_argument(
        "--noise-corr-std-factor",
        type=float,
        default=0.05,
        help=(
            "Std of the optional large-scale correlated altimetry error "
            "component, as a factor of the GMSLR prior std (0 disables). "
            "Represents long-wavelength systematics such as orbit and "
            "reference-frame errors; because it barely averages down, even "
            "small values (e.g. 0.02-0.05) set the error floor for "
            "large-scale averages."
        ),
    )
    parser.add_argument(
        "--noise-corr-scale-factor",
        type=float,
        default=4.0,
        help=(
            "Correlation scale of the correlated altimetry error "
            "component, as a factor of the load scale (default 4.0 = "
            "2000 km at the default load scale). Follows --prior-kernel."
        ),
    )
    parser.add_argument(
        "--ocean-corr",
        type=float,
        default=0.9,
        help=(
            "Magnitude of the long-wavelength anti-correlation between the "
            "ocean dynamic and density fields (0 disables; must lie in "
            "[0, 1); the sign is applied internally."
        ),
    )
    parser.add_argument(
        "--ocean-corr-scale-factor",
        type=float,
        default=0.4,
        help=(
            "Decay scale of the (Dyn, Rho) spectral correlation with "
            "wavenumber, as a factor of the load scale."
        ),
    )
    parser.add_argument(
        "--prior-kernel",
        choices=["heat", "sobolev"],
        default="sobolev",
        help=(
            "Covariance family for the three model priors and any "
            "spatially correlated noise models (applied consistently; "
            "white-noise settings unaffected): 'heat' or "
            "'sobolev' (default)."
        ),
    )
    parser.add_argument(
        "--prior-order",
        type=float,
        default=1.0,
        help=(
            "Common spectral order for --prior-kernel sobolev (must be "
            "positive). Sample roughness combines this order with the "
            "model space order (--load-order): with the default order-2 "
            "spaces, 1.0 gives degree-variance tails ~ l^-5, comparable "
            "to observed mesoscale SSH spectra. Ignored for the heat "
            "kernel."
        ),
    )
    parser.add_argument(
        "--prior-shift", type=float, default=1.0, help="Prior mean shift factor."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Setup directory to save plots
    output_dir = "output_plots_altimetry_bias"
    os.makedirs(output_dir, exist_ok=True)
    figures_to_save = {}

    print("Initializing Earth State and 3-Component Physics Operators...")
    state_dummy = EarthState.from_defaults(lmax=args.lmax)
    points = ocean_altimetry_points(state_dummy, spacing=args.spacing)

    (
        state,
        load_space,
        fp_op,
        continuous_ssh_op,
        continuous_sl_op,
        forward_op,
        scale_mm,
    ) = utils.build_physics_components(
        args.lmax, args.load_order, args.load_scale_km, points, is_surrogate=False
    )

    print("Building analytical measures (with strict global mass conservation)...")
    model_prior, noise_meas, _ = utils.build_measures(
        state,
        load_space,
        args.ice_scale_factor,
        args.ice_std_mm,
        args.ocean_dyn_scale_factor,
        args.ocean_dyn_std_factor,
        args.ocean_rho_scale_factor,
        args.ocean_rho_std_factor,
        args.noise_corr_scale_factor,
        args.noise_std_factor,
        points,
        scale_mm,
        prior_shift=args.prior_shift,
        is_surrogate=False,
        ocean_corr=args.ocean_corr,
        ocean_corr_scale_factor=args.ocean_corr_scale_factor,
        prior_kernel=args.prior_kernel,
        prior_order=args.prior_order,
        noise_corr_std_factor=args.noise_corr_std_factor,
    )

    joint_meas = inf.GaussianMeasure.from_direct_sum([model_prior, noise_meas])
    data_space = noise_meas.domain

    # The TRUE GMSLR uses the Sea Level Operator (SLC)
    true_gmsl_op = utils.true_gmsl_operator(state, load_space, continuous_sl_op)

    # The ESTIMATOR averages the SSH evaluated at the altimetry points
    alt_avg_op = altimetry_averaging_operator(points)
    est_gmsl_op = alt_avg_op @ forward_op

    # Error Operator = True GMSLR - Estimated GMSLR
    err_gmsl_op = true_gmsl_op - est_gmsl_op

    op_true = inf.RowLinearOperator(
        [true_gmsl_op, data_space.zero_operator(inf.EuclideanSpace(1))]
    )
    op_err = inf.RowLinearOperator([err_gmsl_op, -1.0 * alt_avg_op])

    print("Calculating exact analytical moments of the bias...")
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
        print("Drawing a sample to visualize map components...")
        model_sample = model_prior.sample()
        ssh_sample = continuous_ssh_op(model_sample)

        ice_thickness, ocean_dyn, ocean_rho = model_sample

        ocean_mask_raw = state.ocean_projection(value=0.0)
        ocean_mask_mm = scale_mm * ocean_mask_raw
        ice_mask = scale_mm * state.ice_projection(value=0.0)

        # Physical density units for the raw density map (nondimensional
        # density x density_scale = kg/m^3; x1000 -> g/m^3).
        rho_scale_gm3 = 1.0e3 * state.model.parameters.density_scale
        ocean_mask_gm3 = rho_scale_gm3 * ocean_mask_raw

        fig1, ax1 = sl.create_map_figure(figsize=(12, 6))
        _, im1 = sl.plot(
            ice_thickness * ice_mask,
            ax=ax1,
            cmap="seismic",
            symmetric=True,
        )

        im1.colorbar.set_label("Ice Thickness Change (mm)", fontsize=16)
        im1.colorbar.ax.tick_params(labelsize=14)

        ax1.gridliner.xlabel_style = {"size": 12, "color": "black"}
        ax1.gridliner.ylabel_style = {"size": 12, "color": "black"}

        ax1.text(
            0.02,
            0.98,
            "(c)",
            transform=ax1.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3.0),
            zorder=10,
        )

        figures_to_save["bias_ice_thickness"] = fig1

        fig2, ax2 = sl.create_map_figure(figsize=(12, 6))
        _, im2 = sl.plot(
            ocean_dyn * ocean_mask_mm,
            ax=ax2,
            cmap="seismic",
            symmetric=True,
        )

        im2.colorbar.set_label("Sterodynamic SL Change (mm)", fontsize=16)
        im2.colorbar.ax.tick_params(labelsize=14)

        ax2.gridliner.xlabel_style = {"size": 12, "color": "black"}
        ax2.gridliner.ylabel_style = {"size": 12, "color": "black"}

        ax2.text(
            0.02,
            0.98,
            "(a)",
            transform=ax2.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3.0),
            zorder=10,
        )

        figures_to_save["bias_ocean_dynamic"] = fig2

        fig3, ax3 = sl.create_map_figure(figsize=(12, 6))
        _, im3 = sl.plot(
            ocean_rho * ocean_mask_gm3,
            ax=ax3,
            cmap="seismic",
            symmetric=True,
        )

        im3.colorbar.set_label(r"Averaged Density Change (g m$^{-3}$)", fontsize=16)
        im3.colorbar.ax.tick_params(labelsize=14)

        ax3.gridliner.xlabel_style = {"size": 12, "color": "black"}
        ax3.gridliner.ylabel_style = {"size": 12, "color": "black"}

        ax3.text(
            0.02,
            0.98,
            "(b)",
            transform=ax3.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3.0),
            zorder=10,
        )

        figures_to_save["bias_density"] = fig3

        ssh_grid_mm = ssh_sample * ocean_mask_mm
        observed_data_mm = (forward_op(model_sample) + noise_meas.sample()) * scale_mm
        shared_vmax = max(
            np.max(np.abs(ssh_grid_mm.data)), np.max(np.abs(observed_data_mm))
        )

        fig4, ax4 = sl.create_map_figure(figsize=(12, 6))
        _, im4 = sl.plot(
            ssh_grid_mm,
            ax=ax4,
            cmap="seismic",
            vmin=-shared_vmax,
            vmax=shared_vmax,
            colorbar_kwargs={"label": "Continuous SSH Change (mm)"},
        )

        im4.colorbar.set_label("Continuous SSH Change (mm)", fontsize=16)
        im4.colorbar.ax.tick_params(labelsize=14)

        ax4.gridliner.xlabel_style = {"size": 12, "color": "black"}
        ax4.gridliner.ylabel_style = {"size": 12, "color": "black"}

        figures_to_save["bias_ssh_map"] = fig4

        fig5, ax5 = sl.create_map_figure(figsize=(12, 6))
        ax5.set_global()
        _, im5 = sl.plot_points(
            points,
            data=observed_data_mm,
            ax=ax5,
            cmap="seismic",
            vmin=-shared_vmax,
            vmax=shared_vmax,
            s=4,
            edgecolors="none",
            colorbar=True,
            colorbar_kwargs={"label": "Synthetic Altimetry Data (mm)"},
            zorder=5,
        )

        im5.colorbar.set_label("Synthetic Altimetry Data (mm)", fontsize=16)
        im5.colorbar.ax.tick_params(labelsize=14)

        ax5.text(
            0.02,
            0.98,
            "(d)",
            transform=ax5.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3.0),
            zorder=10,
        )

        ax5.gridliner.xlabel_style = {"size": 12, "color": "black"}
        ax5.gridliner.ylabel_style = {"size": 12, "color": "black"}

        figures_to_save["bias_ssh_points"] = fig5

    err_samples = None
    if args.samples > 0:
        print(
            f"Drawing {args.samples} Monte Carlo samples to validate analytical bias..."
        )
        joint_samples_list = joint_meas.samples(args.samples)
        err_samples = np.array(
            [op_err(sample)[0] * scale_mm for sample in joint_samples_list]
        )

    print("Plotting Extended Bias PDFs...")

    def gaussian_pdf(x, mean, std):
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

    fig_pdf, ax = plt.subplots(figsize=(10, 6), layout="constrained")
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
        label=r"Actual error",
    )
    ax.plot(
        x_vals,
        gaussian_pdf(x_vals, 0, alt_noise_std),
        "b",
        linewidth=2,
        label=r"Nominal error (noise only)",
    )

    ax.set_xlabel("Error in GMSLR estimate (mm)", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=11)

    sec_ax = ax.secondary_xaxis(
        "top", functions=(lambda x: x / true_std, lambda x: x * true_std)
    )
    sec_ax.set_xlabel(
        r"Error Standardised by True Signal $\sigma$", fontsize=12, color="darkgreen"
    )
    sec_ax.tick_params(axis="x", colors="darkgreen")

    figures_to_save["bias_pdf"] = fig_pdf

    # -- Save all figures --
    if figures_to_save:
        print(f"\nSaving {len(figures_to_save)} plots to '{output_dir}/'...")
        for name, fig in figures_to_save.items():
            filepath = os.path.join(output_dir, f"{name}.png")
            fig.savefig(filepath, dpi=600, bbox_inches="tight")
            print(f"  Saved: {filepath}")
            plt.close(fig)


if __name__ == "__main__":
    main()
