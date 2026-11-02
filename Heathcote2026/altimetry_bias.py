"""
The error of the standard altimetry estimate of global mean sea level change
(Heathcote et al., 2026).

The model is the ice thickness change, the sterodynamic sea level change eta
and the vertically averaged ocean density change of the set-up
`pyslfp.models.ice_ocean_and_density`, under the prior of
`common.ocean_prior`: global fields conditioned exactly on the conservation
of the ocean's mass. The data are the geocentric sea level change at the
altimetry track points, with the local and large-scale correlated noise of
`common.altimetry_noise`.

The standard estimator of the global mean sea level (GMSL) change is the
cos-latitude-weighted mean of the data. The true GMSL change is the ocean
average of the relative sea level change, the sea level equation's response
to the loads plus eta. The error is the estimate less the true GMSL change,
(C A - B) m + C e, with A the forward operator, B the GMSL change, C the
weighted mean and e the noise. It is linear in the model and the noise, so it
is Gaussian, and its mean and standard deviation follow exactly by
push-forward of the joint measure of model and noise. The script compares
that error with the nominal one, C e alone, which is what the estimator would
make were the geocentric sea level change equal to the GMSL change
everywhere.

Outputs, in output_plots_altimetry_bias/ at 600 dpi:

- bias_pdf.png: the densities of the actual and the nominal error, with a
  secondary axis in units of the true GMSL change's standard deviation, and
  a histogram of Monte Carlo errors with --samples.
- With --plot-maps, from one prior sample: bias_sterodynamic.png (a),
  bias_density.png (b), bias_ice_thickness.png (c), each shown on its
  component's support; bias_geocentric_sl_map.png, the geocentric sea level
  change over the ocean; and bias_altimetry_data.png (d), the synthetic data
  at the track points, on the same colour scale.

The defaults are the paper's resolution; `FAST=1 ./run_all.sh altimetry_bias`
runs at lmax 64 with the points 8 degrees apart.
"""

import argparse

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pygeoinf as inf
import pyslfp as sl
from pyslfp.linear_operators import altimetry_averaging_operator

import common
from common import ETA, ICE, RHO
from figures import error_density, field_label, field_map, panel_label, style_map

OUTPUT_DIR = "output_plots_altimetry_bias"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The error of the altimetry GMSL estimator under the "
        "ice, eta and density model."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Monte Carlo samples drawn to check the analytical error distribution.",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot a prior sample's fields, its geocentric sea level change and "
        "the data.",
    )
    common.add_ocean_arguments(parser)
    common.add_altimetry_noise_arguments(parser)
    common.add_parallel_arguments(parser)
    return parser.parse_args()


def geocentric_maps(physics, model, data_mm):
    """The geocentric sea level change of a model over the ocean, and the
    data at the track points, on one colour scale."""
    mm = physics.scale_mm
    ocean = physics.state.ocean_projection(value=0.0)
    geocentric_mm = mm * physics.geocentric_sea_level(model) * ocean
    vmax = max(np.max(np.abs(geocentric_mm.data)), np.max(np.abs(data_mm)))

    label = "Geocentric Sea Level Change (mm)"
    fig, ax = sl.create_map_figure(figsize=(12, 6))
    _, im = sl.plot(
        geocentric_mm,
        ax=ax,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        colorbar_kwargs={"label": label},
    )
    style_map(ax, im, label)
    geocentric_fig = fig

    label = "Synthetic Altimetry Data (mm)"
    fig, ax = sl.create_map_figure(figsize=(12, 6))
    ax.set_global()
    _, im = sl.plot_points(
        physics.points,
        data=data_mm,
        ax=ax,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        s=4,
        edgecolors="none",
        colorbar=True,
        colorbar_kwargs={"label": label},
        zorder=5,
    )
    style_map(ax, im, label)
    panel_label(ax, "(d)")
    return geocentric_fig, fig


def main():
    args = parse_arguments()
    rng = common.configure(args)
    common.start_run(OUTPUT_DIR, args)
    figures = {}

    print("Building the physics, prior and noise...")
    physics = common.ocean_physics(args)
    setup = physics.setup
    mm = physics.scale_mm
    prior, calib = common.ocean_prior(physics, args, rng)
    noise, _ = common.altimetry_noise(
        physics,
        calib["derived_stds"]["dyn_std"],
        args.prior_order,
        args.noise_std_factor,
        args.noise_corr_std_factor,
        args.noise_corr_scale_factor,
        calib=calib,
    )
    common.print_calibration_report(calib, physics)

    # The error acts on the pair (model, noise): the estimator applied to the
    # noise-free data, less the true GMSL change, plus the estimator applied
    # to the noise.
    true_gmsl = setup.gmsl(physics.fingerprint)
    averaging = altimetry_averaging_operator(physics.points)
    error = inf.RowLinearOperator(
        [averaging @ physics.altimetry - true_gmsl, averaging]
    )
    joint = inf.GaussianMeasure.from_product([prior, noise])

    print("Computing the exact moments of the error...")
    true_std = common.standard_deviations(common.dense(prior, mm * true_gmsl))[0]
    error_measure = common.dense(joint, mm * error)
    err_std = common.standard_deviations(error_measure)[0]
    err_mean = error_measure.expectation[0]
    nominal_std = common.standard_deviations(common.dense(noise, mm * averaging))[0]
    print(
        f"  GMSL error (estimate less true): mean = {err_mean:.4f} mm, std = "
        f"{err_std:.4f} mm; nominal (noise only) std = {nominal_std:.4f} mm; "
        f"true GMSL std = {true_std:.4f} mm"
    )

    if args.plot_maps:
        print("Drawing a sample to show the fields...")
        model = prior.sample(rng=rng)
        shown = physics.display(model)
        for file, name, panel in [
            ("bias_ice_thickness.png", ICE, "(c)"),
            ("bias_sterodynamic.png", ETA, "(a)"),
            ("bias_density.png", RHO, "(b)"),
        ]:
            figures[file] = field_map(shown[name], field_label(name), panel)
        data_mm = mm * (physics.altimetry(model) + noise.sample(rng=rng))
        (
            figures["bias_geocentric_sl_map.png"],
            figures["bias_altimetry_data.png"],
        ) = geocentric_maps(physics, model, data_mm)

    err_samples = None
    if args.samples > 0:
        print(f"Drawing {args.samples} Monte Carlo samples of the error...")
        samples = joint.samples(args.samples, rng=rng, n_jobs=common.jobs(args))
        err_samples = np.array([mm * error(sample)[0] for sample in samples])

    print("Plotting the error densities...")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    error_density(
        ax,
        err_mean,
        err_std,
        nominal_std,
        true_std,
        samples=err_samples,
        linewidth=2.5,
        top_fontsize=12,
    )
    ax.set_xlabel("Error in GMSLR estimate (mm)", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    figures["bias_pdf.png"] = fig

    print(f"\nSaving {len(figures)} plots to '{OUTPUT_DIR}/'...")
    common.save_figures(figures, OUTPUT_DIR, dpi=600)


if __name__ == "__main__":
    main()
