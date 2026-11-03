"""
A Bayesian inversion of synthetic altimetry data (Heathcote et al., 2026).

The model is the ice thickness change, the sterodynamic sea level change eta
and the vertically averaged ocean density change of the set-up
`pyslfp.models.ice_ocean_and_density`, under the prior of
`common.ocean_prior`: global fields, eta and the density anti-correlated at
long wavelengths, conditioned exactly on the conservation of the ocean's
mass. A model drawn from the prior gives synthetic geocentric sea level data
at the altimetry track points, with the noise of `common.altimetry_noise`,
and the linear Gaussian inversion of those data gives the posterior. The
data-space normal equations are solved by conjugate gradients, preconditioned
by the Woodbury inverse of a cheap surrogate problem
(`common.surrogate_preconditioner`): the ice load alone, one pass of the sea
level equation, at --surrogate-degree, under the uncorrelated, unconditioned
product prior and the local noise.

The posterior expectation and the samples behind the pointwise standard
deviations are solved at a relative residual of 0.01 times
--noise-std-factor. Everything computed from the posterior covariance (the
quantities of interest and the covariance maps) goes through solves at
--qoi-rtol instead, with the same preconditioner: a posterior variance is the
prior variance less a correction computed through the solves, so its relative
error is about the tolerance divided by the ratio of the posterior variance
to the prior, and the data reduce the GMSL variance by over 95 per cent
(`common.solve_posterior`).

The quantities of interest are push-forwards of the prior and posterior:

- the global mean sea level (GMSL) change, the ocean average of the relative
  sea level change, against the simple cos-latitude-weighted average of the
  data;
- its split into barystatic and steric parts, with two checks on the true
  model: the residual steric change (the GMSL change less the barystatic)
  against the direct one, which measures the sea level equation's mass
  balance, and the ocean mean of eta against the steric change, whose
  difference, the ocean mean of the dynamic manometric change, vanishes under
  the mass constraint;
- the Tasman Sea mean of the relative sea level change, split into steric
  (SSLC), dynamic manometric (DMSLC) and barystatic manometric (BMSLC) parts
  after Gregory et al. (2019).

Outputs, in output_plots_altimetry_inversion/ at 300 dpi:

- altimetry_metrics.txt: the KL divergences and variance reductions of the
  quantities of interest, with the checks above.
- gmslr_pdf.png (--plot-pdfs): the posterior and simple-averaging densities
  of the GMSL change.
- gmslr_split_corner.png (--plot-gmsl-split): the barystatic/steric split.
- posterior_maps.png, geocentric_sl_data.png, covariance_Ice_WAIS.png and
  covariance_Sterodynamic_North_Atlantic.png (--plot-maps): the true and
  posterior fields, with the pointwise posterior standard deviation from
  --std-samples samples; the geocentric sea level change and the data; and
  the prior and posterior covariance with the field at a point in West
  Antarctica and in the North Atlantic. Each field is shown on its
  component's support.
- regional_corner.png (--plot-regions): the Tasman Sea decomposition, whose
  boundary is then drawn on the maps.

--all selects every plot. The defaults are the paper's resolution;
`FAST=1 ./run_all.sh altimetry_inversion` runs at lmax 64 with the points 8
degrees apart.
"""

import argparse
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import pygeoinf as inf
import pyslfp as sl
from pyslfp.observations import altimetry_averaging_operator

import common
import figures
from common import OCEAN_REGION
from figures import COLORBAR, GRIDLINES, peak

OUTPUT_DIR = "output_plots_altimetry_inversion"
METRICS_FILE = os.path.join(OUTPUT_DIR, "altimetry_metrics.txt")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of altimetry data for the ice, eta and "
        "density model."
    )
    parser.add_argument("--all", action="store_true", help="Make every plot.")
    parser.add_argument(
        "--plot-pdfs", action="store_true", help="Plot the GMSL change densities."
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot the fields, the data and the covariance maps.",
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="Decompose the Tasman Sea mean sea level change.",
    )
    parser.add_argument(
        "--plot-gmsl-split",
        action="store_true",
        help="Split the GMSL change into barystatic and steric parts.",
    )
    parser.add_argument(
        "--std-samples",
        type=int,
        default=0,
        help="Posterior samples for the pointwise standard deviations.",
    )
    common.add_ocean_arguments(parser, surrogate=True)
    common.add_altimetry_noise_arguments(parser)
    common.add_qoi_arguments(parser)
    common.add_parallel_arguments(parser)
    args = parser.parse_args()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True
        args.plot_gmsl_split = True
    return args


# ================================================================ #
#                     Quantities of interest                       #
# ================================================================ #


def gmsl_densities(physics, prior, posterior, data_measure, model, rng):
    print("Pushing forward onto the GMSL change...")
    mm = physics.scale_mm
    true_gmsl = mm * physics.setup.gmsl(physics.fingerprint)
    averaging = mm * altimetry_averaging_operator(physics.points)
    prior_gmsl = common.dense(prior, true_gmsl)
    alt_gmsl = common.dense(data_measure, averaging)
    post_gmsl = common.dense(posterior, true_gmsl)

    kl_div = post_gmsl.kl_divergence(prior_gmsl, rng=rng)
    prior_var = common.variances(prior_gmsl)[0]
    post_var = common.variances(post_gmsl)[0]
    common.write_metrics(
        METRICS_FILE,
        f"{'Target':<16} | {'KL Divergence':<15} | {'Prior Var (mm²)':<17} | "
        f"{'Post Var (mm²)':<16} | {'Variance Reduction':<18}\n"
        + "-" * 92
        + "\n"
        + f"{'GMSLR':<16} | {kl_div:<15.4f} | {prior_var:<17.4f} | "
        f"{post_var:<16.4f} | {common.reduction(prior_var, post_var):>16.4f}%\n",
    )

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    inf.plot_densities(
        [post_gmsl, alt_gmsl],
        truth=true_gmsl(model)[0],
        ax=ax,
        title="",
        xlabel="Global Mean Sea Level Rise (mm)",
        labels=["Bayesian", "Simple averaging"],
        rng=rng,
    )
    return fig


def gmsl_split(physics, prior, posterior, model, n_jobs, rng):
    print("\nPushing forward onto the barystatic/steric GMSL split...")
    split = physics.scale_mm * physics.setup.gmsl_split
    prior_split = common.dense(prior, split, n_jobs=n_jobs)
    post_split = common.dense(posterior, split, n_jobs=n_jobs)
    kl_div = post_split.kl_divergence(prior_split, rng=rng)
    prior_cov = common.covariance_matrix(prior_split)
    post_cov = common.covariance_matrix(post_split)

    common.write_metrics(
        METRICS_FILE,
        "\nGMSLR Split: Barystatic vs Steric\n"
        + "=" * 70
        + "\n"
        + f"Joint KL Divergence:     {kl_div:.4f} nats\n"
        + f"Barystatic-Steric corr:  prior {common.correlation(prior_cov):+.4f} | "
        f"posterior {common.correlation(post_cov):+.4f}\n"
        + "\n".join(common.gmsl_checks(physics, model))
        + "\n"
        + common.variance_table(
            ["Barystatic", "Steric"], np.diag(prior_cov), np.diag(post_cov), 70
        ),
    )

    return figures.corner_figure(
        post_split,
        prior=prior_split,
        truth=split(model),
        labels=["Barystatic SLR (mm)", "Steric SLR (mm)"],
        fill=False,
        rng=rng,
    )


def regional_decomposition(physics, prior, posterior, model, n_jobs, rng):
    print("\nDecomposing the regional sea level change...")
    operator = physics.regional_decomposition(OCEAN_REGION)
    prior_reg = common.dense(prior, operator, n_jobs=n_jobs)
    post_reg = common.dense(posterior, operator, n_jobs=n_jobs)
    common.write_metrics(
        METRICS_FILE,
        f"\nRegional Signal Separation ({OCEAN_REGION})\n"
        + common.REGIONAL_NOTE
        + "\n"
        + "=" * 65
        + "\n"
        + common.covariance_summary(
            post_reg.kl_divergence(prior_reg, rng=rng),
            common.covariance_matrix(prior_reg),
            common.covariance_matrix(post_reg),
            ["SSLC", "DMSLC", "BMSLC"],
        ),
    )

    return figures.corner_figure(
        post_reg,
        prior=prior_reg,
        truth=operator(model),
        labels=["SSLC (mm)", "DMSLC (mm)", "BMSLC (mm)"],
        fill=False,
        rng=rng,
    )


# ================================================================ #
#                              Maps                                #
# ================================================================ #


def posterior_maps(physics, model, posterior, std_samples, n_jobs, rng, boundaries):
    """The true and posterior fields, and the pointwise posterior standard
    deviation if `std_samples` is positive, one row per component."""
    print("\nPlotting the true and posterior fields...")
    true = physics.display(model)
    mean = physics.display(posterior.expectation)
    std = None
    if std_samples > 0:
        print(
            f"\nComputing pointwise standard deviations from {std_samples} "
            "posterior samples..."
        )
        std = physics.display(
            common.pointwise_std(posterior, std_samples, rng=rng, n_jobs=n_jobs)
        )
    limits = {name: max(peak(true[name]), peak(mean[name])) for name in true}
    return figures.state_maps(
        true,
        mean,
        std,
        limits=limits,
        figsize=(16 if std is None else 22, 16),
        state=physics.state,
        boundaries=boundaries,
    )


def geocentric_sea_level_maps(physics, model, data, boundaries):
    """The true geocentric sea level change over the ocean, and the data."""
    print("Plotting the geocentric sea level change and the data...")
    mm = physics.scale_mm
    ocean = physics.state.ocean_projection(value=0.0)
    geocentric_mm = mm * physics.geocentric_sea_level(model) * ocean
    data_mm = mm * data
    vmax = max(peak(geocentric_mm), np.max(np.abs(data_mm)))

    fig, axes = sl.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.1})
    sl.plot(
        geocentric_mm,
        ax=axes[0],
        colorbar=True,
        vmin=-vmax,
        vmax=vmax,
        cmap="seismic",
        colorbar_kwargs={**COLORBAR, "label": "Geocentric Sea Level Change (mm)"},
        gridlines_kwargs=GRIDLINES,
    )
    axes[1].set_global()
    axes[1].coastlines(linewidth=0.5, alpha=0.5, zorder=10)
    sl.plot_points(
        physics.points,
        data=data_mm,
        ax=axes[1],
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        s=6,
        edgecolors="none",
        colorbar=True,
        colorbar_kwargs={**COLORBAR, "label": "Synthetic Altimetry Data (mm)"},
        zorder=5,
    )
    figures.draw_boundaries(physics.state, axes, boundaries)
    return fig


def covariance_maps(physics, prior, posterior, boundaries):
    """The prior and posterior covariance of every field with one field at a
    point, for each of `common.OCEAN_COVARIANCE_POINTS`."""
    print("\nPlotting the covariance maps...")
    output = {}
    for title, component, point, point_name in common.OCEAN_COVARIANCE_POINTS:
        print(f"  Covariance with {title} at {point}...")
        columns = [
            ("Prior", physics.point_covariance(prior, component, point)),
            ("Posterior", physics.point_covariance(posterior, component, point)),
        ]
        output[f"covariance_{title}_{point_name}.png"] = figures.covariance_maps(
            columns,
            component,
            point,
            marker="dot",
            state=physics.state,
            boundaries=boundaries,
        )
    return output


# ================================================================ #
#                              Main                                #
# ================================================================ #


def main():
    args = parse_arguments()
    rng = common.configure(args)
    figures.set_inversion_style()

    common.start_run(
        OUTPUT_DIR,
        args,
        metrics_file=METRICS_FILE,
        heading="Altimetry Inversion Metrics\n===========================\n\n",
    )
    output = {}

    print(f"\nBuilding the physics, prior and noise (lmax={args.lmax})...")
    physics = common.ocean_physics(args)
    print(f"Generated {len(physics.points)} ocean altimetry observation points.")
    prior, calib = common.ocean_prior(physics, args, rng)
    noise, local_noise = common.altimetry_noise(
        physics,
        calib["derived_stds"]["dyn_std"],
        args.prior_order,
        args.noise_std_factor,
        args.noise_corr_std_factor,
        args.noise_corr_scale_factor,
        calib=calib,
    )
    common.print_calibration_report(calib, physics)
    common.print_prior_settings(
        args, args.noise_corr_std_factor, args.noise_corr_scale_factor
    )

    print("Setting up the Bayesian inversion...")
    forward_problem = inf.LinearForwardProblem(physics.altimetry, error=noise)
    model, data = forward_problem.synthetic_model_and_data(prior, rng=rng)
    inversion = inf.LinearGaussianInversion(forward_problem, prior)

    surrogate, surrogate_prior = common.ocean_surrogate(args, physics, calib)
    preconditioner = common.surrogate_preconditioner(
        inversion.normal_operator,
        surrogate.altimetry,
        surrogate_prior,
        local_noise,
        max_jobs=common.jobs(args),
    )

    print("\nSolving for the posterior expectation...")
    posterior, qoi_posterior, _ = common.solve_posterior(
        inversion,
        data,
        preconditioner,
        rtol=0.01 * args.noise_std_factor,
        qoi_rtol=args.qoi_rtol,
    )

    boundaries = [OCEAN_REGION] if args.plot_regions else None
    if args.plot_pdfs:
        output["gmslr_pdf.png"] = gmsl_densities(
            physics, prior, qoi_posterior, noise.translate(data), model, rng
        )

    if args.plot_gmsl_split:
        output["gmslr_split_corner.png"] = gmsl_split(
            physics, prior, qoi_posterior, model, common.jobs(args, 2), rng
        )

    if args.plot_maps:
        output["posterior_maps.png"] = posterior_maps(
            physics,
            model,
            posterior,
            args.std_samples,
            common.jobs(args),
            rng,
            boundaries,
        )
        output["geocentric_sl_data.png"] = geocentric_sea_level_maps(
            physics, model, data, boundaries
        )
        output |= covariance_maps(physics, prior, qoi_posterior, boundaries)

    if args.plot_regions:
        output["regional_corner.png"] = regional_decomposition(
            physics, prior, qoi_posterior, model, common.jobs(args, 3), rng
        )

    if output:
        print(f"\nSaving {len(output)} plots to '{OUTPUT_DIR}/'...")
        common.save_figures(output, OUTPUT_DIR, dpi=300)


if __name__ == "__main__":
    main()
