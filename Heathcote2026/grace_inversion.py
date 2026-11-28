"""
A Bayesian inversion of synthetic GRACE data, compared with the WMB method
(Heathcote et al., 2026).

The model is one field, the direct surface mass load zeta, which is not
divided into its causes. The data are GRACE's coefficients, degrees 2 to
--obs-degree, of the potential change of the sea level equation's response
to zeta, a response that includes the ocean load zeta induces. One load and
one dataset are drawn from the prior and the noise, and the posterior is
solved by conjugate gradients in the data space.

The prior, noise and preconditioner come in two forms
(`common.grace_experiment`):

- --load-prior model with --coefficient-errors, which `run_all.sh` passes.
  The prior of zeta is the image, under the load operator, of the prior of
  the fields of the ocean experiments (the ice, the land water, the
  sterodynamic sea level and the ocean density), and the true load is that
  of a sample of those fields. The noise is the formal error covariance of
  the coefficients of ITSG-Grace2018 for a difference of annual means, a
  full matrix. The normal matrix is assembled and factorised
  (`common.exact_preconditioner`), so each solve takes an iteration or two.
- --load-prior sobolev, the submitted paper's. The prior is a Sobolev
  measure on H^2. The noise is a spatial Sobolev field taken to the
  coefficients by WMB's spectral map, its spectrum solved so that the
  signal-to-noise ratio is --snr-low-degree at --snr-reference-degree and
  one at --snr-crossover-degree; the preconditioner is WMB's diagonal
  approximation of the normal operator. --coefficient-errors may replace
  that noise here too.

The solves for the posterior expectation stop at a relative residual of 0.01
times the ratio of the noise's pointwise standard deviation to the signal's.
Everything computed from the posterior covariance (the densities and tables
of the quantities of interest, the prior-sensitivity tables and the
covariance maps) goes through solves at --qoi-rtol instead, with the same
preconditioner: a posterior variance is the prior variance less a correction
computed through the solves, so its relative error is about the tolerance
divided by the ratio of the posterior variance to the prior, and the data
reduce some variances by over 90 per cent (`common.solve_posterior`).

The WMB method (Wahr, Molenaar and Bryan, 1998) inverts the coefficients
degree by degree through the elastic load Love numbers, ignoring the induced
ocean load and the degrees GRACE does not see. The two are compared on the
averages over the GRACE regions (`common.GRACE_REGIONS`) of the total load,
zeta plus the ocean load rho_w C xi it induces, and, for the Bayesian
estimate alone, on the coefficients of zeta at --target-degree (degree 1 by
default, which GRACE does not observe).

Outputs, in output_plots_grace_inversion/ at 300 dpi (--all selects every
option):

- --plot-pdfs: grace_regional_pdfs.png, per region the posterior and WMB's
  estimate with its noise-only error, against the true value; a table of the
  KL divergence and variance reduction per region.
- --plot-target-degree: grace_degree_{l}_corner.png, the posterior of the
  degree-l coefficients; a table of their KL divergence, total and
  generalised variances and the variance reduction per coefficient.
- --prior-sensitivity, with --load-prior model: the tables of
  `common.posterior_sensitivity` for the regional averages and the degree-l
  coefficients. The prior is built from independent fields; for each
  quantity and field the tables give the share of the posterior variance
  that the field's prior supplies, and the factors, in the operator and the
  Hilbert-Schmidt norms, that bound the change of the variance of the
  estimate's error under a change of that prior, beside the noise's share.
  Then the table of `common.prior_perturbation_sizes`. --paper-bound adds
  the factor of the bound stated on the load space.
- --plot-maps: grace_posterior_maps.png, the true total load, WMB's
  estimate, the posterior mean and, with --std-samples, the pointwise
  posterior standard deviation of the total load, estimated from that many
  randomise-then-optimise samples; grace_covariance_maps.png, the prior and
  posterior covariance of the total load with three points.

The tables go to grace_metrics.txt, in the order above, and are printed. The
defaults are the paper's resolution, which runs for hours;
`FAST=1 ./run_all.sh grace_inversion` runs at lmax 64 and observation
degree 32.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import pygeoinf as inf
import pyslfp as sl

import common
import figures
from figures import COLORBAR, GRIDLINES, draw_region_boundaries

OUTPUT_DIR = "output_plots_grace_inversion"
METRICS_FILE = os.path.join(OUTPUT_DIR, "grace_metrics.txt")

# The points of the covariance maps, (latitude, longitude), one per regime.
# The points of the covariance maps, each on the support of a mass source:
# the gravity experiments look at the land and the ice alone.
COVARIANCE_POINTS = [
    ((-5.0, -60.0), "Land water (Amazon)", common.LAND),
    ((-75.0, -110.0), "Ice (West Antarctica)", common.ICE),
    ((66.0, -46.0), "Ice (South Greenland)", common.ICE),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of synthetic GRACE data, against WMB."
    )
    common.add_grace_arguments(parser)
    # The averages are over the regions themselves, their masks unsmoothed
    # (--smoothing-scale-km smooths them), and WMB's estimate of them is
    # made as it is in practice (`common.wmb_estimator`).
    parser.set_defaults(smoothing_scale_km=0.0)
    common.add_wmb_arguments(parser)
    common.add_qoi_arguments(parser)
    common.add_parallel_arguments(parser)
    parser.add_argument("--all", action="store_true", help="Every option below.")
    parser.add_argument(
        "--plot-pdfs",
        action="store_true",
        help="Densities and metrics of the regional averages, Bayesian and WMB.",
    )
    parser.add_argument(
        "--plot-target-degree",
        action="store_true",
        help="Corner plot and metrics of the coefficients at --target-degree.",
    )
    parser.add_argument("--target-degree", type=int, default=1)
    parser.add_argument(
        "--prior-sensitivity",
        action="store_true",
        help="With --load-prior model: the sensitivity of the posterior "
        "variances of the regional averages and the target degree's "
        "coefficients to the prior of each field "
        "(`common.posterior_sensitivity`).",
    )
    parser.add_argument(
        "--paper-bound",
        action="store_true",
        help="With --prior-sensitivity and --load-prior model: beside the "
        "factors of each field, that of the bound stated on the load space "
        "itself, with the Hilbert-Schmidt norm of the load's prior covariance; "
        "the norm is estimated with random probes to 2 per cent.",
    )
    parser.add_argument(
        "--plot-maps", action="store_true", help="Load and covariance maps."
    )
    parser.add_argument(
        "--std-samples",
        type=int,
        default=0,
        help="Posterior samples for the pointwise standard deviation map.",
    )
    args = parser.parse_args()
    if args.all:
        args.plot_pdfs = args.plot_target_degree = True
        args.prior_sensitivity = args.plot_maps = True
    return args


# ================================================================ #
#                        Regional averages                         #
# ================================================================ #


def regional_densities(
    names, prior, posterior, data_measure, total_average, wmb_average, model, n_jobs
):
    """The posterior of each regional average against its prior and against
    WMB's estimate, whose mean is WMB applied to the data and whose
    covariance is the noise's alone. Returns the table and the figure."""
    wmb_averages = common.dense(data_measure, wmb_average)
    posterior_averages = common.dense(posterior, total_average, n_jobs=n_jobs)
    prior_averages = common.dense(prior, total_average, n_jobs=n_jobs)
    truth = total_average(model)
    prior_vars = common.variances(prior_averages)
    posterior_vars = common.variances(posterior_averages)

    lines = [
        f"{'Region':<16} | {'KL Divergence':<15} | {'Prior Var (mm² EWT)':<19} | "
        f"{'Post Var (mm² EWT)':<18} | {'Variance Reduction':<18}\n",
        "-" * 96 + "\n",
    ]
    nrows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=2, figsize=(14, 5 * nrows), layout="constrained"
    )
    axes = axes.flatten()
    for i, name in enumerate(names):
        select = prior_averages.domain.coordinate_selection([i])
        prior_i = prior_averages.affine_map(select)
        posterior_i = posterior_averages.affine_map(select)
        kl = posterior_i.kl_divergence(prior_i)
        lines.append(
            f"{name:<16} | {kl:<15.2f} | {prior_vars[i]:<19.2f} | "
            f"{posterior_vars[i]:<18.2f} | "
            f"{common.reduction(prior_vars[i], posterior_vars[i]):>16.2f}%\n"
        )
        inf.plot_densities(
            [posterior_i, wmb_averages.affine_map(select)],
            ax=axes[i],
            truth=truth[i],
            xlabel="Regional Average Total Load (mm EWT)",
            title=name,
            labels=["Bayesian", "WMB"],
        )
        if i % 2 == 0:
            axes[i].set_ylabel("Probability density")
        else:
            axes[i].set_ylabel("")
        if i != 1 and axes[i].get_legend() is not None:
            axes[i].get_legend().remove()
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    return "".join(lines), fig


# ================================================================ #
#                     Single-degree coefficients                   #
# ================================================================ #


def degree_coefficients(
    degree, components, prior, posterior, coefficients, model, n_jobs
):
    """The posterior of the coefficients of one degree against their prior.
    Returns the table and the corner plot."""
    prior_c = common.dense(prior, coefficients, n_jobs=n_jobs)
    posterior_c = common.dense(posterior, coefficients, n_jobs=n_jobs)
    table = (
        f"\n\nDegree-{degree} Spherical Harmonic Coefficients\n"
        + "=" * 65
        + "\n"
        + common.covariance_summary(
            posterior_c.kl_divergence(prior_c),
            common.covariance_matrix(prior_c),
            common.covariance_matrix(posterior_c),
            [name for name, _ in components],
        )
    )
    fig = figures.corner_figure(
        posterior_c,
        truth=coefficients(model),
        labels=[label for _, label in components],
    )
    return table, fig


# ================================================================ #
#                               Maps                               #
# ================================================================ #


def posterior_maps(state, panels, std=None):
    """The load maps, row by row: the true load, WMB's estimate and the
    posterior mean, each a (field, title) in `panels`, on the colour scale
    of the first and last of them (values beyond it saturate), then the
    pointwise posterior
    standard deviation `std` on a scale of its own; that panel is left empty
    without it. Each map has its own colour bar beneath it, so that the four
    are drawn at one size."""

    # The scale is that of the true load and the posterior mean, the first
    # and last panels; a panel that exceeds it more than twofold, as WMB's
    # estimate does where the noise survives its filter, takes its own.
    # The limit is the value that all but a two-hundredth of the grid lies
    # within, not the largest: with the ice's std growing towards the edge
    # of the ice sheets a few points there would set the scale for the rest.
    def extent(field):
        return float(np.percentile(np.abs(field.data), 99.5))

    limit = max(extent(panels[0][0]), extent(panels[-1][0]))
    fig, axes = sl.subplots(2, 2, figsize=(18, 12), gridspec_kw={"hspace": 0.10})
    maps = []
    for field, title in panels:
        own = extent(field) if extent(field) > 2.0 * limit else limit
        maps.append(
            (
                field,
                title,
                "Load (mm EWT)",
                {"vmin": -own, "vmax": own, "cmap": "seismic"},
            )
        )
    if std is not None:
        maps.append(
            (
                std,
                "Posterior Std. Deviation",
                "Std. deviation (mm EWT)",
                {"vmin": 0.0, "vmax": extent(std), "cmap": "Blues"},
            )
        )
    else:
        axes[1, 1].set_axis_off()
    for ax, (field, title, label, style) in zip(axes.flat, maps):
        _, image = sl.plot(
            field,
            ax=ax,
            colorbar=True,
            colorbar_kwargs=COLORBAR,
            gridlines_kwargs=GRIDLINES,
            **style,
        )
        ax.set_title(title, fontsize=16)
        draw_region_boundaries(state, ax, common.GRACE_REGIONS)
        image.colorbar.set_label(label, fontsize=14)
        image.colorbar.ax.tick_params(labelsize=12)
    return fig


def covariance_maps(physics, prior, posterior):
    """The posterior covariance of the total load with its value at each of
    COVARIANCE_POINTS, in mm² EWT, and the prior and posterior variances
    there as a line of text for each point.

    Returns (figure, text)."""
    scale = physics.scale_mm**2
    fig, axes = sl.subplots(
        1, len(COVARIANCE_POINTS), figsize=(8.5 * len(COVARIANCE_POINTS), 5.6),
        gridspec_kw={"wspace": 0.08},
    )
    text = "\nVariance of the total load at the points of the covariance maps (mm² EWT)\n"
    for ax, (point, label, component) in zip(axes, COVARIANCE_POINTS):
        _, prior_variance = common.load_point_covariance(
            physics, prior, component, point
        )
        field, variance = common.load_point_covariance(
            physics, posterior, component, point
        )
        text += (
            f"  {label:<26} prior {prior_variance * scale:10.1f}, posterior "
            f"{variance * scale:10.1f}\n"
        )
        _, image = sl.plot(
            field * scale,
            ax=ax,
            colorbar=True,
            cmap="seismic",
            symmetric=True,
            colorbar_kwargs=COLORBAR,
            gridlines_kwargs=GRIDLINES,
        )
        ax.set_title(f"Posterior covariance: {label}", fontsize=16)
        sl.plot_points([point], ax=ax, color="black", gridlines=False, coasts=False)
        # The covariance is local: a region about the point, not the globe.
        import cartopy.crs as ccrs

        lat, lon = point
        ax.set_extent(
            [lon - 40.0, lon + 40.0, max(lat - 25.0, -89.0), min(lat + 25.0, 89.0)],
            crs=ccrs.PlateCarree(),
        )
        image.colorbar.set_label("Covariance (mm² EWT)", fontsize=14)
        image.colorbar.ax.tick_params(labelsize=12)
    return fig, text


def main():
    args = parse_arguments()
    rng = common.configure(args)
    common.start_run(OUTPUT_DIR, args, metrics_file=METRICS_FILE)
    figures.set_inversion_style()

    print(f"Building the physics, prior and noise (lmax={args.lmax})...")
    physics, initial_prior, prior, noise, noise_matrix, noise_info = (
        common.grace_experiment(args, rng)
    )
    state, load_space, scale_mm = physics.state, physics.load_space, physics.scale_mm
    names, average = common.regional_averaging(physics, args)

    forward = common.grace_forward_operator(physics, args)
    total_load = common.total_load_operator(physics)
    wmb = common.wmb_method(state, args)

    print("Drawing a synthetic load and dataset...")
    problem = inf.LinearForwardProblem(forward, error=noise)
    model, data = problem.synthetic_model_and_data(prior, rng=rng)

    # WMB's diagonal approximation of the normal operator needs a noise
    # that is diagonal too; with the coefficients' full error covariance the
    # normal matrix itself is assembled and factorised.
    if noise_matrix is None:
        preconditioner = wmb.bayesian_normal_operator_preconditioner(
            initial_prior, noise
        )
    else:
        print(f"Assembling the normal matrix of {forward.codomain.dim} data...")
        preconditioner = common.exact_preconditioner(
            forward, prior, noise, noise_matrix=noise_matrix, n_jobs=common.jobs(args)
        )

    print("Solving for the posterior...")
    posterior, qoi_posterior, qoi_inversion = common.solve_posterior(
        inf.LinearGaussianInversion(problem, prior),
        data,
        preconditioner,
        rtol=args.qoi_rtol,
        qoi_rtol=args.qoi_rtol,
    )

    # The quantities of interest in mm EWT: the regional averages of the total
    # load, their WMB estimates from the data, and the coefficients of the
    # direct load at the target degree, in the order m = 0, 1, ..., l, then
    # -1, ..., -l.
    total_average = scale_mm * average @ total_load
    wmb_load, wmb_matrix, wmb_description = common.wmb_estimator(
        state,
        load_space,
        args,
        average=scale_mm * average,
        sharp_masks=common.grace_region_masks(state),
        scale=scale_mm,
    )
    wmb_average = inf.LinearOperator.from_matrix(
        forward.codomain, average.codomain, wmb_matrix, form="components"
    )
    print(wmb_description)
    degree = args.target_degree
    coefficients = (
        (scale_mm / load_space.radius)
        * load_space.coefficient_operator(lmax=degree, lmin=degree)
        @ common.direct_load_operator(physics)
    )
    components = [
        (f"({degree},{m})", rf"$\zeta_{{{degree},{m}}}$ (mm EWT)")
        for m in list(range(degree + 1)) + list(range(-1, -degree - 1, -1))
    ]

    output, metrics = {}, []
    if args.plot_pdfs:
        print("Regional averages...")
        table, output["grace_regional_pdfs.png"] = regional_densities(
            names,
            prior,
            qoi_posterior,
            noise.translate(data),
            total_average,
            wmb_average,
            model,
            common.jobs(args, len(names)),
        )
        metrics.append(table)

    if args.plot_target_degree and args.remove_degree_1 and degree == 1:
        # The conditioned prior fixes these coefficients at zero, so neither
        # they nor their posterior have a density.
        print("Degree-1 coefficients are zero under --remove-degree-1; skipped.")
    elif args.plot_target_degree:
        print(f"Degree-{degree} coefficients...")
        table, output[f"grace_degree_{degree}_corner.png"] = degree_coefficients(
            degree,
            components,
            prior,
            qoi_posterior,
            coefficients,
            model,
            common.jobs(args, 2 * degree + 1),
        )
        metrics.append(table)

    independent = noise_info.get("independent")
    if args.prior_sensitivity and independent is None:
        print("--prior-sensitivity needs --load-prior model; skipped.")
    elif args.prior_sensitivity:
        print("Prior sensitivity, field by field...")
        model_hs = None
        if args.paper_bound:
            # A generator of its own, so that the draws that follow do not
            # depend on the option.
            model_hs = prior.hilbert_schmidt_norm(
                method="stochastic",
                rtol=0.02,
                rng=np.random.default_rng(None if args.seed is None else args.seed + 1),
            )
        for title, target, labels in (
            ("regional averages of the total load (mm EWT)", total_average, names),
            (
                f"degree-{degree} coefficients of the direct load (mm EWT)",
                coefficients,
                [name for name, _ in components],
            ),
        ):
            metrics.append(
                common.posterior_sensitivity(
                    target,
                    qoi_inversion,
                    independent,
                    labels,
                    title=title,
                    model_hs=model_hs,
                )
            )
        # WMB's estimate of the same averages, in the same terms. It states
        # no variance for its leakage, so the shares are of the whole of its
        # error variance under the prior.
        metrics.append(
            common.prior_sensitivity(
                common.wmb_error_operator(
                    physics, scale_mm * average, wmb_average, args
                ),
                wmb_average,
                noise,
                independent,
                names,
                title="WMB's regional averages of the total load (mm EWT); "
                "the std is that of its error under the prior",
            )
        )
        metrics.append(common.prior_perturbation_sizes(independent))

    if metrics:
        text = "".join(metrics)
        common.write_metrics(METRICS_FILE, text)
        print(text)
        print(f"saved {METRICS_FILE}")

    if args.plot_maps:
        print("Maps...")
        posterior_total = qoi_posterior.affine_map(total_load)
        panels = [
            (total_load(model), "True Total Load"),
            (wmb_load(data), "WMB Estimate (filtered)"),
            (posterior_total.expectation, "Bayesian Posterior"),
        ]
        std = None
        if args.std_samples > 0:
            print(
                f"Pointwise standard deviation from {args.std_samples} "
                "posterior samples..."
            )
            # The samples are randomise-then-optimise draws of the posterior
            # at the expectation's tolerance, each taken to the total load.
            std = posterior.affine_map(total_load).sample_pointwise_std(
                args.std_samples, rng=rng, n_jobs=common.jobs(args)
            )
            std = std * scale_mm
        output["grace_posterior_maps.png"] = posterior_maps(
            state, [(field * scale_mm, title) for field, title in panels], std
        )
        if physics.fields is not None:
            output["grace_covariance_maps.png"], text = covariance_maps(
                physics, prior, qoi_posterior
            )
            common.write_metrics(METRICS_FILE, text)
            print(text)

    common.save_figures(output, OUTPUT_DIR, dpi=300)


if __name__ == "__main__":
    main()
