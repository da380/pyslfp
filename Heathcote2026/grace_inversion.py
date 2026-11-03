"""
A Bayesian inversion of synthetic GRACE data, compared with the WMB method
(Heathcote et al., 2026).

The model is the direct surface mass load zeta, a global field with a Sobolev
prior. The data are GRACE's coefficients, degrees 2 to --obs-degree, of the
potential change of the sea level equation's response to zeta, a response
that includes the ocean load zeta induces. The noise is a spatial Sobolev
field taken to the coefficients by WMB's spectral map, its spectrum solved so
that the signal-to-noise ratio is --snr-low-degree at --snr-reference-degree
and one at --snr-crossover-degree. One load and one dataset are drawn from
the prior and the noise, and the posterior is solved by conjugate gradients
in the data space, preconditioned by WMB's diagonal approximation of the
normal operator.

The solves for the posterior expectation stop at a relative residual of 0.01
times the ratio of the noise's pointwise standard deviation to the signal's.
Everything computed from the posterior covariance (the densities and tables
of the quantities of interest, the estimator kernels and the covariance
maps) goes through solves at --qoi-rtol instead, with the same
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
- --prior-sensitivity: the estimator kernels, the representers of each
  estimate as a functional of the true load in the absence of noise, against
  the target kernels. estimator_kernels_regions_all.png shows them per
  region for both methods, estimator_kernels_deg{l}_all.png per degree-l
  coefficient for the Bayesian estimate, and estimator_kernel_ratios.png the
  relative kernel errors ||k - t|| / ||t|| in the load space's norm; two
  tables of the norms.
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
from pyslfp.observations import WMBMethod

import common
import figures
from figures import COLORBAR, GRIDLINES, draw_region_boundaries, peak

OUTPUT_DIR = "output_plots_grace_inversion"
METRICS_FILE = os.path.join(OUTPUT_DIR, "grace_metrics.txt")

# The points of the covariance maps, (latitude, longitude), one per regime.
COVARIANCE_POINTS = [
    ((30.0, -45.0), "Ocean"),
    ((-5.0, -60.0), "Hydrology (Amazon)"),
    ((-75.0, -110.0), "Ice (WAIS)"),
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian inversion of synthetic GRACE data, against WMB."
    )
    common.add_grace_arguments(parser)
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
        help="Estimator kernels of the regional averages and the target degree.",
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
#                         Estimator kernels                        #
# ================================================================ #


def kernel_grid(state, rows, titles):
    """Kernel maps, one row per quantity and one column per kernel. `rows`
    holds (row label, panels), each panel a (field, colour limit, colour bar
    label)."""
    fig, axes = sl.subplots(
        len(rows),
        len(titles),
        figsize=(24, 5.5 * len(rows)),
        gridspec_kw={"hspace": 0.15, "wspace": 0.1},
        squeeze=False,
    )
    for i, (_, panels) in enumerate(rows):
        for j, (field, limit, label) in enumerate(panels):
            _, image = sl.plot(
                field,
                ax=axes[i, j],
                cmap="seismic",
                colorbar=True,
                symmetric=True,
                vmin=-limit,
                vmax=limit,
                colorbar_kwargs=COLORBAR,
                gridlines_kwargs=GRIDLINES,
            )
            image.colorbar.set_label(label, fontsize=14)
            draw_region_boundaries(state, axes[i, j], common.GRACE_REGIONS)
    figures.label_rows(axes, [row_label for row_label, _ in rows])
    figures.label_columns(axes, titles)
    return fig


def representers(index, *operators):
    """The representers in the load space of one component of each operator."""
    unit = operators[0].codomain.basis_vector(index)
    return [operator.adjoint(unit) for operator in operators]


def estimator_kernels(
    state,
    names,
    degree,
    components,
    resolution,
    forward,
    total_average,
    wmb_average,
    coefficients,
):
    """
    The estimator kernels: an estimate without noise is a functional of the
    true load, with representer k, to set against the representer t of the
    quantity it estimates. For the Bayesian estimate of B zeta, k - t is the
    representer of B (K A - I), K the Kalman operator and A the forward
    operator; for WMB's, of C A - B, C WMB's averaging of the data. The
    representers and norms are those of the load space; the maps are scaled
    by the target's largest absolute value.

    Returns the two tables, the two kernel figures and the ratio figure.
    """
    load_space = resolution.domain
    bayes_error = total_average @ resolution
    wmb_error = wmb_average @ forward - total_average

    lines = [
        "\n\n" + "=" * 115 + "\n",
        "REGIONAL ESTIMATOR KERNEL METRICS (PRIOR-FREE COMPARISON)\n",
        "-" * 115 + "\n",
        f"{'Region':<16} | {'Target Norm':<15} | {'Bayes Err Norm':<15} | "
        f"{'Bayes Ratio':<15} | {'WMB Err Norm':<15} | {'WMB Ratio':<15}\n",
        "-" * 115 + "\n",
    ]
    region_ratios, region_maps = [], []
    for i, name in enumerate(names):
        target, bayes, wmb = representers(i, total_average, bayes_error, wmb_error)
        target_norm, bayes_norm, wmb_norm = (
            load_space.norm(x) for x in (target, bayes, wmb)
        )
        bayes_ratio, wmb_ratio = bayes_norm / target_norm, wmb_norm / target_norm
        lines.append(
            f"{name:<16} | {target_norm:<15.4e} | {bayes_norm:<15.4e} | "
            f"{bayes_ratio:<15.4f} | {wmb_norm:<15.4e} | {wmb_ratio:<15.4f}\n"
        )
        region_ratios.append((name, bayes_ratio, wmb_ratio))
        scale = peak(target)
        bayes_pct, wmb_pct = bayes * (100.0 / scale), wmb * (100.0 / scale)
        limit = max(peak(bayes_pct), peak(wmb_pct))
        region_maps.append(
            (
                name,
                [
                    (target / scale, 1.0, "Relative Amplitude"),
                    (bayes_pct, limit, "Relative Error (%)"),
                    (wmb_pct, limit, "Relative Error (%)"),
                ],
            )
        )

    degree_error = coefficients @ resolution
    lines += [
        "\n\n" + "=" * 80 + "\n",
        f"DEGREE-{degree} ESTIMATOR KERNEL METRICS (BAYESIAN ONLY)\n",
        "-" * 80 + "\n",
        f"{'Component':<16} | {'Target Norm':<15} | {'Bayes Err Norm':<15} | "
        f"{'Bayes Ratio':<15}\n",
        "-" * 80 + "\n",
    ]
    degree_ratios, degree_maps = [], []
    for i, (name, _) in enumerate(components):
        target, bayes = representers(i, coefficients, degree_error)
        target_norm, bayes_norm = load_space.norm(target), load_space.norm(bayes)
        lines.append(
            f"{name:<16} | {target_norm:<15.4e} | {bayes_norm:<15.4e} | "
            f"{bayes_norm / target_norm:<15.4f}\n"
        )
        degree_ratios.append((name, bayes_norm / target_norm))
        scale = peak(target)
        bayes_pct = bayes * (100.0 / scale)
        degree_maps.append(
            (
                f"Component {name}",
                [
                    (target / scale, 1.0, "Relative Amplitude"),
                    ((target + bayes) / scale, 1.0, "Relative Amplitude"),
                    (bayes_pct, peak(bayes_pct), "Relative Error (%)"),
                ],
            )
        )

    kernel_figures = {
        "estimator_kernels_regions_all.png": kernel_grid(
            state,
            region_maps,
            ["Target Kernel", "Bayesian Kernel Error", "WMB Kernel Error"],
        ),
        f"estimator_kernels_deg{degree}_all.png": kernel_grid(
            state,
            degree_maps,
            ["Target Kernel", "Bayesian Kernel", "Kernel Error"],
        ),
        "estimator_kernel_ratios.png": figures.plot_kernel_error_ratios(
            region_ratios,
            degree_rows=degree_ratios,
            degree_label=f"degree {degree} (Bayesian only)",
        ),
    }
    return "".join(lines), kernel_figures


# ================================================================ #
#                               Maps                               #
# ================================================================ #


def posterior_maps(state, panels, std=None):
    """The load maps, row by row: the true load, WMB's estimate and the
    posterior mean, each a (field, title) in `panels`, on one colour scale
    set by the largest absolute value among them, then the pointwise posterior
    standard deviation `std` on a scale of its own; that panel is left empty
    without it. Each map has its own colour bar beneath it, so that the four
    are drawn at one size."""
    limit = max(peak(field) for field, _ in panels)
    fig, axes = sl.subplots(2, 2, figsize=(18, 12), gridspec_kw={"hspace": 0.10})
    maps = [
        (
            field,
            title,
            "Load (mm EWT)",
            {"vmin": -limit, "vmax": limit, "cmap": "seismic"},
        )
        for field, title in panels
    ]
    if std is not None:
        maps.append(
            (
                std,
                "Posterior Std. Deviation",
                "Std. deviation (mm EWT)",
                {"vmin": 0.0, "cmap": "Blues"},
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


def covariance_maps(prior_total, posterior_total, scale_mm):
    """The prior (left) and posterior (right) covariance of the total load
    with each of COVARIANCE_POINTS, in mm² EWT."""
    fig, axes = sl.subplots(3, 2, figsize=(16, 15), gridspec_kw={"hspace": 0.1})
    for row, (point, label) in enumerate(COVARIANCE_POINTS):
        for col, (measure, kind) in enumerate(
            [(prior_total, "Prior"), (posterior_total, "Posterior")]
        ):
            ax = axes[row, col]
            _, image = sl.plot(
                measure.two_point_covariance(point) * scale_mm**2,
                ax=ax,
                colorbar=True,
                cmap="seismic",
                symmetric=True,
                colorbar_kwargs=COLORBAR,
                gridlines_kwargs=GRIDLINES,
            )
            ax.set_title(f"{kind} Covariance: {label}", fontsize=16)
            sl.plot_points([point], ax=ax, color="black", gridlines=False, coasts=False)
            image.colorbar.set_label("Covariance (mm² EWT)", fontsize=14)
            image.colorbar.ax.tick_params(labelsize=12)
    return fig


def main():
    args = parse_arguments()
    rng = common.configure(args)
    common.start_run(OUTPUT_DIR, args, metrics_file=METRICS_FILE)
    figures.set_inversion_style()

    print(f"Building the physics (lmax={args.lmax})...")
    physics = common.grace_physics(args.lmax, args.load_order, args.load_scale_km)
    state, load_space, scale_mm = physics.state, physics.load_space, physics.scale_mm
    initial_prior, prior, noise_load, noise_info = common.grace_measures(
        physics, args, rng
    )
    names, average = common.regional_averaging(physics, args)

    forward = common.grace_forward_operator(physics, args.obs_degree)
    total_load = common.total_load_operator(physics)
    wmb = WMBMethod(state.model, args.obs_degree)
    wmb_load = wmb.potential_coefficient_to_load_operator(load_space)
    noise = wmb.load_measure_to_observation_measure(noise_load)

    print("Drawing a synthetic load and dataset...")
    problem = inf.LinearForwardProblem(forward, error=noise)
    model, data = problem.synthetic_model_and_data(prior, rng=rng)

    print("Solving for the posterior...")
    posterior, qoi_posterior, qoi_inversion = common.solve_posterior(
        inf.LinearGaussianInversion(problem, prior),
        data,
        wmb.bayesian_normal_operator_preconditioner(initial_prior, noise),
        rtol=0.01 * noise_info["std_factor"],
        qoi_rtol=args.qoi_rtol,
    )

    # The quantities of interest in mm EWT: the regional averages of the total
    # load, their WMB estimates from the data, and the coefficients of the
    # direct load at the target degree, in the order m = 0, 1, ..., l, then
    # -1, ..., -l.
    total_average = scale_mm * average @ total_load
    wmb_average = scale_mm * average @ wmb_load
    degree = args.target_degree
    coefficients = (scale_mm / load_space.radius) * load_space.coefficient_operator(
        lmax=degree, lmin=degree
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

    if args.prior_sensitivity:
        print("Estimator kernels...")
        resolution = qoi_inversion.kalman_operator @ forward - (
            inf.LinearOperator.identity(load_space)
        )
        table, kernel_figures = estimator_kernels(
            state,
            names,
            degree,
            components,
            resolution,
            forward,
            total_average,
            wmb_average,
            coefficients,
        )
        metrics.append(table)
        output.update(kernel_figures)

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
            (wmb_load(data), "WMB Estimate"),
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
        output["grace_covariance_maps.png"] = covariance_maps(
            prior.affine_map(total_load), posterior_total, scale_mm
        )

    common.save_figures(output, OUTPUT_DIR, dpi=300)


if __name__ == "__main__":
    main()
