"""
The errors of the GRACE estimators against the errors they state, on loads
that are not drawn from the prior (Heathcote et al., 2026).

Each estimator of a regional average of the total load comes with a
standard deviation. The Bayesian one's is the posterior's. WMB's is that of
its whole error under the prior: the noise it propagates and what its
filter, its truncation and its neglect of the sea level change do to a load
drawn from the prior (`common.wmb_error_operator`). The two estimators are
then given the same prior information, and both statements are right when
the true load is a sample of the prior. This script asks how the errors
actually made compare with those statements, as distributions, when it is
not. With --prior-truths it is one: the reference case. WMB's errors are
also set against the noise it propagates alone, the uncertainty usually
stated for it.

The experiment is that of `grace_inversion.py` with --load-prior model: one
unknown, the direct load, whose prior is that which the fields of the ocean
experiments give it, and data with the error covariance of the
ITSG-Grace2018 coefficients. The prior's amplitudes and scales are stated
numbers, taken from the first version of the ESA Earth System Model (1995
to 2006). The truths are the year-on-year changes of its version 3.0
(`esm_loads.py`, 2007 to 2020, --truth-years), which no prior has seen:
the water and the ice on the land, and the ocean less its mean over the
ocean, summed into a direct load, the sea level equation supplying the
ocean's total. The ocean's part is what is left once a model of the ocean
has been removed from the data, as is done before a GRACE gravity field is
solved for, and the prior's ocean term describes that residual, which the
inversion marginalises over. How much is left over a year is not known, so
--ocean-case brackets it (`common.OCEAN_CASES`): next to none, the error
series of the same Earth System Model; a quarter of the ocean's load; or
all of it, nothing being removed.

The regions are those of --region-groups, the named regions of
`EarthState.list_all_regions`: by default the drainage basins of Greenland
and of Antarctica and the IPCC's climate regions, land and ocean. The
quantity both estimators are asked for is the average of the total load
over each region itself, its mask unsmoothed (--smoothing-scale-km smooths
it, for both).

WMB's estimate is made as it is in practice. The coefficients are cut off
at --wmb-degree and filtered, and the result is averaged over the region.
The filter (--wmb-filter) is the Gaussian one of Wahr et al. (1998), of
half-width --wmb-radius-km, on the coefficients to degree 60; or the
decorrelating filter DDK of Kusche et al. (2009) at the strength
--wmb-ddk-level (`ddk_filter.py`), on all the coefficients. With --wmb-rescale the average is then
multiplied by the factor that returns the right answer for a load uniform
over the region and absent elsewhere, the scaling of Velicogna and Wahr
(2006). The filter and the factor are part of the estimator, so what they
cost in leakage counts as its error. Degree 1, which the data do not hold,
is left out of WMB's estimate, with no outside estimate put in its place.
The Bayesian estimate uses every coefficient with the weight the noise
covariance gives it, and has no such choices.

Both estimators are linear in the data, so each is assembled once as a
matrix from the data to the regional averages; with the normal matrix
factorised (`common.exact_preconditioner`) a row of the Bayesian one costs
a single solve. An error is then the estimator's error on the load, which
the truth fixes, plus the estimator applied to the noise; the second is
drawn --noise-draws times for each truth, at no cost in solves. Each error
is divided by the standard deviation its estimator states. Were the
statements right, and the errors Gaussian, the ratios would be standard
normal.

Outputs, in output_plots_grace_coverage/ at 200 dpi:

- coverage_metrics.txt: by class of region (ice sheets, land, ocean) and
  overall, for each estimator: the root mean square of the ratios, the
  fractions within one and two stated standard deviations, and the medians
  of the stated standard deviation and of the root mean square error; then
  the same region by region.
- coverage_histograms.png: the ratios of each estimator, by class, against
  the standard normal density.
- coverage_regions.png: region by region, the root mean square error
  against the stated standard deviation, for both estimators.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import pygeoinf as inf
import pyslfp as sl
from pyslfp.core import averaging_operator

import common
import esm_loads
import figures

OUTPUT_DIR = "output_plots_grace_coverage"
METRICS_FILE = os.path.join(OUTPUT_DIR, "coverage_metrics.txt")
DEFAULT_GROUPS = [
    "Mouginot (Greenland)",
    "IMBIE (Antarctica)",
    "AR6 (IPCC Climate Regions)",
]
CLASSES = ("ice sheets", "land", "ocean")
ESTIMATORS = ("WMB", "Bayesian")
COLOURS = {"WMB": "tab:red", "Bayesian": "tab:blue"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The GRACE estimators' errors against their stated "
        "standard deviations, over many regions, on loads of the ESA Earth "
        "System Model."
    )
    common.add_grace_arguments(parser)
    parser.set_defaults(
        load_prior="model", coefficient_errors=[2007, 2008], smoothing_scale_km=0.0
    )
    common.add_wmb_arguments(parser)
    parser.add_argument("--region-groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument(
        "--truth-years",
        nargs="+",
        type=int,
        default=None,
        help="The years of version 3.0 of the Earth System Model whose "
        "successive annual means give the true loads; all those fetched if "
        "omitted.",
    )
    parser.add_argument(
        "--truth-mean",
        choices=("kept", "ice-removed", "removed"),
        default="kept",
        help="A diagnostic of a prior of zero mean against loads that have "
        "one: take from each true load the mean over the true loads of the "
        "land ice (ice-removed) or of the whole load (removed). The errors "
        "are then those of a prior whose mean is that of the truths, with "
        "the stated standard deviations unchanged.",
    )
    parser.add_argument(
        "--prior-truths",
        type=int,
        default=0,
        metavar="N",
        help="The reference case: N true loads drawn from the prior, in place "
        "of those of the Earth System Model. Both estimators' ratios are "
        "then standard normal by construction, and WMB's against its noise "
        "alone show what that leaves out under the prior.",
    )
    parser.add_argument(
        "--noise-draws",
        type=int,
        default=50,
        help="Draws of the noise for each true load.",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Maps for one of the true loads: the load, the two estimates of "
        "it from one draw of the noise, and their errors.",
    )
    parser.add_argument(
        "--map-truth",
        type=int,
        nargs="+",
        default=[0],
        help="With --plot-maps: which of the true loads, counted from zero; "
        "a figure for each.",
    )
    parser.add_argument(
        "--std-samples",
        type=int,
        default=0,
        help="With --plot-maps: posterior samples for the pointwise standard "
        "deviation map.",
    )
    common.add_qoi_arguments(parser)
    common.add_parallel_arguments(parser)
    args = parser.parse_args()
    if args.load_prior != "model" or args.coefficient_errors is None:
        parser.error("the truths are loads of the model: --load-prior model.")
    return args


def true_loads(state, years, ocean="O", share=1.0, mean="kept"):
    """The direct loads of the year-on-year changes of the Earth System
    Model: the water and the ice on the land, and the ocean less its mean
    over the ocean, which moves mass about the ocean without changing its
    total. The ocean's part is the share `share` of the component `ocean`:
    O, the ocean's load, or AOe07, the error of the atmosphere and ocean
    models that are removed from GRACE's data, over the ocean. With `mean`
    "ice-removed" the mean over the changes of the land ice is taken from
    each, and with "removed" the mean of the whole load. Returns (labels,
    loads)."""
    model = state.model
    sea = state.ocean_function
    land = model.constant_grid(1.0) - sea
    components = {}
    for name in ("H", "I", ocean):
        labels, components[name] = esm_loads.load_changes(
            model, name, years=years, version=3
        )
    if mean == "ice-removed":
        ice_mean = sum(components["I"][1:], components["I"][0]) * (
            1.0 / len(components["I"])
        )
        components["I"] = [ice - ice_mean for ice in components["I"]]
    loads = []
    for water, ice, wet in zip(*(components[name] for name in ("H", "I", ocean))):
        ocean_mean = model.integrate(sea * wet) / model.integrate(sea)
        loads.append(
            land * (water + ice)
            + share * sea * (wet - ocean_mean * model.constant_grid(1.0))
        )
    if mean == "removed":
        load_mean = sum(loads[1:], loads[0]) * (1.0 / len(loads))
        loads = [load - load_mean for load in loads]
    return labels, loads


def truth_maps(state, label, truth, wmb, mean, std=None):
    """One true load and its two estimates from one draw of the noise, in mm
    EWT: above, the load, WMB's filtered estimate and the posterior mean;
    below, the pointwise posterior standard deviation (left empty without
    `std`) and the two estimates' errors. The load maps and the error maps
    share the colour scale of the truth and the posterior mean, at the value
    that all but a two-hundredth of the grid lies within."""

    def extent(field):
        return float(np.percentile(np.abs(field.data), 99.5))

    limit = max(extent(truth), extent(mean))
    loads = {"vmin": -limit, "vmax": limit, "cmap": "seismic"}
    panels = [
        (truth, f"True total load ({label})", "Load (mm EWT)", loads),
        (wmb, "WMB estimate (filtered)", "Load (mm EWT)", loads),
        (mean, "Posterior mean", "Load (mm EWT)", loads),
        (
            std,
            "Posterior std. deviation",
            "Std. deviation (mm EWT)",
            None
            if std is None
            else {"vmin": 0.0, "vmax": extent(std), "cmap": "Blues"},
        ),
        (wmb - truth, "WMB estimate less the truth", "Error (mm EWT)", loads),
        (mean - truth, "Posterior mean less the truth", "Error (mm EWT)", loads),
    ]
    fig, axes = sl.subplots(2, 3, figsize=(27, 12), gridspec_kw={"hspace": 0.10})
    for ax, (field, title, colour_label, style) in zip(axes.flat, panels):
        if field is None:
            ax.set_axis_off()
            continue
        _, image = sl.plot(
            field,
            ax=ax,
            colorbar=True,
            colorbar_kwargs=figures.COLORBAR,
            gridlines_kwargs=figures.GRIDLINES,
            **style,
        )
        ax.set_title(title, fontsize=16)
        image.colorbar.set_label(colour_label, fontsize=14)
    return fig


def regions_of(state, groups) -> tuple[list, list, list]:
    """The names, masks and classes of the regions of the named groups. A
    region of Greenland's or Antarctica's basins is of the ice sheets; any
    other is ocean or land by which covers more of it."""
    known = state.list_all_regions()
    model = state.model
    ocean = state.ocean_projection(value=0.0)
    names, masks, classes = [], [], []
    for group in groups:
        if group not in known:
            raise ValueError(f"'{group}' is not one of {sorted(known)}.")
        on_ice = "Greenland" in group or "Antarctica" in group
        for name in known[group]:
            mask = state.get_projection(name, value=0.0)
            area = model.integrate(mask)
            if area <= 0.0:
                continue
            wet = model.integrate(mask * ocean) / area
            names.append(name)
            masks.append(mask)
            classes.append("ice sheets" if on_ice else "ocean" if wet > 0.5 else "land")
    return names, masks, classes


def statistics(ratios, errors, stds) -> list[float]:
    """Of an estimator over a set of regions: the root mean square of the
    ratios, the fractions of them within one and within two, the median
    stated std and the median over regions of the root mean square error.
    `ratios` and `errors` are (cases, regions); `stds` is per region."""
    magnitude = np.abs(ratios)
    return [
        float(np.sqrt(np.mean(ratios**2))),
        float(np.mean(magnitude < 1.0)),
        float(np.mean(magnitude < 2.0)),
        float(np.median(stds)),
        float(np.median(np.sqrt(np.mean(errors**2, axis=0)))),
    ]


def histogram_figure(ratios: dict, classes: np.ndarray):
    """The ratios of error to stated standard deviation of each estimator,
    by class of region, against the standard normal density."""
    fig, axes = plt.subplots(
        len(ESTIMATORS),
        len(CLASSES),
        figsize=(5.5 * len(CLASSES), 4.2 * len(ESTIMATORS)),
        layout="constrained",
        squeeze=False,
    )
    x = np.linspace(-6.0, 6.0, 400)
    bins = np.linspace(-6.0, 6.0, 49)
    for i, estimator in enumerate(ESTIMATORS):
        for j, name in enumerate(CLASSES):
            ax = axes[i, j]
            values = ratios[estimator][:, classes == name].ravel()
            if values.size:
                outside = float(np.mean(np.abs(values) > 6.0))
                ax.hist(
                    np.clip(values, -6.0, 6.0),
                    bins=bins,
                    density=True,
                    color=COLOURS[estimator],
                    alpha=0.6,
                )
                ax.text(
                    0.03,
                    0.95,
                    f"rms {np.sqrt(np.mean(values**2)):.2f}\n"
                    f"beyond 6: {100.0 * outside:.1f}%",
                    transform=ax.transAxes,
                    va="top",
                )
            ax.plot(x, figures.normal_pdf(x, 0.0, 1.0), color="black", linewidth=1.5)
            ax.set_xlim(-6.0, 6.0)
            ax.set_title(f"{estimator}: {name}")
            if i == len(ESTIMATORS) - 1:
                ax.set_xlabel("Error / stated standard deviation")
            if j == 0:
                ax.set_ylabel("Density")
    return fig


def region_figure(errors: dict, stds: dict, classes: np.ndarray):
    """Region by region, the root mean square error of each estimator
    against the standard deviation it states."""
    fig, axes = plt.subplots(
        1, len(CLASSES), figsize=(5.5 * len(CLASSES), 5.0), layout="constrained"
    )
    for ax, name in zip(axes, CLASSES):
        chosen = classes == name
        low, high = np.inf, 0.0
        for estimator in ESTIMATORS:
            rms = np.sqrt(np.mean(errors[estimator][:, chosen] ** 2, axis=0))
            stated = stds[estimator][chosen]
            if not rms.size:
                continue
            ax.loglog(
                stated,
                rms,
                "o",
                color=COLOURS[estimator],
                alpha=0.7,
                label=estimator,
            )
            low = min(low, rms.min(), stated.min())
            high = max(high, rms.max(), stated.max())
        if np.isfinite(low):
            span = [0.7 * low, 1.4 * high]
            ax.plot(span, span, color="black", linewidth=1.0)
            ax.set_xlim(span)
            ax.set_ylim(span)
        ax.set_title(name.capitalize())
        ax.set_xlabel("Stated standard deviation (mm EWT)")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Root mean square error (mm EWT)")
    axes[0].legend(loc="upper left")
    return fig


def main():
    args = parse_arguments()
    rng = common.configure(args)
    common.start_run(OUTPUT_DIR, args, metrics_file=METRICS_FILE)
    figures.set_inversion_style()

    print(f"Building the physics, prior and noise (lmax={args.lmax})...")
    physics, _, prior, noise, noise_matrix, _ = common.grace_experiment(args, rng)
    state, model = physics.state, physics.state.model
    load_space, scale_mm = physics.load_space, physics.scale_mm
    km = 1000.0 / model.parameters.length_scale
    forward = common.grace_forward_operator(physics, args)
    total_load = common.total_load_operator(physics)

    if args.prior_truths > 0:
        print(f"Drawing {args.prior_truths} loads from the prior...")
        labels = [f"draw {i + 1}" for i in range(args.prior_truths)]
        loads = [prior.sample(rng=rng) for _ in labels]
    else:
        print("Reading the loads of the ESA Earth System Model, version 3.0...")
        _, _, ocean, share = common.OCEAN_CASES[args.ocean_case]
        years = args.truth_years or sorted(
            set.intersection(
                *(set(esm_loads.available(c, 3)) for c in ("H", "I", ocean))
            )
        )
        labels, loads = true_loads(state, years, ocean, share, args.truth_mean)
    totals = [total_load(zeta) for zeta in loads]
    clean_data = [forward(zeta) for zeta in loads]

    names, masks, classes = regions_of(state, args.region_groups)
    classes = np.array(classes)
    print(
        f"  {len(labels)} loads ({labels[0]} to {labels[-1]}) and {len(names)} "
        "regions: " + ", ".join(f"{np.sum(classes == c)} {c}" for c in CLASSES)
    )
    sharp_masks = masks
    if args.smoothing_scale_km > 0.0:
        smoother = load_space.heat_measure(args.smoothing_scale_km * km).covariance
        masks = [smoother(mask) for mask in masks]
    average = scale_mm * averaging_operator(state, load_space, masks)

    # WMB's estimator as it is made in practice (`common.wmb_estimator`).
    wmb_load, wmb_matrix, wmb_description = common.wmb_estimator(
        state,
        load_space,
        args,
        average=average,
        sharp_masks=sharp_masks,
        scale=scale_mm,
    )

    print(f"Assembling the normal matrix of {forward.codomain.dim} data...")
    preconditioner = common.exact_preconditioner(
        forward, prior, noise, noise_matrix=noise_matrix, n_jobs=common.jobs(args)
    )
    inversion = inf.LinearGaussianInversion(
        inf.LinearForwardProblem(forward, error=noise), prior
    ).with_solver(inf.CGSolver(rtol=args.qoi_rtol, preconditioner=preconditioner))
    # Each row is one solve with the factorised normal matrix; worker
    # processes would each be sent that matrix, for nothing.
    print("Assembling the estimators...")
    target = average @ total_load
    matrices = {
        "Bayesian": (target @ inversion.kalman_operator).matrix(
            form="components", by="rows"
        ),
        "WMB": wmb_matrix,
    }
    posterior = inf.GaussianMeasure(prior.domain, covariance=inversion.covariance)
    # WMB's error is its estimate of the noise-free data less the truth, a
    # linear function of the load, plus the noise it propagates.
    wmb_noise = np.einsum("ij,jk,ik->i", wmb_matrix, noise_matrix, wmb_matrix)
    wmb_average = inf.LinearOperator.from_matrix(
        forward.codomain, average.codomain, wmb_matrix, form="components"
    )
    wmb_leakage = common.variances(
        common.dense(
            prior,
            common.wmb_error_operator(physics, average, wmb_average, args),
            n_jobs=common.jobs(args, len(names)),
        )
    )
    stds = {
        "Bayesian": common.standard_deviations(common.dense(posterior, target)),
        "WMB": np.sqrt(wmb_noise + wmb_leakage),
    }
    noise_alone = np.sqrt(wmb_noise)

    # The errors: on each load, without noise, and with each of the draws.
    truth = np.array([average(sigma) for sigma in totals])
    clean = np.array(clean_data)
    factor = np.linalg.cholesky(noise_matrix)
    draws = (
        rng.standard_normal((len(labels) * args.noise_draws, len(factor))) @ factor.T
    ).reshape(len(labels), args.noise_draws, -1)
    errors, ratios, load_errors = {}, {}, {}
    for estimator, matrix in matrices.items():
        load_errors[estimator] = clean @ matrix.T - truth
        with_noise = load_errors[estimator][:, None, :] + draws @ matrix.T
        errors[estimator] = with_noise.reshape(-1, len(names))
        ratios[estimator] = errors[estimator] / stds[estimator]

    columns = ["rms ratio", "within 1", "within 2", "median std", "median rms"]
    width = 30 + 12 * len(columns)
    text = (
        "The GRACE estimators' errors against their stated standard deviations\n"
        + (
            f"{len(labels)} loads drawn from the prior (the ocean's part "
            if args.prior_truths > 0
            else f"{len(labels)} loads of the ESA Earth System Model, version 3.0 "
            f"({labels[0]} to {labels[-1]}; the ocean's part "
        )
        + {"removed": "removed", "partly": "a quarter left", "full": "in full"}[
            args.ocean_case
        ]
        + (
            ""
            if args.prior_truths > 0
            else {
                "kept": "",
                "ice-removed": "; less the mean of the land ice over the loads",
                "removed": "; less the mean over the loads",
            }[args.truth_mean]
        )
        + f"), {args.noise_draws} draws of the noise "
        f"for each, data to degree {args.obs_degree},\n"
        f"{len(names)} regions, their masks "
        + (
            f"smoothed over {args.smoothing_scale_km:.0f} km.\n"
            if args.smoothing_scale_km > 0.0
            else "unsmoothed.\n"
        )
        + wmb_description
        + "\n"
        "A ratio is an error over the standard deviation its estimator states; "
        "for a standard normal the\nroot mean square is 1 and the fractions "
        "within 1 and 2 are 0.683 and 0.954. Stds and errors in mm EWT.\n"
        "WMB's stated standard deviation is that of its whole error under the "
        "prior, noise and leakage; the\nrows 'WMB, noise alone' set the same "
        "errors against the noise it propagates.\n\n"
    )
    text += (
        f"{'':<30}" + "".join(f"{c:>12}" for c in columns) + "\n" + "-" * width + "\n"
    )
    for label in CLASSES + ("all",):
        chosen = np.ones(len(names), bool) if label == "all" else classes == label
        if not chosen.any():
            continue
        for estimator in ESTIMATORS:
            row = statistics(
                ratios[estimator][:, chosen],
                errors[estimator][:, chosen],
                stds[estimator][chosen],
            )
            text += f"{label + ', ' + estimator:<30}" + "".join(
                f"{v:>12.3f}" for v in row
            )
            text += "\n"
            if estimator == "WMB":
                row = statistics(
                    (errors["WMB"] / noise_alone)[:, chosen],
                    errors["WMB"][:, chosen],
                    noise_alone[chosen],
                )
                text += f"{label + ', WMB, noise alone':<30}" + "".join(
                    f"{v:>12.3f}" for v in row
                )
                text += "\n"
    text += "\n\nRegion by region: stated std, rms error, and rms error without noise\n"
    text += f"{'region':<26}{'class':<12}{'WMB noise std':>16}" + "".join(
        f"{e + ' ' + c:>16}" for e in ESTIMATORS for c in ("std", "rms", "load")
    )
    text += "\n"
    for i, name in enumerate(names):
        text += f"{name[:25]:<26}{classes[i]:<12}{noise_alone[i]:>16.3f}"
        for estimator in ESTIMATORS:
            values = (
                stds[estimator][i],
                np.sqrt(np.mean(errors[estimator][:, i] ** 2)),
                np.sqrt(np.mean(load_errors[estimator][:, i] ** 2)),
            )
            text += "".join(f"{v:>16.3f}" for v in values)
        text += "\n"
    common.write_metrics(METRICS_FILE, text)
    print(text[: text.index("Region by region")])
    print(f"saved {METRICS_FILE}")

    output = {
        "coverage_histograms.png": histogram_figure(ratios, classes),
        "coverage_regions.png": region_figure(errors, stds, classes),
    }
    for chosen in args.map_truth if args.plot_maps else []:
        # One of the loads, inverted from one draw of the noise as it would
        # be in practice: its misfit to the prior's prediction says how far
        # from the prior this load is.
        print(f"Maps for the load {labels[chosen]}...")
        data = forward.codomain.add(clean_data[chosen], noise.sample(rng=rng))
        estimate, _, _ = common.solve_posterior(
            inf.LinearGaussianInversion(
                inf.LinearForwardProblem(forward, error=noise), prior
            ),
            data,
            preconditioner,
            rtol=args.qoi_rtol,
            qoi_rtol=args.qoi_rtol,
        )
        std = None
        if args.std_samples > 0:
            std = estimate.affine_map(total_load).sample_pointwise_std(
                args.std_samples, rng=rng, n_jobs=common.jobs(args)
            )
            std = std * scale_mm
        output[f"coverage_maps_{labels[chosen]}.png"] = truth_maps(
            state,
            labels[chosen],
            totals[chosen] * scale_mm,
            wmb_load(data) * scale_mm,
            total_load(estimate.expectation) * scale_mm,
            std,
        )
    common.save_figures(output, OUTPUT_DIR, dpi=200)


if __name__ == "__main__":
    main()
