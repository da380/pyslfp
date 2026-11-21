"""
The errors of the GRACE estimators against the errors they state, over many
regions and loads (Heathcote et al., 2026).

Each estimator of a regional average of the total load comes with a
standard deviation: WMB's is the noise it propagates, and the Bayesian
one's is the posterior's. This script asks how the errors actually made
compare with those statements, as distributions.

The truths are the year-on-year changes of the ESA Earth System Model
(`esm_loads.py`), each built into a gravitationally self-consistent load as
in `grace_heldout.py`. The prior is the three-field prior from stated
numbers (`common.stated_budget_prior`): it is not fitted to the truths
here, though its amplitudes and scales are representative of them, and no
truth is a sample of it. The noise is that of `common.grace_noise_model`.

The regions are those of --region-groups, the named regions of
`EarthState.list_all_regions`: by default the drainage basins of Greenland
and of Antarctica and the IPCC's climate regions, land and ocean. Their
masks are smoothed over --smoothing-scale-km.

Both estimators are linear in the data, so each is assembled once as a
matrix from the data to the regional averages. An error is then the
estimator's error on the load, which the truth fixes, plus the estimator
applied to the noise; the second is drawn --noise-draws times for each
truth, at no cost in solves. Each error is divided by the standard
deviation its estimator states. Were the statements right, and the errors
Gaussian, the ratios would be standard normal.

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
        "standard deviations, over many regions and loads."
    )
    parser.add_argument("--lmax", type=int, default=256)
    parser.add_argument("--obs-degree", type=int, default=60)
    parser.add_argument(
        "--grace-convention", choices=("stokes", "orthonormal"), default="stokes"
    )
    parser.add_argument("--region-groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument(
        "--smoothing-scale-km",
        type=float,
        default=300.0,
        help="Smoothing of the region masks; zero leaves them sharp.",
    )
    parser.add_argument("--prior-order", type=float, default=common.BUDGET_PRIOR_ORDER)
    parser.add_argument(
        "--noise-draws",
        type=int,
        default=50,
        help="Draws of the noise for each true load.",
    )
    common.add_grace_noise_model_arguments(parser)
    common.add_qoi_arguments(parser)
    common.add_parallel_arguments(parser)
    return parser.parse_args()


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

    print(f"Building the physics (lmax={args.lmax})...")
    physics = common.budget_physics(args.lmax)
    state, model = physics.state, physics.state.model
    load_space, scale_mm = physics.load_space, physics.scale_mm
    km = 1000.0 / model.parameters.length_scale

    prior, fields, info = common.stated_budget_prior(physics, order=args.prior_order)
    for name, values in info.items():
        print(
            f"  prior of {name}: std {values['load_std_mm']:.0f} mm EWT, scale "
            f"{values['scale_km']:.0f} km, order {values['order']:.2f}"
        )

    wmb = common.wmb_method(state, args)
    noise, invariant_noise, _ = common.grace_noise_model(
        state, load_space, physics.response_space, args
    )
    wmb_load = wmb.potential_coefficient_to_load_operator(load_space)
    observe = common.grace_coefficients(
        physics.response_space, model, args.obs_degree, args.grace_convention
    )

    print("Reading the loads of the ESA Earth System Model...")
    components = {}
    for key, name in [("water", "H"), ("ice", "I"), ("ocean", "O")]:
        labels, components[key] = esm_loads.load_changes(model, name)
    totals, clean_data = [], []
    for k in range(len(labels)):
        water, ice, ocean = (components[key][k] for key in ("water", "ice", "ocean"))
        zeta = sum(common.field_loads(physics, water, ice, ocean).values())
        totals.append(physics.load_to_total_load(zeta))
        clean_data.append(observe(physics.fingerprint(zeta)))

    names, masks, classes = regions_of(state, args.region_groups)
    classes = np.array(classes)
    print(
        f"  {len(labels)} loads and {len(names)} regions: "
        + ", ".join(f"{np.sum(classes == c)} {c}" for c in CLASSES)
    )
    if args.smoothing_scale_km > 0.0:
        smoother = load_space.heat_measure(args.smoothing_scale_km * km).covariance
        masks = [smoother(mask) for mask in masks]
    average = scale_mm * averaging_operator(state, load_space, masks)

    print("Assembling the estimators...")
    forward = physics.forward(args)
    if args.coefficient_errors is None:
        preconditioner = wmb.bayesian_normal_operator_preconditioner(
            common.invariant_load_measure(physics, fields), invariant_noise
        )
    else:
        print(f"  the normal matrix of {forward.codomain.dim} data...")
        preconditioner = common.exact_preconditioner(
            forward, prior, noise, n_jobs=common.jobs(args)
        )
    inversion = inf.LinearGaussianInversion(
        inf.LinearForwardProblem(forward, error=noise), prior
    ).with_solver(inf.CGSolver(rtol=args.qoi_rtol, preconditioner=preconditioner))
    target = average @ physics.total_load
    workers = common.jobs(args, len(names))
    matrices = {
        "Bayesian": (target @ inversion.kalman_operator).matrix(
            form="components", by="rows", n_jobs=workers
        ),
        "WMB": (average @ wmb_load).matrix(form="components", by="rows"),
    }
    posterior = inf.GaussianMeasure(prior.domain, covariance=inversion.covariance)
    stds = {
        "Bayesian": common.standard_deviations(
            common.dense(posterior, target, n_jobs=workers)
        ),
        "WMB": common.standard_deviations(common.dense(noise, average @ wmb_load)),
    }

    # The errors: on each load, without noise, and with each of the draws.
    truth = np.array([average(sigma) for sigma in totals])
    clean = np.array(clean_data)
    draws = np.array(
        [noise.sample(rng=rng) for _ in range(len(labels) * args.noise_draws)]
    ).reshape(len(labels), args.noise_draws, -1)
    errors, ratios, load_errors = {}, {}, {}
    for estimator, matrix in matrices.items():
        load_errors[estimator] = clean @ matrix.T - truth
        with_noise = load_errors[estimator][:, None, :] + draws @ matrix.T
        errors[estimator] = with_noise.reshape(-1, len(names))
        ratios[estimator] = errors[estimator] / stds[estimator]

    columns = ["rms ratio", "within 1", "within 2", "median std", "median rms"]
    width = 24 + 12 * len(columns)
    text = (
        "The GRACE estimators' errors against their stated standard deviations\n"
        f"{len(labels)} loads, {args.noise_draws} draws of the noise for each, "
        f"{len(names)} regions smoothed over {args.smoothing_scale_km:.0f} km.\n"
        "A ratio is an error over the standard deviation its estimator states; "
        "for a standard normal the\nroot mean square is 1 and the fractions "
        "within 1 and 2 are 0.683 and 0.954. Stds and errors in mm EWT.\n\n"
    )
    text += (
        f"{'':<24}" + "".join(f"{c:>12}" for c in columns) + "\n" + "-" * width + "\n"
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
            text += f"{label + ', ' + estimator:<24}" + "".join(
                f"{v:>12.3f}" for v in row
            )
            text += "\n"
    text += "\n\nRegion by region: stated std, rms error, and rms error without noise\n"
    text += f"{'region':<26}{'class':<12}" + "".join(
        f"{e + ' ' + c:>16}" for e in ESTIMATORS for c in ("std", "rms", "load")
    )
    text += "\n"
    for i, name in enumerate(names):
        text += f"{name[:25]:<26}{classes[i]:<12}"
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

    common.save_figures(
        {
            "coverage_histograms.png": histogram_figure(ratios, classes),
            "coverage_regions.png": region_figure(errors, stds, classes),
        },
        OUTPUT_DIR,
        dpi=200,
    )


if __name__ == "__main__":
    main()
