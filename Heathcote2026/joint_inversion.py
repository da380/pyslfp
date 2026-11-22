"""
The joint experiment of Heathcote et al. (2026): satellite altimetry and
GRACE gravimetry, alone and together.

The model is the set-up `pyslfp.models.ice_ocean_and_density` that
common.py builds: an ice thickness change over the ice sheets, the
sterodynamic sea level change eta and a vertically averaged ocean density
change delta rho. The fields are global, and the prior is conditioned exactly
on the conservation of the ocean's mass. One truth and one noise realisation
are drawn, and the model is inverted three ways: from the altimetry along
the tracks alone, from GRACE's potential coefficients alone, and from both.
Altimetry sees the geocentric sea level change, the response to every load
plus eta itself; GRACE sees the potential of the response to the total load.
The data-space normal equations are solved by conjugate gradients,
preconditioned by the Woodbury inverse of a cheap surrogate problem.

Each posterior expectation, and the samples behind the pointwise standard
deviations, are solved at a relative residual of 0.01 times the smaller of
the two noise/signal std factors. Everything computed from the posterior
covariances (the quantities of interest and the covariance maps) goes
through solves at --qoi-rtol instead, with the same preconditioners. A
posterior variance is the prior variance less a correction computed through
the solves, so its relative error is about the tolerance divided by the
ratio of the posterior variance to the prior; GRACE reduces the barystatic
variances by over 99 per cent, and at the expectation's tolerance the joint
barystatic variance would come out above the GRACE-only one
(`common.solve_posterior`).

The three posteriors are compared with the prior and with each other on

- the global mean sea level change (GMSL), with the simple average of the
  altimetry data alongside (--plot-pdfs);
- its split into barystatic and steric parts (--plot-gmsl-split), with a
  check of the sea level equation's mass balance and of the mass constraint
  on the truth;
- the Tasman Sea mean sea level change, split into steric, dynamic
  manometric and barystatic manometric parts (--plot-regions);
- maps of the posterior expectation and pointwise standard deviation, and of
  the covariance with the ice thickness change in West Antarctica and with
  eta in the North Atlantic (--plot-maps).

Outputs, in output_plots_joint_inversion/ at 300 dpi:

- joint_metrics.txt: the variance reductions and KL divergences of each
  case, with the checks on the truth; with --prior-sensitivity, the tables
  of `common.ocean_prior_sensitivity` for each case;
- gmslr_pdf.png;
- gmslr_split_corner_{altimetry,grace,joint}.png, gmslr_split_ellipses.png;
- posterior_maps_joint.png, and posterior_maps_{altimetry,grace}.png with
  --map-all-cases;
- covariance_comparison_Ice_WAIS.png,
  covariance_comparison_Sterodynamic_North_Atlantic.png;
- regional_corner_{altimetry,grace,joint}.png, regional_variance_dots.png.

The defaults are the paper's resolution; `FAST=1 ./run_all.sh
joint_inversion` runs at lmax 64, observation degree 32, with the altimetry
points 8 degrees apart.
"""

import argparse
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pygeoinf as inf
from pyslfp.observations import altimetry_averaging_operator

import common
import figures
from common import OCEAN_REGION
from figures import peak

OUTPUT_DIR = "output_plots_joint_inversion"
METRICS_FILE = os.path.join(OUTPUT_DIR, "joint_metrics.txt")

# The three inversions, in the order they are solved and tabulated: the
# estimator's name in the tables, the prefix of its table columns, and the
# title of its map column.
CASES = {
    "altimetry": ("Alt-Only", "Alt", "Altimetry-Only Posterior"),
    "grace": ("GRACE-Only", "GRACE", "GRACE-Only Posterior"),
    "joint": ("Joint", "Joint", "Joint Posterior"),
}


# ================================================================ #
#                         Command line                             #
# ================================================================ #


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint Bayesian inversion (ice, eta, delta rho) of altimetry "
        "and GRACE, against each sensor alone."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="All the outputs below except --map-all-cases.",
    )
    parser.add_argument(
        "--plot-pdfs", action="store_true", help="The GMSL densities of each case."
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Maps of the posterior and of covariances (the joint posterior only "
        "unless --map-all-cases).",
    )
    parser.add_argument(
        "--map-all-cases",
        action="store_true",
        help="Map the posteriors of all three cases; implies --plot-maps.",
    )
    parser.add_argument(
        "--plot-regions",
        action="store_true",
        help="The three-way decomposition of the Tasman Sea mean sea level change.",
    )
    parser.add_argument(
        "--plot-gmsl-split",
        action="store_true",
        help="The barystatic/steric split of the GMSL change.",
    )
    parser.add_argument(
        "--prior-sensitivity",
        action="store_true",
        help="The sensitivity of the posterior variances of the GMSL change, "
        "its split and the regional decomposition to the prior of each "
        "field, for each case (`common.prior_sensitivity`); one more solve "
        "for each quantity and case.",
    )
    parser.add_argument(
        "--std-samples",
        type=int,
        default=0,
        help="Samples for the pointwise standard deviation of each mapped posterior.",
    )
    parser.add_argument(
        "--truth-years",
        nargs=2,
        type=int,
        default=None,
        help="Take the true model from the change between these two years "
        "of the ESA Earth System Model and the EN4 analyses "
        "(`common.model_truth`), not from the prior. Needs --ocean-prior "
        "steric.",
    )
    common.add_ocean_arguments(parser, surrogate=True)
    common.add_altimetry_noise_arguments(parser, prefix="alt-")
    common.add_joint_grace_noise_arguments(parser)
    common.add_gia_arguments(parser)
    common.add_preconditioner_arguments(parser)
    common.add_qoi_arguments(parser)
    common.add_parallel_arguments(parser)
    args = parser.parse_args()
    if args.all:
        args.plot_pdfs = args.plot_maps = args.plot_regions = True
        args.plot_gmsl_split = args.prior_sensitivity = True
    if args.map_all_cases:
        args.plot_maps = True
    return args


def append_metrics(lines: list[str]) -> None:
    common.write_metrics(METRICS_FILE, "\n".join(lines) + "\n")


# ================================================================ #
#                           Figures                                #
# ================================================================ #


def zoom_densities(
    ax,
    measures,
    /,
    *,
    exclude=(),
    n_std: float = 4.0,
    true_value: float | None = None,
    pad_frac: float = 0.05,
    inset_bounds=(0.04, 0.50, 0.32, 0.45),
    inset_label: str = "full range",
    **plot_kwargs,
):
    """
    Zooms the x-axis of a `plot_densities` plot of scalar Gaussian measures
    onto those not in `exclude`, to `n_std` standard deviations about their
    means and the true value, and draws the full range in an inset. A much
    wider measure would otherwise flatten the narrow ones.
    """
    stats = [
        (float(np.asarray(m.expectation)[0]), float(common.standard_deviations(m)[0]))
        for m in measures
    ]
    kept = [s for i, s in enumerate(stats) if i not in set(exclude) and s[1] > 0]
    if not kept:
        raise ValueError("No measure with a positive standard deviation to zoom on.")
    lo = min(mu - n_std * sd for mu, sd in kept)
    hi = max(mu + n_std * sd for mu, sd in kept)
    if true_value is not None:
        lo, hi = min(lo, true_value), max(hi, true_value)
    pad = pad_frac * (hi - lo)
    lo, hi = lo - pad, hi + pad

    full_xlim = ax.get_xlim()
    ax.set_xlim(lo, hi)
    visible = [sd for mu, sd in stats if sd > 0 and lo <= mu <= hi]
    if visible:
        ax.set_ylim(0.0, 1.05 / (min(visible) * np.sqrt(2.0 * np.pi)))

    inset = ax.inset_axes(inset_bounds)
    inf.plot_densities(
        list(measures), ax=inset, truth=true_value, title="", xlabel="", **plot_kwargs
    )
    if inset.get_legend() is not None:
        inset.get_legend().remove()
    inset.set_ylabel("")
    inset.set_xlim(*full_xlim)
    inset.tick_params(labelsize=8, labelleft=False, left=False)
    inset.set_title(inset_label, fontsize=9, pad=3)
    y0, y1 = inset.get_ylim()
    inset.indicate_inset(
        (lo, y0, hi - lo, y1 - y0), edgecolor="0.3", alpha=0.8, linewidth=1.0
    )


# ================================================================ #
#                         Diagnostics                              #
# ================================================================ #

# The measures below are Gaussian with assembled covariances, so the KL
# divergences and the plots are exact; the generator is passed for any route
# that would sample instead.


def gmsl_densities(physics, prior, posteriors, true_model, alt_noise, alt_data, rng):
    """The GMSL of the prior and of each posterior, with the simple average
    of the altimetry data, whose spread is that of the noise alone."""
    print("\nPlotting GMSLR PDFs...")
    gmsl = physics.setup.gmsl(physics.fingerprint) * physics.scale_mm
    average = altimetry_averaging_operator(physics.points) * physics.scale_mm

    prior_gmsl = common.dense(prior, gmsl)
    simple = common.dense(alt_noise.translate(alt_data), average)
    pushed = {case: common.dense(post, gmsl) for case, post in posteriors.items()}
    true_value = gmsl(true_model)[0]

    prior_var = common.variances(prior_gmsl)[0]
    lines = [
        f"{'Target':<12} | {'Estimator':<12} | {'KL Div':<10} | {'Prior Var':<12} | "
        f"{'Post Var':<12} | {'Reduction'}",
        "-" * 80,
    ]
    for case, (name, _, _) in CASES.items():
        var = common.variances(pushed[case])[0]
        lines.append(
            f"{'GMSLR':<12} | {name:<12} | "
            f"{pushed[case].kl_divergence(prior_gmsl, rng=rng):<10.4f} | "
            f"{prior_var:<12.4f} | {var:<12.4f} | "
            f"{common.reduction(prior_var, var):>6.2f}%"
        )
    append_metrics(lines)

    measures = [simple, *pushed.values()]
    labels = ["Simple averaging", "Altimetry-Only", "GRACE-Only", "Joint"]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    inf.plot_densities(
        measures,
        truth=true_value,
        ax=ax,
        title="",
        xlabel="Global Mean Sea Level Rise (mm)",
        labels=labels,
        rng=rng,
    )
    zoom_densities(
        ax,
        measures,
        exclude=(labels.index("GRACE-Only"),),
        true_value=true_value,
        labels=labels,
        rng=rng,
    )
    return {"gmslr_pdf.png": fig}


def gmsl_split(physics, prior, posteriors, true_model, args, rng):
    """The barystatic and steric GMSL changes of the prior and each
    posterior, with the checks of `common.gmsl_checks` on the truth."""
    print("\nPushing forward onto the 2D GMSLR split (barystatic, steric)...")
    split = physics.setup.gmsl_split * physics.scale_mm
    n_jobs = common.jobs(args, 2)
    prior_split = common.dense(prior, split, n_jobs=n_jobs)
    splits = {
        case: common.dense(post, split, n_jobs=n_jobs)
        for case, post in posteriors.items()
    }
    true_split = split(true_model)

    prior_cov = common.covariance_matrix(prior_split)
    covs = [common.covariance_matrix(s) for s in splits.values()]
    bar = "-" * 125
    lines = [
        "\n\nGMSLR Split: Barystatic vs Steric",
        "=" * 125,
        " | ".join(
            [f"{'Metric':<22}", f"{'Prior':<12}"]
            + [f"{name:<12}" for name, _, _ in CASES.values()]
        ),
        bar,
        " | ".join(
            [f"{'KL Div vs prior':<22}", f"{'-':<12}"]
            + [
                f"{s.kl_divergence(prior_split, rng=rng):<12.4f}"
                for s in splits.values()
            ]
        ),
        " | ".join(
            [f"{'Barystatic-Steric corr':<22}"]
            + [f"{common.correlation(c):<+12.4f}" for c in [prior_cov, *covs]]
        ),
        *common.gmsl_checks(physics, true_model),
        bar,
        common.comparison_header(column for _, column, _ in CASES.values()),
        bar,
    ]
    for i, name in enumerate(["Barystatic", "Steric"]):
        lines.append(
            common.comparison_row(name, prior_cov[i, i], [c[i, i] for c in covs])
        )
    lines.append(bar)
    append_metrics(lines)

    labels = ["Barystatic SLR (mm)", "Steric SLR (mm)"]
    figs = {
        f"gmslr_split_corner_{case}.png": figures.corner_figure(
            s, prior=prior_split, truth=true_split, labels=labels, fill=False, rng=rng
        )
        for case, s in splits.items()
    }
    figs["gmslr_split_ellipses.png"] = figures.plot_gmsl_split_ellipses(
        prior_split, *splits.values(), true_values=true_split
    )
    return figs


def state_maps(physics, posteriors, true_model, args, rng, boundaries):
    """The true state, the posterior expectations and, with --std-samples,
    the pointwise standard deviations, for the joint posterior or, with
    --map-all-cases, all three. The samples are drawn from the posteriors
    solved at the expectation's tolerance."""
    print("\nGenerating 3-component spatial maps...")
    mapped = ["joint"] + (["altimetry", "grace"] if args.map_all_cases else [])
    true_fields = physics.display(true_model)
    expectations = {
        case: physics.display(posteriors[case].expectation) for case in mapped
    }
    limits = {
        name: max(
            peak(true_fields[name]), *(peak(e[name]) for e in expectations.values())
        )
        for name in true_fields
    }

    stds = dict.fromkeys(mapped)
    std_limits = None
    if args.std_samples > 0:
        for case in mapped:
            print(
                f"  Computing pointwise standard deviation from "
                f"{args.std_samples} {CASES[case][2]} samples..."
            )
            stds[case] = physics.display(
                common.pointwise_std(
                    posteriors[case],
                    args.std_samples,
                    rng=rng,
                    n_jobs=common.jobs(args),
                )
            )
        std_limits = {
            name: max(float(np.max(s[name].data)) for s in stds.values())
            for name in true_fields
        }

    figs = {}
    for case in mapped:
        title = CASES[case][2]
        print(f"  Rendering {title} maps...")
        figs[f"posterior_maps_{case}.png"] = figures.state_maps(
            true_fields,
            expectations[case],
            stds[case],
            limits=limits,
            std_limits=std_limits,
            figsize=(14 if stds[case] is None else 22, 15),
            mean_title=title,
            row_titles=True,
            state=physics.state,
            boundaries=boundaries,
        )
    return figs


def covariance_maps(physics, prior, posteriors, boundaries):
    """The covariance of the prior and of each posterior with the ice
    thickness change at a point of West Antarctica and with eta at a point
    of the North Atlantic (`common.OCEAN_COVARIANCE_POINTS`)."""
    cases = [
        ("Prior", prior),
        ("GRACE-Only Posterior", posteriors["grace"]),
        ("Altimetry-Only Posterior", posteriors["altimetry"]),
        ("Joint Posterior", posteriors["joint"]),
    ]
    print(
        "\nGenerating Point-wise Covariance Maps "
        f"({' vs '.join(title for title, _ in cases)})..."
    )
    figs = {}
    for field_name, component, point, point_name in common.OCEAN_COVARIANCE_POINTS:
        print(f"  Evaluating perturbation in {field_name} at {point}...")
        columns = [
            (title, physics.point_covariance(measure, component, point))
            for title, measure in cases
        ]
        figs[f"covariance_comparison_{field_name}_{point_name}.png"] = (
            figures.covariance_maps(
                columns,
                component,
                point,
                wspace=0.1,
                state=physics.state,
                boundaries=boundaries,
            )
        )
    return figs


def regional_decomposition(physics, prior, posteriors, true_model, args, rng):
    """The Tasman Sea mean sea level change of the prior and each posterior,
    split into steric, dynamic manometric and barystatic manometric parts."""
    print("\nDecomposing Regional Sea Level Signals (3-way)...")
    decomposition = physics.regional_decomposition(OCEAN_REGION)
    n_jobs = common.jobs(args, 3)

    true_values = decomposition(true_model)
    print("  -> Prior push-forward (3 covariance actions)...")
    prior_dec = common.dense(prior, decomposition, n_jobs=n_jobs)
    decs = {}
    for case, post in posteriors.items():
        print(f"  -> {CASES[case][0]} push-forward (3 posterior solves)...")
        decs[case] = common.dense(post, decomposition, n_jobs=n_jobs)

    prior_cov = common.covariance_matrix(prior_dec)
    covs = [common.covariance_matrix(d) for d in decs.values()]
    kls = [d.kl_divergence(prior_dec, rng=rng) for d in decs.values()]
    names = ["SSLC", "DMSLC", "BMSLC"]
    bar = "-" * 135
    lines = [
        f"\n\nRegional Signal Separation ({OCEAN_REGION})",
        common.REGIONAL_NOTE,
        "=" * 135,
        f"{'Metric':<22} | {'Prior':<12} | {'Alt-Only':<12} | {'Alt Red%':<10} | "
        f"{'GRACE-Only':<12} | {'GRACE Red%':<10} | {'Joint':<12} | {'Joint Red%'}",
        bar,
        f"{'Joint KL Div (nats)':<22} | {'-':<12} | {kls[0]:<12.4f} | {'-':<10} | "
        f"{kls[1]:<12.4f} | {'-':<10} | {kls[2]:<12.4f} | {'-'}",
        common.comparison_row(
            "Total Var (Trace) mm²", np.trace(prior_cov), [np.trace(c) for c in covs]
        ),
        common.comparison_row(
            "Generalised Var (Det)",
            np.linalg.det(prior_cov),
            [np.linalg.det(c) for c in covs],
            fmt=".4e",
        ),
        bar,
        common.comparison_header(column for _, column, _ in CASES.values()),
        bar,
    ]
    for i, name in enumerate(names):
        lines.append(
            common.comparison_row(name, prior_cov[i, i], [c[i, i] for c in covs])
        )
    lines.append(bar)
    append_metrics(lines)

    labels = ["SSLC (mm)", "DMSLC (mm)", "BMSLC (mm)"]
    figs = {
        f"regional_corner_{case}.png": figures.corner_figure(
            d, prior=prior_dec, truth=true_values, labels=labels, fill=False, rng=rng
        )
        for case, d in decs.items()
    }
    figs["regional_variance_dots.png"] = figures.plot_component_variance_dots(
        names, np.diag(prior_cov), *(np.diag(c) for c in covs)
    )
    return figs


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
        heading="Joint Inversion (Alt + GRACE) Metrics\n"
        "======================================\n\n",
    )

    # ------------------ The exact problem ------------------
    print(f"\nBuilding the exact physics, prior and noise (lmax={args.lmax})...")
    physics = common.ocean_physics(args)
    print(f"Generated {len(physics.points)} ocean altimetry observation points.")
    prior, calib = common.ocean_prior(physics, args, rng)
    stds = calib["derived_stds"]

    # The altimetry noise is a local part and a large-scale correlated part;
    # the preconditioners see the local part alone.
    alt_parts = {}
    alt_noise, alt_local_noise = common.altimetry_noise(
        physics,
        args.prior_order,
        calib=calib,
        parts=alt_parts,
        **common.altimetry_noise_options(args, "alt_"),
    )
    grace_noise, grace_info = common.joint_grace_noise(
        physics, stds["ice_std"], args.prior_order, args
    )
    common.print_calibration_report(calib, physics)
    common.print_prior_settings(
        args, args.alt_noise_corr_std_mm, args.alt_noise_corr_scale_km
    )

    altimetry = physics.altimetry
    grace = physics.grace(args)
    # The column of the two, with the sea level equation solved once.
    joint = physics.joint(args)
    # The error of the GIA correction is one model's error in both data
    # sets, so in the joint problem it is correlated between them. The
    # surrogate preconditioner does not see it; the blocks correct for it.
    joint_noise = inf.GaussianMeasure.from_product(
        [alt_noise, grace_noise], domain=joint.codomain
    )
    grace_total = grace_noise
    gia = {case: None for case in CASES}
    if args.gia_years > 0.0:
        gia = {
            "altimetry": common.ocean_gia_error(physics, args),
            "grace": common.ocean_gia_error(physics, args, altimetry=False, grace=True),
            "joint": common.ocean_gia_error(physics, args, grace=True),
        }
        alt_noise = alt_noise + gia["altimetry"]
        grace_total = grace_noise + gia["grace"]
        joint_noise = joint_noise + gia["joint"]
    problems = {
        "altimetry": inf.LinearForwardProblem(altimetry, error=alt_noise),
        "grace": inf.LinearForwardProblem(grace, error=grace_total),
        "joint": inf.LinearForwardProblem(joint, error=joint_noise),
    }

    if args.truth_years is None:
        print("\nDrawing the synthetic truth and data...")
        true_model, joint_data = problems["joint"].synthetic_model_and_data(
            prior, rng=rng
        )
    else:
        print(
            f"\nThe truth is the change from {args.truth_years[0]} to "
            f"{args.truth_years[1]} of the models; drawing the noise..."
        )
        true_model = common.model_truth(physics, args, args.truth_years)
        joint_data = joint.codomain.add(joint(true_model), joint_noise.sample(rng=rng))
    data = {"altimetry": joint_data[0], "grace": joint_data[1], "joint": joint_data}

    # ------------------ The preconditioners ------------------
    surrogates, blocks = {}, {}
    if args.preconditioner == "surrogate":
        surrogate, surrogate_prior = common.ocean_surrogate(args, physics, calib)
        surrogate_altimetry = surrogate.altimetry
        surrogate_grace = surrogate.grace(args)
        surrogates = {
            "altimetry": (surrogate_altimetry, alt_local_noise),
            "grace": (surrogate_grace, grace_noise),
            "joint": (
                inf.ColumnLinearOperator([surrogate_altimetry, surrogate_grace]),
                inf.GaussianMeasure.from_product(
                    [alt_local_noise, grace_noise], domain=joint.codomain
                ),
            ),
        }

    # ------------------ Posteriors ------------------
    print("\nSolving for Posteriors...")
    rtol = 0.01 * min(
        1.0,
        args.alt_noise_std_mm / common.sea_level_std_mm(args),
        grace_info["std_factor"],
    )
    posteriors, qoi_posteriors, qoi_inversions = {}, {}, {}
    for case, (name, _, _) in CASES.items():
        print(f" -> Solving {name}...")
        inversion = inf.LinearGaussianInversion(problems[case], prior)
        if args.preconditioner == "blocks":
            preconditioner = common.block_preconditioner(
                case,
                physics,
                args,
                prior,
                calib,
                alt_local=alt_local_noise,
                alt_parts=alt_parts,
                grace_noise=grace_noise,
                grace_matrix=grace_info.get("covariance"),
                gia=gia[case],
                rng=rng,
                cache=blocks,
            )
        else:
            forward, noise = surrogates[case]
            preconditioner = common.surrogate_preconditioner(
                inversion.normal_operator,
                forward,
                surrogate_prior,
                noise,
                max_jobs=common.jobs(args),
            )
        posteriors[case], qoi_posteriors[case], qoi_inversions[case] = (
            common.solve_posterior(
                inversion, data[case], preconditioner, rtol=rtol, qoi_rtol=args.qoi_rtol
            )
        )

    # ------------------ Diagnostics ------------------
    boundaries = [OCEAN_REGION] if args.plot_regions else None
    figs = {}
    if args.plot_pdfs:
        figs |= gmsl_densities(
            physics,
            prior,
            qoi_posteriors,
            true_model,
            alt_noise,
            data["altimetry"],
            rng,
        )
    if args.plot_gmsl_split:
        figs |= gmsl_split(physics, prior, qoi_posteriors, true_model, args, rng)
    if args.plot_maps:
        figs |= state_maps(physics, posteriors, true_model, args, rng, boundaries)
        figs |= covariance_maps(physics, prior, qoi_posteriors, boundaries)
    if args.plot_regions:
        figs |= regional_decomposition(
            physics, prior, qoi_posteriors, true_model, args, rng
        )

    if args.prior_sensitivity:
        print("\nPrior sensitivity, field by field...")
        for case, (name, _, _) in CASES.items():
            common.write_metrics(
                METRICS_FILE,
                common.ocean_prior_sensitivity(
                    physics,
                    qoi_inversions[case],
                    calib,
                    OCEAN_REGION,
                    case=name,
                    sizes=case == list(CASES)[-1],
                ),
            )

    if figs:
        print(f"\nSaving {len(figs)} plots to '{OUTPUT_DIR}/'...")
        common.save_figures(figs, OUTPUT_DIR, dpi=300)


if __name__ == "__main__":
    main()
