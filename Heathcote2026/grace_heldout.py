"""
A held-out test of the GRACE estimators on realistic loads (Heathcote et
al., 2026).

The loads are year-on-year changes of the hydrology (H), land ice (I) and
ocean (O) components of the updated ESA Earth System Model
(`esm_loads.py`), a synthetic but realistic model of surface mass
variability that is independent of any inversion of GRACE data. They are
split in two. The earlier changes are used to build the prior, and the
later ones are held out as truths: none of them is seen in building the
prior, and none is a sample of it.

The model has three fields, the ice thickness change over the ice sheets,
the water stored on the rest of the land and the ocean's own load
(`common.BudgetPhysics`), so that the load may jump at the coast and at the
edge of an ice sheet. The prior gives each field the amplitude and the
spectrum of the corresponding component of the training loads
(`common.budget_prior`).

For each held-out change a true direct load is built from the components,
each confined to where it belongs: the hydrology and the land ice to the
land, and the ocean, less its mean, to the ocean (`direct_load`). The
Earth System Model is given only as coefficients to degree 180, so its
land signals ring across the coast; confining them removes that. The sea
level equation then gives the ocean's response to the land's mass change,
in place of the uniform layer of the Earth System Model, so that the true
total load is gravitationally self-consistent. With --truth total the
summed components, less their mean, are taken as the total load instead,
and the direct load that produces it is found by iteration.

Synthetic data are made from the direct load with the forward operator and
a draw of the noise, and the two estimators are applied: WMB, and the
posterior expectation. They are compared on the averages of the total load
over the GRACE regions, and the Bayesian estimate alone on the degree-1
coefficients of the total load.
What is reported is each estimator's error beside the standard deviation it
states: the posterior's, or the noise WMB propagates.

WMB's error is also assessed as in `grace_bias.py`, with the fitted prior in
place of that script's. Its error on a load is linear in the load, so under
the prior its law is exact; with the noise added, that is the error the
prior predicts for WMB, to be set against the error WMB states and against
the errors realised on the held-out loads, which the prior has not seen.

The noise is that of `common.grace_noise_model`: a spatial field of errors
with root mean square --noise-std-mm and correlation scale --noise-scale-km,
whose standard deviation may follow a map (--noise-std-map), taken to the
coefficients as the field of a mascon solution is; and, optionally, the
error of the background corrections (--corrections-std-mm) and that of the
GIA correction, from an ensemble of models (--gia-years).

Outputs, in output_plots_grace_heldout/ at 300 dpi:

- heldout_metrics.txt: the fitted prior; per quantity, the truth, the
  errors of the two estimates and their stated standard deviations, field
  by field, and their root mean squares over the held-out fields.
- heldout_errors.png: per region, the errors of the two estimates for each
  held-out field, with bars of one stated standard deviation.
- heldout_wmb_error_pdfs.png: per region, the density of WMB's error under
  the prior against that of the error it states, with the errors realised
  on the held-out fields marked along the axis.
- With --plot-maps, for the held-out field --map-field:
  heldout_maps.png, the true total load, WMB's estimate and the posterior
  expectation on one colour scale, and beneath them a sample of the prior's
  total load on that scale and the errors of the two estimates;
  heldout_fields.png, the posterior expectation of each of the three fields
  on its support.
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
from figures import COLORBAR, GRIDLINES, draw_region_boundaries
from grace_mascon import error_figure, table

OUTPUT_DIR = "output_plots_grace_heldout"
METRICS_FILE = os.path.join(OUTPUT_DIR, "heldout_metrics.txt")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The GRACE estimators on held-out loads of the ESA Earth "
        "System Model, with a prior built from the rest."
    )
    parser.add_argument("--lmax", type=int, default=256)
    parser.add_argument("--obs-degree", type=int, default=60)
    parser.add_argument(
        "--grace-convention", choices=("stokes", "orthonormal"), default="stokes"
    )
    parser.add_argument(
        "--held-out",
        type=int,
        default=5,
        help="The number of changes held out as truths, the latest ones.",
    )
    parser.add_argument(
        "--smoothing-scale-km",
        type=float,
        default=500.0,
        help="Smoothing of the region masks; zero leaves them sharp.",
    )
    parser.add_argument(
        "--truth",
        choices=("confined", "total"),
        default="confined",
        help="How the true loads are built from the components; see above.",
    )
    common.add_grace_noise_model_arguments(parser)
    parser.add_argument("--min-prior-order", type=float, default=1.05)
    parser.add_argument(
        "--prior-order",
        type=float,
        default=None,
        help="The order of each field's prior, stated and not fitted; its "
        "scale and amplitude are still fitted to the training loads.",
    )
    parser.add_argument(
        "--fit-degree",
        type=int,
        default=None,
        help="The highest degree of the training loads the spectra are "
        "fitted to; all of them if omitted.",
    )
    parser.add_argument(
        "--prior-mean",
        action="store_true",
        help="Take each field's expectation as the mean of its training loads.",
    )
    parser.add_argument(
        "--modulate",
        choices=("pointwise", "margin"),
        default=None,
        help="Let the standard deviation vary over the support: following the "
        "variability of the training loads point by point, or, for the ice "
        "sheets alone, growing towards their margins at a fitted rate.",
    )
    parser.add_argument("--modulation-scale-km", type=float, default=300.0)
    parser.add_argument(
        "--plot-maps", action="store_true", help="Maps of one field's estimates."
    )
    parser.add_argument(
        "--map-field",
        type=int,
        default=-1,
        help="The index of the held-out field of the maps; the last by default.",
    )
    common.add_qoi_arguments(parser)
    common.add_parallel_arguments(parser)
    return parser.parse_args()


def prior_summary(info: dict) -> str:
    lines = ["Prior fitted to the training loads\n", "-" * 72 + "\n"]
    lines.append(
        f"{'field':<14}{'order':>8}{'scale (km)':>12}{'std on support':>18}"
        f"{'rms of mean':>14}   (mm EWT)\n"
    )
    for name, values in info.items():
        lines.append(
            f"{name:<14}{values['order']:>8.2f}{values['scale_km']:>12.0f}"
            f"{values['load_std_mm']:>18.1f}{values['mean_rms_mm']:>14.1f}\n"
        )
    growth = info[common.ICE].get("margin_growth")
    if growth is not None:
        lines.append(
            f"ice sheets: std grows as 1 + {growth:.2f} p towards the margin\n"
        )
    return "".join(lines)


def wmb_error_densities(names, means, stds, nominal_stds, true_stds, errors):
    """Per region, the density of WMB's error under the prior, noise
    included, against that of the error it states, as in `grace_bias.py`;
    the rows of `errors` are the errors realised on the held-out loads,
    marked along the axis."""
    nrows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=2, figsize=(14, 5 * nrows), layout="constrained"
    )
    axes = np.atleast_1d(axes).flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        figures.error_density(ax, means[i], stds[i], nominal_stds[i], true_stds[i])
        ax.lines[0].set_label("Error under the prior")
        ax.lines[1].set_label("Stated error (noise only)")
        ax.plot(
            errors[:, i],
            np.zeros(len(errors)),
            "|",
            color="black",
            markersize=22,
            markeredgewidth=2,
            label="Held-out errors",
        )
        ax.set_ylim(bottom=0.0)
        ax.set_title(name)
        ax.set_xlabel("Error (mm EWT)")
        if i == 1:
            ax.legend(loc="best")
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    return fig


def wmb_error_table(names, stated, mean, signal_std, clean_error) -> str:
    """WMB's error by region: what it states, what the prior predicts, and
    what is realised on the held-out loads. The noise is not drawn: its
    variance, which WMB states, is added to the part fixed by the load."""
    realised = np.sqrt(np.mean(np.square(clean_error), axis=0))
    width = 34 + 14 * len(names)
    lines = ["\n\nWMB's error: stated, under the prior, and realised\n"]
    lines.append("=" * width + "\n")
    lines.append(f"{'':<34}" + "".join(f"{n:>14}" for n in names) + "\n")
    lines.append("-" * width + "\n")
    for label, row in [
        ("stated std (noise only)", stated),
        ("prior: mean", mean),
        ("prior: std, load part", signal_std),
        ("prior: std, with noise", np.hypot(signal_std, stated)),
        ("held out: rms, load part", realised),
        ("held out: rms, with noise", np.hypot(realised, stated)),
        ("actual/stated, prior", np.hypot(signal_std, stated) / stated),
        ("actual/stated, held out", np.hypot(realised, stated) / stated),
    ]:
        lines.append(f"{label:<34}" + "".join(f"{v:>14.3f}" for v in row) + "\n")
    return "".join(lines)


def map_grid(state, rows, /, *, regions=True):
    """Maps in rows. Each panel is (field, title, label, limit): a field in
    its display unit, drawn on a symmetric colour scale saturating at
    `limit`, which several panels may share."""
    ncols = max(len(row) for row in rows)
    fig, axes = sl.subplots(
        len(rows),
        ncols,
        figsize=(8.0 * ncols, 5.2 * len(rows)),
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
        squeeze=False,
    )
    for i, row in enumerate(rows):
        for j in range(ncols):
            if j >= len(row):
                axes[i, j].set_axis_off()
                continue
            field, title, label, limit = row[j]
            _, image = sl.plot(
                field,
                ax=axes[i, j],
                cmap="seismic",
                vmin=-limit,
                vmax=limit,
                colorbar=True,
                colorbar_kwargs=COLORBAR,
                gridlines_kwargs=GRIDLINES,
            )
            axes[i, j].set_title(title, fontsize=15)
            image.colorbar.set_label(label, fontsize=13)
            if regions:
                draw_region_boundaries(state, axes[i, j], common.GRACE_REGIONS)
    return fig


def saturation(field, percentile: float = 99.5) -> float:
    """A colour limit below a field's largest value, so that a few extreme
    points do not flatten the rest of the map."""
    return float(np.percentile(np.abs(field.data), percentile))


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

    print("Reading the loads of the ESA Earth System Model...")
    components = {}
    for key, name in [("water", "H"), ("ice", "I"), ("ocean", "O")]:
        labels, components[key] = esm_loads.load_changes(model, name)
    n_test = args.held_out
    if not 0 < n_test < len(labels) - 1:
        raise ValueError("--held-out must leave at least two training loads.")
    # The training loads of each field, confined to its support.
    training = {name: [] for name in physics.setup.names}
    for k in range(len(labels) - n_test):
        parts = common.field_loads(
            physics,
            components["water"][k],
            components["ice"][k],
            components["ocean"][k],
        )
        for name, load in parts.items():
            training[name].append(load)
    test_labels = [label[:4] for label in labels[-n_test:]]
    print(
        f"  training on the changes to {', '.join(l[:4] for l in labels[:-n_test])}; "
        f"holding out {', '.join(test_labels)}"
    )

    prior, fields, info = common.budget_prior(
        physics,
        training,
        mean=args.prior_mean,
        modulate=args.modulate,
        modulation_scale=args.modulation_scale_km * km / load_space.radius,
        min_order=args.min_prior_order,
        order=args.prior_order,
        fit_degree=args.fit_degree,
    )
    print(prior_summary(info))

    # The noise, and a stand-in with a diagonal covariance for the
    # preconditioner.
    wmb = common.wmb_method(state, args)
    noise, invariant_noise, noise_parts = common.grace_noise_model(
        state, load_space, physics.response_space, args
    )
    wmb_load = wmb.potential_coefficient_to_load_operator(load_space)

    # The held-out truths: a direct load and its total load for each.
    ones = model.constant_grid(1.0)
    sphere = model.integrate(ones)
    observe = common.grace_coefficients(
        physics.response_space, model, args.obs_degree, args.grace_convention
    )
    print("Building the true loads and making the data...")
    totals, clean_data, data = [], [], []
    for label, k in zip(test_labels, range(len(labels) - n_test, len(labels))):
        water, ice, ocean = (components[key][k] for key in ("water", "ice", "ocean"))
        if args.truth == "confined":
            zeta = sum(common.field_loads(physics, water, ice, ocean).values())
            sigma = physics.load_to_total_load(zeta)
            note = ""
        else:
            sigma = water + ice + ocean
            sigma = sigma - (model.integrate(sigma) / sphere) * ones
            zeta, residual = common.direct_load_of_total(
                physics.load_to_total_load, sigma
            )
            note = f", total load reproduced to {residual:.1e}"
        rms = np.sqrt(model.integrate(sigma * sigma) / sphere) * scale_mm
        print(f"  {label}: rms of the total load {rms:.1f} mm EWT{note}")
        totals.append(sigma)
        clean_data.append(observe(physics.fingerprint(zeta)))
        data.append(clean_data[-1] + noise.sample(rng=rng))

    # The quantities of interest, in mm EWT, as functionals of a total load:
    # the regional averages and the degree-1 coefficients (m = 0, 1, -1).
    names = list(common.GRACE_REGIONS)
    weights = [
        state.get_projection(region, value=0.0)
        for region in common.GRACE_REGIONS.values()
    ]
    if args.smoothing_scale_km > 0.0:
        smoother = load_space.heat_measure(args.smoothing_scale_km * km).covariance
        weights = [smoother(w) for w in weights]
    average = scale_mm * averaging_operator(state, load_space, weights)
    degree_one = (scale_mm / load_space.radius) * load_space.coefficient_operator(
        lmax=1, lmin=1
    )
    stacked = inf.ColumnLinearOperator([average, degree_one])
    quantities = stacked.codomain.coordinate_projection() @ stacked
    all_names = names + ["(1,0)", "(1,1)", "(1,-1)"]
    n_regions = len(names)

    print("Assembling the estimators...")
    forward = physics.forward(args)
    preconditioner = wmb.bayesian_normal_operator_preconditioner(
        common.invariant_load_measure(physics, fields), invariant_noise
    )
    inversion = inf.LinearGaussianInversion(
        inf.LinearForwardProblem(forward, error=noise), prior
    ).with_solver(inf.CGSolver(rtol=args.qoi_rtol, preconditioner=preconditioner))
    target = quantities @ physics.total_load
    workers = common.jobs(args, target.codomain.dim)
    bayes_matrix = (target @ inversion.kalman_operator).matrix(
        form="components", by="rows", n_jobs=workers
    )
    wmb_matrix = (average @ wmb_load).matrix(form="components", by="rows")

    posterior = inf.GaussianMeasure(prior.domain, covariance=inversion.covariance)
    bayes_std = common.standard_deviations(
        common.dense(posterior, target, n_jobs=workers)
    )
    prior_std = common.standard_deviations(common.dense(prior, target, n_jobs=workers))
    wmb_std = common.standard_deviations(common.dense(noise, average @ wmb_load))
    # What each part of the noise adds to WMB's regional averages.
    part_stds = {
        name: common.standard_deviations(common.dense(part, average @ wmb_load))
        for name, part in noise_parts.items()
    }

    # WMB's error on a load, before the noise: its estimate from clean data
    # less the true average. Its law under the prior is exact.
    wmb_load_error = common.dense(
        prior,
        (average @ wmb_load) @ forward - average @ physics.total_load,
        n_jobs=common.jobs(args, n_regions),
    )
    wmb_load_error_std = common.standard_deviations(wmb_load_error)

    # The estimate is affine in the data: the prior's expectation, pushed to
    # the quantities, plus the matrix applied to the data less what that
    # expectation predicts.
    prior_mean = prior.expectation
    predicted = forward(prior_mean)
    offset = target(prior_mean) - bayes_matrix @ predicted

    truth = np.array([quantities(sigma) for sigma in totals])
    bayes = np.array([bayes_matrix @ d + offset for d in data])
    bayes_clean = np.array([bayes_matrix @ d + offset for d in clean_data])
    wmb_estimates = np.array([wmb_matrix @ d for d in data])
    wmb_clean = np.array([wmb_matrix @ d for d in clean_data])

    text = (
        "Held-out test: loads of the ESA Earth System Model\n"
        f"Held-out changes: {', '.join(test_labels)}\n"
        "All values in mm EWT. 'stated std' is the standard deviation the "
        "estimator reports: the posterior's, or WMB's propagated noise.\n\n"
        + prior_summary(info)
        + "\nPrior std of the quantities: "
        + ", ".join(f"{n} {v:.2f}" for n, v in zip(all_names, prior_std))
    )
    text += table(
        "Bayesian estimates",
        all_names,
        test_labels,
        truth,
        bayes,
        bayes_std,
        bayes_clean,
    )
    text += table(
        "WMB estimates (regional averages only)",
        names,
        test_labels,
        truth[:, :n_regions],
        wmb_estimates,
        wmb_std,
        wmb_clean,
    )
    text += "\n\nThe noise in WMB's regional averages, by part (std, mm EWT)\n"
    text += f"{'':<16}" + "".join(f"{n:>14}" for n in names) + "\n"
    for name, values in part_stds.items():
        text += f"{name:<16}" + "".join(f"{v:>14.3f}" for v in values) + "\n"
    text += wmb_error_table(
        names,
        wmb_std,
        wmb_load_error.expectation,
        wmb_load_error_std,
        wmb_clean - truth[:, :n_regions],
    )
    common.write_metrics(METRICS_FILE, text)
    print(text)
    print(f"saved {METRICS_FILE}")

    output = {
        "heldout_errors.png": error_figure(
            names,
            test_labels,
            truth[:, :n_regions],
            wmb_estimates,
            bayes[:, :n_regions],
            wmb_std,
            bayes_std[:n_regions],
        ),
        "heldout_wmb_error_pdfs.png": wmb_error_densities(
            names,
            wmb_load_error.expectation,
            np.hypot(wmb_load_error_std, wmb_std),
            wmb_std,
            prior_std[:n_regions],
            wmb_estimates - truth[:, :n_regions],
        ),
    }
    if args.plot_maps:
        k = args.map_field
        label = test_labels[k]
        print(f"Maps for the held-out change to {label}...")
        expectation = prior.domain.add(
            prior_mean, inversion.kalman_operator(data[k] - predicted)
        )
        true_total = totals[k] * scale_mm
        wmb_total = wmb_load(data[k]) * scale_mm
        bayes_total = physics.total_load(expectation) * scale_mm
        sample_total = physics.total_load(prior.sample(rng=rng)) * scale_mm
        limit = saturation(true_total)
        wmb_error, bayes_error = wmb_total - true_total, bayes_total - true_total
        error_limit = max(saturation(wmb_error), saturation(bayes_error))
        unit = "Load (mm EWT)"
        output["heldout_maps.png"] = map_grid(
            state,
            [
                [
                    (true_total, f"True total load ({label})", unit, limit),
                    (wmb_total, "WMB estimate", unit, limit),
                    (bayes_total, "Posterior expectation", unit, limit),
                ],
                [
                    (sample_total, "A sample of the prior", unit, limit),
                    (wmb_error, "WMB error", "Error (mm EWT)", error_limit),
                    (bayes_error, "Bayesian error", "Error (mm EWT)", error_limit),
                ],
            ],
        )
        shown = physics.setup.support(expectation)
        densities = physics.densities
        output["heldout_fields.png"] = map_grid(
            state,
            [
                [
                    (
                        field * (densities[name] * scale_mm),
                        f"Posterior expectation: {name.replace('_', ' ')}",
                        unit,
                        saturation(field * (densities[name] * scale_mm), 99.9),
                    )
                    for name, field in zip(physics.setup.names, shown)
                ]
            ],
            regions=False,
        )
    common.save_figures(output, OUTPUT_DIR, dpi=200)


if __name__ == "__main__":
    main()
