"""
An out-of-prior test of the GRACE estimators (Heathcote et al., 2026).

In the other GRACE experiments the true load is drawn from the prior of the
inversion, which favours the Bayesian estimate. Here it is not. The true
total load is taken from a mascon solution of real GRACE and GRACE-FO data
(--mascon-file): one field, or several, each the change in mass over an
interval. Such a field is far from a sample of a rotationally invariant
Gaussian prior: it does not vanish on average, and its amplitude varies by
orders of magnitude between the ice sheet margins and the open ocean. It is
used only as a realistic load whose value is known exactly, not as a
statement about the Earth.

For each field the direct load that produces it through the sea level
equation is found (`common.direct_load_of_total`), synthetic data are made
from it with the forward operator and a draw of the noise, and the two
estimators are applied: WMB, and the posterior expectation under the prior
and noise of the other GRACE experiments. Both are compared with the truth
on the averages of the total load over the GRACE regions
(`common.GRACE_REGIONS`), and the Bayesian estimate alone on the degree-1
coefficients of the total load, which GRACE does not observe.

Both estimators are linear in the data, so each is assembled once as a
matrix from the data to the quantities of interest, at the cost of one solve
per quantity for the Bayesian one, and then applied to every field. The
errors are reported with and without the noise. Without it, the error of an
estimate is the pairing of its estimator kernel error with the true load,
which is what `grace_inversion.py --prior-sensitivity` bounds; here it is
evaluated.

Outputs, in output_plots_grace_mascon/ at 300 dpi:

- mascon_metrics.txt: per quantity, the truth, the errors of the two
  estimates and their stated standard deviations, field by field, and their
  root mean squares over the fields.
- mascon_errors.png: per region, the errors of the two estimates for each
  field, with bars of one stated standard deviation.
- With --plot-maps, mascon_maps.png: for the field --map-field, the true
  total load, WMB's estimate, the posterior expectation and the estimator
  errors' standard deviations are not shown; see grace_inversion.py.

The prior is not fitted to the fields. Its pointwise standard deviation is
--direct-std-m, as in the other GRACE experiments, and the fields' root mean
squares are printed beside it.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import pygeoinf as inf

import common
import figures
from grace_inversion import posterior_maps

OUTPUT_DIR = "output_plots_grace_mascon"
METRICS_FILE = os.path.join(OUTPUT_DIR, "mascon_metrics.txt")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The GRACE estimators applied to loads from a mascon solution."
    )
    parser.add_argument(
        "--mascon-file",
        required=True,
        help="netCDF file of one load or several; see common.read_load_fields.",
    )
    parser.add_argument("--mascon-variable", default="lwe_thickness")
    parser.add_argument(
        "--plot-maps", action="store_true", help="Maps of one field's estimates."
    )
    parser.add_argument(
        "--map-field",
        type=int,
        default=-1,
        help="The index of the field of the maps; the last by default.",
    )
    common.add_grace_arguments(parser)
    common.add_qoi_arguments(parser)
    common.add_parallel_arguments(parser)
    args = parser.parse_args()
    if args.prior_shift != 0.0:
        parser.error("the test takes the prior as it is: --prior-shift must be 0.")
    return args


def error_figure(names, labels, truth, wmb, bayes, wmb_std, bayes_std):
    """Per region, the error of each estimate for each field, with a bar of
    one stated standard deviation: WMB's nominal one and the posterior's."""
    nrows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=2, figsize=(14, 4.5 * nrows), layout="constrained"
    )
    axes = np.atleast_1d(axes).flatten()
    x = np.arange(len(labels))
    for i, name in enumerate(names):
        ax = axes[i]
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.errorbar(
            x - 0.12,
            wmb[:, i] - truth[:, i],
            yerr=wmb_std[i],
            fmt="o",
            mfc="white",
            color="tab:red",
            capsize=3,
            label="WMB",
        )
        ax.errorbar(
            x + 0.12,
            bayes[:, i] - truth[:, i],
            yerr=bayes_std[i],
            fmt="o",
            color="tab:blue",
            capsize=3,
            label="Bayesian",
        )
        ax.set_title(name)
        ax.set_ylabel("Error (mm EWT)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 6 else 0)
        if i == 1:
            ax.legend(loc="best")
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    return fig


def rms(values, axis=0):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def table(title, names, labels, truth, estimates, stds, noise_free):
    """One estimator's errors, field by field and as root mean squares over
    the fields; `stds` is its stated standard deviation per quantity."""
    error = estimates - truth
    width = 16 + 14 * len(names)
    lines = [f"\n\n{title}\n", "=" * width + "\n"]
    lines.append(f"{'':<16}" + "".join(f"{n:>14}" for n in names) + "\n")
    lines.append("-" * width + "\n")
    for k, label in enumerate(labels):
        lines.append(
            f"{'truth ' + label:<16}" + "".join(f"{v:>14.3f}" for v in truth[k]) + "\n"
        )
        lines.append(
            f"{'error ' + label:<16}" + "".join(f"{v:>14.3f}" for v in error[k]) + "\n"
        )
    lines.append("-" * width + "\n")
    for label, row in [
        ("truth rms", rms(truth)),
        ("error rms", rms(error)),
        ("  without noise", rms(noise_free - truth)),
        ("stated std", stds),
        ("rms error/std", rms(error / stds)),
    ]:
        lines.append(f"{label:<16}" + "".join(f"{v:>14.3f}" for v in row) + "\n")
    return "".join(lines)


def main():
    args = parse_arguments()
    rng = common.configure(args)
    common.start_run(OUTPUT_DIR, args, metrics_file=METRICS_FILE)
    figures.set_inversion_style()

    print(f"Building the physics (lmax={args.lmax})...")
    physics, initial_prior, prior, noise, noise_matrix, noise_info = (
        common.grace_experiment(args, rng)
    )
    state, load_space, scale_mm = physics.state, physics.load_space, physics.scale_mm
    names, average = common.regional_averaging(physics, args)

    forward = common.grace_forward_operator(physics, args)
    total_load = common.total_load_operator(physics)
    wmb = common.wmb_method(state, args)
    wmb_load = wmb.potential_coefficient_to_load_operator(load_space)

    print(f"Reading the loads of {args.mascon_file}...")
    labels, totals = common.read_load_fields(
        physics, args.mascon_file, variable=args.mascon_variable
    )
    print(
        f"  the prior's pointwise std is {noise_info['signal_std'] * scale_mm:.1f} "
        "mm EWT, "
        f"the noise's {noise_info['noise_std'] * scale_mm:.2f} mm"
    )

    print("Finding the direct loads and making the data...")
    directs, clean_data, data = [], [], []
    for label, sigma in zip(labels, totals):
        zeta, residual = common.direct_load_of_total(total_load, sigma)
        print(f"  {label}: total load reproduced to {residual:.1e}")
        directs.append(zeta)
        clean_data.append(forward(zeta))
        data.append(clean_data[-1] + noise.sample(rng=rng))

    # The quantities of interest, in mm EWT: the regional averages of the
    # total load and its degree-1 coefficients, in the order m = 0, 1, -1.
    degree_one = (scale_mm / load_space.radius) * load_space.coefficient_operator(
        lmax=1, lmin=1
    )
    degree_names = ["(1,0)", "(1,1)", "(1,-1)"]
    stacked = inf.ColumnLinearOperator([scale_mm * average, degree_one])
    quantities = stacked.codomain.coordinate_projection() @ stacked
    n_regions = len(names)

    # Both estimators as matrices from the data to the quantities. The
    # Bayesian one costs a solve per quantity, at the tolerance of the
    # posterior covariance; the expectation of the prior is zero, so the
    # estimate has no constant term.
    print("Assembling the estimators...")
    problem = inf.LinearForwardProblem(forward, error=noise)
    if noise_matrix is None:
        preconditioner = wmb.bayesian_normal_operator_preconditioner(
            initial_prior, noise
        )
    else:
        preconditioner = common.exact_preconditioner(
            forward, prior, noise, noise_matrix=noise_matrix, n_jobs=common.jobs(args)
        )
    inversion = inf.LinearGaussianInversion(problem, prior).with_solver(
        inf.CGSolver(rtol=args.qoi_rtol, preconditioner=preconditioner)
    )
    target = quantities @ total_load
    bayes_matrix = (target @ inversion.kalman_operator).matrix(
        form="components", by="rows", n_jobs=common.jobs(args, target.codomain.dim)
    )
    wmb_matrix = (scale_mm * average @ wmb_load).matrix(form="components", by="rows")

    posterior = inf.GaussianMeasure(load_space, covariance=inversion.covariance)
    bayes_std = common.standard_deviations(
        common.dense(posterior, target, n_jobs=common.jobs(args, target.codomain.dim))
    )
    prior_std = common.standard_deviations(
        common.dense(prior, target, n_jobs=common.jobs(args, target.codomain.dim))
    )
    wmb_std = common.standard_deviations(
        common.dense(noise, scale_mm * average @ wmb_load)
    )

    truth = np.array([quantities(sigma) for sigma in totals])
    bayes = np.array([bayes_matrix @ d for d in data])
    bayes_clean = np.array([bayes_matrix @ d for d in clean_data])
    wmb_estimates = np.array([wmb_matrix @ d for d in data])
    wmb_clean = np.array([wmb_matrix @ d for d in clean_data])

    all_names = list(names) + degree_names
    text = (
        "Out-of-prior test: loads from a mascon solution\n"
        f"Fields: {', '.join(labels)}\n"
        "All values in mm EWT. 'stated std' is the standard deviation the "
        "estimator reports: the posterior's, or WMB's propagated noise.\n"
        f"Prior std of the quantities: "
        + ", ".join(f"{n} {v:.3f}" for n, v in zip(all_names, prior_std))
    )
    text += table(
        "Bayesian estimates",
        all_names,
        labels,
        truth,
        bayes,
        bayes_std,
        bayes_clean,
    )
    text += table(
        "WMB estimates (regional averages only)",
        names,
        labels,
        truth[:, :n_regions],
        wmb_estimates,
        wmb_std,
        wmb_clean,
    )
    common.write_metrics(METRICS_FILE, text)
    print(text)
    print(f"saved {METRICS_FILE}")

    output = {
        "mascon_errors.png": error_figure(
            names,
            labels,
            truth[:, :n_regions],
            wmb_estimates,
            bayes[:, :n_regions],
            wmb_std,
            bayes_std[:n_regions],
        )
    }
    if args.plot_maps:
        k = args.map_field
        print(f"Maps for the field {labels[k]}...")
        expectation = inversion.kalman_operator(data[k])
        panels = [
            (totals[k], f"True Total Load ({labels[k]})"),
            (wmb_load(data[k]), "WMB Estimate"),
            (total_load(expectation), "Bayesian Posterior"),
        ]
        output["mascon_maps.png"] = posterior_maps(
            state, [(field * scale_mm, title) for field, title in panels]
        )
    common.save_figures(output, OUTPUT_DIR, dpi=300)


if __name__ == "__main__":
    main()
