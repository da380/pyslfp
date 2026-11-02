"""
The error of the WMB estimate of regional mass change (Heathcote et al., 2026).

The model is the direct surface mass load zeta, a global field with a Sobolev
prior. The quantities of interest are the averages over the GRACE regions
(`common.GRACE_REGIONS`) of the total load: zeta plus the ocean load
rho_w C xi that it induces through the sea level equation. The WMB method
(Wahr, Molenaar and Bryan, 1998) estimates them from GRACE's potential
coefficients, degrees 2 to --obs-degree, by inverting each degree through the
elastic load Love numbers. It ignores the induced ocean load and the degrees
GRACE does not see, and reports as its error only the propagated data noise.

The error is the estimate less the true average, (C A - B) zeta + C e, with
A the forward operator, B the true averages, C WMB's averaging of the data
and e the data noise. It is linear in the pair (zeta, e), so its law is the
push-forward of the product of the prior and the noise, computed exactly.
The script compares, region by region, the actual error with the nominal
one, C e alone. The actual error has zero mean unless the prior is shifted
(--prior-shift).

Outputs, in output_plots_grace_bias/ at 600 dpi:

- grace_bias_pdfs.png: per region, the densities of the actual and the
  nominal error in mm of equivalent water thickness (EWT), with a second axis
  in units of the true average's prior standard deviation. With --samples, a
  histogram of Monte Carlo errors checks the actual error's density.
- With --plot-maps, grace_bias_direct_load.png and grace_bias_induced_load.png:
  a prior sample of the direct load, and the ocean load it induces, with the
  region boundaries.

The defaults are the paper's resolution, which runs for a long time;
`FAST=1 ./run_all.sh grace_bias` runs at lmax 64 and observation degree 32.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import pygeoinf as inf
from pyslfp.linear_operators import WMBMethod

import common
from figures import error_density, field_map

OUTPUT_DIR = "output_plots_grace_bias"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The error of the WMB regional averages of the total load."
    )
    common.add_grace_arguments(parser)
    common.add_parallel_arguments(parser)
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Monte Carlo samples to check the analytic error densities.",
    )
    parser.add_argument(
        "--plot-maps",
        action="store_true",
        help="Plot a prior sample of the direct load and the load it induces.",
    )
    return parser.parse_args()


def plot_error_densities(
    names, means, stds, nominal_stds, true_stds, /, *, samples=None
):
    """Per region, the actual error against the nominal error; the top axis
    divides the error by the true average's standard deviation."""
    nrows = int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=2, figsize=(14, 5 * nrows), layout="constrained"
    )
    axes = axes.flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        error_density(
            ax,
            means[i],
            stds[i],
            nominal_stds[i],
            true_stds[i],
            samples=None if samples is None else samples[:, i],
        )
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Error (mm)", fontsize=12)
        if i == 1:
            ax.legend(loc="best", fontsize=9)
    for ax in axes[len(names) :]:
        ax.set_visible(False)
    return fig


def main():
    args = parse_arguments()
    rng = common.configure(args)
    common.start_run(OUTPUT_DIR, args)

    physics = common.grace_physics(args.lmax, args.load_order, args.load_scale_km)
    state, load_space, scale_mm = physics.state, physics.load_space, physics.scale_mm
    _, prior, noise_load, _ = common.grace_measures(physics, args, rng)
    names, average = common.regional_averaging(physics, args)

    wmb = WMBMethod(state.model, args.obs_degree)
    noise = wmb.load_measure_to_observation_measure(noise_load)
    total_load = common.total_load_operator(physics)
    forward = common.grace_forward_operator(physics, args.obs_degree)
    true_average = average @ total_load
    wmb_average = average @ wmb.potential_coefficient_to_load_operator(load_space)

    figures = {}
    if args.plot_maps:
        print("Plotting a prior sample of the direct load and the load it induces...")
        direct = prior.sample(rng=rng)
        induced = total_load(direct) - direct
        for name, field, label, panel in [
            ("direct", direct, "Direct Load (mm EWT)", "(a)"),
            ("induced", induced, "Induced Load (mm EWT)", "(b)"),
        ]:
            figures[f"grace_bias_{name}_load.png"] = field_map(
                field * scale_mm,
                label,
                panel,
                state=state,
                regions=common.GRACE_REGIONS,
            )

    # WMB's error as a function of (zeta, e): WMB applied to the data,
    # GRACE's coefficients of the response plus the noise, less the true
    # average.
    error = inf.RowLinearOperator([wmb_average @ forward - true_average, wmb_average])
    joint = inf.GaussianMeasure.from_product([prior, noise])

    print("Computing the error distributions...")
    true_stds = common.standard_deviations(common.dense(prior, true_average)) * scale_mm
    errors = common.dense(joint, error)
    error_means = errors.expectation * scale_mm
    error_stds = common.standard_deviations(errors) * scale_mm
    nominal_stds = (
        common.standard_deviations(common.dense(noise, wmb_average)) * scale_mm
    )

    error_samples = None
    if args.samples > 0:
        print(f"Drawing {args.samples} Monte Carlo samples...")
        draws = joint.samples(args.samples, rng=rng, n_jobs=common.jobs(args))
        error_samples = np.array([error(draw) for draw in draws]) * scale_mm

    print("Regional averages of the total load, in mm EWT (error = WMB less true):")
    for i, name in enumerate(names):
        line = (
            f"  {name:<16} true std {true_stds[i]:6.3f}, WMB error mean "
            f"{error_means[i]:+6.3f}, std {error_stds[i]:6.3f} (nominal "
            f"{nominal_stds[i]:6.3f})"
        )
        if error_samples is not None:
            line += f"; Monte Carlo error std {np.std(error_samples[:, i]):6.3f}"
        print(line)

    figures["grace_bias_pdfs.png"] = plot_error_densities(
        names, error_means, error_stds, nominal_stds, true_stds, samples=error_samples
    )
    common.save_figures(figures, OUTPUT_DIR, dpi=600)


if __name__ == "__main__":
    main()
