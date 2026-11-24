#!/usr/bin/env python
"""
truncation_check.py
===================
How the data and the quantities of interest of the joint experiment depend
on the truncation degree, by forward modelling alone.

A model is drawn from the prior at the highest of --degrees, the reference.
Its fields are truncated to each lower degree, the forward problem is solved
there with that degree's own grids and masks, and the predicted data are set
against the reference's:

- the altimetry, at the reference's points, as the difference in mm and
  against the local error of a point (--alt-noise-std-mm);
- GRACE's coefficients, as the difference in the metric of their error
  covariance: the root mean square over the coefficients of the whitened
  difference, which is one for a draw of the noise itself;
- global mean sea level, its barystatic and steric parts and the parts of
  the regional mean of --region, in mm.

What a degree leaves out is of two kinds, and both are in these numbers: the
part of the fields above the degree, and the error of the calculation on a
coarser grid with coarser masks. The reference is itself truncated, so this
shows convergence up to its degree and no further.

The options are those of joint_inversion.py (run_all.sh gives the revised
configuration); --lmax is ignored in favour of --degrees.

    python truncation_check.py --degrees 63 127 255 511 --samples 3 <options>
"""

import argparse
import copy
import os
import time

import numpy as np
from scipy.linalg import cholesky, solve_triangular

import common
import joint_inversion

OUTPUT_DIR = "output_plots_truncation_check"
METRICS_FILE = os.path.join(OUTPUT_DIR, "truncation_metrics.txt")


def parse_arguments() -> argparse.Namespace:
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[63, 127, 255, 511],
        help="Truncation degrees; the highest is the reference.",
    )
    extra.add_argument(
        "--samples", type=int, default=3, help="Models drawn from the prior."
    )
    extra.add_argument(
        "--region",
        default=joint_inversion.OCEAN_REGION,
        help="Ocean region of the regional decomposition.",
    )
    own, rest = extra.parse_known_args()
    import sys

    sys.argv = [sys.argv[0]] + rest
    args = joint_inversion.parse_arguments()
    args.degrees, args.samples, args.region = (
        sorted(own.degrees),
        own.samples,
        own.region,
    )
    return args


def quantities(physics, region: str):
    """Global mean sea level, its barystatic and steric parts, and the parts
    of the regional mean, in mm: (names, operator from the model)."""
    setup, mm = physics.setup, physics.scale_mm
    parts = [
        setup.gmsl(physics.fingerprint) * mm,
        setup.gmsl_split * mm,
        physics.regional_decomposition(region),
    ]
    names = ["GMSL", "barystatic", "steric", "SSLC", "DMSLC", "BMSLC"]

    def values(model):
        return np.concatenate([np.atleast_1d(part(model)) for part in parts])

    return names, values


def truncations(reference, physics):
    """The map taking a model of the reference to that of `physics`: each
    field's coefficients up to the lower degree."""
    transfers = [
        reference.setup.component(name).space.degree_transfer_operator(
            physics.setup.component(name).space
        )
        for name in reference.setup.names
    ]
    return lambda model: tuple(t(field) for t, field in zip(transfers, model))


def main():
    args = parse_arguments()
    rng = common.configure(args)
    common.start_run(OUTPUT_DIR, args, metrics_file=METRICS_FILE)
    top = args.degrees[-1]

    def physics_at(degree, points=None):
        local = copy.copy(args)
        local.lmax = degree
        start = time.perf_counter()
        physics = common.ocean_physics(local, points=points)
        return physics, local, time.perf_counter() - start

    print(f"Building the reference (lmax={top})...")
    reference, reference_args, seconds = physics_at(top)
    print(f"  {len(reference.points)} altimetry points; built in {seconds:.0f} s")
    prior, calib = common.ocean_prior(reference, reference_args, rng)
    _, info = common.joint_grace_noise(
        reference, calib["derived_stds"]["ice_std"], args.prior_order, reference_args
    )
    if "covariance" not in info:
        raise SystemExit("truncation_check.py needs --grace-coefficient-errors.")
    lower = cholesky(info["covariance"], lower=True)
    alt_std = args.alt_noise_std_mm
    mm = reference.scale_mm

    print(f"Drawing {args.samples} model(s) from the prior...")
    models = [prior.sample(rng=rng) for _ in range(args.samples)]

    results = {}
    names = None
    for degree in reversed(args.degrees):
        if degree == top:
            physics, local, seconds = reference, reference_args, 0.0
            truncate = lambda model: model  # noqa: E731
        else:
            physics, local, seconds = physics_at(degree, reference.points)
            truncate = truncations(reference, physics)
        forward = physics.joint(local)
        names, values = quantities(physics, args.region)
        start = time.perf_counter()
        rows = []
        for model in models:
            part = truncate(model)
            altimetry, grace = forward(part)
            rows.append((np.asarray(altimetry) * mm, np.asarray(grace), values(part)))
        each = (time.perf_counter() - start) / len(models)
        results[degree] = rows
        print(f"  lmax={degree}: built in {seconds:.0f} s, {each:.1f} s a model")

    lines = []
    out = lines.append
    out(
        f"Truncation check: {args.samples} model(s) drawn from the prior at degree "
        f"{top}, truncated to each degree; differences from degree {top}."
    )
    out(
        f"{len(reference.points)} altimetry points (spacing {args.spacing} degrees), "
        f"GRACE to degree {args.obs_degree} ({len(lower)} coefficients)."
    )
    out(
        "Altimetry: root mean square and largest difference over the points, in mm, "
        f"and the first over the local error of a point ({alt_std} mm)."
    )
    out(
        "GRACE: root mean square over the coefficients of the difference whitened by "
        "the error covariance (one for a draw of the noise)."
    )
    out("")
    signal_alt = np.sqrt(np.mean([np.mean(r[0] ** 2) for r in results[top]]))
    signal_grace = np.sqrt(
        np.mean(
            [
                np.mean(solve_triangular(lower, r[1], lower=True) ** 2)
                for r in results[top]
            ]
        )
    )
    out(
        f"Size of the data at degree {top}: altimetry {signal_alt:.2f} mm rms; "
        f"GRACE {signal_grace:.2f} whitened rms."
    )
    out("")
    header = (
        f"{'degree':>6} | {'alt rms':>8} {'alt max':>8} {'rms/noise':>9} | "
        f"{'GRACE whitened':>14} | "
    ) + " ".join(f"{name:>10}" for name in names)
    out(header)
    out("-" * len(header))
    reference_rows = results[top]
    for degree in args.degrees:
        alt_sq, alt_max, grace_sq = [], [], []
        qoi = []
        for row, ref in zip(results[degree], reference_rows):
            difference = row[0] - ref[0]
            alt_sq.append(np.mean(difference**2))
            alt_max.append(np.max(np.abs(difference)))
            white = solve_triangular(lower, row[1] - ref[1], lower=True)
            grace_sq.append(np.mean(white**2))
            qoi.append(row[2] - ref[2])
        alt_rms = np.sqrt(np.mean(alt_sq))
        qoi_rms = np.sqrt(np.mean(np.array(qoi) ** 2, axis=0))
        out(
            f"{degree:>6} | {alt_rms:>8.3f} {np.max(alt_max):>8.3f} "
            f"{alt_rms / alt_std:>9.3f} | {np.sqrt(np.mean(grace_sq)):>14.4f} | "
            + " ".join(f"{value:>10.4f}" for value in qoi_rms)
        )
    out("")
    out(
        "The last six columns are root mean square differences over the models of the "
        f"quantities of interest, in mm ({args.region} for the last three)."
    )
    values = np.array([row[2] for row in reference_rows])
    out(
        f"Their sizes at degree {top} (rms over the models, mm): "
        + ", ".join(
            f"{name} {value:.3f}"
            for name, value in zip(names, np.sqrt(np.mean(values**2, axis=0)))
        )
    )
    text = "\n".join(lines)
    print("\n" + text)
    with open(METRICS_FILE, "a") as handle:
        handle.write("\n" + text + "\n")
    print(f"\nsaved {METRICS_FILE}")


if __name__ == "__main__":
    main()
