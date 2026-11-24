"""
Figures of the GRACE noise model against the signal of the prior
(Heathcote et al., 2026).

The noise is that of `common.joint_grace_noise` with
--grace-coefficient-errors: the formal error covariance of ITSG-Grace2018's
coefficients for a difference of annual means (`grace_covariance.py`),
with any white load floor. The signal is GRACE's data for draws of the
prior of the ocean experiments, through the full forward operator.

Outputs, in output_plots_grace_noise/ at 200 dpi:

- grace_noise_spectrum.png: by degree, the standard deviation of a
  coefficient of the signal and of the noise (root mean square over the
  orders, the median coefficient, and the range over the orders), and the
  signal-to-noise ratio the same ways, and the share of each degree's
  coefficients in which the signal exceeds the noise.
- grace_noise_coefficients.png: by degree and order, the standard
  deviation of the noise and the signal-to-noise ratio.
- grace_noise_maps.png: as fields of equivalent water thickness to the
  observed degree, through the load Love numbers and unfiltered: a draw of
  the noise, a draw of the signal, and the pointwise standard deviation of
  the noise.
- grace_noise_metrics.txt: the numbers of the first figure.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from pyshtools.legendre import PlmBar

import pyslfp as sl

import common
from figures import COLORBAR, GRIDLINES

OUTPUT_DIR = "output_plots_grace_noise"
METRICS_FILE = os.path.join(OUTPUT_DIR, "grace_noise_metrics.txt")
SIGNAL, NOISE, MUTED = "#2a78d6", "#eb6834", "#898781"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figures of the GRACE noise model against the prior's signal."
    )
    parser.add_argument("--samples", type=int, default=40)
    common.add_ocean_arguments(parser, surrogate=True)
    common.add_joint_grace_noise_arguments(parser)
    common.add_parallel_arguments(parser)
    args = parser.parse_args()
    if args.grace_coefficient_errors is None:
        parser.error("--grace-coefficient-errors is required.")
    return args


def labels(obs_degree: int):
    """The degree, the order and whether it is a sine term of each of
    pyslfp's coefficients of degrees 2 to `obs_degree`."""
    degrees, orders, sine = [], [], []
    for l in range(2, obs_degree + 1):
        degrees += [l] * (2 * l + 1)
        orders += list(range(l + 1)) + list(range(1, l + 1))
        sine += [False] * (l + 1) + [True] * l
    return np.array(degrees), np.array(orders), np.array(sine)


def pointwise_std(covariance, degrees, orders, sine, to_mm, latitudes, longitudes):
    """The standard deviation, at each point of a grid, of the field with
    4pi-normalised coefficients `to_mm` times the data, whose covariance is
    `covariance`."""
    scaled = covariance * to_mm[:, None] * to_mm[None, :]
    phase = np.where(
        sine[None, :],
        np.sin(np.outer(longitudes, orders)),
        np.cos(np.outer(longitudes, orders)),
    )
    index = (degrees * (degrees + 1)) // 2 + orders
    result = np.empty((len(latitudes), len(longitudes)))
    for row, latitude in enumerate(latitudes):
        basis = PlmBar(degrees.max(), np.sin(latitude))[index][None, :] * phase
        result[row] = np.sqrt(np.einsum("ij,ij->i", basis @ scaled, basis))
    return result


def main():
    args = parse_arguments()
    rng = common.configure(args)
    common.start_run(OUTPUT_DIR, args, metrics_file=METRICS_FILE)
    physics = common.ocean_physics(args)
    prior, calib = common.ocean_prior(physics, args, rng)
    forward = physics.grace(args)
    noise, info = common.joint_grace_noise(
        physics, calib["derived_stds"]["ice_std"], args.prior_order, args
    )
    covariance = info["covariance"]
    p = physics.state.model.parameters
    scale_mm = 1000.0 * p.length_scale / p.water_density
    wmb = common.wmb_method(physics.state, args)
    to_load = wmb.potential_coefficient_to_load_operator(physics.load_space)
    # mm EWT in a 4pi-normalised coefficient of the load per unit datum.
    to_mm = (
        wmb.potential_coefficient_to_load_coefficient_operator().diagonals()[0]
        * scale_mm
        / np.sqrt(4.0 * np.pi)
    )
    to_mm = np.abs(to_mm)
    degrees, orders, sine = labels(args.obs_degree)
    ls = np.arange(2, args.obs_degree + 1)

    draws = [prior.sample(rng=rng) for _ in range(args.samples)]
    signal = np.array([forward(draw) for draw in draws])
    signal_variance = np.array([np.mean(signal[:, degrees == l] ** 2) for l in ls])[
        degrees - 2
    ]
    noise_variance = np.diag(covariance)
    signal_mm = np.sqrt(signal_variance) * to_mm
    noise_mm = np.sqrt(noise_variance) * to_mm
    ratio = signal_mm / noise_mm

    def by_degree(values, reduce):
        return np.array([reduce(values[degrees == l]) for l in ls])

    def rms(values):
        return np.sqrt(np.mean(values**2))

    signal_l = by_degree(signal_mm, rms)
    noise_rms = by_degree(noise_mm, rms)
    noise_median = by_degree(noise_mm, np.median)
    noise_min, noise_max = by_degree(noise_mm, np.min), by_degree(noise_mm, np.max)
    share = by_degree(ratio, lambda v: np.mean(v > 1.0))

    text = (
        "\n\nGRACE noise against the prior's signal, by degree: the std of a "
        "coefficient in mm EWT\n(4pi-normalised, through the load Love "
        "numbers), and the share of the degree's\ncoefficients in which the "
        "signal exceeds the noise.\n"
        f"{'l':>4}{'signal':>10}{'noise rms':>11}{'median':>9}{'min':>9}"
        f"{'max':>9}{'s/n rms':>9}{'s/n med':>9}{'share':>7}\n"
    )
    for i, l in enumerate(ls):
        text += (
            f"{l:4d}{signal_l[i]:10.3f}{noise_rms[i]:11.3f}{noise_median[i]:9.3f}"
            f"{noise_min[i]:9.3f}{noise_max[i]:9.3f}{signal_l[i] / noise_rms[i]:9.2f}"
            f"{signal_l[i] / noise_median[i]:9.2f}{share[i]:7.2f}\n"
        )

    def total(values):
        return np.sqrt(np.sum(values**2))

    text += (
        f"\nPointwise std to degree {args.obs_degree}, unfiltered: signal "
        f"{total(signal_mm):.1f} mm EWT, noise {total(noise_mm):.1f} mm EWT "
        "(root mean square over the sphere)\n"
    )
    print(text)
    common.write_metrics(METRICS_FILE, text)
    figures = {}

    # ---------------- spectra ----------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    ax = axes[0]
    ax.fill_between(
        ls,
        noise_min,
        noise_max,
        color=NOISE,
        alpha=0.15,
        lw=0,
        label="Noise, range over the orders",
    )
    ax.plot(ls, noise_rms, color=NOISE, lw=2, label="Noise, RMS over the orders")
    ax.plot(
        ls, noise_median, color=NOISE, lw=2, ls="--", label="Noise, median coefficient"
    )
    ax.plot(ls, signal_l, color=SIGNAL, lw=2, label="Signal (prior)")
    ax.set_yscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Std of a coefficient (mm EWT)")
    ax.set_title("Signal and noise by degree")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.axhline(1.0, color=MUTED, lw=1)
    ax.plot(ls, signal_l / noise_rms, color=NOISE, lw=2, label="Against the RMS noise")
    ax.plot(
        ls,
        signal_l / noise_median,
        color=NOISE,
        lw=2,
        ls="--",
        label="Against the median coefficient's noise",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Signal-to-noise ratio (amplitude)")
    ax.set_title("Signal-to-noise ratio by degree")
    ax.legend(frameon=False)
    ax = axes[2]
    ax.plot(ls, share, color=SIGNAL, lw=2)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Share of the degree's coefficients")
    ax.set_title("Coefficients with signal above noise")
    for ax in axes:
        ax.grid(True, color="#e1e0d9", lw=0.6)
        ax.set_xlim(2, args.obs_degree)
    figures["grace_noise_spectrum.png"] = fig

    # ---------------- by degree and order ----------------
    def triangle(values):
        """Rows are degrees; columns orders, sine terms at negative orders."""
        grid = np.full((args.obs_degree + 1, 2 * args.obs_degree + 1), np.nan)
        grid[degrees, args.obs_degree + np.where(sine, -orders, orders)] = values
        return grid

    extent = [
        -args.obs_degree - 0.5,
        args.obs_degree + 0.5,
        args.obs_degree + 0.5,
        -0.5,
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    image = axes[0].imshow(
        triangle(noise_mm),
        extent=extent,
        aspect="auto",
        cmap="Blues",
        norm=LogNorm(vmin=noise_mm.min(), vmax=noise_mm.max()),
    )
    fig.colorbar(image, ax=axes[0], label="Std of the noise (mm EWT)")
    axes[0].set_title("Noise by degree and order")
    limit = max(ratio.max(), 1.0 / ratio.min())
    image = axes[1].imshow(
        triangle(ratio),
        extent=extent,
        aspect="auto",
        cmap="RdBu",
        norm=LogNorm(vmin=1.0 / limit, vmax=limit),
    )
    fig.colorbar(image, ax=axes[1], label="Signal-to-noise ratio (amplitude)")
    axes[1].set_title("Signal-to-noise ratio by degree and order")
    for ax in axes:
        ax.set_xlabel("Order (sine terms negative)")
        ax.set_ylabel("Degree")
    figures["grace_noise_coefficients.png"] = fig

    # ---------------- maps ----------------
    latitudes = np.deg2rad(np.arange(-88.75, 90.0, 2.5))
    longitudes = np.deg2rad(np.arange(-178.75, 180.0, 2.5))
    std = pointwise_std(covariance, degrees, orders, sine, to_mm, latitudes, longitudes)
    noise_field = to_load(noise.sample(rng=rng)) * scale_mm
    signal_field = to_load(signal[0]) * scale_mm
    limit = float(np.percentile(np.abs(noise_field.data), 99.5))
    fig, axes = sl.subplots(1, 3, figsize=(25, 5.6), gridspec_kw={"wspace": 0.08})
    for ax, field, title in (
        (axes[0], noise_field, "A draw of the noise"),
        (axes[1], signal_field, "A draw of the prior, as GRACE's data"),
    ):
        _, image = sl.plot(
            field,
            ax=ax,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            colorbar=True,
            colorbar_kwargs=COLORBAR,
            gridlines_kwargs=GRIDLINES,
        )
        ax.set_title(f"{title}, to degree {args.obs_degree}")
        image.colorbar.set_label("mm EWT")
    import cartopy.crs as ccrs

    image = axes[2].pcolormesh(
        np.rad2deg(longitudes),
        np.rad2deg(latitudes),
        std,
        cmap="Blues",
        vmin=0.0,
        transform=ccrs.PlateCarree(),
    )
    axes[2].coastlines(lw=0.6)
    axes[2].set_global()
    bar = fig.colorbar(image, ax=axes[2], **COLORBAR)
    bar.set_label("mm EWT")
    axes[2].set_title(f"Pointwise std of the noise, to degree {args.obs_degree}")
    figures["grace_noise_maps.png"] = fig

    for name, fig in figures.items():
        path = os.path.join(OUTPUT_DIR, name)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()
