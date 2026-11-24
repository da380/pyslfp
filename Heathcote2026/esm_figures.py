"""
Figures of the loads of the updated ESA Earth System Model (Heathcote et
al., 2026), as `esm_loads.py` reads them: the year-on-year changes of its
hydrology (H), land ice (I) and ocean (O) components.

Outputs, in output_plots_esm/ at 200 dpi:

- esm_spectra.png: the root mean square coefficient of each component's
  changes by degree, to degree 180, with the Sobolev spectrum fitted to it
  (`common.fit_spectrum`) and, for comparison, the spectrum of a Sobolev
  measure of order 3 at 250 km on a Sobolev space of order 2 at 500 km, the
  prior of the original GRACE experiments, matched at low degree.
- esm_variability.png: the root mean square over the changes of the summed
  load at each point, on a logarithmic scale, and their mean.
- esm_example.png: the three components of one change (--example).
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import pyslfp as sl

import common
import esm_loads
from figures import COLORBAR, GRIDLINES

OUTPUT_DIR = "output_plots_esm"
NAMES = {"H": "Hydrology", "I": "Land ice", "O": "Ocean"}
COLOURS = {"H": "tab:green", "I": "tab:blue", "O": "tab:orange"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figures of the ESA model's loads.")
    parser.add_argument("--lmax", type=int, default=180)
    parser.add_argument(
        "--example", type=int, default=-1, help="The index of the change shown."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    common.start_run(OUTPUT_DIR, args)
    physics = common.budget_physics(args.lmax)
    model, space, scale_mm = physics.state.model, physics.load_space, physics.scale_mm
    radius_km = space.radius * model.parameters.length_scale / 1000.0
    labels, components = None, {}
    for name in NAMES:
        labels, components[name] = esm_loads.load_changes(model, name)
    degrees = np.arange(space.lmax + 1)
    k = degrees * (degrees + 1.0) / space.radius**2
    figures = {}

    # ---------------- spectra ----------------
    fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
    total_variance = None
    for name, fields in components.items():
        variance = common.degree_variances(space, fields) * scale_mm**2
        order, scale = common.fit_spectrum(variance, space.radius)
        fitted = (1.0 + scale**2 * k) ** (-order)
        fitted *= np.exp(np.mean(np.log(variance[2:]) - np.log(fitted[2:])))
        ax.loglog(
            degrees[1:], np.sqrt(variance[1:]), color=COLOURS[name], label=NAMES[name]
        )
        ax.loglog(
            degrees[1:],
            np.sqrt(fitted[1:]),
            color=COLOURS[name],
            linestyle="--",
            label=f"  fit: order {order:.2f}, {scale / space.radius * radius_km:.0f} km",
        )
        total_variance = variance if total_variance is None else total_variance
    summed = [sum(components[name][i] for name in NAMES) for i in range(len(labels))]
    variance = common.degree_variances(space, summed) * scale_mm**2
    ax.loglog(degrees[1:], np.sqrt(variance[1:]), color="black", label="Sum")
    km = 1.0 / radius_km * space.radius
    original = (1.0 + (250.0 * km) ** 2 * k) ** -3 * (1.0 + (500.0 * km) ** 2 * k) ** -2
    original *= variance[2:9].mean() / original[2:9].mean()
    ax.loglog(
        degrees[1:],
        np.sqrt(original[1:]),
        color="tab:red",
        linestyle=":",
        linewidth=2,
        label="Original prior (matched at degrees 2 to 8)",
    )
    ax.set_xlabel("Degree")
    ax.set_ylabel("Root mean square coefficient (mm EWT)")
    ax.set_ylim(1.0e-3, None)
    ax.set_title("Year-on-year changes of the ESA Earth System Model")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    figures["esm_spectra.png"] = fig

    # ---------------- variability and mean ----------------
    stack = np.array([field.data for field in summed]) * scale_mm
    rms = summed[0].copy()
    rms.data[:] = np.log10(np.maximum(np.sqrt(np.mean(stack**2, axis=0)), 1.0e-3))
    mean = summed[0].copy()
    mean.data[:] = np.mean(stack, axis=0)
    fig, axes = sl.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"wspace": 0.08})
    _, image = sl.plot(
        rms,
        ax=axes[0],
        cmap="magma_r",
        vmin=0.0,
        vmax=float(np.percentile(rms.data, 99.9)),
        colorbar=True,
        colorbar_kwargs=COLORBAR,
        gridlines_kwargs=GRIDLINES,
    )
    axes[0].set_title(f"RMS of the {len(summed)} changes (H + I + O)")
    image.colorbar.set_label(r"$\log_{10}$ of the RMS in mm EWT")
    limit = float(np.percentile(np.abs(mean.data), 99.5))
    _, image = sl.plot(
        mean,
        ax=axes[1],
        cmap="seismic",
        vmin=-limit,
        vmax=limit,
        colorbar=True,
        colorbar_kwargs=COLORBAR,
        gridlines_kwargs=GRIDLINES,
    )
    axes[1].set_title("Mean of the changes")
    image.colorbar.set_label("mm EWT")
    figures["esm_variability.png"] = fig

    # ---------------- one change ----------------
    fig, axes = sl.subplots(1, 3, figsize=(25, 5.6), gridspec_kw={"wspace": 0.08})
    for ax, name in zip(axes, NAMES):
        field = components[name][args.example] * scale_mm
        limit = float(np.percentile(np.abs(field.data), 99.5))
        _, image = sl.plot(
            field,
            ax=ax,
            cmap="seismic",
            vmin=-limit,
            vmax=limit,
            colorbar=True,
            colorbar_kwargs=COLORBAR,
            gridlines_kwargs=GRIDLINES,
        )
        ax.set_title(f"{NAMES[name]}: change to {labels[args.example][:4]}")
        image.colorbar.set_label("mm EWT")
    figures["esm_example.png"] = fig

    common.save_figures(figures, OUTPUT_DIR, dpi=200)


if __name__ == "__main__":
    main()
