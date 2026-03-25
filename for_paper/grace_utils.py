"""
grace_utils.py
==============
Shared utilities, physics initializations, and plotting for Bayesian GRACE inversions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
import pyslfp as sl


def build_physics_components(lmax, load_order, load_scale_km):
    """Initializes the Earth model, Sobolev spaces, and fingerprint operator."""
    fp = sl.FingerPrint(
        lmax=lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    load_to_water_thickness_mm = 1000 * fp.length_scale / fp.water_density
    load_space_scale = load_scale_km * 1000 / fp.length_scale

    finger_print_operator = fp.as_sobolev_linear_operator(load_order, load_space_scale)
    load_space = finger_print_operator.domain
    response_space = finger_print_operator.codomain

    return (
        fp,
        load_space,
        response_space,
        finger_print_operator,
        load_to_water_thickness_mm,
    )


def build_measures(
    fp,
    load_space,
    direct_scale_km,
    direct_std_m,
    noise_scale_factor,
    noise_std_factor,
    remove_deg_1=False,
):
    """Constructs the prior and noise Gaussian measures."""
    direct_load_measure_scale = direct_scale_km * 1000 / fp.length_scale
    direct_load_measure_std = fp.water_density * direct_std_m / fp.length_scale

    initial_direct_load_prior = (
        load_space.point_value_scaled_heat_kernel_gaussian_measure(
            direct_load_measure_scale, std=direct_load_measure_std
        )
    )

    constraint_lmax = 1 if remove_deg_1 else 0
    constraint_operator = load_space.to_coefficient_operator(constraint_lmax)
    constraint_subspace = inf.LinearSubspace.from_kernel(constraint_operator)
    direct_load_prior = constraint_subspace.condition_gaussian_measure(
        initial_direct_load_prior
    )

    noise_load_measure_scale = noise_scale_factor * direct_load_measure_scale
    noise_load_measure_std = noise_std_factor * direct_load_measure_std
    noise_load_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        noise_load_measure_scale, std=noise_load_measure_std
    )

    return initial_direct_load_prior, direct_load_prior, noise_load_measure


def build_total_load_operator(fp, response_space, load_space, finger_print_operator):
    """Builds the operator linking direct load to total physical load (including SLE)."""
    sea_level_projection = response_space.subspace_projection(0)
    sea_level_to_load = sl.sea_level_change_to_load_operator(
        fp, sea_level_projection.codomain, load_space
    )
    induced_load_operator = (
        sea_level_to_load @ sea_level_projection @ finger_print_operator
    )
    return load_space.identity_operator() + induced_load_operator


def get_regional_averaging(fp, load_space, smoothing_scale_km=None):
    """Sets up the averaging operator, optionally applying a spatial smoothing filter."""
    sle_factor = -1.0 / (fp.water_density * fp.ocean_area)
    selected_regions = [
        "Greenland/Iceland",
        "W.Antarctica",
        "S.Indic-Ocean",
        "South-American-Monsoon",
    ]
    target_regions = {
        region: fp.regionmask_projection(region, value=0) * sle_factor
        for region in selected_regions
    }

    region_names = list(target_regions.keys())
    weighting_functions = list(target_regions.values())

    # Apply spatial smoothing via the strictly unscaled heat kernel covariance
    if smoothing_scale_km is not None and smoothing_scale_km > 0:
        smoothing_scale = smoothing_scale_km * 1000 / fp.length_scale
        smoothing_measure = load_space.heat_kernel_gaussian_measure(smoothing_scale)
        smoothing_operator = smoothing_measure.covariance
        # Physically smooth the functions before passing them to the averager
        weighting_functions = [smoothing_operator(wf) for wf in weighting_functions]

    averaging_operator = sl.averaging_operator(load_space, weighting_functions)

    return region_names, averaging_operator, weighting_functions


def plot_regional_pdfs(results_dict, title, region_names, true_averages_mm):
    """Helper function to plot a 2x2 grid of regional PDFs comparing multiple estimators."""

    def gaussian_pdf(x, mean, std):
        return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

    ncols = 2
    nrows = int(np.ceil(len(region_names) / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), layout="constrained"
    )
    axes_flat = axes.flatten()

    if len(results_dict) == 2 and "Bayesian" in results_dict and "WMB" in results_dict:
        colors = ["blue", "red"]
    else:
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results_dict)))

    for i, region in enumerate(region_names):
        ax = axes_flat[i]
        true_val = true_averages_mm[i]

        all_means = [res["means"][i] for res in results_dict.values()] + [true_val]
        max_std = max([res["stds"][i] for res in results_dict.values()])
        plot_min = min(all_means) - 3.5 * max_std
        plot_max = max(all_means) + 3.5 * max_std
        x_vals = np.linspace(plot_min, plot_max, 400)

        for c_idx, (label, res) in enumerate(results_dict.items()):
            mean = res["means"][i]
            std = res["stds"][i]
            y_vals = gaussian_pdf(x_vals, mean, std)
            ax.plot(
                x_vals,
                y_vals,
                color=colors[c_idx],
                linewidth=2.5,
                label=rf"{label} ($\mu$={mean:.2f}, $\sigma$={std:.3f})",
            )
            ax.fill_between(x_vals, 0, y_vals, color=colors[c_idx], alpha=0.15)

        ax.axvline(
            true_val,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"True Value: {true_val:.2f}",
        )
        ax.set_title(f"{region}", fontsize=14)
        ax.set_xlabel("Regional Average Mass (mm EWT)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=9)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=18, fontweight="bold")
