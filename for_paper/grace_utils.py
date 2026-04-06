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
    /,
    *,
    remove_degree_1=False,
    prior_shift=0.0,
):
    """Constructs the prior and noise Gaussian measures."""
    direct_load_measure_scale = direct_scale_km * 1000 / fp.length_scale
    direct_load_measure_std = fp.water_density * direct_std_m / fp.length_scale

    initial_direct_load_prior = (
        load_space.point_value_scaled_heat_kernel_gaussian_measure(
            direct_load_measure_scale, std=direct_load_measure_std
        )
    )

    constraint_lmax = 1 if remove_degree_1 else 0
    constraint_operator = load_space.to_coefficient_operator(constraint_lmax)
    constraint_subspace = inf.LinearSubspace.from_kernel(constraint_operator)
    direct_load_prior = constraint_subspace.condition_gaussian_measure(
        initial_direct_load_prior
    )

    if prior_shift != 0.0:
        offset_shape = direct_load_prior.sample()
        direct_load_prior = direct_load_prior.affine_mapping(
            translation=offset_shape * prior_shift
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
    sle_factor = 1.0 / (fp.water_density * fp.ocean_area)
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

    if smoothing_scale_km is not None and smoothing_scale_km > 0:
        smoothing_scale = smoothing_scale_km * 1000 / fp.length_scale
        smoothing_measure = load_space.heat_kernel_gaussian_measure(smoothing_scale)
        smoothing_operator = smoothing_measure.covariance
        weighting_functions = [smoothing_operator(wf) for wf in weighting_functions]

    averaging_operator = sl.averaging_operator(load_space, weighting_functions)

    return region_names, averaging_operator, weighting_functions


def plot_regional_pdfs(results_dict, region_names, true_averages_mm):
    """Helper function to plot a 2x2 grid of regional PDFs comparing multiple estimators"""

    class MockMeasure:
        def __init__(self, m, s):
            self.mean = np.array([m])
            self.cov = np.array([[s**2]])

    ncols = 2
    nrows = int(np.ceil(len(region_names) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows), layout="constrained"
    )
    axes_flat = axes.flatten()

    labels = list(results_dict.keys())

    for i, region in enumerate(region_names):
        ax = axes_flat[i]
        true_val = true_averages_mm[i]

        measures = [
            MockMeasure(res["means"][i], res["stds"][i])
            for res in results_dict.values()
        ]

        inf.plot_1d_distributions(
            measures,
            true_value=true_val,
            ax=ax,
            xlabel="Regional Average Mass (mm EWT)",
            title=f"{region}",
            posterior_labels=labels,
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
