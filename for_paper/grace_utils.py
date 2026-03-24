"""
grace_utils.py
==============
Shared utilities and physics initializations for Bayesian GRACE inversions.
"""

import numpy as np
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
    remove_deg_0=True,
):
    """Constructs the prior and noise Gaussian measures."""
    direct_load_measure_scale = direct_scale_km * 1000 / fp.length_scale
    direct_load_measure_std = fp.water_density * direct_std_m / fp.length_scale

    # 1. The invariant initial measure (required for the preconditioner)
    initial_direct_load_prior = (
        load_space.point_value_scaled_heat_kernel_gaussian_measure(
            direct_load_measure_scale, std=direct_load_measure_std
        )
    )

    # 2. The conditioned measure (required for the actual inversion)
    direct_load_prior = initial_direct_load_prior
    if remove_deg_0:
        constraint_operator = load_space.to_coefficient_operator(0)
        constraint_subspace = inf.LinearSubspace.from_kernel(constraint_operator)
        direct_load_prior = constraint_subspace.condition_gaussian_measure(
            initial_direct_load_prior
        )

    noise_load_measure_scale = noise_scale_factor * direct_load_measure_scale
    noise_load_measure_std = noise_std_factor * direct_load_measure_std
    noise_load_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        noise_load_measure_scale, std=noise_load_measure_std
    )

    # Return BOTH priors
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
    total_load_operator = load_space.identity_operator() + induced_load_operator
    return total_load_operator


def get_regional_averaging(fp, load_space):
    """Sets up the averaging operator for the AR6 test regions."""
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
    averaging_operator = sl.averaging_operator(load_space, weighting_functions)

    return region_names, averaging_operator
