"""
joint_extended_utils.py
=======================
Shared utilities, physics initializations, and Bayesian measures for
extended Joint (Altimetry + GRACE) inversions using a 3-component model.
"""

import numpy as np
import pygeoinf as inf

from pyslfp.state import EarthState
from pyslfp.linear_operators import (
    FingerPrintOperator,
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    ocean_density_change_to_load_operator,
    sea_surface_height_operator,
    ice_projection_operator,
    ocean_projection_operator,
    ocean_average_operator,
    averaging_operator,
    grace_observation_operator,
    WMBMethod,
)


def build_physics_components(
    lmax, load_order, load_scale_km, points, obs_degree, is_surrogate=False
):
    """
    Constructs the 3-component physical operators for Joint Inversion.
    Model Space: [Ice Thickness Change, Ocean Dynamic Topography, Ocean Density Change]
    """
    state = EarthState.from_defaults(lmax=lmax)
    scale_mm = 1000.0 * state.model.parameters.length_scale
    load_scale = load_scale_km * 1000.0 / state.model.parameters.length_scale

    max_iters = 1 if is_surrogate else None

    finger_print_operator = FingerPrintOperator(
        state,
        load_parameters=(load_order, load_scale),
        response_parameters=(load_order, load_scale),
        max_iterations=max_iters,
    )
    load_space = finger_print_operator.domain

    # 1. Operators to convert each component to a surface mass load
    ice_to_load = ice_thickness_change_to_load_operator(state, load_space, load_space)

    if is_surrogate:
        ocean_dyn_to_load = load_space.zero_operator(load_space)
        ocean_rho_to_load = load_space.zero_operator(load_space)
    else:
        ocean_dyn_to_load = sea_level_change_to_load_operator(
            state, load_space, load_space
        )
        ocean_rho_to_load = ocean_density_change_to_load_operator(
            state, load_space, load_space
        )

    # Joint to Total Load: [Ice, OceanDyn, OceanRho] -> Total Mass Load
    joint_to_load = inf.RowLinearOperator(
        [ice_to_load, ocean_dyn_to_load, ocean_rho_to_load]
    )
    joint_space = joint_to_load.domain

    # --- A. ALTIMETRY TRACK (SSH) ---
    op1_alt = inf.ColumnLinearOperator(
        [joint_to_load, joint_space.subspace_projection(1)]
    )
    static_ssh_op = sea_surface_height_operator(state, finger_print_operator.codomain)
    barystatic_ssh_op = static_ssh_op @ finger_print_operator

    op2_ssh = inf.BlockDiagonalLinearOperator(
        [barystatic_ssh_op, load_space.identity_operator()]
    )
    op3_alt = inf.RowLinearOperator(
        [load_space.identity_operator(), load_space.identity_operator()]
    )

    continuous_ssh_operator = op3_alt @ op2_ssh @ op1_alt
    point_eval = load_space.point_evaluation_operator(points)
    alt_track = point_eval @ continuous_ssh_operator

    # --- B. GRACE TRACK ---
    # GRACE observes the response to the Total Mass Load directly.
    grace_obs_op = grace_observation_operator(
        finger_print_operator.codomain, obs_degree
    )
    grace_track = grace_obs_op @ finger_print_operator @ joint_to_load

    # --- C. TRUE SEA LEVEL TRACK (For GMSL) ---
    barystatic_sl_op = (
        finger_print_operator.codomain.subspace_projection(0) @ finger_print_operator
    )
    op2_sl = inf.BlockDiagonalLinearOperator(
        [barystatic_sl_op, load_space.identity_operator()]
    )
    continuous_sl_operator = op3_alt @ op2_sl @ op1_alt

    # --- D. FULL JOINT FORWARD OPERATOR ---
    joint_forward_operator = inf.ColumnLinearOperator([alt_track, grace_track])

    return {
        "state": state,
        "load_space": load_space,
        "fp_op": finger_print_operator,
        "alt_track": alt_track,
        "grace_track": grace_track,
        "joint_forward": joint_forward_operator,
        "continuous_ssh": continuous_ssh_operator,
        "continuous_sl": continuous_sl_operator,
        "scale_mm": scale_mm,
    }


def true_gmsl_operator(state, load_space, continuous_sl_operator):
    """Returns the true spatial integration of continuous Sea Level."""
    true_avg_weight = state.ocean_projection(value=0.0) / state.ocean_area
    true_avg_op = averaging_operator(state, load_space, [true_avg_weight])
    return true_avg_op @ continuous_sl_operator


def build_measures(
    state,
    load_space,
    ice_scale_factor,
    ice_std_mm,
    ocean_dyn_scale_factor,
    ocean_dyn_std_factor,
    ocean_rho_scale_factor,
    ocean_rho_std_factor,
    alt_noise_scale_factor,
    alt_noise_std_factor,
    grace_noise_scale_km,
    grace_noise_std_factor,
    obs_degree,
    points,
    scale_mm,
    prior_shift=0.0,
    is_surrogate=False,
):
    """Constructs the 3-component joint prior and dual-sensor noise measures."""

    # --- 1. PRIORS ---
    ice_scale = load_space.scale * ice_scale_factor
    ice_std = ice_std_mm / scale_mm
    ice_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_scale, std=ice_std
    )

    GMSL_weighting_function = (
        -state.model.parameters.ice_density
        * state.one_minus_ocean_function
        / (state.model.parameters.water_density * state.ocean_area)
    )
    B = averaging_operator(state, load_space, [GMSL_weighting_function])
    GMSL_prior_std = np.sqrt(
        ice_prior.affine_mapping(operator=B).covariance.matrix(dense=True)[0, 0]
    )

    ocean_dyn_scale = load_space.scale * ocean_dyn_scale_factor
    ocean_dyn_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_dyn_scale, std=ocean_dyn_std_factor * GMSL_prior_std
    )

    ocean_rho_scale = load_space.scale * ocean_rho_scale_factor
    mean_ocean_depth = state.model.integrate(state.sea_level) / state.ocean_area
    ocean_rho_std = (ocean_rho_std_factor * GMSL_prior_std) * (
        state.model.parameters.water_density / mean_ocean_depth
    )
    ocean_rho_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_rho_scale, std=ocean_rho_std
    )

    unmasked_prior = inf.GaussianMeasure.from_direct_sum(
        [ice_prior, ocean_dyn_prior, ocean_rho_prior]
    )

    # --- PHYSICAL CONDITIONING ---
    model_prior = unmasked_prior
    if not is_surrogate:
        ice_proj = ice_projection_operator(state, load_space)
        ocean_proj = ocean_projection_operator(state, load_space)
        model_prior = model_prior.affine_mapping(
            operator=inf.BlockDiagonalLinearOperator([ice_proj, ocean_proj, ocean_proj])
        )

        avg_op = ocean_average_operator(state, load_space)
        dyn_to_load = sea_level_change_to_load_operator(state, load_space, load_space)
        rho_to_load = ocean_density_change_to_load_operator(
            state, load_space, load_space
        )

        mass_constraint_op = inf.RowLinearOperator(
            [
                load_space.zero_operator(codomain=avg_op.codomain),
                avg_op @ dyn_to_load,
                avg_op @ rho_to_load,
            ]
        )

        mass_subspace = inf.AffineSubspace.from_linear_equation(
            operator=mass_constraint_op,
            value=avg_op.codomain.zero,
            solver=inf.CholeskySolver(galerkin=True),
        )
        model_prior = mass_subspace.condition_gaussian_measure(model_prior)

    if prior_shift != 0.0:
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(prior_shift, model_prior.sample())
        )

    if is_surrogate:
        return {"model_prior": model_prior, "unmasked_prior": unmasked_prior}

    # --- 2. NOISE MEASURES ---
    # Altimetry Noise
    alt_noise_std = alt_noise_std_factor * GMSL_prior_std
    if alt_noise_scale_factor == 0.0:
        alt_noise_meas = inf.GaussianMeasure.from_standard_deviation(
            inf.EuclideanSpace(len(points)), alt_noise_std
        )
    else:
        alt_noise_meas = load_space.point_value_scaled_heat_kernel_gaussian_measure(
            load_space.scale * alt_noise_scale_factor, std=alt_noise_std
        ).affine_mapping(operator=load_space.point_evaluation_operator(points))

    # GRACE Noise
    grace_spatial_scale = (
        grace_noise_scale_km * 1000.0 / state.model.parameters.length_scale
    )
    grace_spatial_noise = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        grace_spatial_scale, std=grace_noise_std_factor * ice_std
    )
    wmb = WMBMethod(state.model, obs_degree)
    grace_noise_meas = wmb.load_measure_to_observation_measure(grace_spatial_noise)

    joint_noise_meas = inf.GaussianMeasure.from_direct_sum(
        [alt_noise_meas, grace_noise_meas]
    )

    return {
        "model_prior": model_prior,
        "unmasked_prior": unmasked_prior,
        "alt_noise": alt_noise_meas,
        "grace_noise": grace_noise_meas,
        "joint_noise": joint_noise_meas,
        "wmb": wmb,
        "gmsl_std": GMSL_prior_std,
        "ice_std": ice_std,
        "ice_scale": ice_scale,
    }


def regional_decomposition_operators(state, load_space, finger_print_operator, regions):
    """Decomposes the 3-component state into regional signals (Dyn, Rho, SLE)."""
    masks = [state.get_projection(r, value=0.0) for r in regions]
    avg_op = averaging_operator(state, load_space, masks)
    joint_space = inf.HilbertSpaceDirectSum([load_space, load_space, load_space])

    op_dyn = avg_op @ joint_space.subspace_projection(1)
    op_rho = avg_op @ joint_space.subspace_projection(2)

    ice_to_load = ice_thickness_change_to_load_operator(state, load_space, load_space)
    barystatic_sl_op = (
        finger_print_operator.codomain.subspace_projection(0) @ finger_print_operator
    )
    op_ice_fp = (
        avg_op @ barystatic_sl_op @ ice_to_load @ joint_space.subspace_projection(0)
    )

    return op_dyn, op_rho, op_ice_fp
