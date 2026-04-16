"""
joint_utils.py
==============
Shared utilities, physics initializations, and Bayesian measures for
Joint Satellite Altimetry and GRACE gravimetry inversions.
"""

import numpy as np
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev, Lebesgue
from pyslfp.state import EarthState, EarthModel
from pyslfp.linear_operators import (
    FingerPrintOperator,
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    ice_projection_operator,
    ocean_projection_operator,
    averaging_operator,
    remove_ocean_average_operator,
    grace_observation_operator,
    WMBMethod,
)


def build_physics_components(
    lmax, load_order, load_scale_km, points, obs_degree, is_surrogate=False
):
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

    # --- 1. Assembly: Joint Space -> [Total Load, Ocean] ---
    ice_to_load = ice_thickness_change_to_load_operator(state, load_space, load_space)
    if is_surrogate:
        ocean_to_load = load_space.zero_operator(load_space)
    else:
        ocean_to_load = sea_level_change_to_load_operator(state, load_space, load_space)

    joint_to_load = inf.RowLinearOperator([ice_to_load, ocean_to_load])
    joint_space = inf.HilbertSpaceDirectSum([load_space, load_space])

    op1 = inf.ColumnLinearOperator([joint_to_load, joint_space.subspace_projection(1)])

    # --- 2. Physics: [Total Load, Ocean] -> [SLE Response, Ocean] ---
    op2 = inf.BlockDiagonalLinearOperator(
        [finger_print_operator, load_space.identity_operator()]
    )

    # --- 3a. Altimetry Track: [SLE Response, Ocean] -> Discrete SSH Points ---
    static_ssh_op = sea_surface_height_operator(state, finger_print_operator.codomain)
    alt_continuous_op = inf.RowLinearOperator(
        [static_ssh_op, load_space.identity_operator()]
    )
    point_eval = load_space.point_evaluation_operator(points)
    alt_track = point_eval @ alt_continuous_op

    # --- 3b. GRACE Track: [SLE Response, Ocean] -> Potential SH Coefficients ---
    grace_obs_op = grace_observation_operator(
        finger_print_operator.codomain, obs_degree
    )
    grace_track = inf.RowLinearOperator(
        [grace_obs_op, load_space.zero_operator(codomain=grace_obs_op.codomain)]
    )

    # --- 4. Combine Tracks: [SLE Response, Ocean] -> [Alt Data, GRACE Data] ---
    op3 = inf.ColumnLinearOperator([alt_track, grace_track])

    joint_forward_operator = op3 @ op2 @ op1
    continuous_ssh_operator = alt_continuous_op @ op2 @ op1

    # Return as a dictionary for clean extraction in the main script
    return {
        "state": state,
        "load_space": load_space,
        "fp_op": finger_print_operator,
        "alt_track": alt_track @ op2 @ op1,  # Isolated Alt Forward
        "grace_track": grace_track @ op2 @ op1,  # Isolated GRACE Forward
        "joint_forward": joint_forward_operator,
        "continuous_ssh": continuous_ssh_operator,
        "scale_mm": scale_mm,
    }


def build_measures(
    state,
    load_space,
    ice_scale_km,
    ice_std_mm,
    ocean_scale_km,
    ocean_std_factor,
    alt_noise_std_factor,
    grace_noise_scale_km,
    grace_noise_std_mm,
    obs_degree,
    points,
    scale_mm,
    prior_shift=0.0,
):
    # --- 1. PRIORS ---
    ice_scale = ice_scale_km * 1000.0 / state.model.parameters.length_scale
    ice_std = ice_std_mm / scale_mm
    ice_thickness_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_scale, std=ice_std
    )

    GMSL_weighting_function = (
        -state.model.parameters.ice_density
        * state.ice_projection(value=0.0)
        / (state.model.parameters.water_density * state.ocean_area)
    )

    B = averaging_operator(state, load_space, [GMSL_weighting_function])
    GMSL_prior_measure = ice_thickness_prior.affine_mapping(operator=B)
    GMSL_prior_std = np.sqrt(GMSL_prior_measure.covariance.matrix(dense=True)[0, 0])

    ocean_scale = ocean_scale_km * 1000.0 / state.model.parameters.length_scale
    ocean_std = ocean_std_factor * GMSL_prior_std
    ocean_thickness_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_scale, std=ocean_std
    )
    ocean_thickness_prior = ocean_thickness_prior.affine_mapping(
        operator=remove_ocean_average_operator(state, load_space)
    )

    model_prior = inf.GaussianMeasure.from_direct_sum(
        [ice_thickness_prior, ocean_thickness_prior]
    )

    model_prior = model_prior.affine_mapping(
        operator=inf.BlockDiagonalLinearOperator(
            [
                ice_projection_operator(state, load_space),
                ocean_projection_operator(state, load_space),
            ]
        )
    )

    if prior_shift != 0.0:
        offset_shape = model_prior.sample()
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(prior_shift, offset_shape)
        )

    # --- 2. NOISE MEASURES ---
    n_points = len(points)
    alt_noise_std = alt_noise_std_factor * GMSL_prior_std
    alt_data_space = inf.EuclideanSpace(n_points)
    alt_noise_meas = inf.GaussianMeasure.from_standard_deviations(
        alt_data_space, np.full(n_points, alt_noise_std)
    )

    grace_spatial_scale = (
        grace_noise_scale_km * 1000.0 / state.model.parameters.length_scale
    )
    grace_spatial_std = grace_noise_std_mm / scale_mm

    if load_space.lmax < obs_degree:
        if isinstance(load_space, Sobolev):
            noise_load_space = Sobolev(
                obs_degree,
                load_space.order,
                load_space.scale,
                radius=state.model.parameters.mean_sea_floor_radius,
                grid=state.model.grid,
            )
        else:
            noise_load_space = Lebesgue(
                obs_degree,
                radius=state.model.parameters.mean_sea_floor_radius,
                grid=state.model.grid,
            )
    else:
        noise_load_space = load_space

    grace_spatial_noise_meas = (
        noise_load_space.point_value_scaled_heat_kernel_gaussian_measure(
            grace_spatial_scale, std=grace_spatial_std
        )
    )

    if state.model.lmax < obs_degree:
        wmb_model = EarthModel(
            obs_degree, parameters=state.model.parameters, grid=state.model.grid_name
        )
    else:
        wmb_model = state.model

    wmb = WMBMethod(wmb_model, obs_degree)
    grace_noise_meas = wmb.load_measure_to_observation_measure(grace_spatial_noise_meas)

    joint_noise_meas = inf.GaussianMeasure.from_direct_sum(
        [alt_noise_meas, grace_noise_meas]
    )

    # Return as a dictionary for clean extraction
    return {
        "model_prior": model_prior,
        "alt_noise": alt_noise_meas,
        "grace_noise": grace_noise_meas,
        "joint_noise": joint_noise_meas,
        "wmb": wmb,
        "gmsl_std": GMSL_prior_std,
        "ice_scale": ice_scale,
        "ice_std": ice_std,
    }


def true_gmsl_operator(state, load_space, continuous_ssh_operator):
    true_avg_weight = state.ocean_projection(value=0.0) / state.ocean_area
    true_avg_op = averaging_operator(state, load_space, [true_avg_weight])
    return true_avg_op @ continuous_ssh_operator


def regional_decomposition_operators(state, load_space, finger_print_operator, regions):
    masks = [state.get_projection(r, value=0.0) for r in regions]
    avg_op = averaging_operator(state, load_space, masks)
    joint_space = inf.HilbertSpaceDirectSum([load_space, load_space])

    op_dynamic = avg_op @ joint_space.subspace_projection(1)

    ice_to_load = ice_thickness_change_to_load_operator(state, load_space, load_space)
    static_ssh_op = sea_surface_height_operator(state, finger_print_operator.codomain)

    op_ice_fingerprint = (
        avg_op
        @ static_ssh_op
        @ finger_print_operator
        @ ice_to_load
        @ joint_space.subspace_projection(0)
    )

    return op_dynamic, op_ice_fingerprint
