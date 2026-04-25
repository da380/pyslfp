"""
altimetry_utils.py
==================
Shared utilities, physics initializations, and Bayesian measures for
satellite altimetry inversions and bias evaluation.
"""

import numpy as np
import scipy.sparse as sps


import pygeoinf as inf

from pygeoinf import GaussianMeasure
from pygeoinf.symmetric_space.sphere import Lebesgue

from pyslfp.state import EarthState
from pyslfp.linear_operators import (
    FingerPrintOperator,
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    ice_projection_operator,
    ocean_projection_operator,
    averaging_operator,
    remove_ocean_average_operator,
)


def build_physics_components(
    lmax, load_order, load_scale_km, points, is_surrogate=False
):
    """
    Constructs the physical operators for a specific SH degree.
    If is_surrogate=True, builds a "Physics-Lite" model that bypasses the iterative
    Sea Level Equation and zeroes ocean dynamics.
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

    # Assembly: Joint Space -> Load Space
    ice_to_load = ice_thickness_change_to_load_operator(state, load_space, load_space)
    if is_surrogate:
        ocean_to_load = load_space.zero_operator(load_space)
    else:
        ocean_to_load = sea_level_change_to_load_operator(state, load_space, load_space)

    joint_to_load = inf.RowLinearOperator([ice_to_load, ocean_to_load])

    # Assembly: [Ice, Ocean] -> Continuous SSH Field
    joint_space = inf.HilbertSpaceDirectSum([load_space, load_space])
    op1 = inf.ColumnLinearOperator([joint_to_load, joint_space.subspace_projection(1)])

    static_ssh_op = sea_surface_height_operator(state, finger_print_operator.codomain)
    ssh_inclusion = static_ssh_op.codomain.order_inclusion_operator(load_space.order)

    op2 = inf.BlockDiagonalLinearOperator(
        [
            ssh_inclusion @ static_ssh_op @ finger_print_operator,
            load_space.identity_operator(),
        ]
    )

    op3 = inf.RowLinearOperator(
        [load_space.identity_operator(), load_space.identity_operator()]
    )

    continuous_ssh_operator = op3 @ op2 @ op1

    # Extract discrete points
    point_eval = load_space.point_evaluation_operator(points)
    forward_operator = point_eval @ continuous_ssh_operator

    return (
        state,
        load_space,
        finger_print_operator,
        continuous_ssh_operator,
        forward_operator,
        scale_mm,
    )


def build_measures(
    state,
    load_space,
    ice_scale_factor,
    ice_std_mm,
    ocean_scale_factor,
    ocean_std_factor,
    noise_scale_factor,
    noise_std_factor,
    points,
    scale_mm,
    /,
    *,
    prior_shift=0.0,
):
    """Constructs the joint prior and observation noise measures."""
    ice_scale = load_space.scale * ice_scale_factor
    ice_std = ice_std_mm / scale_mm

    ice_thickness_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_scale, std=ice_std
    )
    ice_thickness_prior = ice_thickness_prior.affine_mapping(
        operator=ice_projection_operator(state, load_space)
    )

    GMSL_weighting_function = (
        -state.model.parameters.ice_density
        * state.one_minus_ocean_function
        / (state.model.parameters.water_density * state.ocean_area)
    )

    B = averaging_operator(state, load_space, [GMSL_weighting_function])
    GMSL_prior_measure = ice_thickness_prior.affine_mapping(operator=B)
    GMSL_prior_std = np.sqrt(GMSL_prior_measure.covariance.matrix(dense=True)[0, 0])

    ocean_scale = load_space.scale * ocean_scale_factor
    ocean_std = ocean_std_factor * GMSL_prior_std

    ocean_thickness_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_scale, std=ocean_std
    )

    ocean_thickness_prior = ocean_thickness_prior.affine_mapping(
        operator=ocean_projection_operator(state, load_space)
        @ remove_ocean_average_operator(state, load_space)
    )

    model_prior = inf.GaussianMeasure.from_direct_sum(
        [ice_thickness_prior, ocean_thickness_prior]
    )

    if prior_shift != 0.0:
        offset_shape = model_prior.sample()
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(prior_shift, offset_shape)
        )

    noise_std = noise_std_factor * GMSL_prior_std
    if noise_scale_factor == 0.0:
        n_points = len(points)
        data_space = inf.EuclideanSpace(n_points)
        noise_meas = inf.GaussianMeasure.from_standard_deviation(data_space, noise_std)
    else:
        continuous_noise_meas = (
            load_space.point_value_scaled_heat_kernel_gaussian_measure(
                load_space.scale * noise_scale_factor, std=noise_std
            )
        )
        noise_meas = continuous_noise_meas.affine_mapping(
            operator=load_space.point_evaluation_operator(points)
        )

    return model_prior, noise_meas, GMSL_prior_std


def true_gmsl_operator(state, load_space, continuous_ssh_operator):
    """Returns the true spatial integration of continuous SSH."""
    true_avg_weight = state.ocean_projection(value=0.0) / state.ocean_area
    true_avg_op = averaging_operator(state, load_space, [true_avg_weight])
    return true_avg_op @ continuous_ssh_operator


def regional_decomposition_operators(state, load_space, finger_print_operator, regions):
    """
    Builds operators to isolate the Dynamic Ocean Topography and Ice Melt (SLE)
    contributions to regional sea level changes.
    """

    # Resolve the physical regions into SHGrid masks
    masks = [state.get_projection(r, value=0.0) for r in regions]
    avg_op = averaging_operator(state, load_space, masks)

    joint_space = inf.HilbertSpaceDirectSum([load_space, load_space])

    # 1. Dynamic Ocean Topography Operator (Subspace 1 -> Average)
    op_dynamic = avg_op @ joint_space.subspace_projection(1)

    # 2. Ice Melt (SLE Fingerprint) Operator (Subspace 0 -> Load -> SLE -> SSH -> Average)
    ice_to_load = ice_thickness_change_to_load_operator(state, load_space, load_space)
    static_ssh_op = sea_surface_height_operator(state, finger_print_operator.codomain)
    ssh_inclusion = static_ssh_op.codomain.order_inclusion_operator(load_space.order)

    op_ice_fingerprint = (
        avg_op
        @ ssh_inclusion
        @ static_ssh_op
        @ finger_print_operator
        @ ice_to_load
        @ joint_space.subspace_projection(0)
    )

    return op_dynamic, op_ice_fingerprint
