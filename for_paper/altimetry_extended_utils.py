"""
altimetry_extended_utils.py
===========================
Shared utilities, physics initializations, and Bayesian measures for
satellite altimetry inversions with separated ocean dynamic and steric components.
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
)
from pyslfp.linear_operators.utils import spatial_multiplication_operator


def build_physics_components(
    lmax, load_order, load_scale_km, points, is_surrogate=False
):
    """
    Constructs the 3-component physical operators.
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

    # 2. Extract specific physical responses needed for Sea Surface Height
    # Map [Ice, Dyn, Rho] -> [Total Load, OceanDyn]
    # The Rho component is dropped here as it only contributes to the Total Mass Load
    op1 = inf.ColumnLinearOperator([joint_to_load, joint_space.subspace_projection(1)])

    # 3. Resolve the Sea Surface Height Components
    # Barystatic SSH from SLE
    static_ssh_op = sea_surface_height_operator(state, finger_print_operator.codomain)
    slc_operator = static_ssh_op @ finger_print_operator

    # Map [Total Load, OceanDyn] -> [Barystatic SSH, Dynamic Topo]
    op2 = inf.BlockDiagonalLinearOperator(
        [slc_operator, load_space.identity_operator()]
    )

    # Combine into Total Continuous SSH (Barystatic + Dynamic)
    op3 = inf.RowLinearOperator(
        [load_space.identity_operator(), load_space.identity_operator()]
    )

    continuous_ssh_operator = op3 @ op2 @ op1

    # Extract discrete altimetry points
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
    ocean_dyn_scale_factor,
    ocean_dyn_std_factor,
    ocean_rho_scale_factor,
    ocean_rho_std_factor,
    noise_scale_factor,
    noise_std_factor,
    points,
    scale_mm,
    /,
    *,
    prior_shift=0.0,
    is_surrogate=False,
):
    """
    Constructs the 3-component joint prior and observation noise measures.
    Skips the spatial projection and mass conservation conditioning if is_surrogate=True.
    """

    # --- 1. ICE PRIOR ---
    ice_scale = load_space.scale * ice_scale_factor
    ice_std = ice_std_mm / scale_mm
    ice_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_scale, std=ice_std
    )

    # Calculate GMSL variance to scale ocean and noise priors appropriately
    GMSL_weighting_function = (
        -state.model.parameters.ice_density
        * state.one_minus_ocean_function
        / (state.model.parameters.water_density * state.ocean_area)
    )
    B = averaging_operator(state, load_space, [GMSL_weighting_function])
    GMSL_prior_measure = ice_prior.affine_mapping(operator=B)
    GMSL_prior_std = np.sqrt(GMSL_prior_measure.covariance.matrix(dense=True)[0, 0])

    # --- 2. OCEAN DYNAMIC PRIOR ---
    ocean_dyn_scale = load_space.scale * ocean_dyn_scale_factor
    ocean_dyn_std = ocean_dyn_std_factor * GMSL_prior_std
    ocean_dyn_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_dyn_scale, std=ocean_dyn_std
    )

    # --- 3. OCEAN DENSITY PRIOR ---
    ocean_rho_scale = load_space.scale * ocean_rho_scale_factor
    ocean_rho_std = ocean_rho_std_factor * GMSL_prior_std
    ocean_rho_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_rho_scale, std=ocean_rho_std
    )

    # --- JOINT MEASURE ---
    model_prior = inf.GaussianMeasure.from_direct_sum(
        [ice_prior, ocean_dyn_prior, ocean_rho_prior]
    )

    # --- PHYSICAL CONDITIONING (EXACT ONLY) ---
    if not is_surrogate:
        # A. Spatial Boundary Masking
        ice_proj = ice_projection_operator(state, load_space)
        ocean_proj = ocean_projection_operator(state, load_space)

        proj_op = inf.BlockDiagonalLinearOperator([ice_proj, ocean_proj, ocean_proj])
        model_prior = model_prior.affine_mapping(operator=proj_op)

        # B. Strict Mass Conservation via Affine Subspace
        avg_op = ocean_average_operator(state, load_space)

        dyn_to_load = sea_level_change_to_load_operator(state, load_space, load_space)
        rho_to_load = ocean_density_change_to_load_operator(
            state, load_space, load_space
        )

        # B([Ice, Dyn, Rho]) = 0*Ice + Avg(Load(Dyn)) + Avg(Load(Rho))
        zero_op = load_space.zero_operator(codomain=avg_op.codomain)
        mass_constraint_op = inf.RowLinearOperator(
            [zero_op, avg_op @ dyn_to_load, avg_op @ rho_to_load]
        )

        zero_val = avg_op.codomain.zero

        # Build the physical manifold constraint
        mass_subspace = inf.AffineSubspace.from_linear_equation(
            operator=mass_constraint_op,
            value=zero_val,
            solver=inf.CholeskySolver(
                galerkin=True
            ),  # 1D constraint, Cholesky is trivial
        )

        # Condition the spatial prior strictly onto this subspace
        model_prior = mass_subspace.condition_gaussian_measure(model_prior)

    # --- NOISE MODEL ---
    noise_std = noise_std_factor * GMSL_prior_std
    if noise_scale_factor == 0.0 or is_surrogate:
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

    # Prior shift
    if prior_shift != 0.0:
        offset_shape = model_prior.sample()
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(prior_shift, offset_shape)
        )

    return model_prior, noise_meas, GMSL_prior_std
