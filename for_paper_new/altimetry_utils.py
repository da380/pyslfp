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
    l2_products_operator,
)


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
    ice_to_load = ice_thickness_change_to_load_operator(
        state, load_space, load_space
    ) @ ice_projection_operator(state, load_space)

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

    joint_to_load = inf.RowLinearOperator(
        [ice_to_load, ocean_dyn_to_load, ocean_rho_to_load]
    )
    joint_space = joint_to_load.domain

    # Map [Ice, Dyn, Rho] -> [Total Load, OceanDyn]
    op1 = inf.ColumnLinearOperator([joint_to_load, joint_space.subspace_projection(1)])

    # --- A. Resolve the Sea Surface Height Components (For Altimetry) ---
    static_ssh_op = sea_surface_height_operator(state, finger_print_operator.codomain)
    barystatic_ssh_op = static_ssh_op @ finger_print_operator

    op2_ssh = inf.BlockDiagonalLinearOperator(
        [barystatic_ssh_op, load_space.identity_operator()]
    )

    op3 = inf.RowLinearOperator(
        [load_space.identity_operator(), load_space.identity_operator()]
    )

    continuous_ssh_operator = op3 @ op2_ssh @ op1

    # --- B. Resolve the True Sea Level Components (For GMSL Integration) ---
    # Sea Level Change is strictly subspace 0 of the SLE response!
    barystatic_sl_op = (
        finger_print_operator.codomain.subspace_projection(0) @ finger_print_operator
    )

    op2_sl = inf.BlockDiagonalLinearOperator(
        [barystatic_sl_op, load_space.identity_operator()]
    )

    continuous_sl_operator = op3 @ op2_sl @ op1

    # --- Extract discrete altimetry points from SSH ---
    point_eval = load_space.point_evaluation_operator(points)
    forward_operator = point_eval @ continuous_ssh_operator

    return (
        state,
        load_space,
        finger_print_operator,
        continuous_ssh_operator,
        continuous_sl_operator,
        forward_operator,
        scale_mm,
    )


def true_gmsl_operator(state, load_space, continuous_sl_operator):
    """Returns the true spatial integration of continuous Sea Level."""
    true_avg_weight = state.ocean_projection(value=0.0) / state.ocean_area
    true_avg_op = averaging_operator(state, load_space, [true_avg_weight])
    return true_avg_op @ continuous_sl_operator


def barystatic_gmsl_weighting(state):
    """
    Weighting function whose average against an ice thickness change field
    returns the implied barystatic GMSL change (global mass conservation).
    """
    return (
        -state.model.parameters.ice_density
        * state.one_minus_ocean_function
        / (state.model.parameters.water_density * state.ocean_area)
    )


def effective_steric_scale(state):
    """
    Conversion factor from vertically averaged density change to effective
    steric sea level:

        eta_s_eff = effective_steric_scale * delta_rho_w
                  = (mean_ocean_depth / rho_w) * delta_rho_w.

    This is a fixed linear relabelling used (i) to specify the density prior
    std in height units and (ii) to express density fields in mm for plots
    and regional averages. The model parameter itself remains delta_rho_w.
    """
    eta0 = state.sea_level * state.ocean_projection(value=0.0)
    mean_ocean_depth = state.model.integrate(eta0) / state.ocean_area
    return mean_ocean_depth / state.model.parameters.water_density


def gmsl_split_operators(state, load_space, continuous_sl_operator):
    """
    Operators for the 2D push-forward of the model onto GMSL change split
    into barystatic and ocean-dynamic parts.

    Returns (bary_op, dyn_op, dyn_direct_op), each mapping the joint model
    space [Ice, Dyn, Rho] to a scalar:

      bary_op       : barystatic GMSL change, evaluated directly from global
                      mass conservation as (minus) the grounded ice mass
                      change spread uniformly over the oceans.
      dyn_op        : the lumped ocean-dynamic GMSL contribution (dynamic
                      topography plus steric), defined as the residual
                      true GMSL - barystatic, so that the two coordinates
                      sum to the true GMSL by construction.
      dyn_direct_op : the ocean average of the dynamic topography alone.
                      Under exact SLE mass conservation and the ocean mass
                      constraint on (Dyn, Rho), this equals dyn_op; the
                      difference when both are applied to a given model is
                      a direct diagnostic of how well those constraints are
                      satisfied.
    """
    joint_space = continuous_sl_operator.domain

    bary_avg_op = l2_products_operator(load_space, [barystatic_gmsl_weighting(state)])
    bary_op = bary_avg_op @ joint_space.subspace_projection(0)

    total_op = true_gmsl_operator(state, load_space, continuous_sl_operator)
    dyn_op = total_op - bary_op

    ocean_weight = state.ocean_projection(value=0.0) / state.ocean_area
    ocean_avg_op = averaging_operator(state, load_space, [ocean_weight])
    dyn_direct_op = ocean_avg_op @ joint_space.subspace_projection(1)

    return bary_op, dyn_op, dyn_direct_op


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

    Note: ocean_rho_std_factor sets the effective steric sea level std as a
    fraction of the OCEAN DYNAMIC TOPOGRAPHY std (not, as previously, of the
    GMSL std). ocean_dyn_std_factor remains referenced to the GMSL std.
    """

    # --- 1. ICE PRIOR ---
    ice_scale = load_space.scale * ice_scale_factor
    ice_std = ice_std_mm / scale_mm
    ice_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_scale, std=ice_std
    )

    # Calculate GMSL variance to scale ocean and noise priors appropriately
    B = averaging_operator(state, load_space, [barystatic_gmsl_weighting(state)])
    GMSL_prior_measure = ice_prior.affine_mapping(operator=B)
    GMSL_prior_std = np.sqrt(GMSL_prior_measure.covariance.matrix(dense=True)[0, 0])

    # --- 2. OCEAN DYNAMIC PRIOR ---
    ocean_dyn_scale = load_space.scale * ocean_dyn_scale_factor
    ocean_dyn_std = ocean_dyn_std_factor * GMSL_prior_std
    ocean_dyn_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_dyn_scale, std=ocean_dyn_std
    )

    # --- 3. OCEAN DENSITY PRIOR ---
    # The model parameter is the vertically averaged density change,
    # delta_rho_w, with a constant pointwise std within the oceans. That std
    # is specified through the effective steric sea level (a fixed linear
    # relabelling; see effective_steric_scale) as a FRACTION OF THE OCEAN
    # DYNAMIC TOPOGRAPHY STD, so the steric/dynamic ratio is a statement
    # about the two ocean-interior processes. Previously this fraction was
    # referenced to the GMSL std: with the default dyn factor of 2.0, the
    # new default of 0.25 reproduces the old default of 0.5 exactly.
    ocean_rho_scale = load_space.scale * ocean_rho_scale_factor

    effective_steric_std = ocean_rho_std_factor * ocean_dyn_std
    ocean_rho_std = effective_steric_std / effective_steric_scale(state)

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
            solver=inf.CholeskySolver(galerkin=True),
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
