"""
joint_utils.py
=======================
Shared utilities, physics initializations, and Bayesian measures for
Joint (Altimetry + GRACE) inversions using a mass / volume split of the
ocean state.

Model space (three fields on the sphere):

    m = [ Dh, zeta, drho ]

    Dh   : ice thickness change, supported on the (grounded) ice sheets.
    zeta : ocean dynamic sea level -- vertical motion of the sea surface
           associated with currents alone. It integrates to zero over the
           oceans (pure redistribution of ocean mass) but locally applies
           a load, equivalent to an ocean bottom pressure change
           rho_w * g * zeta.
    drho : vertically averaged ocean density change. It is paired with the
           steric sea level

               eta_s = -(D / rho_w) * C * drho,

           with D the local ocean depth and C the ocean function, chosen so
           that the mass per vertical column is pointwise unchanged
           (rho_w * eta_s + D * drho = 0). The steric component therefore
           applies no load and is invisible to gravimetry.

Forward physics:

    sigma = rho_i * (1 - C) * X_ice * Dh + rho_w * C * zeta   (direct load)
    (SLC, u, phi, omega) = F(sigma)                           (sea level equation)
    SSH = SLC + u + psi(omega)/g + zeta + eta_s               (altimetry)
    GRACE = G(F(sigma))                                       (gravimetry)
    SL  = SLC + zeta + eta_s                                  (relative sea level)

GRACE therefore observes the ice and dynamic components only, while the
altimetry sees all three: the joint inversion demonstrates the separation
of the mass and steric contributions to sea level change.
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
    l2_products_operator,
)


def steric_sea_level_operator(state, load_space):
    """
    Maps the vertically averaged density change to the associated steric
    sea level change:

        eta_s = -(D / rho_w) * C * drho,

    where D = state.sea_level is the local ocean depth and C the ocean
    function. The scaling is such that the mass per vertical column is
    pointwise unchanged (rho_w * eta_s + D * drho = 0): the steric
    component changes sea level without applying any load.
    """
    return (
        -1.0 / state.model.parameters.water_density
    ) * ocean_density_change_to_load_operator(state, load_space, load_space)


def steric_depth_field(state):
    """
    Returns (D / rho_w) * C as an SHGrid: the pointwise magnitude
    |d eta_s / d drho| of the steric relabelling. Useful for converting
    (positive) density standard deviation fields into steric sea level
    units; note the local steric std therefore scales with depth, being
    muted in shallow seas relative to the mean-depth value.
    """
    return state.sea_level * state.ocean_function / state.model.parameters.water_density


def build_physics_components(
    lmax, load_order, load_scale_km, points, obs_degree, is_surrogate=False
):
    """
    Constructs the 3-component physical operators for Joint Inversion.
    Model Space: [Ice Thickness Change, Ocean Dynamic Sea Level, Ocean Density Change]

    Only the ice and dynamic sea level components apply a load; the density
    component is column-mass neutral and enters the altimetry through the
    steric sea level alone, and is invisible to GRACE. The ice component is
    projected onto the ice sheets before loading, consistently with the
    prior masking.

    In surrogate mode the dynamic sea level load is switched off (one-way
    physics for preconditioning) and no steric term is added, so the density
    component is invisible to the surrogate forward: the surrogate operator
    tree is exactly the pre-steric construction, as required for the
    Woodbury preconditioner setup.
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

    # 1. Operators to convert the loading components to a surface mass load
    ice_to_load = ice_thickness_change_to_load_operator(
        state, load_space, load_space
    ) @ ice_projection_operator(state, load_space)

    if is_surrogate:
        ocean_dyn_to_load = load_space.zero_operator(load_space)
    else:
        ocean_dyn_to_load = sea_level_change_to_load_operator(
            state, load_space, load_space
        )

    # The density component applies no load by construction.
    ocean_rho_to_load = load_space.zero_operator(load_space)

    # Joint to Total Load: [Ice, OceanDyn, OceanRho] -> Total Mass Load
    joint_to_load = inf.RowLinearOperator(
        [ice_to_load, ocean_dyn_to_load, ocean_rho_to_load]
    )
    joint_space = joint_to_load.domain

    # --- A. ALTIMETRY TRACK (SSH) ---
    static_ssh_op = sea_surface_height_operator(state, finger_print_operator.codomain)
    barystatic_ssh_op = static_ssh_op @ finger_print_operator

    if is_surrogate:
        # The surrogate reproduces the pre-steric operator tree exactly: the
        # density component is invisible to the surrogate forward (its load
        # is zero and no steric term is added). This keeps the operator tree
        # used by the Woodbury preconditioner identical to the previously
        # validated construction. Note the domain is still the full
        # 3-component joint space, so the preconditioner pairs correctly
        # with the 3-component surrogate prior.
        op1 = inf.ColumnLinearOperator(
            [joint_to_load, joint_space.subspace_projection(1)]
        )
        op2_ssh = inf.BlockDiagonalLinearOperator(
            [barystatic_ssh_op, load_space.identity_operator()]
        )
        op3 = inf.RowLinearOperator(
            [load_space.identity_operator(), load_space.identity_operator()]
        )
    else:
        # Steric relabelling of the density component (no load, direct SSH term)
        steric_op = steric_sea_level_operator(state, load_space)

        # Map [Ice, Dyn, Rho] -> [Total Load, Dyn, Rho]
        op1 = inf.ColumnLinearOperator(
            [
                joint_to_load,
                joint_space.subspace_projection(1),
                joint_space.subspace_projection(2),
            ]
        )
        op2_ssh = inf.BlockDiagonalLinearOperator(
            [barystatic_ssh_op, load_space.identity_operator(), steric_op]
        )
        op3 = inf.RowLinearOperator(
            [
                load_space.identity_operator(),
                load_space.identity_operator(),
                load_space.identity_operator(),
            ]
        )

    continuous_ssh_operator = op3 @ op2_ssh @ op1
    point_eval = load_space.point_evaluation_operator(points)
    alt_track = point_eval @ continuous_ssh_operator

    # --- B. GRACE TRACK ---
    # GRACE observes the response to the Total Mass Load directly, and so is
    # sensitive to the ice and dynamic sea level components only.
    grace_obs_op = grace_observation_operator(
        finger_print_operator.codomain, obs_degree
    )
    grace_track = grace_obs_op @ finger_print_operator @ joint_to_load

    # --- C. TRUE SEA LEVEL TRACK (For GMSL) ---
    barystatic_sl_op = (
        finger_print_operator.codomain.subspace_projection(0) @ finger_print_operator
    )
    if is_surrogate:
        op2_sl = inf.BlockDiagonalLinearOperator(
            [barystatic_sl_op, load_space.identity_operator()]
        )
    else:
        op2_sl = inf.BlockDiagonalLinearOperator(
            [barystatic_sl_op, load_space.identity_operator(), steric_op]
        )
    continuous_sl_operator = op3 @ op2_sl @ op1

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
    true_avg_op = l2_products_operator(load_space, [true_avg_weight])
    return true_avg_op @ continuous_sl_operator


def barystatic_gmsl_weighting(state):
    """
    Weighting function whose L2 product with an ice thickness change field
    returns the implied barystatic GMSL change (global mass conservation).
    The weighting is restricted to grounded ice, consistently with the
    forward operator, which projects the ice component onto the ice sheets
    before loading.
    """
    return (
        -state.model.parameters.ice_density
        * state.one_minus_ocean_function
        * state.ice_projection(value=0.0)
        / (state.model.parameters.water_density * state.ocean_area)
    )


def effective_steric_scale(state):
    """
    Magnitude of the mean-depth steric relabelling:

        effective_steric_scale = mean_ocean_depth / rho_w,

    so that a density prior std specified through an effective steric sea
    level std converts as drho_std = steric_std / effective_steric_scale.
    This is a positive, magnitude-only conversion used to set the density
    prior; the physical (negative) sign of the steric response lives in
    steric_sea_level_operator. The actual pointwise steric std scales with
    the local depth, D(x) / mean_ocean_depth (see steric_depth_field).
    """
    eta0 = state.sea_level * state.ocean_projection(value=0.0)
    mean_ocean_depth = state.model.integrate(eta0) / state.ocean_area
    return mean_ocean_depth / state.model.parameters.water_density


def gmsl_split_operators(state, load_space, continuous_sl_operator):
    """
    Operators for the 2D push-forward of the model onto GMSL change split
    into barystatic and steric parts.

    Returns (bary_op, steric_op, steric_direct_op, dyn_avg_op), each mapping
    the joint model space [Ice, Dyn, Rho] to a scalar:

      bary_op         : barystatic GMSL change, evaluated directly from
                        global mass conservation as (minus) the grounded ice
                        mass change spread uniformly over the oceans.
      steric_op       : the steric GMSL contribution, defined as the residual
                        true GMSL - barystatic, so that the two coordinates
                        sum to the true GMSL by construction. The load of the
                        dynamic sea level cancels its direct ocean-mean term
                        identically, so this residual equals the ocean mean
                        of the steric sea level up to SLE solver convergence.
      steric_direct_op: the ocean average of the steric sea level alone. The
                        difference steric_op - steric_direct_op applied to a
                        given model diagnoses the SLE mass balance.
      dyn_avg_op      : the ocean average of the dynamic sea level, which is
                        zero under the prior constraint; its value on a given
                        model diagnoses how well the constraint is satisfied.
    """
    joint_space = continuous_sl_operator.domain

    bary_avg_op = l2_products_operator(load_space, [barystatic_gmsl_weighting(state)])
    bary_op = bary_avg_op @ joint_space.subspace_projection(0)

    total_op = true_gmsl_operator(state, load_space, continuous_sl_operator)
    steric_op = total_op - bary_op

    ocean_weight = state.ocean_projection(value=0.0) / state.ocean_area
    ocean_avg_op = l2_products_operator(load_space, [ocean_weight])
    steric_direct_op = (
        ocean_avg_op
        @ steric_sea_level_operator(state, load_space)
        @ joint_space.subspace_projection(2)
    )
    dyn_avg_op = ocean_avg_op @ joint_space.subspace_projection(1)

    return bary_op, steric_op, steric_direct_op, dyn_avg_op


def build_measures(
    state,
    load_space,
    ice_scale_factor,
    ice_std_mm,
    ocean_dyn_scale_factor,
    ocean_dyn_std_mm,
    ocean_rho_scale_factor,
    steric_std_mm,
    alt_noise_scale_factor,
    alt_noise_std_mm,
    grace_noise_scale_km,
    grace_noise_std_mm,
    obs_degree,
    points,
    scale_mm,
    prior_shift=0.0,
    is_surrogate=False,
):
    """
    Constructs the 3-component joint prior and dual-sensor noise measures.

    All amplitudes are specified pointwise in mm:

      ice_std_mm         : ice thickness change std (over the ice sheets).
      ocean_dyn_std_mm   : ocean dynamic sea level std.
      steric_std_mm      : effective steric sea level std at the mean ocean
                           depth, converted to a density std via the
                           mean-depth relabelling (see effective_steric_scale);
                           the actual steric std varies with the local depth.
      alt_noise_std_mm   : altimetry per-point noise std.
      grace_noise_std_mm : GRACE spatial noise std (water-equivalent height),
                           mapped to the observed coefficients via WMBMethod.

    The mass conservation constraint is <zeta>_O = 0 on the dynamic sea
    level alone; the density component is column-mass neutral by
    construction and is left unconstrained (its ocean mean carries the
    steric GMSL signal).
    """

    # --- 1. PRIORS ---
    ice_scale = load_space.scale * ice_scale_factor
    ice_std = ice_std_mm / scale_mm
    ice_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ice_scale, std=ice_std
    )

    # Implied barystatic GMSL prior std, for reporting and reference. Using
    # the grounded-ice weighting makes this equal to the barystatic std of
    # the masked prior (the mask is idempotent under the weighting).
    B = l2_products_operator(load_space, [barystatic_gmsl_weighting(state)])
    GMSL_prior_std = np.sqrt(
        ice_prior.affine_mapping(operator=B).covariance.matrix(dense=True)[0, 0]
    )

    ocean_dyn_scale = load_space.scale * ocean_dyn_scale_factor
    ocean_dyn_std = ocean_dyn_std_mm / scale_mm
    ocean_dyn_prior = load_space.point_value_scaled_heat_kernel_gaussian_measure(
        ocean_dyn_scale, std=ocean_dyn_std
    )

    # Vertically averaged density change: constant pointwise std within the
    # oceans, specified through the effective steric sea level at the mean
    # ocean depth (a fixed, magnitude-only relabelling).
    ocean_rho_scale = load_space.scale * ocean_rho_scale_factor
    ocean_rho_std = (steric_std_mm / scale_mm) / effective_steric_scale(state)
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

        # Strict mass conservation: the dynamic sea level is a pure
        # redistribution of ocean mass, B([Ice, Dyn, Rho]) = <Dyn>_O = 0.
        # The density component applies no load, so no further constraint
        # is required.
        avg_op = ocean_average_operator(state, load_space)

        mass_constraint_op = inf.RowLinearOperator(
            [
                load_space.zero_operator(codomain=avg_op.codomain),
                avg_op,
                load_space.zero_operator(codomain=avg_op.codomain),
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
    alt_noise_std = alt_noise_std_mm / scale_mm
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
        grace_spatial_scale, std=grace_noise_std_mm / scale_mm
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
    """
    Decomposes the 3-component state into regional signals:
    (ocean dynamic SL, steric SL, barystatic GRD SL).

    The barystatic GRD component is the regional average of the sea level
    response to the ice load alone (including its Gravitation, Rotation and
    Deformation fingerprint); the small SLE response to the dynamic sea
    level load is not included in the decomposition.
    """
    masks = [state.get_projection(r, value=0.0) for r in regions]
    avg_op = averaging_operator(state, load_space, masks)
    joint_space = inf.HilbertSpaceDirectSum([load_space, load_space, load_space])

    op_dyn = avg_op @ joint_space.subspace_projection(1)
    op_steric = (
        avg_op
        @ steric_sea_level_operator(state, load_space)
        @ joint_space.subspace_projection(2)
    )

    ice_to_load = ice_thickness_change_to_load_operator(
        state, load_space, load_space
    ) @ ice_projection_operator(state, load_space)
    barystatic_sl_op = (
        finger_print_operator.codomain.subspace_projection(0) @ finger_print_operator
    )
    op_ice_fp = (
        avg_op @ barystatic_sl_op @ ice_to_load @ joint_space.subspace_projection(0)
    )

    return op_dyn, op_steric, op_ice_fp
