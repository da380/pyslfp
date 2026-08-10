"""
joint_utils.py
=======================
Shared utilities, physics initializations, and Bayesian measures for
extended Joint (Altimetry + GRACE) inversions using a 3-component model.

Model space (three fields on the sphere):

    m = [ Dh, eta, drho ]

      Dh   : ice thickness change (grounded ice),
      eta  : ocean dynamic sea surface height change (the TOTAL dynamic
             perturbation to the sea surface), applying the load rho_w*C*eta,
      drho : vertically averaged ocean density change, applying the load
             D*C*drho (the column mass change at fixed column thickness).

The prior enforces <rho_w*C*eta + D*C*drho>_O = 0, so the combined ocean
load is a pure redistribution of ocean mass. GRACE observes the response to
the total load; altimetry observes the GRD response plus the direct dynamic
term eta.

For reporting, the ocean state is post-processed into the standard split:

    eta_s = -(D / rho_w) * C * drho     (steric sea level; loads nothing)
    zeta  = eta - eta_s                 (ocean dynamic sea level, the mass /
                                         manometric part; its load equals
                                         the total ocean load)

so that quantities of interest can be expressed as barystatic, steric and
ocean-dynamic contributions while the model itself retains the simple
(eta, drho) parameterisation.

Optionally, the (eta, drho) pair can be given a scale-dependent
anti-correlation in the prior (see build_measures; requires pygeoinf >=
1.8.4), reflecting that at long wavelengths dynamic sea surface height
changes are predominantly steric in origin.

The three prior marginals can also be switched from heat-kernel to Sobolev
(Matern-type) covariances with a single common spectral order, giving
rougher, power-law-tailed fields while keeping the per-field length scales
and pointwise stds; the spatially correlated noise measures follow the same
family (see build_measures).
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


def build_physics_components(
    lmax, load_order, load_scale_km, points, obs_degree, is_surrogate=False
):
    """
    Constructs the 3-component physical operators for Joint Inversion.
    Model Space: [Ice Thickness Change, Ocean Dynamic SSH Change, Ocean Density Change]
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
    # The ice component is projected onto the ice sheets before loading,
    # consistently with the prior masking and with altimetry_utils.
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


def steric_sea_level_operator(state, load_space):
    """
    Maps the vertically averaged density change to the associated steric
    sea level change:

        eta_s = -(D / rho_w) * C * drho,

    where D = state.sea_level is the local ocean depth and C the ocean
    function. The scaling is such that the mass per vertical column is
    pointwise unchanged (rho_w * eta_s + D * drho = 0): the steric part of
    the sea surface height change loads nothing and so has no induced GRD
    response. Used for post-processing the (eta, drho) model state into
    the standard steric / ocean-dynamic split.
    """
    return (
        -1.0 / state.model.parameters.water_density
    ) * ocean_density_change_to_load_operator(state, load_space, load_space)


def effective_steric_scale(state):
    """
    Magnitude of the mean-depth steric relabelling:

        effective_steric_scale = mean_ocean_depth / rho_w,

    so that a density prior std specified through an effective steric sea
    level std converts as drho_std = steric_std / effective_steric_scale.
    This is a positive, magnitude-only conversion; the physical (negative)
    sign of the steric response lives in steric_sea_level_operator, and any
    signed scalar relabelling should apply the minus sign explicitly.
    """
    eta0 = state.sea_level * state.ocean_projection(value=0.0)
    mean_ocean_depth = state.model.integrate(eta0) / state.ocean_area
    return mean_ocean_depth / state.model.parameters.water_density


def gmsl_split_operators(state, load_space, continuous_sl_operator):
    """
    Operators for the 2D push-forward of the model onto GMSL change split
    into barystatic and steric parts.

    Returns (bary_op, steric_op, steric_direct_op, dyn_direct_op), each
    mapping the joint model space [Ice, Dyn, Rho] to a scalar:

      bary_op       : barystatic GMSL change, evaluated directly from global
                      mass conservation as (minus) the grounded ice mass
                      change spread uniformly over the oceans.
      steric_op       : the steric GMSL contribution, defined as the
                        residual true GMSL - barystatic, so that the two
                        coordinates sum to the true GMSL by construction.
                        Because the ocean load's SLE response removes
                        exactly the ocean-mean sea level that the direct
                        dynamic term adds back through its mass part, this
                        residual equals the ocean mean of the steric sea
                        level, -<(D/rho_w)*C*drho>_O, up to SLE solver
                        convergence and independently of the prior mass
                        constraint.
      steric_direct_op: the ocean average of the steric sea level alone,
                        evaluated directly from the density component. The
                        difference steric_op - steric_direct_op applied to
                        a given model diagnoses the SLE mass balance.
      dyn_direct_op   : the ocean average of the dynamic sea surface height
                        alone. Under the prior mass constraint this equals
                        the direct steric average (their difference is the
                        ocean mean of the manometric part, <zeta>_O), so
                        dyn_direct_op - steric_direct_op diagnoses how well
                        the constraint is satisfied.
    """
    joint_space = continuous_sl_operator.domain

    bary_avg_op = l2_products_operator(load_space, [barystatic_gmsl_weighting(state)])
    bary_op = bary_avg_op @ joint_space.subspace_projection(0)

    total_op = true_gmsl_operator(state, load_space, continuous_sl_operator)
    steric_op = total_op - bary_op

    ocean_weight = state.ocean_projection(value=0.0) / state.ocean_area
    ocean_avg_op = averaging_operator(state, load_space, [ocean_weight])
    steric_direct_op = (
        ocean_avg_op
        @ steric_sea_level_operator(state, load_space)
        @ joint_space.subspace_projection(2)
    )
    dyn_direct_op = ocean_avg_op @ joint_space.subspace_projection(1)

    return bary_op, steric_op, steric_direct_op, dyn_direct_op


def build_measures(
    state,
    load_space,
    ice_scale_factor,
    ice_std_mm,
    ocean_dyn_scale_factor,
    ocean_dyn_std_factor,
    ocean_rho_scale_factor,
    ocean_rho_std_factor,
    alt_noise_corr_scale_factor,
    alt_noise_std_factor,
    grace_noise_scale_km,
    grace_noise_std_factor,
    obs_degree,
    points,
    scale_mm,
    prior_shift=0.0,
    is_surrogate=False,
    ocean_corr=0.0,
    ocean_corr_scale_factor=0.4,
    prior_kernel="heat",
    prior_order=1.0,
    alt_noise_corr_std_factor=0.0,
):
    """
    Constructs the 3-component joint prior and dual-sensor noise measures.

    Note: ocean_rho_std_factor sets the effective steric sea level std as a
    fraction of the OCEAN DYNAMIC SSH std (not, as previously, of the
    GMSL std). ocean_dyn_std_factor remains referenced to the GMSL std.

    If ocean_corr > 0 (exact model only; requires pygeoinf >= 1.8.4), the
    (Dyn, Rho) marginals are combined into a correlated invariant measure
    with the scale-dependent spectral correlation

        r(k) = -ocean_corr * exp(-(corr_scale**2) * k),

    with corr_scale = load_space.scale * ocean_corr_scale_factor and k the
    Laplacian eigenvalue: a strong anti-correlation at long wavelengths
    (dynamic SSH changes there are predominantly steric) that decays towards
    independence at short wavelengths (mesoscale dynamic perturbations).
    The marginal distributions of each component are unchanged, so the std
    settings above retain their meaning, and the anti-correlation acts in
    concert with the mass constraint, which the conditioning then enforces
    exactly. The unmasked prior returned for the Woodbury preconditioner is
    always the plain uncorrelated direct sum, so the preconditioner
    construction is unaffected.
    The three model priors use point-value-scaled heat-kernel covariances
    by default. With prior_kernel="sobolev" they instead use Sobolev
    (Matern-type) covariances proportional to
    (1 + scale**2 * Laplacian)**(-prior_order) with respect to the model
    space inner product, sharing a single common order across the fields
    while retaining the per-field length scales and pointwise stds. Note
    that sample spectra therefore decay with the COMBINED exponent
    prior_order + (model space order): per-coefficient variance
    ~ (1 + s**2 k)**(-prior_order) x (space weight)**(-1). With the default
    order-2 load spaces, prior_order = 1 gives degree-variance tails
    ~ l**(-5), comparable to observed mesoscale SSH spectral slopes, and
    any positive order yields a finite pointwise variance. Any spatially
    correlated observation noise uses the same covariance family (the
    white-noise settings are unchanged), keeping the spectral
    signal-to-noise comparison within one family, and the surrogate prior
    uses the same kernel family so the preconditioner stays matched.

    The altimetry noise is the sum of a local (uncorrelated) component on
    the track points, with std alt_noise_std_factor x the GMSL prior std,
    and an optional large-scale correlated error component with std
    alt_noise_corr_std_factor x the GMSL prior std (0 disables) at
    correlation scale load_space.scale x alt_noise_corr_scale_factor,
    representing long-wavelength systematics such as orbit and
    reference-frame errors. The correlated component barely averages down
    and so sets an irreducible error floor on large-scale functionals such
    as GMSL. The returned alt_precond_noise and joint_precond_noise
    measures contain the local component alone (with the GRACE noise
    unchanged) for use in the preconditioner constructions.
    """

    if not 0.0 <= ocean_corr < 1.0:
        raise ValueError(
            "ocean_corr must lie in [0, 1); the anti-correlation sign is "
            "applied internally."
        )
    if prior_kernel not in ("heat", "sobolev"):
        raise ValueError("prior_kernel must be 'heat' or 'sobolev'.")
    if prior_kernel == "sobolev" and prior_order <= 0.0:
        raise ValueError(
            "prior_order must be positive. Sample spectra combine this "
            "order with the model space order, so the pointwise variance "
            "is well defined for any positive value on the (order > 1) "
            "spaces used here."
        )
    if alt_noise_corr_std_factor < 0.0:
        raise ValueError("alt_noise_corr_std_factor must be non-negative.")

    # Common covariance family for the three model priors AND the
    # spatially correlated noise measures (white-noise settings are
    # unchanged), so the spectral signal-to-noise comparison stays within
    # one family. The surrogate is built through the same path, so the
    # preconditioner stays matched to the exact prior family.
    if prior_kernel == "sobolev":

        def load_measure(scale, std):
            return load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
                prior_order, scale, std=std
            )

    else:

        def load_measure(scale, std):
            return load_space.point_value_scaled_heat_kernel_gaussian_measure(
                scale, std=std
            )

    # --- 1. PRIORS ---
    ice_scale = load_space.scale * ice_scale_factor
    ice_std = ice_std_mm / scale_mm
    ice_prior = load_measure(ice_scale, ice_std)

    B = averaging_operator(state, load_space, [barystatic_gmsl_weighting(state)])
    GMSL_prior_std = np.sqrt(
        ice_prior.affine_mapping(operator=B).covariance.matrix(dense=True)[0, 0]
    )

    ocean_dyn_scale = load_space.scale * ocean_dyn_scale_factor
    ocean_dyn_std = ocean_dyn_std_factor * GMSL_prior_std
    ocean_dyn_prior = load_measure(ocean_dyn_scale, ocean_dyn_std)

    # Vertically averaged density change: prior std specified through the
    # effective steric sea level as a fraction of the dynamic SSH std
    # (see altimetry_utils for discussion; 0.25 x dyn = old 0.5 x GMSL).
    ocean_rho_scale = load_space.scale * ocean_rho_scale_factor
    effective_steric_std = ocean_rho_std_factor * ocean_dyn_std
    ocean_rho_std = effective_steric_std / effective_steric_scale(state)
    ocean_rho_prior = load_measure(ocean_rho_scale, ocean_rho_std)

    unmasked_prior = inf.GaussianMeasure.from_direct_sum(
        [ice_prior, ocean_dyn_prior, ocean_rho_prior]
    )

    # --- PHYSICAL CONDITIONING ---
    model_prior = unmasked_prior
    if ocean_corr > 0.0 and not is_surrogate:
        # Correlated (Dyn, Rho) pair with unchanged marginals: block-diagonal
        # spectral correlation matrix with the ice component independent.
        # Imported lazily so the scripts continue to run on older pygeoinf
        # when the correlation is disabled. Note unmasked_prior (used for
        # the surrogate preconditioner) stays uncorrelated.
        from pygeoinf.symmetric_space.symmetric_space import (
            CorrelatedInvariantGaussianMeasure,
        )

        corr_scale = load_space.scale * ocean_corr_scale_factor

        def spectral_correlation(k):
            r = -ocean_corr * np.exp(-(corr_scale**2) * k)
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, r],
                    [0.0, r, 1.0],
                ]
            )

        model_prior = CorrelatedInvariantGaussianMeasure.from_invariant_measures(
            [ice_prior, ocean_dyn_prior, ocean_rho_prior], spectral_correlation
        )
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
    # Altimetry noise: a local (uncorrelated) part on the track points,
    # plus an optional large-scale correlated error representing
    # long-wavelength systematics (orbit, reference-frame and large-scale
    # correction residuals). The correlated part barely averages down, so
    # it sets an irreducible error floor on large-scale functionals such
    # as GMSL. The preconditioners are built from the local part alone
    # (alt_precond_noise / joint_precond_noise below); the low-rank
    # discrepancy this leaves is absorbed by a few extra CG iterations.
    alt_noise_std = alt_noise_std_factor * GMSL_prior_std
    alt_precond_noise_meas = inf.GaussianMeasure.from_standard_deviation(
        inf.EuclideanSpace(len(points)), alt_noise_std
    )
    alt_noise_meas = alt_precond_noise_meas
    if alt_noise_corr_std_factor > 0.0:
        alt_corr_noise_meas = load_measure(
            load_space.scale * alt_noise_corr_scale_factor,
            alt_noise_corr_std_factor * GMSL_prior_std,
        ).affine_mapping(operator=load_space.point_evaluation_operator(points))
        alt_noise_meas = alt_noise_meas + alt_corr_noise_meas

    # GRACE Noise
    grace_spatial_scale = (
        grace_noise_scale_km * 1000.0 / state.model.parameters.length_scale
    )
    grace_spatial_noise = load_measure(
        grace_spatial_scale, grace_noise_std_factor * ice_std
    )
    wmb = WMBMethod(state.model, obs_degree)
    grace_noise_meas = wmb.load_measure_to_observation_measure(grace_spatial_noise)

    joint_noise_meas = inf.GaussianMeasure.from_direct_sum(
        [alt_noise_meas, grace_noise_meas]
    )
    joint_precond_noise_meas = inf.GaussianMeasure.from_direct_sum(
        [alt_precond_noise_meas, grace_noise_meas]
    )

    return {
        "model_prior": model_prior,
        "unmasked_prior": unmasked_prior,
        "alt_noise": alt_noise_meas,
        "grace_noise": grace_noise_meas,
        "joint_noise": joint_noise_meas,
        "alt_precond_noise": alt_precond_noise_meas,
        "joint_precond_noise": joint_precond_noise_meas,
        "wmb": wmb,
        "gmsl_std": GMSL_prior_std,
        "ice_std": ice_std,
        "ice_scale": ice_scale,
    }


def regional_decomposition_operators(state, load_space, finger_print_operator, regions):
    """
    Decomposes the 3-component state into regional sea level signals:
    (ocean dynamic SL, steric SL, barystatic GRD SL).

      ocean dynamic SL : the mass (manometric) part of the dynamic signal,
                         zeta = eta - eta_s, PLUS the regional GRD response
                         to the total ocean load -- which is exactly the
                         load of zeta, since the steric part is column-mass
                         neutral and applies no load.
      steric SL        : eta_s = -(D / rho_w) * C * drho, which loads
                         nothing and so has no induced response.
      barystatic GRD SL: the sea level response to the (grounded) ice load,
                         including its Gravitation, Rotation and
                         Deformation fingerprint.

    By linearity of the sea level equation the three coordinates sum
    exactly to the regional average of the true sea level change.
    """
    masks = [state.get_projection(r, value=0.0) for r in regions]
    avg_op = averaging_operator(state, load_space, masks)
    joint_space = inf.HilbertSpaceDirectSum([load_space, load_space, load_space])

    steric_op_field = steric_sea_level_operator(
        state, load_space
    ) @ joint_space.subspace_projection(2)
    op_steric = avg_op @ steric_op_field

    barystatic_sl_op = (
        finger_print_operator.codomain.subspace_projection(0) @ finger_print_operator
    )

    dyn_to_load = sea_level_change_to_load_operator(state, load_space, load_space)
    rho_to_load = ocean_density_change_to_load_operator(state, load_space, load_space)
    ocean_load_op = dyn_to_load @ joint_space.subspace_projection(
        1
    ) + rho_to_load @ joint_space.subspace_projection(2)

    zeta_field_op = joint_space.subspace_projection(1) - steric_op_field
    op_dyn = avg_op @ (zeta_field_op + barystatic_sl_op @ ocean_load_op)

    ice_to_load = ice_thickness_change_to_load_operator(
        state, load_space, load_space
    ) @ ice_projection_operator(state, load_space)
    op_ice_fp = (
        avg_op @ barystatic_sl_op @ ice_to_load @ joint_space.subspace_projection(0)
    )

    return op_dyn, op_steric, op_ice_fp
