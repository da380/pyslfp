"""
altimetry_utils.py
===========================
Shared utilities, physics initializations, and Bayesian measures for
satellite altimetry inversions with separated ocean dynamic and steric
components.

Model space (three fields on the sphere):

    m = [ Dh, eta, drho ]

      Dh   : ice thickness change (grounded ice),
      eta  : sterodynamic sea level change (the TOTAL dynamic
             perturbation to the sea surface), applying the load rho_w*C*eta,
      drho : vertically averaged ocean density change, applying the load
             D*C*drho (the column mass change at fixed column thickness).

The prior enforces <rho_w*C*eta + D*C*drho>_O = 0, so the combined ocean
load is a pure redistribution of ocean mass.

For reporting, the ocean state is post-processed following the
terminology of Gregory et al. (2019), with the manometric change further
split into dynamic and barystatic parts:

    eta_s = -(D / rho_w) * C * drho     (steric sea level change, SSLC;
                                         loads nothing)
    zeta  = eta - eta_s                 (dynamic manometric sea level
                                         change, DMSLC; its load equals
                                         the total ocean load)

together with the barystatic manometric sea level change (BMSLC), the
GRD response to the land-ice load. eta itself is the sterodynamic sea
level change (DMSLC + SSLC), and quantities of interest are expressed as
barystatic, steric and dynamic-manometric contributions while the model
retains the simple (eta, drho) parameterisation.

Optionally, the (eta, drho) pair can be given a scale-dependent
anti-correlation in the prior, reflecting that at long wavelengths
dynamic sea surface height changes are predominantly steric in origin.

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
)


def build_physics_components(
    lmax, load_order, load_scale_km, points, is_surrogate=False
):
    """
    Constructs the 3-component physical operators.
    Model Space: [Ice Thickness Change, Sterodynamic SL Change, Ocean Density Change]
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

    # --- B. Resolve the True Sea Level Components (For GMSLR Integration) ---
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
        point_eval,
    )


def true_gmsl_operator(state, load_space, continuous_sl_operator):
    """Returns the true spatial integration of continuous Sea Level."""
    true_avg_weight = state.ocean_projection(value=0.0) / state.ocean_area
    true_avg_op = load_space.l2_products_operator([true_avg_weight])
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


def steric_gmsl_functional(state, load_space):
    """
    The steric GMSL contribution <eta_s>_O as a linear functional of the
    vertically averaged density change alone (the ocean average of
    steric_sea_level_operator). Identical to the steric_direct_op of
    gmsl_split_operators without the model-space projection.
    """
    ocean_weight = state.ocean_projection(value=0.0) / state.ocean_area
    return load_space.l2_products_operator([ocean_weight]) @ steric_sea_level_operator(
        state, load_space
    )


def _functional_std(measure, functional):
    """Std of a scalar linear functional of a Gaussian measure."""
    return float(np.sqrt(measure.directional_variance(functional.adjoint(np.ones(1)))))


def build_conditioned_prior(
    state,
    load_space,
    load_measure,
    ice_scale,
    ocean_dyn_scale,
    ocean_rho_scale,
    gmsl_bary_steric_ratio,
    ocean_dyn_std,
    steric_dyn_std_ratio,
    *,
    is_surrogate=False,
    ocean_corr=0.0,
    corr_scale=None,
    derived_stds=None,
):
    """
    Derives the three pointwise prior stds from the sterodynamic anchor
    and assembles the (optionally correlated) joint prior with masking
    and the mass-conservation conditioning. Shared by altimetry_utils
    and joint_utils build_measures.

    The chain is triangular, with a single dimensioned input:

      ocean_dyn_std          : pointwise std of the sterodynamic field
                               (pre-mass-constraint; nondimensional here),
      steric_dyn_std_ratio   : mean-depth steric sea level std as a
                               fraction of the sterodynamic std, giving
                               rho_std = ratio * dyn_std /
                               effective_steric_scale,
      gmsl_bary_steric_ratio : ratio of the barystatic to steric GMSL
                               prior stds, giving the ice std.

    The steric GMSL std used in the last step is the REALISED value
    under the mass constraint. Writing X = <eta>_O and Y = <eta_s>_O
    with pre-constraint stds a, b and correlation rho_XY under the
    masked prior, the constraint functional g is proportional to X - Y
    in the continuum, so the (rank-one) conditioning gives in closed form

      sigma_S = a*b * sqrt((1 - rho_XY**2) / (a**2 + b**2 - 2*rho_XY*a*b)).

    At finite lmax, g (assembled through the load operators) and X - Y
    differ by grid-product truncation, so instead of using the closed
    form the code applies the identical conditioning machinery to a
    masked ocean-pair measure and reads sigma_S off directly -- exact
    for the realised prior at the working discretisation, with
    (a, b, rho_XY) reported for the closed-form interpretation. sigma_S is strictly positive for any positive
    amplitudes since ocean_corr < 1. The ice marginal is uncorrelated
    with the ocean pair and absent from the constraint, so the
    barystatic GMSL std of the realised prior is exactly
    gmsl_bary_steric_ratio * sigma_S, closing the chain without
    circularity; both realised stds are verified directly on the final
    conditioned measure.

    If derived_stds is given ({"ice_std", "dyn_std", "rho_std"},
    nondimensional), the derivation is skipped and those amplitudes are
    used directly: the surrogate path, so preconditioner amplitudes
    match the exact model instead of being re-derived at surrogate
    resolution.

    Returns (model_prior, unmasked_prior, calib): the conditioned (or,
    for the surrogate, unconditioned) prior, the uncorrelated
    direct-sum prior at the same amplitudes for the Woodbury
    preconditioner, and the calibration dictionary.
    """
    if ocean_corr > 0.0 and corr_scale is None:
        raise ValueError("corr_scale is required when ocean_corr > 0.")

    if derived_stds is not None:
        ice_std = derived_stds["ice_std"]
        dyn_std = derived_stds["dyn_std"]
        rho_std = derived_stds["rho_std"]
        calib = {"derived_stds": dict(derived_stds)}
        calibrating = False
    else:
        if min(gmsl_bary_steric_ratio, ocean_dyn_std, steric_dyn_std_ratio) <= 0.0:
            raise ValueError(
                "gmsl_bary_steric_ratio, ocean_dyn_std_mm and "
                "steric_dyn_std_ratio must all be positive."
            )
        calibrating = True
        dyn_std = ocean_dyn_std
        es = effective_steric_scale(state)
        rho_std = steric_dyn_std_ratio * dyn_std / es

        ice_proj = ice_projection_operator(state, load_space)
        ocean_proj = ocean_projection_operator(state, load_space)
        bary_fn = (
            load_space.l2_products_operator([barystatic_gmsl_weighting(state)])
            @ ice_proj
        )
        q_ice = _functional_std(load_measure(ice_scale, 1.0), bary_fn)

        # Masked ocean-pair measure at the actual amplitudes; its blocks
        # coincide exactly with the ocean blocks of the joint prior (the
        # ice component is spectrally decoupled), so the rank-one
        # conditioning below reproduces the realised constrained values.
        dyn_prior_cal = load_measure(ocean_dyn_scale, dyn_std)
        rho_prior_cal = load_measure(ocean_rho_scale, rho_std)
        if ocean_corr > 0.0:
            from pygeoinf.symmetric_space.symmetric_space import (
                CorrelatedInvariantGaussianMeasure,
            )

            def pair_correlation(k):
                r = -ocean_corr * np.exp(-(corr_scale**2) * k)
                return np.array([[1.0, r], [r, 1.0]])

            pair = CorrelatedInvariantGaussianMeasure.from_invariant_measures(
                [dyn_prior_cal, rho_prior_cal], pair_correlation
            )
        else:
            pair = inf.GaussianMeasure.from_direct_sum([dyn_prior_cal, rho_prior_cal])
        pair = pair.affine_mapping(
            operator=inf.BlockDiagonalLinearOperator([ocean_proj, ocean_proj])
        )
        pair_space = pair.domain

        ocean_weight = state.ocean_projection(value=0.0) / state.ocean_area
        avg_op = ocean_average_operator(state, load_space)
        dyn_to_load = sea_level_change_to_load_operator(state, load_space, load_space)
        rho_to_load = ocean_density_change_to_load_operator(
            state, load_space, load_space
        )
        x_fn = load_space.l2_products_operator([ocean_weight]) @ (
            pair_space.subspace_projection(0)
        )
        y_fn = steric_gmsl_functional(state, load_space) @ (
            pair_space.subspace_projection(1)
        )
        g_fn = inf.RowLinearOperator([avg_op @ dyn_to_load, avg_op @ rho_to_load])
        pair_subspace = inf.AffineSubspace.from_linear_equation(
            operator=g_fn,
            value=avg_op.codomain.zero,
            solver=inf.CholeskySolver(galerkin=True),
        )
        sigma_s = _functional_std(pair_subspace.condition_gaussian_measure(pair), y_fn)
        # Narrative constants for the closed-form interpretation.
        d_x = x_fn.adjoint(np.ones(1))
        d_y = y_fn.adjoint(np.ones(1))
        a = float(np.sqrt(pair.directional_variance(d_x)))
        b = float(np.sqrt(pair.directional_variance(d_y)))
        rho_xy = float(pair.directional_covariance(d_x, d_y)) / (a * b)
        u_eta, u_rho = a / dyn_std, b / rho_std
        ice_std = gmsl_bary_steric_ratio * sigma_s / q_ice
        calib = {
            "constants": {
                "q_ice": q_ice,
                "u_eta": u_eta,
                "u_rho": u_rho,
                "rho_xy": rho_xy,
                "effective_steric_scale": es,
            },
            "gmsl": {
                "barystatic_std": gmsl_bary_steric_ratio * sigma_s,
                "steric_std": sigma_s,
                "eta_mean_std_unconstrained": a,
                "bary_steric_ratio": gmsl_bary_steric_ratio,
            },
        }

    def assemble():
        ice_prior = load_measure(ice_scale, ice_std)
        dyn_prior = load_measure(ocean_dyn_scale, dyn_std)
        rho_prior = load_measure(ocean_rho_scale, rho_std)
        unmasked = inf.GaussianMeasure.from_direct_sum(
            [ice_prior, dyn_prior, rho_prior]
        )
        if ocean_corr > 0.0 and not is_surrogate:
            # Correlated (Dyn, Rho) pair with unchanged marginals;
            # imported lazily so the scripts continue to run on older
            # pygeoinf when the correlation is disabled. The unmasked
            # prior (used for the surrogate preconditioner) stays
            # uncorrelated.
            from pygeoinf.symmetric_space.symmetric_space import (
                CorrelatedInvariantGaussianMeasure,
            )

            def spectral_correlation(k):
                r = -ocean_corr * np.exp(-(corr_scale**2) * k)
                return np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, r],
                        [0.0, r, 1.0],
                    ]
                )

            joint = CorrelatedInvariantGaussianMeasure.from_invariant_measures(
                [ice_prior, dyn_prior, rho_prior], spectral_correlation
            )
        else:
            joint = unmasked
        return joint, unmasked

    joint, unmasked_prior = assemble()
    calib["derived_stds"] = {
        "ice_std": ice_std,
        "dyn_std": dyn_std,
        "rho_std": rho_std,
    }

    if is_surrogate:
        return joint, unmasked_prior, calib

    # --- masking and mass-conservation conditioning (exact model) ---
    ice_proj = ice_projection_operator(state, load_space)
    ocean_proj = ocean_projection_operator(state, load_space)
    proj_op = inf.BlockDiagonalLinearOperator([ice_proj, ocean_proj, ocean_proj])
    masked = joint.affine_mapping(operator=proj_op)

    avg_op = ocean_average_operator(state, load_space)
    dyn_to_load = sea_level_change_to_load_operator(state, load_space, load_space)
    rho_to_load = ocean_density_change_to_load_operator(state, load_space, load_space)
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
    model_prior = mass_subspace.condition_gaussian_measure(masked)

    if calibrating:
        # Direct verification of the closed forms on the conditioned
        # measure, with the QoI-style functionals used downstream.
        joint_space = masked.domain
        steric_check_fn = steric_gmsl_functional(state, load_space) @ (
            joint_space.subspace_projection(2)
        )
        bary_check_fn = load_space.l2_products_operator(
            [barystatic_gmsl_weighting(state)]
        ) @ joint_space.subspace_projection(0)
        calib["gmsl"]["steric_std_check"] = _functional_std(
            model_prior, steric_check_fn
        )
        calib["gmsl"]["barystatic_std_check"] = _functional_std(
            model_prior, bary_check_fn
        )

    return model_prior, unmasked_prior, calib


def print_calibration_report(calib, scale_mm, density_scale):
    """Prints the derived amplitudes and realised GMSL-level statistics."""
    d = calib["derived_stds"]
    if "constants" not in calib:
        print(
            "Prior amplitudes (supplied, underived): ice std = "
            f"{d['ice_std'] * scale_mm:.3f} mm, sterodynamic std = "
            f"{d['dyn_std'] * scale_mm:.3f} mm, density std = "
            f"{d['rho_std'] * density_scale:.4e} kg m^-3"
        )
        return
    c, g = calib["constants"], calib["gmsl"]
    print(
        "Prior amplitudes: sterodynamic std = "
        f"{d['dyn_std'] * scale_mm:.3f} mm (input), density std = "
        f"{d['rho_std'] * density_scale:.4e} kg m^-3 (mean-depth steric std "
        f"= {d['rho_std'] * c['effective_steric_scale'] * scale_mm:.3f} mm), "
        f"ice std = {d['ice_std'] * scale_mm:.3f} mm"
    )
    print(
        "  realised GMSL prior stds under the mass constraint: barystatic = "
        f"{g['barystatic_std'] * scale_mm:.3f} mm "
        f"(verified {g['barystatic_std_check'] * scale_mm:.3f}), steric = "
        f"{g['steric_std'] * scale_mm:.3f} mm "
        f"(verified {g['steric_std_check'] * scale_mm:.3f}); "
        f"barystatic/steric ratio = {g['bary_steric_ratio']:.2f}"
    )
    print(
        f"  <eta>_O -- <eta_s>_O prior correlation rho_XY = {c['rho_xy']:.3f}; "
        "unconstrained <eta>_O std = "
        f"{g['eta_mean_std_unconstrained'] * scale_mm:.3f} mm"
    )
    floor = calib.get("noise", {}).get("alt_corr_gmsl_floor")
    if floor is not None:
        print(
            "  correlated altimetry error component: implied GMSL error "
            f"floor = {floor * scale_mm:.3f} mm"
        )


def gmsl_split_operators(state, load_space, continuous_sl_operator):
    """
    Operators for the 2D push-forward of the model onto the GMSLR split
    into barystatic and steric parts.

    Returns (bary_op, steric_op, steric_direct_op, dyn_direct_op), each
    mapping the joint model space [Ice, Dyn, Rho] to a scalar:

      bary_op         : barystatic GMSLR, evaluated directly from
                        global mass conservation as (minus) the grounded
                        ice mass change spread uniformly over the oceans.
      steric_op       : the steric GMSLR contribution, defined as the
                        residual true GMSLR - barystatic, so that the two
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
      dyn_direct_op   : the ocean average of the sterodynamic sea level
                        change alone. Under the prior mass constraint this equals
                        the direct steric average (their difference is the
                        ocean mean of the dynamic manometric part, DMSLC), so
                        dyn_direct_op - steric_direct_op diagnoses how well
                        the constraint is satisfied.
    """
    joint_space = continuous_sl_operator.domain

    bary_avg_op = load_space.l2_products_operator([barystatic_gmsl_weighting(state)])
    bary_op = bary_avg_op @ joint_space.subspace_projection(0)

    total_op = true_gmsl_operator(state, load_space, continuous_sl_operator)
    steric_op = total_op - bary_op

    ocean_weight = state.ocean_projection(value=0.0) / state.ocean_area
    ocean_avg_op = load_space.l2_products_operator([ocean_weight])
    steric_direct_op = (
        ocean_avg_op
        @ steric_sea_level_operator(state, load_space)
        @ joint_space.subspace_projection(2)
    )
    dyn_direct_op = ocean_avg_op @ joint_space.subspace_projection(1)

    return bary_op, steric_op, steric_direct_op, dyn_direct_op


def regional_decomposition_operators(state, load_space, finger_print_operator, regions):
    """
    Decomposes the 3-component state into regional sea level signals
    (SSLC, DMSLC, BMSLC), following Gregory et al. (2019) with the
    manometric change split into dynamic and barystatic parts:

      SSLC  : steric sea level change, eta_s = -(D / rho_w) * C * drho,
              which loads nothing and so has no induced response.
      DMSLC : dynamic manometric sea level change -- the manometric part
              of the dynamic signal, zeta = eta - eta_s, PLUS the regional
              GRD response to the total ocean load, which is exactly the
              load of zeta since the steric part is column-mass neutral
              and applies no load.
      BMSLC : barystatic manometric sea level change -- the sea level
              response to the (grounded) ice load, including its
              Gravitation, Rotation and Deformation fingerprint.

    By linearity of the sea level equation the three coordinates sum
    exactly to the regional average of the true sea level change.
    """
    masks = [state.get_projection(r, value=0.0) for r in regions]
    avg_op = averaging_operator(state, load_space, masks)
    joint_space = inf.HilbertSpaceDirectSum([load_space, load_space, load_space])

    steric_op_field = steric_sea_level_operator(
        state, load_space
    ) @ joint_space.subspace_projection(2)
    op_sslc = avg_op @ steric_op_field

    barystatic_sl_op = (
        finger_print_operator.codomain.subspace_projection(0) @ finger_print_operator
    )

    dyn_to_load = sea_level_change_to_load_operator(state, load_space, load_space)
    rho_to_load = ocean_density_change_to_load_operator(state, load_space, load_space)
    ocean_load_op = dyn_to_load @ joint_space.subspace_projection(
        1
    ) + rho_to_load @ joint_space.subspace_projection(2)

    zeta_field_op = joint_space.subspace_projection(1) - steric_op_field
    op_dmslc = avg_op @ (zeta_field_op + barystatic_sl_op @ ocean_load_op)

    ice_to_load = ice_thickness_change_to_load_operator(
        state, load_space, load_space
    ) @ ice_projection_operator(state, load_space)
    op_bmslc = (
        avg_op @ barystatic_sl_op @ ice_to_load @ joint_space.subspace_projection(0)
    )

    return op_sslc, op_dmslc, op_bmslc


def build_measures(
    state,
    load_space,
    ice_scale_factor,
    gmsl_bary_steric_ratio,
    ocean_dyn_scale_factor,
    ocean_dyn_std_mm,
    ocean_rho_scale_factor,
    steric_dyn_std_ratio,
    noise_corr_scale_factor,
    noise_std_factor,
    points,
    scale_mm,
    /,
    *,
    prior_shift=0.0,
    is_surrogate=False,
    ocean_corr=0.0,
    ocean_corr_scale_factor=0.4,
    prior_kernel="heat",
    prior_order=1.0,
    noise_corr_std_factor=0.0,
    point_evaluation_operator=None,
    derived_stds=None,
):
    """
    Constructs the 3-component joint prior and observation noise measures.
    Skips the spatial projection and mass conservation conditioning if is_surrogate=True.

    The amplitudes form a triangular chain anchored on one dimensioned
    number (see build_conditioned_prior):

      ocean_dyn_std_mm       : pointwise std of the sterodynamic field
                               (mm, pre-mass-constraint),
      steric_dyn_std_ratio   : mean-depth steric sea level std as a
                               fraction of the sterodynamic std; sets
                               the density amplitude,
      gmsl_bary_steric_ratio : ratio of the barystatic to steric GMSL
                               prior stds, the steric value being the
                               REALISED one under the mass constraint;
                               sets the ice amplitude.

    The derived pointwise stds and the realised GMSL-level statistics
    are reported in the returned calibration dictionary; the per-field
    length scales are unchanged inputs. If derived_stds is given
    ({"ice_std", "dyn_std", "rho_std"}, nondimensional), the derivation
    is skipped and those amplitudes are used directly -- the surrogate
    path, so the preconditioner amplitudes match the exact model instead
    of being re-derived at surrogate resolution. The altimetry noise
    factors below are referenced to the sterodynamic pointwise std, the
    field altimetry actually observes.

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
    exactly. The surrogate prior is always uncorrelated so that the Woodbury
    preconditioner construction is unaffected.
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
    the track points, with std noise_std_factor x the sterodynamic pointwise prior std, and
    an optional large-scale correlated error component with std
    noise_corr_std_factor x the sterodynamic pointwise prior std (0 disables) at correlation
    scale load_space.scale x noise_corr_scale_factor, representing
    long-wavelength systematics such as orbit and reference-frame errors.
    The correlated component barely averages down and so sets an
    irreducible error floor on large-scale functionals such as GMSLR; the
    implied GMSL floor (the exact std of its ocean mean) is recorded in
    calib["noise"] and printed with the calibration report. The
    surrogate always uses the local component alone, so the Woodbury
    preconditioner is built from the uncorrelated noise. If
    point_evaluation_operator is supplied (the one already built inside
    build_physics_components), it is reused for the correlated component
    instead of constructing a second, functionally identical dense
    operator.
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
    if noise_corr_std_factor < 0.0:
        raise ValueError("noise_corr_std_factor must be non-negative.")

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

    # --- PRIORS: amplitude derivation, assembly, masking and conditioning ---
    ice_scale = load_space.scale * ice_scale_factor
    ocean_dyn_scale = load_space.scale * ocean_dyn_scale_factor
    ocean_rho_scale = load_space.scale * ocean_rho_scale_factor

    model_prior, _, calib = build_conditioned_prior(
        state,
        load_space,
        load_measure,
        ice_scale,
        ocean_dyn_scale,
        ocean_rho_scale,
        gmsl_bary_steric_ratio,
        ocean_dyn_std_mm / scale_mm,
        steric_dyn_std_ratio,
        is_surrogate=is_surrogate,
        ocean_corr=ocean_corr,
        corr_scale=load_space.scale * ocean_corr_scale_factor,
        derived_stds=derived_stds,
    )

    # --- NOISE MODEL ---
    # Two-component altimetry noise: a local (uncorrelated) part on the
    # track points, plus an optional large-scale correlated error
    # representing long-wavelength systematics (orbit, reference-frame and
    # large-scale correction residuals). The correlated part barely
    # averages down, so it sets an irreducible error floor on large-scale
    # functionals such as GMSLR. The surrogate always uses the local part
    # alone, so the Woodbury preconditioner is built from the uncorrelated
    # noise; the low-rank discrepancy this leaves is absorbed by a few
    # extra CG iterations.
    noise_std = noise_std_factor * calib["derived_stds"]["dyn_std"]
    n_points = len(points)
    data_space = inf.EuclideanSpace(n_points)
    noise_meas = inf.GaussianMeasure.from_standard_deviation(data_space, noise_std)
    if noise_corr_std_factor > 0.0 and not is_surrogate:
        if point_evaluation_operator is None:
            point_evaluation_operator = load_space.point_evaluation_operator(points)
        corr_field_meas = load_measure(
            load_space.scale * noise_corr_scale_factor,
            noise_corr_std_factor * calib["derived_stds"]["dyn_std"],
        )
        # Irreducible error floor this component sets on GMSL-type
        # averages: the exact std of its ocean mean, reported with the
        # calibration so the floor is a deliberate choice.
        gmsl_fn = load_space.l2_products_operator(
            [state.ocean_projection(value=0.0) / state.ocean_area]
        )
        calib.setdefault("noise", {})["alt_corr_gmsl_floor"] = _functional_std(
            corr_field_meas, gmsl_fn
        )
        corr_noise_meas = corr_field_meas.affine_mapping(
            operator=point_evaluation_operator
        )
        noise_meas = noise_meas + corr_noise_meas

    # Prior shift
    if prior_shift != 0.0:
        offset_shape = model_prior.sample()
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(prior_shift, offset_shape)
        )

    return model_prior, noise_meas, calib
