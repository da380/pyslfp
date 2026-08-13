"""
joint_utils.py
=======================
Shared utilities, physics initializations, and Bayesian measures for
extended Joint (Altimetry + GRACE) inversions using a 3-component model.

Model space (three fields on the sphere):

    m = [ Dh, eta, drho ]

      Dh   : ice thickness change (grounded ice),
      eta  : sterodynamic sea level change (the TOTAL dynamic
             perturbation to the sea surface), applying the load rho_w*C*eta,
      drho : vertically averaged ocean density change, applying the load
             D*C*drho (the column mass change at fixed column thickness).

The prior enforces <rho_w*C*eta + D*C*drho>_O = 0, so the combined ocean
load is a pure redistribution of ocean mass. GRACE observes the response to
the total load; altimetry observes the GRD response plus the direct dynamic
term eta.

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
anti-correlation in the prior (see build_measures; requires pygeoinf >=
1.8.4), reflecting that at long wavelengths dynamic sea surface height
changes are predominantly steric in origin.

The three prior marginals can also be switched from heat-kernel to Sobolev
(Matern-type) covariances with a single common spectral order, giving
rougher, power-law-tailed fields while keeping the per-field length scales
and pointwise stds; the spatially correlated noise measures follow the same
family (see build_measures).
"""

import pygeoinf as inf

from pyslfp.state import EarthState
from pyslfp.linear_operators import (
    FingerPrintOperator,
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    ocean_density_change_to_load_operator,
    sea_surface_height_operator,
    ice_projection_operator,
    grace_observation_operator,
    WMBMethod,
)

# Post-processing helpers shared verbatim with the altimetry pipeline,
# re-exported here so the joint scripts keep using joint_utils.<name>.
# The redundant "x as x" aliases mark these as intentional re-exports:
# without them, unused-import cleanup (ruff/pyflakes F401, IDE "optimise
# imports") strips the ones this module does not call itself, silently
# breaking the scripts.
from altimetry_utils import (
    build_conditioned_prior as build_conditioned_prior,
    print_calibration_report as print_calibration_report,
    barystatic_gmsl_weighting as barystatic_gmsl_weighting,
    effective_steric_scale as effective_steric_scale,
    gmsl_split_operators as gmsl_split_operators,
    regional_decomposition_operators as regional_decomposition_operators,
    steric_sea_level_operator as steric_sea_level_operator,
    true_gmsl_operator as true_gmsl_operator,
)


def build_physics_components(
    lmax, load_order, load_scale_km, points, obs_degree, is_surrogate=False
):
    """
    Constructs the 3-component physical operators for Joint Inversion.
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

    # --- C. TRUE SEA LEVEL TRACK (For GMSLR) ---
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
        "point_eval": point_eval,
        "scale_mm": scale_mm,
    }


def build_measures(
    state,
    load_space,
    ice_scale_factor,
    gmsl_barystatic_std_mm,
    ocean_dyn_scale_factor,
    steric_dmslc_ratio,
    ocean_rho_scale_factor,
    gmsl_steric_std_mm,
    alt_noise_corr_scale_factor,
    alt_noise_std_factor,
    grace_noise_scale_km,
    grace_noise_std_mm,
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
    point_evaluation_operator=None,
    derived_stds=None,
    calibrate_constrained=True,
):
    """
    Constructs the 3-component joint prior and dual-sensor noise measures.

    The three relative amplitudes are set through GMSL-level targets
    rather than pointwise stds (see altimetry_utils.build_conditioned_prior):

      gmsl_barystatic_std_mm : prior std of the barystatic GMSL change (mm),
      gmsl_steric_std_mm     : prior std of the steric GMSL change (mm),
      steric_dmslc_ratio     : ratio of the ocean-integrated steric to
                               dynamic-manometric (zeta = eta - eta_s)
                               variance.

    The pointwise ice, sterodynamic and density stds are DERIVED from
    these targets (and reported in the returned "calib" entry); the
    per-field length scales are unchanged inputs. By default the ocean
    amplitudes are refined so the realised POST-constraint steric std and
    variance ratio match the targets (calibrate_constrained; the
    barystatic std is constraint-invariant). If derived_stds is given
    ({"ice_std", "dyn_std", "rho_std"}, nondimensional), the calibration
    is skipped and those amplitudes are used directly -- the surrogate
    path, so the preconditioner amplitudes match the exact model instead
    of being re-derived at surrogate resolution. The altimetry noise
    factors below remain referenced to the barystatic GMSL std, which is
    now the input target itself; the GRACE spatial noise std is an
    absolute value in mm (grace_noise_std_mm), no longer a factor of the
    (now derived) ice std.

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
    the track points, with std alt_noise_std_factor x the barystatic GMSLR prior std,
    and an optional large-scale correlated error component with std
    alt_noise_corr_std_factor x the barystatic GMSLR prior std (0 disables) at
    correlation scale load_space.scale x alt_noise_corr_scale_factor,
    representing long-wavelength systematics such as orbit and
    reference-frame errors. The correlated component barely averages down
    and so sets an irreducible error floor on large-scale functionals such
    as GMSLR. The returned alt_precond_noise and joint_precond_noise
    measures contain the local component alone (with the GRACE noise
    unchanged) for use in the preconditioner constructions. If
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

    # --- 1. PRIORS: calibration, assembly, masking and conditioning ---
    ice_scale = load_space.scale * ice_scale_factor
    ocean_dyn_scale = load_space.scale * ocean_dyn_scale_factor
    ocean_rho_scale = load_space.scale * ocean_rho_scale_factor
    gmsl_barystatic_std = gmsl_barystatic_std_mm / scale_mm

    model_prior, unmasked_prior, calib = build_conditioned_prior(
        state,
        load_space,
        load_measure,
        ice_scale,
        ocean_dyn_scale,
        ocean_rho_scale,
        gmsl_barystatic_std,
        gmsl_steric_std_mm / scale_mm,
        steric_dmslc_ratio,
        is_surrogate=is_surrogate,
        ocean_corr=ocean_corr,
        corr_scale=load_space.scale * ocean_corr_scale_factor,
        derived_stds=derived_stds,
        calibrate_constrained=calibrate_constrained,
    )
    ice_std = calib["derived_stds"]["ice_std"]

    if prior_shift != 0.0:
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(prior_shift, model_prior.sample())
        )

    if is_surrogate:
        return {
            "model_prior": model_prior,
            "unmasked_prior": unmasked_prior,
            "calib": calib,
        }

    # --- 2. NOISE MEASURES ---
    # Altimetry noise: a local (uncorrelated) part on the track points,
    # plus an optional large-scale correlated error representing
    # long-wavelength systematics (orbit, reference-frame and large-scale
    # correction residuals). The correlated part barely averages down, so
    # it sets an irreducible error floor on large-scale functionals such
    # as GMSL. The preconditioners are built from the local part alone
    # (alt_precond_noise / joint_precond_noise below); the low-rank
    # discrepancy this leaves is absorbed by a few extra CG iterations.
    alt_noise_std = alt_noise_std_factor * gmsl_barystatic_std
    alt_precond_noise_meas = inf.GaussianMeasure.from_standard_deviation(
        inf.EuclideanSpace(len(points)), alt_noise_std
    )
    alt_noise_meas = alt_precond_noise_meas
    if alt_noise_corr_std_factor > 0.0:
        if point_evaluation_operator is None:
            point_evaluation_operator = load_space.point_evaluation_operator(points)
        alt_corr_noise_meas = load_measure(
            load_space.scale * alt_noise_corr_scale_factor,
            alt_noise_corr_std_factor * gmsl_barystatic_std,
        ).affine_mapping(operator=point_evaluation_operator)
        alt_noise_meas = alt_noise_meas + alt_corr_noise_meas

    # GRACE Noise (absolute spatial std in mm)
    grace_spatial_scale = (
        grace_noise_scale_km * 1000.0 / state.model.parameters.length_scale
    )
    grace_spatial_noise = load_measure(
        grace_spatial_scale, grace_noise_std_mm / scale_mm
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
        "gmsl_std": gmsl_barystatic_std,
        "ice_std": ice_std,
        "ice_scale": ice_scale,
        "calib": calib,
    }
