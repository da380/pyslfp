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

import warnings

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


def ocean_depth_moments(state, load_space):
    """
    First and second moments of D / rho_w over the oceans:

        m1 = <D/rho_w>_O          (equals effective_steric_scale),
        m2 = <(D/rho_w)**2>_O,

    so that a density prior of pointwise std rho_std induces a steric sea
    level field with ocean-mean-depth std rho_std * m1 and rms-depth std
    rho_std * sqrt(m2).
    """
    f = (
        state.sea_level
        * state.ocean_projection(value=0.0)
        / state.model.parameters.water_density
    )
    m1 = effective_steric_scale(state)
    m2 = float(load_space.l2_products_operator([f])(f)[0]) / state.ocean_area
    return m1, m2


def pointwise_pair_correlation(
    load_measure,
    dyn_scale,
    rho_scale,
    ocean_corr,
    corr_scale,
    load_space,
    point=(0.0, 0.0),
):
    """
    Magnitude of the pointwise prior correlation between the sterodynamic
    and density fields under the correlated invariant pair (unit stds; the
    correlation is scale-free). Computed exactly from one covariance
    action on a Dirac representer per component; by invariance the value
    is independent of the evaluation point (masking, which preserves
    interior values, is applied downstream). Returns 0 when ocean_corr = 0.
    """
    if ocean_corr == 0.0:
        return 0.0
    from pygeoinf.symmetric_space.symmetric_space import (
        CorrelatedInvariantGaussianMeasure,
    )

    def spectral_correlation(k):
        r = -ocean_corr * np.exp(-(corr_scale**2) * k)
        return np.array([[1.0, r], [r, 1.0]])

    pair = CorrelatedInvariantGaussianMeasure.from_invariant_measures(
        [load_measure(dyn_scale, 1.0), load_measure(rho_scale, 1.0)],
        spectral_correlation,
    )
    dirac = load_space.dirac_representation(point)
    ev = load_space.point_evaluation_operator([point])
    c_dyn = pair.covariance([dirac, load_space.zero])
    c_rho = pair.covariance([load_space.zero, dirac])
    v_dyn = float(ev(c_dyn[0])[0])
    v_rho = float(ev(c_rho[1])[0])
    cross = float(ev(c_dyn[1])[0])
    return abs(cross) / np.sqrt(v_dyn * v_rho)


def _functional_std(measure, functional):
    """Std of a scalar linear functional of a Gaussian measure."""
    return float(
        np.sqrt(
            measure.affine_mapping(operator=functional).covariance.matrix(dense=True)[
                0, 0
            ]
        )
    )


def ocean_stds_from_ratio(gmsl_steric_std, steric_dmslc_ratio, u_rho, m1, m2, rho_bar):
    """
    Solves the (sigma_S, R) targets for the pointwise (dyn_std, rho_std).

    rho_std follows from the steric GMSL std alone (the density marginal
    is unchanged by the correlation, and indicator masking preserves
    ocean-interior values). With s_tilde = rho_std * m1 (mean-depth
    steric std), s_bar = rho_std * sqrt(m2) (rms-depth) and rho_bar the
    pointwise (eta, drho) correlation magnitude, the pre-constraint
    ocean-integrated variance ratio

        R = E||eta_s||^2_L2(O) / E||zeta||^2_L2(O)
          = s_bar**2 / (dyn_std**2 - 2*rho_bar*dyn_std*s_tilde + s_bar**2)

    is solved for dyn_std, taking the LARGER root (the energetic-eta
    regime with long-wavelength steric cancellation, continuous with the
    previous parameterisation; the smaller root is a low-energy-eta
    regime). R is attainable only up to
    R_max = s_bar**2 / (s_bar**2 - (rho_bar*s_tilde)**2): steric can
    dominate the integrated variance only insofar as the anti-correlation
    lets eta cancel against eta_s within zeta.
    """
    rho_std = gmsl_steric_std / u_rho
    s_tilde = rho_std * m1
    s_bar = rho_std * np.sqrt(m2)
    disc = (rho_bar * s_tilde) ** 2 + s_bar**2 * (1.0 / steric_dmslc_ratio - 1.0)
    if disc < 0.0:
        r_max = s_bar**2 / (s_bar**2 - (rho_bar * s_tilde) ** 2)
        raise ValueError(
            f"steric_dmslc_ratio = {steric_dmslc_ratio:.4f} is not attainable "
            f"at these prior shapes: the maximum ratio is {r_max:.4f} "
            "(raise ocean_corr, extend the correlation scale, or lower the "
            "target)."
        )
    dyn_std = rho_bar * s_tilde + np.sqrt(disc)
    return dyn_std, rho_std, s_tilde, s_bar


def build_conditioned_prior(
    state,
    load_space,
    load_measure,
    ice_scale,
    ocean_dyn_scale,
    ocean_rho_scale,
    gmsl_barystatic_std,
    gmsl_steric_std,
    steric_dmslc_ratio,
    *,
    is_surrogate=False,
    ocean_corr=0.0,
    corr_scale=None,
    derived_stds=None,
    calibrate_constrained=True,
    refine_rtol=1.0e-3,
    refine_max_iterations=8,
):
    """
    Calibrates the three pointwise prior stds from the GMSL-level targets
    and assembles the (optionally correlated) joint prior with masking and
    the mass-conservation conditioning. Shared by altimetry_utils and
    joint_utils build_measures.

    Targets (nondimensional):
      gmsl_barystatic_std : prior std of the barystatic GMSL change,
      gmsl_steric_std     : prior std of the steric GMSL change <eta_s>_O,
      steric_dmslc_ratio  : ratio of the ocean-integrated steric variance
                            E||eta_s||^2_L2(O) to the dynamic-manometric
                            variance E||zeta||^2_L2(O), zeta = eta - eta_s.

    The identity GMSL = barystatic + <eta_s>_O holds exactly under the
    mass constraint (the dynamic degree of freedom has zero ocean mean),
    so the two std targets fix the ice and density amplitudes, and the
    ratio -- necessarily a spatially resolved statement -- fixes the
    sterodynamic amplitude through ocean_stds_from_ratio. The barystatic
    functional is composed with the ice projection, so the target refers
    to the realised (masked) prior; the previous unmasked referencing
    overstated the realised barystatic std by the all-land to ice-sheet
    variance ratio of the weighting.

    If calibrate_constrained (exact model only), the density and
    sterodynamic amplitudes are refined by a fixed-point iteration so
    that the REALISED post-constraint sigma_S and R match the targets
    (sigma_B is exactly constraint-invariant); otherwise the targets
    refer to the masked, unconstrained measure and the realised
    post-constraint values are reported in the returned dictionary. The
    post-constraint values are computed exactly: the constraint is a
    rank-one conditioning, so second moments shift by closed-form Schur
    corrections requiring one covariance action each.

    If derived_stds is given (a dict with ice_std, dyn_std, rho_std), the
    calibration is skipped and those amplitudes are used directly: the
    surrogate path, so preconditioner amplitudes match the exact model
    instead of being re-derived at surrogate resolution.

    Returns (model_prior, unmasked_prior, calib): the conditioned
    (or, for the surrogate, unconditioned) prior, the uncorrelated
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
        s_tilde = s_bar = rho_bar = None
        calibrating = False
    else:
        if min(gmsl_barystatic_std, gmsl_steric_std, steric_dmslc_ratio) <= 0.0:
            raise ValueError(
                "gmsl_barystatic_std_mm, gmsl_steric_std_mm and "
                "steric_dmslc_ratio must all be positive."
            )
        calibrating = True

        ice_proj = ice_projection_operator(state, load_space)
        ocean_proj = ocean_projection_operator(state, load_space)

        # Calibration functionals composed with the projections, so the
        # targets are exact for the realised masked prior at the working
        # discretisation.
        bary_fn_field = load_space.l2_products_operator(
            [barystatic_gmsl_weighting(state)]
        )
        steric_fn_field = steric_gmsl_functional(state, load_space)
        q_ice = _functional_std(load_measure(ice_scale, 1.0), bary_fn_field @ ice_proj)
        u_rho = _functional_std(
            load_measure(ocean_rho_scale, 1.0), steric_fn_field @ ocean_proj
        )
        m1, m2 = ocean_depth_moments(state, load_space)
        rho_bar = pointwise_pair_correlation(
            load_measure,
            ocean_dyn_scale,
            ocean_rho_scale,
            ocean_corr,
            corr_scale,
            load_space,
        )
        r_max = m2 / (m2 - (rho_bar * m1) ** 2) if rho_bar > 0.0 else 1.0

        ice_std = gmsl_barystatic_std / q_ice
        dyn_std, rho_std, s_tilde, s_bar = ocean_stds_from_ratio(
            gmsl_steric_std, steric_dmslc_ratio, u_rho, m1, m2, rho_bar
        )
        calib = {
            "targets": {
                "gmsl_barystatic_std": gmsl_barystatic_std,
                "gmsl_steric_std": gmsl_steric_std,
                "steric_dmslc_ratio": steric_dmslc_ratio,
            },
            "constants": {
                "q_ice": q_ice,
                "u_rho": u_rho,
                "m1": m1,
                "m2": m2,
                "rho_bar": rho_bar,
                "R_max": r_max,
            },
        }

    def assemble(dyn_std_i, rho_std_i):
        ice_prior = load_measure(ice_scale, ice_std)
        dyn_prior = load_measure(ocean_dyn_scale, dyn_std_i)
        rho_prior = load_measure(ocean_rho_scale, rho_std_i)
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

    if is_surrogate:
        model_prior, unmasked_prior = assemble(dyn_std, rho_std)
        calib["derived_stds"] = {
            "ice_std": ice_std,
            "dyn_std": dyn_std,
            "rho_std": rho_std,
        }
        calib.setdefault("post", None)
        return model_prior, unmasked_prior, calib

    # --- masking and mass-conservation conditioning (exact model) ---
    ice_proj = ice_projection_operator(state, load_space)
    ocean_proj = ocean_projection_operator(state, load_space)
    proj_op = inf.BlockDiagonalLinearOperator([ice_proj, ocean_proj, ocean_proj])

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
    steric_op = steric_sea_level_operator(state, load_space)

    def condition_and_diagnose(dyn_std_i, rho_std_i, s_tilde_i, s_bar_i):
        joint, unmasked = assemble(dyn_std_i, rho_std_i)
        masked = joint.affine_mapping(operator=proj_op)
        constrained = mass_subspace.condition_gaussian_measure(masked)
        if not calibrating:
            return constrained, unmasked, None

        joint_space = masked.domain
        bary_fn = load_space.l2_products_operator(
            [barystatic_gmsl_weighting(state)]
        ) @ joint_space.subspace_projection(0)
        steric_fn = steric_gmsl_functional(state, load_space) @ (
            joint_space.subspace_projection(2)
        )
        # Rank-one Schur corrections for the ocean-integrated second
        # moments: for the constraint functional g, E_constr||f||^2 =
        # E0||f||^2 - ||Cov0(f(.), g)||^2_L2(O) / Var0(g), with the
        # cross-covariance field from one covariance action on the
        # adjoint representer of g.
        var_g = _functional_std(masked, mass_constraint_op) ** 2
        h = masked.covariance(mass_constraint_op.adjoint(np.ones(1)))
        h_etas = steric_op(h[2])
        h_zeta = h[1] - h_etas

        def l2_norm_sq(field):
            return float(load_space.l2_products_operator([field])(field)[0])

        a_ocean = state.ocean_area
        e0_etas = a_ocean * s_bar_i**2
        e0_zeta = a_ocean * (
            dyn_std_i**2 - 2.0 * rho_bar * dyn_std_i * s_tilde_i + s_bar_i**2
        )
        diagnostics = {
            "pre": {
                "gmsl_barystatic_std": _functional_std(masked, bary_fn),
                "gmsl_steric_std": _functional_std(masked, steric_fn),
                "steric_dmslc_ratio": e0_etas / e0_zeta,
            },
            "post": {
                "gmsl_barystatic_std": _functional_std(constrained, bary_fn),
                "gmsl_steric_std": _functional_std(constrained, steric_fn),
                "steric_dmslc_ratio": (e0_etas - l2_norm_sq(h_etas) / var_g)
                / (e0_zeta - l2_norm_sq(h_zeta) / var_g),
            },
        }
        return constrained, unmasked, diagnostics

    if not calibrating:
        model_prior, unmasked_prior, _ = condition_and_diagnose(
            dyn_std, rho_std, None, None
        )
        return model_prior, unmasked_prior, calib

    # Fixed-point refinement: rescale the pre-constraint (sigma_S, R)
    # targets by the target-to-realised ratios until the post-constraint
    # values match the requested ones. sigma_B is constraint-invariant
    # (the ice component is uncorrelated with the ocean pair and absent
    # from the constraint), so only the ocean amplitudes iterate.
    target_s, target_r = gmsl_steric_std, steric_dmslc_ratio
    iterations, converged = 0, True
    while True:
        model_prior, unmasked_prior, diagnostics = condition_and_diagnose(
            dyn_std, rho_std, s_tilde, s_bar
        )
        if not calibrate_constrained:
            break
        err_s = diagnostics["post"]["gmsl_steric_std"] / gmsl_steric_std - 1.0
        err_r = diagnostics["post"]["steric_dmslc_ratio"] / steric_dmslc_ratio - 1.0
        if max(abs(err_s), abs(err_r)) < refine_rtol:
            break
        iterations += 1
        if iterations >= refine_max_iterations:
            warnings.warn(
                "post-constraint amplitude refinement did not converge to "
                f"rtol = {refine_rtol:g} in {refine_max_iterations} "
                "iterations; using the last iterate."
            )
            converged = False
            break
        target_s /= 1.0 + err_s
        target_r /= 1.0 + err_r
        try:
            dyn_std, rho_std, s_tilde, s_bar = ocean_stds_from_ratio(
                target_s,
                target_r,
                calib["constants"]["u_rho"],
                calib["constants"]["m1"],
                calib["constants"]["m2"],
                rho_bar,
            )
        except ValueError:
            warnings.warn(
                "post-constraint refinement pushed the variance ratio "
                "beyond the attainable ceiling; falling back to the "
                "pre-constraint calibration."
            )
            dyn_std, rho_std, s_tilde, s_bar = ocean_stds_from_ratio(
                gmsl_steric_std,
                steric_dmslc_ratio,
                calib["constants"]["u_rho"],
                calib["constants"]["m1"],
                calib["constants"]["m2"],
                rho_bar,
            )
            model_prior, unmasked_prior, diagnostics = condition_and_diagnose(
                dyn_std, rho_std, s_tilde, s_bar
            )
            converged = False
            break

    calib["derived_stds"] = {
        "ice_std": ice_std,
        "dyn_std": dyn_std,
        "rho_std": rho_std,
    }
    calib.update(diagnostics)
    calib["refinement_iterations"] = iterations
    calib["refinement_converged"] = converged
    return model_prior, unmasked_prior, calib


def print_calibration_report(calib, scale_mm):
    """Prints the derived amplitudes and realised GMSL-level statistics."""
    d = calib["derived_stds"]
    if "constants" not in calib:
        print(
            "Prior amplitudes (supplied, uncalibrated): ice std = "
            f"{d['ice_std'] * scale_mm:.3f} mm, sterodynamic std = "
            f"{d['dyn_std'] * scale_mm:.3f} mm, density std = "
            f"{d['rho_std']:.4e}"
        )
        return
    print(
        "Calibrated prior amplitudes: ice std = "
        f"{d['ice_std'] * scale_mm:.3f} mm, sterodynamic std = "
        f"{d['dyn_std'] * scale_mm:.3f} mm, density std = "
        f"{d['rho_std']:.4e} (mean-depth steric std = "
        f"{d['rho_std'] * calib['constants']['m1'] * scale_mm:.3f} mm)"
    )
    c = calib["constants"]
    print(
        f"  pointwise (eta, drho) correlation = {c['rho_bar']:.3f}; "
        f"attainable steric/DMSLC ratio ceiling = {c['R_max']:.3f}"
    )
    for tag, label in (("pre", "unconstrained"), ("post", "constrained")):
        v = calib.get(tag)
        if v is None:
            continue
        print(
            f"  {label}: barystatic GMSL std = "
            f"{v['gmsl_barystatic_std'] * scale_mm:.3f} mm, steric GMSL std = "
            f"{v['gmsl_steric_std'] * scale_mm:.3f} mm, steric/DMSLC "
            f"variance ratio = {v['steric_dmslc_ratio']:.4f}"
        )
    if calib.get("refinement_iterations", 0) > 0:
        print(
            "  (ocean amplitudes refined to post-constraint targets in "
            f"{calib['refinement_iterations']} iteration(s))"
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
    gmsl_barystatic_std_mm,
    ocean_dyn_scale_factor,
    steric_dmslc_ratio,
    ocean_rho_scale_factor,
    gmsl_steric_std_mm,
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
    calibrate_constrained=True,
):
    """
    Constructs the 3-component joint prior and observation noise measures.
    Skips the spatial projection and mass conservation conditioning if is_surrogate=True.

    The three relative amplitudes are set through GMSL-level targets
    rather than pointwise stds (see build_conditioned_prior):

      gmsl_barystatic_std_mm : prior std of the barystatic GMSL change (mm),
      gmsl_steric_std_mm     : prior std of the steric GMSL change (mm),
      steric_dmslc_ratio     : ratio of the ocean-integrated steric to
                               dynamic-manometric (zeta = eta - eta_s)
                               variance.

    The pointwise ice, sterodynamic and density stds are DERIVED from
    these targets (and reported in the returned calibration dictionary);
    the per-field length scales are unchanged inputs. By default the
    density and sterodynamic amplitudes are refined so the realised
    POST-constraint steric std and variance ratio match the targets
    (calibrate_constrained; the barystatic std is constraint-invariant).
    If derived_stds is given ({"ice_std", "dyn_std", "rho_std"},
    nondimensional), the calibration is skipped and those amplitudes are
    used directly -- the surrogate path, so the preconditioner amplitudes
    match the exact model instead of being re-derived at surrogate
    resolution. The altimetry noise factors below remain referenced to
    the barystatic GMSL std, which is now the input target itself.

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
    the track points, with std noise_std_factor x the barystatic GMSLR prior std, and
    an optional large-scale correlated error component with std
    noise_corr_std_factor x the barystatic GMSLR prior std (0 disables) at correlation
    scale load_space.scale x noise_corr_scale_factor, representing
    long-wavelength systematics such as orbit and reference-frame errors.
    The correlated component barely averages down and so sets an
    irreducible error floor on large-scale functionals such as GMSLR. The
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

    # --- PRIORS: calibration, assembly, masking and conditioning ---
    ice_scale = load_space.scale * ice_scale_factor
    ocean_dyn_scale = load_space.scale * ocean_dyn_scale_factor
    ocean_rho_scale = load_space.scale * ocean_rho_scale_factor
    gmsl_barystatic_std = gmsl_barystatic_std_mm / scale_mm

    model_prior, _, calib = build_conditioned_prior(
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
    noise_std = noise_std_factor * gmsl_barystatic_std
    n_points = len(points)
    data_space = inf.EuclideanSpace(n_points)
    noise_meas = inf.GaussianMeasure.from_standard_deviation(data_space, noise_std)
    if noise_corr_std_factor > 0.0 and not is_surrogate:
        if point_evaluation_operator is None:
            point_evaluation_operator = load_space.point_evaluation_operator(points)
        corr_noise_meas = load_measure(
            load_space.scale * noise_corr_scale_factor,
            noise_corr_std_factor * gmsl_barystatic_std,
        ).affine_mapping(operator=point_evaluation_operator)
        noise_meas = noise_meas + corr_noise_meas

    # Prior shift
    if prior_shift != 0.0:
        offset_shape = model_prior.sample()
        model_prior = model_prior.affine_mapping(
            translation=model_prior.domain.multiply(prior_shift, offset_shape)
        )

    return model_prior, noise_meas, calib
