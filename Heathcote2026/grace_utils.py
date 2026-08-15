"""
grace_utils.py
==============
Shared utilities, physics initializations, and plotting for Bayesian GRACE inversions.

The direct load prior and the spatial noise measure can optionally use
Sobolev (Matern-type) covariances in place of the default heat kernels,
giving rougher, power-law-tailed fields while keeping the correlation scale
and pointwise std settings. The noise takes its own Sobolev exponent, by
default solved together with its amplitude from a specified low-degree
signal-to-noise ratio and the degree at which the spectral SNR crosses one
(see build_measures and resolve_noise_amplitude).
"""

import pygeoinf as inf


from pyslfp import EarthState
from pyslfp.linear_operators import (
    FingerPrintOperator,
    sea_level_change_to_load_operator,
    averaging_operator,
)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def build_physics_components(lmax, load_order, load_scale_km):
    """Initialises the Earth state, Sobolev spaces, and fingerprint operator."""
    state = EarthState.from_defaults(lmax=lmax)

    length_scale = state.model.parameters.length_scale
    water_density = state.model.parameters.water_density

    load_to_water_thickness_mm = 1000 * length_scale / water_density
    load_space_scale = load_scale_km * 1000 / length_scale

    finger_print_operator = FingerPrintOperator(
        state, load_parameters=(load_order, load_space_scale)
    )

    load_space = finger_print_operator.domain
    response_space = finger_print_operator.codomain

    return (
        state,
        load_space,
        response_space,
        finger_print_operator,
        load_to_water_thickness_mm,
    )


def invariant_coefficient_profile(
    kernel, order, scale, space_order, space_scale, degrees, /, *, radius=1.0
):
    """Unnormalised per-coefficient L2 variances c_l at the given degrees."""
    degrees = np.asarray(degrees, dtype=float)
    lam = degrees * (degrees + 1.0) / radius**2
    space_weight = (1.0 + space_scale**2 * lam) ** (-space_order)
    if kernel == "sobolev":
        return (1.0 + scale**2 * lam) ** (-order) * space_weight
    if kernel == "heat":
        return np.exp(-(scale**2) * lam) * space_weight
    raise ValueError("kernel must be 'heat' or 'sobolev'.")


def resolve_noise_amplitude(
    load_space,
    prior_kernel,
    prior_order,
    noise_order,
    signal_scale,
    signal_std,
    noise_scale,
    /,
    *,
    noise_std_factor=None,
    snr_crossover_degree=None,
    snr_low_degree=None,
    snr_reference_degree=2.0,
    obs_degree=None,
    factor_reference_std=None,
    std_to_mm=None,
    label="spatial noise",
):
    """
    Resolves the noise spectrum (exponent and amplitude) and reports the
    spectral signal-to-noise structure of a (signal, noise) pair of
    invariant kernel measures on the same space.

    The per-coefficient L2 variances are

        signal: c_l = A_s (1 + s_s^2 lam_l)^(-q_s) w_l^(-1),
        noise:  c_l = B   (1 + s_n^2 lam_l)^(-q_n) w_l^(-1),

    with lam_l = l(l+1)/radius^2 and w_l = (1 + s0^2 lam_l)^p the load
    space's spectral weight, which cancels in the per-coefficient ratio

        SNR(l) = (A_s / B) (1 + s_s^2 lam_l)^(-q_s) (1 + s_n^2 lam_l)^(q_n).

    A_s is fixed by the signal's pointwise std; the noise is then
    resolved in one of three mutually exclusive ways:

      snr_low_degree + snr_crossover_degree :
          two log-linear conditions -- amplitude SNR = snr_low_degree
          at degree snr_reference_degree, and SNR = 1 at the crossover
          -- solve the pair (q_n, B) in closed form,

              q_n = (q_s DL_s - ln rho_var) / DL_n,
              DL_x = ln(1 + s_x^2 lam_{l*}) - ln(1 + s_x^2 lam_ref),

          (sobolev only: the exponent is the second dof). The noise
          must be a legitimate random field, q_n + p > 1, which caps
          the low-degree contrast at
          rho_amp < exp{[q_s DL_s + (p - 1) DL_n] / 2}; infeasible
          requests raise with the cap reported.

      noise_order + snr_crossover_degree :
          fixed exponent; only B is solved from SNR(l*) = 1.

      noise_std_factor :
          legacy amplitude mode; the pointwise noise std is this factor
          of factor_reference_std (default signal_std), with exponent
          noise_order (default prior_order).

    Writing lam = l(l+1), the log-derivative of SNR in lam has
    numerator -[(q_s s_s^2 - q_n s_n^2) + (q_s - q_n) s_s^2 s_n^2 lam],
    affine in lam, so SNR is monotonically decreasing -- and the
    crossover unique -- iff q_n <= q_s and q_n s_n^2 <= q_s s_s^2 (heat
    family: s_n <= s_s). This is enforced in the crossover modes and
    reported otherwise. Heat kernels have no exponent, so they support
    only the legacy and fixed-shape crossover modes.

    Returns a dict with the resolution 'mode', the spectral
    'kernel_amplitude' B, the implied pointwise 'noise_std' at the
    space's lmax, its 'std_factor' relative to factor_reference_std,
    the effective 'noise_order', the 'crossover_degree' (given, or
    solved in legacy mode if reached, else None), the amplitude SNR at
    the reference degree, the observation-band variance fraction and
    std (l <= obs_degree, the only part the WMB map transmits),
    band-edge NSR, 'legitimate_field' and 'monotone' flags, and a
    'plot_smoothing_scale' (heat kernel with half-power at the
    crossover) for map smoothing.
    """
    if noise_std_factor is not None and (
        snr_crossover_degree is not None or snr_low_degree is not None
    ):
        raise ValueError(
            "noise_std_factor is mutually exclusive with the SNR "
            "conditions (snr_crossover_degree / snr_low_degree)."
        )
    if noise_std_factor is None and snr_crossover_degree is None:
        raise ValueError("Give either noise_std_factor or snr_crossover_degree.")
    if noise_order is not None and snr_low_degree is not None:
        raise ValueError(
            "noise_order and snr_low_degree both fix the noise "
            "exponent; give at most one."
        )
    if prior_kernel not in ("heat", "sobolev"):
        raise ValueError("prior_kernel must be 'heat' or 'sobolev'.")
    if prior_kernel == "heat" and (
        noise_order is not None or snr_low_degree is not None
    ):
        raise ValueError(
            "noise_order / snr_low_degree apply to the sobolev family "
            "only (the heat kernel has no exponent)."
        )

    space_order = load_space.order
    space_scale = load_space.scale
    lmax = load_space.lmax
    radius = load_space.radius
    q_s = prior_order

    degrees = np.arange(lmax + 1, dtype=float)
    weights = 2.0 * degrees + 1.0

    def lam(ell):
        ell = np.asarray(ell, dtype=float)
        return ell * (ell + 1.0) / radius**2

    def log_kernel(scale, ell):
        return float(np.log1p(scale**2 * lam(ell)))

    # Signal spectrum and its kernel amplitude A_s (from the pointwise
    # normalisation of the point-value-scaled prior).
    prof_s = invariant_coefficient_profile(
        prior_kernel,
        q_s,
        signal_scale,
        space_order,
        space_scale,
        degrees,
        radius=radius,
    )
    z_s = float(np.sum(weights * prof_s))
    kernel_amplitude_signal = 4.0 * np.pi * signal_std**2 / z_s

    band_degree = int(min(obs_degree, lmax)) if obs_degree is not None else lmax
    reference = signal_std if factor_reference_std is None else factor_reference_std
    ref_degree = float(snr_reference_degree)
    solved_exponent = False

    if snr_crossover_degree is not None:
        lstar = float(snr_crossover_degree)
        if not 0.0 < lstar <= band_degree:
            raise ValueError(
                "snr_crossover_degree must lie in (0, "
                f"min(obs_degree, lmax)] = (0, {band_degree}]."
            )
        if snr_low_degree is not None:
            # --- two-condition solve for (q_n, B) ---
            if ref_degree < 1.0:
                raise ValueError("snr_reference_degree must be >= 1.")
            if not ref_degree < lstar:
                raise ValueError(
                    "snr_reference_degree must lie below " "snr_crossover_degree."
                )
            if not snr_low_degree > 1.0:
                raise ValueError(
                    "snr_low_degree (the amplitude SNR at the reference "
                    "degree) must exceed 1."
                )
            ln_rho_var = 2.0 * np.log(snr_low_degree)
            dl_s = log_kernel(signal_scale, lstar) - log_kernel(
                signal_scale, ref_degree
            )
            dl_n = log_kernel(noise_scale, lstar) - log_kernel(noise_scale, ref_degree)
            q_n = float((q_s * dl_s - ln_rho_var) / dl_n)
            solved_exponent = True
            if q_n + space_order <= 1.0 + 1.0e-12:
                rho_cap = np.exp((q_s * dl_s + (space_order - 1.0) * dl_n) / 2.0)
                raise ValueError(
                    f"The solved noise exponent q_n = {q_n:.3f} gives an "
                    "illegitimate field (needs q_n + p > 1 with p = "
                    f"{space_order:.1f}). For this crossover degree and "
                    "these scales the low-degree amplitude SNR must be "
                    f"below {rho_cap:.2f}."
                )
        else:
            q_n = q_s if noise_order is None else noise_order

        if prior_kernel == "sobolev":
            lin_0 = q_s * signal_scale**2 - q_n * noise_scale**2
            lin_1 = (q_s - q_n) * signal_scale**2 * noise_scale**2
            monotone = lin_0 >= 0.0 and lin_1 >= 0.0 and (lin_0 + lin_1) > 0.0
        else:
            monotone = noise_scale < signal_scale
        if not monotone:
            hint = (
                " (reduce noise_scale relative to signal_scale)"
                if solved_exponent
                else ""
            )
            raise ValueError(
                "The SNR crossover requires a monotonically decreasing "
                "signal-to-noise ratio: q_n <= q_s and q_n * s_n^2 <= "
                "q_s * s_s^2 for the sobolev family (s_n < s_s for "
                f"heat), not both equalities{hint}."
            )
        if prior_kernel == "sobolev":
            log_b = (
                np.log(kernel_amplitude_signal)
                - q_s * log_kernel(signal_scale, lstar)
                + q_n * log_kernel(noise_scale, lstar)
            )
        else:
            q_n = None
            log_b = (
                np.log(kernel_amplitude_signal)
                - signal_scale**2 * float(lam(lstar))
                + noise_scale**2 * float(lam(lstar))
            )
        kernel_amplitude = float(np.exp(log_b))
        crossover_degree = lstar
        mode = "two-condition" if solved_exponent else "fixed-order-crossover"
    else:
        # --- legacy amplitude mode ---
        q_n = (
            (q_s if noise_order is None else noise_order)
            if (prior_kernel == "sobolev")
            else None
        )
        noise_std = noise_std_factor * reference
        if prior_kernel == "sobolev":
            lin_0 = q_s * signal_scale**2 - q_n * noise_scale**2
            lin_1 = (q_s - q_n) * signal_scale**2 * noise_scale**2
            monotone = lin_0 >= 0.0 and lin_1 >= 0.0 and (lin_0 + lin_1) > 0.0
        else:
            monotone = noise_scale < signal_scale
        mode = "amplitude"

    prof_n = invariant_coefficient_profile(
        prior_kernel,
        0.0 if q_n is None else q_n,
        noise_scale,
        space_order,
        space_scale,
        degrees,
        radius=radius,
    )
    z_n = float(np.sum(weights * prof_n))
    if mode == "amplitude":
        kernel_amplitude = 4.0 * np.pi * noise_std**2 / z_n
    else:
        noise_std = float(np.sqrt(kernel_amplitude * z_n / (4.0 * np.pi)))

    # Per-coefficient NSR from the kernel-only profiles (the space
    # weight cancels; passing space_order=0 removes it).
    kern_s = invariant_coefficient_profile(
        prior_kernel,
        q_s,
        signal_scale,
        0.0,
        space_scale,
        degrees,
        radius=radius,
    )
    kern_n = invariant_coefficient_profile(
        prior_kernel,
        0.0 if q_n is None else q_n,
        noise_scale,
        0.0,
        space_scale,
        degrees,
        radius=radius,
    )
    nsr = (kernel_amplitude * kern_n) / (kernel_amplitude_signal * kern_s)
    if mode == "amplitude":
        above = np.nonzero(nsr[1:] >= 1.0)[0]
        crossover_degree = float(above[0] + 1) if above.size else None
    snr_amp_ref = float(1.0 / np.sqrt(nsr[int(round(ref_degree))]))

    legitimate = True if q_n is None else bool(q_n + space_order > 1.0 + 1.0e-12)
    band_fraction = float(
        np.sum(weights[: band_degree + 1] * prof_n[: band_degree + 1]) / z_n
    )
    info = {
        "mode": mode,
        "kernel_amplitude": float(kernel_amplitude),
        "noise_std": float(noise_std),
        "std_factor": float(noise_std / reference),
        "noise_order": q_n,
        "crossover_degree": crossover_degree,
        "snr_low_degree": snr_amp_ref,
        "snr_reference_degree": ref_degree,
        "band_degree": band_degree,
        "band_variance_fraction": band_fraction,
        "band_std": float(noise_std * np.sqrt(band_fraction)),
        "nsr_band_edge": float(nsr[band_degree]),
        "monotone": monotone,
        "legitimate_field": legitimate,
        "plot_smoothing_scale": (
            radius
            * float(
                np.sqrt(np.log(2.0) / (crossover_degree * (crossover_degree + 1.0)))
            )
            if crossover_degree is not None
            else None
        ),
    }

    if std_to_mm:
        in_mm = lambda x: f"{x * std_to_mm:.3f} mm"  # noqa: E731
    else:
        in_mm = lambda x: f"{x:.4e}"  # noqa: E731
    family = "sobolev" if prior_kernel == "sobolev" else "heat kernels"
    print(
        f"{label}: {family}, scale ratio s_n/s_s = " f"{noise_scale / signal_scale:.2f}"
    )
    if mode == "two-condition":
        print(
            f"  solved from amplitude SNR = {snr_amp_ref:.2f} at degree "
            f"{ref_degree:g} and crossover at degree {crossover_degree:g}:"
        )
        print(
            f"  noise order q_n = {q_n:+.3f} "
            f"(q_n + p = {q_n + space_order:.2f} > 1: legitimate field)"
        )
    elif mode == "fixed-order-crossover":
        order_txt = "" if q_n is None else f" with fixed order q_n = {q_n:.2f}"
        print(
            f"  amplitude from SNR crossover at degree "
            f"{crossover_degree:g}{order_txt}: amplitude SNR = "
            f"{snr_amp_ref:.2f} at degree {ref_degree:g}"
        )
        if not legitimate:
            print(
                "  WARNING: q_n + p <= 1 -- the noise is not a legitimate "
                "field and its pointwise std is lmax-dominated."
            )
    else:
        cross_txt = (
            f"NSR crosses one at degree {crossover_degree:g}"
            if crossover_degree is not None
            else f"NSR stays below one up to lmax (NSR(lmax) = {nsr[-1]:.3f})"
        )
        print(f"  amplitude mode: {cross_txt}")
        if not monotone:
            print(
                "  WARNING: the signal-to-noise ratio is not monotone in "
                "degree for these orders/scales; any crossover need not "
                "be unique."
            )
        if not legitimate:
            print(
                "  WARNING: q_n + p <= 1 -- the noise is not a legitimate "
                "field and its pointwise std is lmax-dominated."
            )
    print(
        f"  pointwise std = {in_mm(noise_std)} "
        f"({info['std_factor']:.4f} x reference std); "
        f"{100 * band_fraction:.1f}% of the noise variance lies in the "
        f"band l <= {band_degree} at lmax = {lmax} "
        f"(band std = {in_mm(info['band_std'])})"
    )
    print(
        f"  noise/signal amplitude = {np.sqrt(info['nsr_band_edge']):.1f} "
        f"at l = {band_degree}"
    )
    return info


def build_measures(
    state,
    load_space,
    direct_scale_km,
    direct_std_m,
    noise_scale_factor,
    noise_std_factor,
    /,
    *,
    remove_degree_1=False,
    prior_shift=0.0,
    prior_kernel="heat",
    prior_order=1.0,
    noise_order=None,
    snr_crossover_degree=None,
    snr_low_degree=None,
    snr_reference_degree=2.0,
    obs_degree=None,
):
    """
    Constructs the prior and noise Gaussian measures.

    The direct load prior uses a point-value-scaled heat-kernel covariance
    by default. With prior_kernel="sobolev" it instead uses a Sobolev
    (Matern-type) covariance proportional to
    (1 + scale**2 * Laplacian)**(-prior_order) with respect to the load
    space inner product. Note that sample spectra therefore decay with the
    COMBINED exponent prior_order + (load space order): with the default
    order-2 spaces, prior_order = 1 gives degree-variance tails ~ l**(-5),
    and any positive order yields a finite pointwise variance. The
    correlation scale and pointwise std settings retain their meaning under
    either family.

    The spatial noise measure uses the same covariance family (mixed
    families -- a power-law signal against an exponentially decaying
    noise -- would make the spectral signal-to-noise ratio non-monotone
    and are not supported) with its own Sobolev exponent, representing
    -- very crudely -- the spatially correlated uncertainty of the
    GRACE corrections (GIA, geocentre, low-degree replacements,
    leakage) rather than the formal measurement covariance. It is
    resolved in one of three mutually exclusive ways (see
    resolve_noise_amplitude for the algebra and validation):

      snr_low_degree + snr_crossover_degree :
          default; the exponent and amplitude are solved in closed form
          from the amplitude SNR at a low reference degree and the
          degree where the per-coefficient variances are equal. The
          noise is required to be a legitimate random field
          (q_n + p > 1), which caps the achievable low-degree SNR,

      noise_order + snr_crossover_degree :
          fixed exponent, amplitude solved from the crossover alone,

      noise_std_factor :
          legacy; pointwise noise std as a factor of the prior's, with
          exponent noise_order (default prior_order).

    Solved-mode noise is constructed directly from its spectral kernel
    amplitude (invariant_gaussian_measure), so no pointwise
    normalisation of the rough field is involved; the implied pointwise
    and observation-band stds (l <= obs_degree, the only part the WMB
    map transmits) are reported. Both measures remain invariant with
    analytic spectral structure, so the WMB preconditioner construction
    is unchanged. The resolved spectrum and diagnostics are returned as
    a fifth element and printed.
    """
    if prior_kernel not in ("heat", "sobolev"):
        raise ValueError("prior_kernel must be 'heat' or 'sobolev'.")
    if prior_kernel == "sobolev" and prior_order <= 0.0:
        raise ValueError(
            "prior_order must be positive. Sample spectra combine this "
            "order with the load space order, so the pointwise variance "
            "is well defined for any positive value on the (order > 1) "
            "spaces used here."
        )

    length_scale = state.model.parameters.length_scale
    water_density = state.model.parameters.water_density

    direct_load_measure_scale = direct_scale_km * 1000 / length_scale
    direct_load_measure_std = water_density * direct_std_m / length_scale

    # Common covariance family for the direct load prior AND the spatial
    # noise measure, so the spectral signal-to-noise crossover is well
    # defined under either family.
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

    initial_direct_load_prior = load_measure(
        direct_load_measure_scale, direct_load_measure_std
    )

    if remove_degree_1:
        constraint_lmax = 1
        constraint_operator = load_space.to_coefficient_operator(constraint_lmax)
        constraint_subspace = inf.LinearSubspace.from_kernel(constraint_operator)
        direct_load_prior = constraint_subspace.condition_gaussian_measure(
            initial_direct_load_prior
        )
    else:
        direct_load_prior = initial_direct_load_prior

    if prior_shift != 0.0:
        offset_shape = direct_load_prior.sample()
        direct_load_prior = direct_load_prior.affine_mapping(
            translation=offset_shape * prior_shift
        )

    noise_load_measure_scale = noise_scale_factor * direct_load_measure_scale
    noise_info = resolve_noise_amplitude(
        load_space,
        prior_kernel,
        prior_order,
        noise_order,
        direct_load_measure_scale,
        direct_load_measure_std,
        noise_load_measure_scale,
        noise_std_factor=noise_std_factor,
        snr_crossover_degree=snr_crossover_degree,
        snr_low_degree=snr_low_degree,
        snr_reference_degree=snr_reference_degree,
        obs_degree=obs_degree,
        std_to_mm=1000.0 * length_scale / water_density,
        label="GRACE spatial noise",
    )
    if noise_info["mode"] == "amplitude":
        # Legacy pointwise-normalised construction (identical measures
        # to the historical behaviour).
        if prior_kernel == "sobolev":
            noise_load_measure = (
                load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
                    noise_info["noise_order"],
                    noise_load_measure_scale,
                    std=noise_info["noise_std"],
                )
            )
        else:
            noise_load_measure = load_measure(
                noise_load_measure_scale, noise_info["noise_std"]
            )
    else:
        # Solved modes: the spectrum is fixed directly by its kernel
        # amplitude, with no pointwise normalisation of the rough field.
        _b = noise_info["kernel_amplitude"]
        if prior_kernel == "sobolev":
            _q = noise_info["noise_order"]
            noise_load_measure = load_space.invariant_gaussian_measure(
                lambda k, b=_b, s=noise_load_measure_scale, q=_q: (
                    b * (1.0 + s * s * k) ** (-q)
                )
            )
        else:
            noise_load_measure = load_space.invariant_gaussian_measure(
                lambda k, b=_b, s=noise_load_measure_scale: (b * np.exp(-(s * s) * k))
            )

    return (
        initial_direct_load_prior,
        direct_load_prior,
        noise_load_measure,
        noise_load_measure_scale,
        noise_info,
    )


def build_total_load_operator(state, response_space, load_space, finger_print_operator):
    """Builds the operator linking direct load to total physical load (including SLE)."""
    sea_level_projection = response_space.subspace_projection(0)
    sea_level_to_load = sea_level_change_to_load_operator(
        state, sea_level_projection.codomain, load_space
    )
    induced_load_operator = (
        sea_level_to_load @ sea_level_projection @ finger_print_operator
    )
    return load_space.identity_operator() + induced_load_operator


def get_regional_averaging(
    state, load_space, /, *, regions_dict=None, smoothing_scale_km=None
):
    """Sets up the averaging operator using specialised geophysical basins."""

    if regions_dict is None:
        regions_dict = {
            "GRL (S Basin)": ["SE", "SW"],
            "WAIS": ["F-G", "Ep-F", "J-Jpp", "G-H", "H-Hp"],
            "Gulf of Mexico": "Gulf of Mexico",
            "GBM basin": "4030025450",
        }

    region_names = list(regions_dict.keys())

    weighting_functions = [
        state.get_projection(raw_names, value=0.0)
        for raw_names in regions_dict.values()
    ]

    if smoothing_scale_km is not None and smoothing_scale_km > 0:
        smoothing_scale = (
            smoothing_scale_km * 1000 / state.model.parameters.length_scale
        )
        smoothing_measure = load_space.heat_kernel_gaussian_measure(smoothing_scale)
        smoothing_operator = smoothing_measure.covariance
        weighting_functions = [smoothing_operator(wf) for wf in weighting_functions]

    avg_operator = averaging_operator(state, load_space, weighting_functions)

    return (
        region_names,
        avg_operator,
        weighting_functions,
        regions_dict,
    )


def draw_region_boundaries(state, ax, regions_dict, **kwargs):
    """
    Helper to plot all boundaries defined in a regions dictionary.
    Handles nested lists automatically for composite regions.
    """
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("linewidth", 2.0)

    raw_regions = []
    for val in regions_dict.values():
        if isinstance(val, str):
            raw_regions.append(val)
        else:
            raw_regions.extend(val)

    state.plot_boundaries(ax, raw_regions, **kwargs)


# ---------------------------------------------------------------------------
# Comparison-summary figures.
# Styled via rc_context so the calling script's global rcParams do not
# leak in; sized for single-column use. Style block duplicated in
# joint_utils to keep the two pipelines independent.
# ---------------------------------------------------------------------------

_SUMMARY_RC = {
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "font.family": "sans-serif",
}
_C_BAYES = "#0072B2"  # Okabe-Ito blue
_C_WMB = "#D55E00"  # Okabe-Ito vermillion


def plot_kernel_error_ratios(
    region_rows,
    /,
    *,
    degree_rows=None,
    degree_label=None,
):
    """
    Dumbbell chart of relative estimator-kernel errors ||k - t|| / ||t||
    in the load-space norm: Bayesian (filled) vs WMB (open) per region,
    with an optional Bayesian-only cluster for single-degree
    coefficients below a separator. Smaller is better. region_rows is a
    sequence of (name, bayes_ratio, wmb_ratio); degree_rows of
    (name, bayes_ratio). Returns the figure.
    """
    region_rows = list(region_rows)
    degree_rows = list(degree_rows) if degree_rows else []
    n_deg = len(degree_rows)
    n_total = len(region_rows) + n_deg
    ys_reg = np.arange(len(region_rows))[::-1] + (n_deg + 1.0 if n_deg else 0.0)
    ys_deg = np.arange(n_deg)[::-1]

    ratios = [b for _, b, _ in region_rows]
    ratios += [w for _, _, w in region_rows]
    ratios += [b for _, b in degree_rows]
    xmax = max(0.5, 1.12 * max(ratios))

    with mpl.rc_context(_SUMMARY_RC):
        fig, ax = plt.subplots(
            figsize=(4.8, 1.0 + 0.30 * (n_total + (1 if n_deg else 0))),
            layout="constrained",
        )
        for y, (_, bayes, wmb) in zip(ys_reg, region_rows):
            ax.plot([bayes, wmb], [y, y], color="0.6", lw=1.1, zorder=1)
            ax.plot(bayes, y, "o", ms=5, color=_C_BAYES, zorder=2)
            ax.plot(
                wmb,
                y,
                "o",
                ms=5,
                mec=_C_WMB,
                mfc="white",
                mew=1.4,
                zorder=2,
            )
        for y, (_, bayes) in zip(ys_deg, degree_rows):
            ax.plot(bayes, y, "o", ms=5, color=_C_BAYES, zorder=2)
        if n_deg:
            ax.axhline(n_deg, color="0.8", lw=0.7, ls=":")
            if degree_label:
                ax.text(
                    0.98 * xmax,
                    ys_deg[0] + 0.55,
                    degree_label,
                    fontsize=7.5,
                    color="0.4",
                    ha="right",
                )
        ax.set_yticks(np.concatenate([ys_reg, ys_deg]))
        ax.set_yticklabels([r[0] for r in region_rows] + [d[0] for d in degree_rows])
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(-0.6, ys_reg[0] + 1.05)
        ax.set_xlabel("Relative estimator-kernel error  $\\|k - t\\|\\, /\\, \\|t\\|$")
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.annotate(
            "$\\leftarrow$ better",
            xy=(0.02 * xmax, ys_reg[0] + 0.62),
            fontsize=7.5,
            color="0.4",
        )
        handles = [
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=5,
                color=_C_BAYES,
                label="Bayesian",
            ),
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=5,
                mec=_C_WMB,
                mfc="white",
                mew=1.4,
                label="WMB",
            ),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False)
    return fig
