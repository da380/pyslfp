"""
grace_utils.py
==============
Shared utilities, physics initializations, and plotting for Bayesian GRACE inversions.

The direct load prior and the spatial noise measure can optionally use
Sobolev (Matern-type) covariances in place of the default heat kernels,
giving rougher, power-law-tailed fields while keeping the correlation scale
and pointwise std settings (see build_measures).
"""

import pygeoinf as inf


from pyslfp import EarthState
from pyslfp.linear_operators import (
    FingerPrintOperator,
    sea_level_change_to_load_operator,
    averaging_operator,
)


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
    either family. The spatial noise measure uses the SAME covariance
    family (with its own scale and std factors), keeping the spectral
    signal-to-noise crossover well defined: within a common family the
    noise-to-signal spectral ratio is monotone in degree, whereas mixed
    families (a power-law signal against an exponentially decaying noise)
    would make it non-monotone. Both measures remain invariant with
    analytic spectral structure under either family, so the WMB
    preconditioner construction is unchanged.
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
    noise_load_measure_std = noise_std_factor * direct_load_measure_std
    noise_load_measure = load_measure(noise_load_measure_scale, noise_load_measure_std)

    return (
        initial_direct_load_prior,
        direct_load_prior,
        noise_load_measure,
        noise_load_measure_scale,
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
