"""
Shared physics, priors, noise models, inversion helpers and tables for the
experiments of Heathcote et al. (2026). FORMULATION.md sets out the
formulation they implement; the terms are those of Gregory et al. (2019).

Two families of experiment share this module:

- **GRACE** (grace_bias, grace_inversion): the model is the direct surface
  mass load, a global field; GRACE observes the potential change of the
  sea level equation's response to it.
- **Ocean** (altimetry_bias, altimetry_inversion, joint_inversion): the model
  is the ice thickness change, the sterodynamic sea level change eta and the
  vertically averaged ocean density change, the set-up
  `pyslfp.models.ice_ocean_and_density`. The fields are global, the loads
  carry their supports, and the prior is conditioned exactly on the
  conservation of the ocean's mass. The spaces follow the regularity of the
  physics: the fields in H^2, the loads in H^(1/4), the response in H^(5/4).

Every random field (prior or noise) is built on the space of order
`field_order` at the load scale, so that its statistics do not depend on the
spaces of the loads and the response.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

import pygeoinf as inf
from pyslfp import EarthState, set_num_threads
from pyslfp.linear_operators import (
    FingerPrintOperator,
    WMBMethod,
    geocentric_sea_level_operator,
    grace_observation_operator,
    ocean_altimetry_points,
)
from pyslfp.linear_operators.loads import sea_level_change_to_load_operator
from pyslfp.linear_operators.physics import sobolev_load_space
from pyslfp.linear_operators.spatial import ocean_average_operator
from pyslfp.linear_operators.utils import averaging_operator
from pyslfp.models import (
    GEOCENTRIC_SEA_LEVEL,
    OCEAN_MASS,
    ModelSetup,
    barystatic_gmsl_operator,
    ice_ocean_and_density,
)

ICE, ETA, RHO = "ice_sheets", "sterodynamic_sea_level", "ocean_density"

# The regions of the GRACE experiments, by `EarthState.get_projection` names.
GRACE_REGIONS = {
    "GRL (S Basin)": ["SE", "SW"],
    "WAIS": ["F-G", "Ep-F", "J-Jpp", "G-H", "H-Hp"],
    "Gulf of Mexico": "Gulf of Mexico",
    "GBM basin": "4030025450",
}

# The region of the ocean experiments' regional decomposition.
OCEAN_REGION = "Tasman Sea"

# The covariance maps of the ocean inversions: the name of the perturbed field
# in the file name, its component, the point (latitude, longitude) and the
# point's name in the file name.
OCEAN_COVARIANCE_POINTS = [
    ("Ice", ICE, (-78.0, -110.0), "WAIS"),
    ("Sterodynamic", ETA, (30.0, -45.0), "North_Atlantic"),
]


# ================================================================ #
#                         Command line                             #
# ================================================================ #


def add_parallel_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parallel", action="store_true", help="Use worker processes.")
    parser.add_argument(
        "--max-jobs", type=int, default=os.cpu_count() or 1, help="Worker processes."
    )
    parser.add_argument(
        "--serial-threads",
        type=int,
        default=None,
        help="Threads for the transforms when running serially.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for every random draw; unseeded if omitted.",
    )


def add_qoi_arguments(parser: argparse.ArgumentParser) -> None:
    """The tolerance of the push-forwards of the posterior; see
    `solve_posterior`."""
    parser.add_argument(
        "--qoi-rtol",
        type=float,
        default=1.0e-6,
        help="Relative residual of the conjugate-gradient solves that push the "
        "posterior covariance onto the quantities of interest, the estimator "
        "kernels and the covariance maps.",
    )


def add_grace_arguments(parser: argparse.ArgumentParser) -> None:
    """The resolution, prior and noise of the GRACE experiments."""
    parser.add_argument("--lmax", type=int, default=256)
    parser.add_argument("--obs-degree", type=int, default=100)
    parser.add_argument("--load-order", type=float, default=2.0)
    parser.add_argument("--load-scale-km", type=float, default=500.0)
    parser.add_argument(
        "--smoothing-scale-km",
        type=float,
        default=None,
        help="Smoothing of the region masks; --load-scale-km if omitted.",
    )
    parser.add_argument("--direct-scale-km", type=float, default=250.0)
    parser.add_argument("--direct-std-m", type=float, default=0.01)
    parser.add_argument("--prior-order", type=float, default=3.0)
    parser.add_argument("--prior-shift", type=float, default=0.0)
    parser.add_argument(
        "--remove-degree-1",
        action="store_true",
        help="Condition the prior on a vanishing degree 1.",
    )
    parser.add_argument("--noise-scale-factor", type=float, default=0.5)
    parser.add_argument("--snr-low-degree", type=float, default=10.0)
    parser.add_argument("--snr-reference-degree", type=float, default=2.0)
    parser.add_argument("--snr-crossover-degree", type=float, default=64.0)


def add_ocean_arguments(
    parser: argparse.ArgumentParser, *, surrogate: bool = False
) -> None:
    """The resolution and prior of the ocean experiments."""
    parser.add_argument("--lmax", type=int, default=256)
    if surrogate:
        parser.add_argument("--surrogate-degree", type=int, default=32)
    parser.add_argument(
        "--field-order",
        type=float,
        default=2.0,
        help="Sobolev order of the model fields (and of every random field).",
    )
    parser.add_argument(
        "--load-order",
        type=float,
        default=0.25,
        help="Sobolev order of the loads, below 1/2; the response is one order above.",
    )
    parser.add_argument("--load-scale-km", type=float, default=500.0)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--ice-scale-factor", type=float, default=1.0)
    parser.add_argument("--gmsl-bary-steric-ratio", type=float, default=1.7)
    parser.add_argument("--ocean-dyn-scale-factor", type=float, default=0.2)
    parser.add_argument("--ocean-dyn-std-mm", type=float, default=4.0)
    parser.add_argument("--ocean-rho-scale-factor", type=float, default=1.0)
    parser.add_argument("--steric-dyn-std-ratio", type=float, default=0.75)
    parser.add_argument("--ocean-corr", type=float, default=0.9)
    parser.add_argument("--ocean-corr-scale-factor", type=float, default=0.4)
    parser.add_argument("--prior-order", type=float, default=3.0)
    parser.add_argument("--prior-shift", type=float, default=0.0)


def add_altimetry_noise_arguments(
    parser: argparse.ArgumentParser, *, prefix: str = ""
) -> None:
    parser.add_argument(f"--{prefix}noise-std-factor", type=float, default=0.5)
    parser.add_argument(f"--{prefix}noise-corr-std-factor", type=float, default=0.01)
    parser.add_argument(f"--{prefix}noise-corr-scale-factor", type=float, default=4.0)


def add_joint_grace_noise_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--obs-degree", type=int, default=100)
    parser.add_argument("--grace-noise-scale-km", type=float, default=250.0)
    parser.add_argument("--grace-snr-low-degree", type=float, default=10.0)
    parser.add_argument("--grace-snr-reference-degree", type=float, default=2.0)
    parser.add_argument("--grace-snr-crossover-degree", type=float, default=64.0)


def configure(args: argparse.Namespace) -> np.random.Generator:
    """Sets the thread count and returns the random generator of the run."""
    if not args.parallel and args.serial_threads is not None:
        set_num_threads(args.serial_threads)
    return np.random.default_rng(args.seed)


def jobs(args: argparse.Namespace, tasks: Optional[int] = None) -> Optional[int]:
    """Worker processes for a call site with `tasks` independent tasks (no
    limit if omitted): up to --max-jobs with --parallel, otherwise None,
    which runs it serially."""
    if not args.parallel:
        return None
    return args.max_jobs if tasks is None else min(tasks, args.max_jobs)


# ================================================================ #
#                            GRACE                                 #
# ================================================================ #


@dataclass
class GracePhysics:
    """The GRACE experiments' physics: the direct load on `load_space`, the
    fingerprint operator to a Lebesgue response, and `scale_mm`, which takes
    a load to millimetres of equivalent water thickness."""

    state: EarthState
    load_space: object
    fingerprint: FingerPrintOperator
    scale_mm: float

    @property
    def response_space(self):
        return self.fingerprint.codomain


def grace_physics(lmax: int, load_order: float, load_scale_km: float) -> GracePhysics:
    state = EarthState.from_defaults(lmax=lmax)
    p = state.model.parameters
    fingerprint = FingerPrintOperator(
        state, load_parameters=(load_order, load_scale_km * 1000.0 / p.length_scale)
    )
    return GracePhysics(
        state,
        fingerprint.domain,
        fingerprint,
        1000.0 * p.length_scale / p.water_density,
    )


def grace_forward_operator(physics: GracePhysics, obs_degree: int):
    """GRACE's coefficients of the potential change of the response to the
    direct load. The fingerprint operator's response already includes the
    ocean load the direct load induces."""
    return (
        grace_observation_operator(physics.response_space, obs_degree)
        @ physics.fingerprint
    )


def total_load_operator(physics: GracePhysics):
    """The direct load plus the ocean load it induces, rho_w C xi."""
    sea_level = physics.response_space.field_operator("relative_sea_level_change")
    to_load = sea_level_change_to_load_operator(
        physics.state, sea_level.codomain, physics.load_space
    )
    return inf.LinearOperator.identity(physics.load_space) + (
        to_load @ sea_level @ physics.fingerprint
    )


def regional_averaging(physics: GracePhysics, args: argparse.Namespace):
    """
    The averaging operator over the GRACE regions (`GRACE_REGIONS`), with
    weights the region masks smoothed by the heat kernel at
    --smoothing-scale-km, or at --load-scale-km if that is omitted; a scale
    of zero leaves the masks sharp. The smoothing keeps the weights within
    the load space's resolution, as GRACE's own is.

    Returns (names, operator).
    """
    state, load_space = physics.state, physics.load_space
    weights = [
        state.get_projection(names, value=0.0) for names in GRACE_REGIONS.values()
    ]
    scale_km = (
        args.load_scale_km
        if args.smoothing_scale_km is None
        else args.smoothing_scale_km
    )
    if scale_km > 0.0:
        scale = scale_km * 1000.0 / state.model.parameters.length_scale
        smoother = load_space.heat_measure(scale).covariance
        weights = [smoother(w) for w in weights]
    return list(GRACE_REGIONS), averaging_operator(state, load_space, weights)


def _coefficient_profile(order, scale, space_order, space_scale, degrees, *, radius):
    lam = degrees * (degrees + 1.0) / radius**2
    return (1.0 + scale**2 * lam) ** (-order) * (1.0 + space_scale**2 * lam) ** (
        -space_order
    )


def resolve_noise_amplitude(
    load_space,
    prior_order,
    signal_scale,
    signal_std,
    noise_scale,
    snr_crossover_degree,
    snr_low_degree,
    /,
    *,
    snr_reference_degree=2.0,
    obs_degree=None,
    std_to_mm=None,
    label="spatial noise",
):
    """
    The spectrum of a spatial noise field against a signal field, both
    invariant Sobolev measures on `load_space`: the noise order and amplitude
    solved in closed form so that the amplitude signal-to-noise ratio is
    `snr_low_degree` at `snr_reference_degree` and one at
    `snr_crossover_degree`. The noise must be a legitimate random field
    (order above 1), which caps the achievable low-degree ratio. `signal_std`
    is in the units of the noise field.
    """
    space_order = load_space.order
    space_scale = load_space.length_scale
    lmax = load_space.lmax
    radius = load_space.radius
    q_s = prior_order
    degrees = np.arange(lmax + 1, dtype=float)
    weights = 2.0 * degrees + 1.0

    def log_kernel(scale, ell):
        return float(np.log1p(scale**2 * ell * (ell + 1.0) / radius**2))

    prof_s = _coefficient_profile(
        q_s, signal_scale, space_order, space_scale, degrees, radius=radius
    )
    z_s = float(np.sum(weights * prof_s))
    amplitude_signal = 4.0 * np.pi * signal_std**2 / z_s

    band_degree = int(min(obs_degree, lmax)) if obs_degree is not None else lmax
    ref_degree = float(snr_reference_degree)
    lstar = float(snr_crossover_degree)
    if ref_degree < 1.0:
        raise ValueError("snr_reference_degree must be >= 1.")
    if not ref_degree < lstar:
        raise ValueError("snr_reference_degree must lie below snr_crossover_degree.")
    if not snr_low_degree > 1.0:
        raise ValueError("snr_low_degree must exceed 1.")

    dl_s = log_kernel(signal_scale, lstar) - log_kernel(signal_scale, ref_degree)
    dl_n = log_kernel(noise_scale, lstar) - log_kernel(noise_scale, ref_degree)
    q_n = float((q_s * dl_s - 2.0 * np.log(snr_low_degree)) / dl_n)
    if q_n <= 1.0 + 1.0e-12:
        cap = np.exp((q_s * dl_s - dl_n) / 2.0)
        raise ValueError(
            f"The solved noise order q_n = {q_n:.3f} gives an illegitimate field "
            f"(needs q_n > 1); the low-degree amplitude SNR must be below {cap:.2f}."
        )
    lin_0 = q_s * signal_scale**2 - q_n * noise_scale**2
    lin_1 = (q_s - q_n) * signal_scale**2 * noise_scale**2
    monotone = lin_0 >= 0.0 and lin_1 >= 0.0 and (lin_0 + lin_1) > 0.0
    if not monotone:
        raise ValueError(
            "The SNR crossover needs a decreasing signal-to-noise ratio: "
            "q_n <= q_s and q_n s_n^2 <= q_s s_s^2."
        )
    amplitude = float(
        np.exp(
            np.log(amplitude_signal)
            - q_s * log_kernel(signal_scale, lstar)
            + q_n * log_kernel(noise_scale, lstar)
        )
    )
    prof_n = _coefficient_profile(
        q_n, noise_scale, space_order, space_scale, degrees, radius=radius
    )
    z_n = float(np.sum(weights * prof_n))
    noise_std = float(np.sqrt(amplitude * z_n / (4.0 * np.pi)))
    kern_s = _coefficient_profile(
        q_s, signal_scale, 0.0, space_scale, degrees, radius=radius
    )
    kern_n = _coefficient_profile(
        q_n, noise_scale, 0.0, space_scale, degrees, radius=radius
    )
    nsr = (amplitude * kern_n) / (amplitude_signal * kern_s)
    band_fraction = float(
        np.sum(weights[: band_degree + 1] * prof_n[: band_degree + 1]) / z_n
    )
    info = {
        "kernel_amplitude": amplitude,
        "noise_std": noise_std,
        "std_factor": float(noise_std / signal_std),
        "noise_order": q_n,
        "crossover_degree": lstar,
        "snr_low_degree": float(1.0 / np.sqrt(nsr[int(round(ref_degree))])),
        "snr_reference_degree": ref_degree,
        "band_degree": band_degree,
        "band_variance_fraction": band_fraction,
        "band_std": float(noise_std * np.sqrt(band_fraction)),
        "nsr_band_edge": float(nsr[band_degree]),
        "monotone": monotone,
    }
    in_mm = (
        (lambda x: f"{x * std_to_mm:.3f} mm") if std_to_mm else (lambda x: f"{x:.4e}")
    )
    print(f"{label}: Sobolev, scale ratio s_n/s_s = {noise_scale / signal_scale:.2f}")
    print(
        f"  solved from amplitude SNR = {info['snr_low_degree']:.2f} at degree "
        f"{ref_degree:g} and crossover at degree {lstar:g}:"
    )
    print(f"  noise order q_n = {q_n:+.3f} (q_n > 1: legitimate field)")
    print(
        f"  pointwise std = {in_mm(noise_std)} ({info['std_factor']:.4f} x signal std); "
        f"{100 * band_fraction:.1f}% of the noise variance lies in the band "
        f"l <= {band_degree} at lmax = {lmax} (band std = {in_mm(info['band_std'])})"
    )
    print(
        f"  noise/signal amplitude = {np.sqrt(info['nsr_band_edge']):.1f} at l = {band_degree}"
    )
    return info


def spatial_noise_measure(space, info: dict, noise_scale: float):
    """The invariant noise field the solved spectrum describes."""
    b, q = info["kernel_amplitude"], info["noise_order"]
    return space.invariant_measure(lambda k: b * (1.0 + noise_scale**2 * k) ** (-q))


def grace_measures(physics: GracePhysics, args: argparse.Namespace, rng):
    """The direct-load prior and the GRACE noise of the GRACE experiments.

    Returns (initial_prior, prior, noise_load_measure, noise_info): the
    unconditioned prior, the prior used (conditioned on a vanishing degree 1
    with --remove-degree-1, then shifted), the spatial noise field and its
    solved spectrum."""
    if args.prior_order <= 1.0:
        raise ValueError("--prior-order must exceed one.")
    p = physics.state.model.parameters
    space = physics.load_space
    scale = args.direct_scale_km * 1000.0 / p.length_scale
    std = p.water_density * args.direct_std_m / p.length_scale
    initial = space.sobolev_measure(args.prior_order, scale, pointwise_std=std)
    prior = initial
    if args.remove_degree_1:
        constraint = space.coefficient_operator(lmax=1, lmin=1).with_traits(
            inf.Traits.SURJECTIVE
        )
        prior = prior.condition(constraint, np.zeros(constraint.codomain.dim))
    prior = shifted(prior, args.prior_shift, rng)
    noise_scale = args.noise_scale_factor * scale
    info = resolve_noise_amplitude(
        space,
        args.prior_order,
        scale,
        std,
        noise_scale,
        args.snr_crossover_degree,
        args.snr_low_degree,
        snr_reference_degree=args.snr_reference_degree,
        obs_degree=args.obs_degree,
        std_to_mm=physics.scale_mm,
        label="GRACE spatial noise",
    )
    return initial, prior, spatial_noise_measure(space, info, noise_scale), info


def shifted(measure, shift: float, rng):
    """The measure translated by `shift` times one of its own samples."""
    if shift == 0.0:
        return measure
    return measure.translate(measure.domain.scale(shift, measure.sample(rng=rng)))


# ================================================================ #
#                       Ocean experiments                          #
# ================================================================ #


@dataclass
class OceanPhysics:
    """The ocean experiments' physics.

    `setup` is the ice + eta + delta-rho set-up on the H^`field_order`
    fields; `fingerprint` maps its loads (in H^`load_order`) to the response
    (one order above); `geocentric_sea_level` is the continuous geocentric
    sea level change, in the response's field space, and `altimetry` its
    values at the track `points`. `field_point_evaluation` evaluates a field
    of the fields' space at the points, for noise fields. `scale_mm` takes a
    length to millimetres.
    """

    state: EarthState
    setup: ModelSetup
    fingerprint: FingerPrintOperator
    points: list
    geocentric_sea_level: inf.LinearOperator
    altimetry: inf.LinearOperator
    field_point_evaluation: inf.LinearOperator
    scale_mm: float

    @property
    def field_space(self):
        return self.setup.component(ICE).space

    @property
    def load_space(self):
        return self.setup.load_space

    @property
    def display_scales(self) -> dict:
        """The factors taking each field to its display unit: mm for the ice
        thickness change and eta, g m^-3 for delta rho."""
        density = 1.0e3 * self.state.model.parameters.density_scale
        return {ICE: self.scale_mm, ETA: self.scale_mm, RHO: density}

    def display(self, model) -> dict:
        """The fields of a model vector on their supports and in display
        units, by component name. The fields are global, but only their
        restrictions to the supports are physical, so only those are shown."""
        scales = self.display_scales
        return {
            name: field * scales[name]
            for name, field in zip(self.setup.names, self.setup.support(model))
        }

    def point_covariance(self, measure, component: str, point) -> dict:
        """The covariance under `measure` of each field with `component` at
        `point` (latitude, longitude): the covariance applied to the
        model-space representer of that point evaluation. By component name,
        on the supports and in the product of the two fields' display units."""
        dirac = self.field_space.dirac(point).representer
        test = self.setup.space.inclusion(component)(dirac)
        scale = self.display_scales[component]
        return {
            name: field * scale
            for name, field in self.display(measure.covariance(test)).items()
        }

    def regional_decomposition(self, region: str) -> inf.LinearOperator:
        """The `region` mean of the relative sea level change, split into its
        steric, dynamic manometric and barystatic manometric parts (SSLC,
        DMSLC, BMSLC), in mm."""
        parts = self.setup.regional_decomposition(self.fingerprint, region)
        combined = inf.ColumnLinearOperator(list(parts)) * self.scale_mm
        return combined.codomain.coordinate_projection() @ combined

    def grace(self, obs_degree: int) -> inf.LinearOperator:
        """GRACE's coefficients of the potential change, from the model."""
        return (
            grace_observation_operator(self.fingerprint.codomain, obs_degree)
            @ self.fingerprint
            @ self._loads
        )

    @property
    def _loads(self):
        return self.setup.load_operator


@dataclass
class SurrogateOceanPhysics(OceanPhysics):
    """The cheap surrogate of the preconditioners: the ice load alone, one
    pass of the sea level equation, at a lower degree."""

    ice_only_loads: inf.LinearOperator = None

    @property
    def _loads(self):
        return self.ice_only_loads


def ocean_physics(
    args: argparse.Namespace,
    /,
    *,
    points: Optional[Sequence[tuple[float, float]]] = None,
    surrogate: bool = False,
) -> OceanPhysics:
    """The physics of the ocean experiments at --lmax, or, for the surrogate
    of the preconditioners, at --surrogate-degree. The altimetry points are
    the regular grid at --spacing over the ice-free ocean within 66 degrees
    of the equator, unless given; the surrogate takes those of the exact
    physics, so that the two share the data space."""
    lmax = args.surrogate_degree if surrogate else args.lmax
    state = EarthState.from_defaults(lmax=lmax)
    if points is None:
        points = ocean_altimetry_points(state, spacing=args.spacing)
    p = state.model.parameters
    scale = args.load_scale_km * 1000.0 / p.length_scale
    fingerprint = FingerPrintOperator(
        state,
        load_parameters=(args.load_order, scale),
        response_parameters=(args.load_order + 1.0, scale),
        max_iterations=1 if surrogate else None,
    )
    field_space = sobolev_load_space(state.model, args.field_order, scale)
    setup = ice_ocean_and_density(
        state,
        fingerprint.domain,
        space=field_space,
        observable_space=fingerprint.field_space,
    )
    loads = setup.load_operator
    if surrogate:
        loads = loads @ setup.space.inclusion(ICE) @ setup.space.projection(ICE)
    geocentric = (
        geocentric_sea_level_operator(state, fingerprint.codomain) @ fingerprint @ loads
        + setup.observables[GEOCENTRIC_SEA_LEVEL]
    )
    point_evaluation = fingerprint.field_space.point_evaluation_operator(points)
    fields = dict(
        state=state,
        setup=setup,
        fingerprint=fingerprint,
        points=list(points),
        geocentric_sea_level=geocentric,
        altimetry=point_evaluation @ geocentric,
        field_point_evaluation=field_space.point_evaluation_operator(points),
        scale_mm=1000.0 * p.length_scale,
    )
    if surrogate:
        return SurrogateOceanPhysics(**fields, ice_only_loads=loads)
    return OceanPhysics(**fields)


def _std(measure, functional) -> float:
    """The standard deviation of a scalar functional under a measure."""
    return float(np.sqrt(measure.directional_variance(functional.adjoint(np.ones(1)))))


def _field_measure(physics: OceanPhysics, args: argparse.Namespace, name, std):
    """The prior of one field alone: a Sobolev field of order --prior-order
    on the fields' space with pointwise std `std`."""
    length = physics.field_space.length_scale
    factors = {
        ICE: args.ice_scale_factor,
        ETA: args.ocean_dyn_scale_factor,
        RHO: args.ocean_rho_scale_factor,
    }
    return physics.field_space.sobolev_measure(
        args.prior_order, length * factors[name], pointwise_std=std
    )


def product_prior(physics: OceanPhysics, args: argparse.Namespace, stds: dict):
    """The uncorrelated, unconditioned product of the fields' priors at the
    pointwise stds `stds` (keyed as calib["derived_stds"]): the prior of the
    preconditioners' surrogate."""
    by_name = {ICE: stds["ice_std"], ETA: stds["dyn_std"], RHO: stds["rho_std"]}
    return inf.GaussianMeasure.from_product(
        [
            _field_measure(physics, args, name, by_name[name])
            for name in physics.setup.names
        ],
        domain=physics.setup.space,
    )


def ocean_prior(physics: OceanPhysics, args: argparse.Namespace, rng):
    """
    The prior of the ocean experiments: Sobolev fields of order
    `--prior-order` on the fields' space, eta and delta rho anti-correlated
    at long wavelengths, conditioned exactly on the conservation of the
    ocean's mass, then shifted. The fields are global and the prior is not
    masked.

    The amplitudes form a triangular chain with one dimensioned input,
    `--ocean-dyn-std-mm`, the pointwise std of eta: the density std follows
    from `--steric-dyn-std-ratio` and the mean ocean depth, and the ice std
    from `--gmsl-bary-steric-ratio`, the ratio of the barystatic to the
    realised steric GMSL std under the mass constraint. The ice is
    uncorrelated with the ocean fields and absent from the constraint, so the
    chain closes without iteration.

    Returns (prior, calib): the prior and the calibration record.
    """
    if not 0.0 <= args.ocean_corr < 1.0:
        raise ValueError("--ocean-corr must lie in [0, 1).")
    if args.prior_order <= 1.0:
        raise ValueError("--prior-order must exceed one.")
    if (
        min(
            args.gmsl_bary_steric_ratio,
            args.ocean_dyn_std_mm,
            args.steric_dyn_std_ratio,
        )
        <= 0.0
    ):
        raise ValueError(
            "The prior's amplitude ratios and the eta std must be positive."
        )
    setup = physics.setup
    field_space = physics.field_space
    state = physics.state
    p = state.model.parameters
    corr_scale = field_space.length_scale * args.ocean_corr_scale_factor

    def correlated(stds):
        if args.ocean_corr == 0.0:
            return inf.GaussianMeasure.from_product(
                [_field_measure(physics, args, n, stds[n]) for n in setup.names],
                domain=setup.space,
            )

        def correlation(k):
            r = -args.ocean_corr * np.exp(-(corr_scale**2) * k)
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, r], [0.0, r, 1.0]])

        variances = [
            _field_measure(physics, args, n, stds[n]).covariance.eigenvalues
            for n in setup.names
        ]
        matrices = np.array([correlation(k) for k in field_space.laplacian_eigenvalues])
        return field_space.correlated_measure_from_correlations(
            variances, matrices, labels=list(setup.names)
        )

    def conditioned(measure_):
        constraint = setup.constraints[OCEAN_MASS]
        return measure_.condition(constraint, np.zeros(constraint.codomain.dim))

    dyn_std = args.ocean_dyn_std_mm / physics.scale_mm
    effective_steric_scale = state.mean_ocean_depth / p.water_density
    rho_std = args.steric_dyn_std_ratio * dyn_std / effective_steric_scale

    # The barystatic GMSL per unit ice std, its support built into the load.
    q_ice = _std(
        _field_measure(physics, args, ICE, 1.0),
        barystatic_gmsl_operator(state, setup.component(ICE)),
    )

    # The realised steric GMSL std under the constraint. The ice std is a
    # placeholder, the ice being uncorrelated and absent from the constraint.
    trial = correlated({ICE: 1.0, ETA: dyn_std, RHO: rho_std})
    sigma_s = _std(conditioned(trial), setup.steric_gmsl)
    eta_mean = ocean_average_operator(state, field_space) @ setup.space.projection(ETA)
    a = _std(trial, eta_mean)
    b = _std(trial, setup.steric_gmsl)
    d_x = eta_mean.adjoint(np.ones(1))
    d_y = setup.steric_gmsl.adjoint(np.ones(1))
    rho_xy = float(trial.directional_covariance(d_x, d_y)) / (a * b)
    ice_std = args.gmsl_bary_steric_ratio * sigma_s / q_ice

    prior = conditioned(correlated({ICE: ice_std, ETA: dyn_std, RHO: rho_std}))
    calib = {
        "constants": {
            "q_ice": q_ice,
            "u_eta": a / dyn_std,
            "u_rho": b / rho_std,
            "rho_xy": rho_xy,
            "effective_steric_scale": effective_steric_scale,
        },
        "gmsl": {
            "barystatic_std": args.gmsl_bary_steric_ratio * sigma_s,
            "steric_std": sigma_s,
            "eta_mean_std_unconstrained": a,
            "bary_steric_ratio": args.gmsl_bary_steric_ratio,
            "steric_std_check": _std(prior, setup.steric_gmsl),
            "barystatic_std_check": _std(prior, setup.barystatic_gmsl()),
        },
        "derived_stds": {"ice_std": ice_std, "dyn_std": dyn_std, "rho_std": rho_std},
    }
    return shifted(prior, args.prior_shift, rng), calib


def ocean_surrogate(args: argparse.Namespace, physics: OceanPhysics, calib: dict):
    """The surrogate of the preconditioners at --surrogate-degree, on the
    exact physics' points, and its prior (`product_prior` at the calibrated
    amplitudes). Returns (surrogate, prior)."""
    print(
        f"\nBuilding the surrogate (lmax={args.surrogate_degree}) for preconditioning..."
    )
    surrogate = ocean_physics(args, points=physics.points, surrogate=True)
    return surrogate, product_prior(surrogate, args, calib["derived_stds"])


def print_calibration_report(calib: dict, physics: OceanPhysics) -> None:
    """Prints the derived amplitudes and the realised GMSL statistics."""
    d, c, g = calib["derived_stds"], calib["constants"], calib["gmsl"]
    mm = physics.scale_mm
    density_scale = physics.state.model.parameters.density_scale
    print(
        f"Prior amplitudes: sterodynamic std = {d['dyn_std'] * mm:.3f} mm (input), "
        f"density std = {d['rho_std'] * density_scale:.4e} kg m^-3 (mean-depth steric "
        f"std = {d['rho_std'] * c['effective_steric_scale'] * mm:.3f} mm), "
        f"ice std = {d['ice_std'] * mm:.3f} mm"
    )
    print(
        "  realised GMSL prior stds under the mass constraint: barystatic = "
        f"{g['barystatic_std'] * mm:.3f} mm (verified "
        f"{g['barystatic_std_check'] * mm:.3f}), steric = "
        f"{g['steric_std'] * mm:.3f} mm (verified "
        f"{g['steric_std_check'] * mm:.3f}); barystatic/steric ratio = "
        f"{g['bary_steric_ratio']:.2f}"
    )
    print(
        f"  <eta>_O -- <eta_s>_O prior correlation rho_XY = {c['rho_xy']:.3f}; "
        f"unconstrained <eta>_O std = {g['eta_mean_std_unconstrained'] * mm:.3f} mm"
    )
    floor = calib.get("noise", {}).get("alt_corr_gmsl_floor")
    if floor is not None:
        print(
            "  correlated altimetry error component: implied GMSL error floor = "
            f"{floor * mm:.3f} mm"
        )


def print_prior_settings(
    args: argparse.Namespace, noise_corr_std_factor: float, noise_corr_scale_factor
) -> None:
    """Prints the settings of the prior's correlation and of the correlated
    altimetry noise."""
    if args.ocean_corr > 0.0:
        print(
            f"Correlated (sterodynamic, density) prior: r0 = {args.ocean_corr:.2f}, "
            f"correlation scale factor = {args.ocean_corr_scale_factor:.2f}"
        )
    print(f"Sobolev priors: common order = {args.prior_order:.2f}")
    if noise_corr_std_factor > 0.0:
        print(
            f"Correlated altimetry error component: std factor = "
            f"{noise_corr_std_factor:.3f}, scale factor = {noise_corr_scale_factor:.1f}"
        )


def altimetry_noise(
    physics: OceanPhysics,
    dyn_std: float,
    prior_order: float,
    std_factor: float,
    corr_std_factor: float,
    corr_scale_factor: float,
    /,
    *,
    calib: Optional[dict] = None,
):
    """
    The altimetry noise: a local part, independent at the points with std
    `std_factor` x the eta std, plus, if `corr_std_factor` is positive, a
    large-scale correlated part, a Sobolev field of the prior's order on the
    fields' space at `corr_scale_factor` x the load scale with std
    `corr_std_factor` x the eta std, evaluated at the points. The correlated
    part barely averages down and so sets a floor on the error of GMSL-type
    averages, recorded in `calib["noise"]`.

    Returns (noise, local): the noise and its local part, which is all the
    preconditioners' surrogate sees.
    """
    data_space = inf.EuclideanSpace(len(physics.points))
    local = inf.GaussianMeasure.from_standard_deviation(
        data_space, std_factor * dyn_std
    )
    noise = local
    if corr_std_factor > 0.0:
        field_space = physics.field_space
        field = field_space.sobolev_measure(
            prior_order,
            field_space.length_scale * corr_scale_factor,
            pointwise_std=corr_std_factor * dyn_std,
        )
        if calib is not None:
            mean = ocean_average_operator(physics.state, field_space)
            calib.setdefault("noise", {})["alt_corr_gmsl_floor"] = _std(field, mean)
        noise = local + field.affine_map(physics.field_point_evaluation)
    return noise, local


def joint_grace_noise(
    physics: OceanPhysics,
    ice_std: float,
    prior_order: float,
    args: argparse.Namespace,
):
    """
    The GRACE noise of the joint experiment, solved against the ice load:
    the signal is the load of the ice field, rho_i times the ice std, a load
    as the noise field is. The noise field is built on the fields' space and
    taken to GRACE's coefficients by WMB's spectral map.

    Returns (data_noise, info).
    """
    state = physics.state
    p = state.model.parameters
    field_space = physics.field_space
    ice_scale = field_space.length_scale * args.ice_scale_factor
    noise_scale = args.grace_noise_scale_km * 1000.0 / p.length_scale
    info = resolve_noise_amplitude(
        field_space,
        prior_order,
        ice_scale,
        p.ice_density * ice_std,
        noise_scale,
        args.grace_snr_crossover_degree,
        args.grace_snr_low_degree,
        snr_reference_degree=args.grace_snr_reference_degree,
        obs_degree=args.obs_degree,
        std_to_mm=1000.0 * p.length_scale / p.water_density,
        label="GRACE spatial noise",
    )
    wmb = WMBMethod(state.model, args.obs_degree)
    noise = wmb.load_measure_to_observation_measure(
        spatial_noise_measure(field_space, info, noise_scale)
    )
    return noise, info


def gmsl_checks(physics: OceanPhysics, model) -> list[str]:
    """
    Two checks on a model, in mm. The sea level equation conserves mass, so
    the GMSL change less its barystatic part (the steric residual) equals the
    steric part taken directly, whatever the prior. The mass constraint makes
    the ocean mean of eta equal the steric part, so that the dynamic
    manometric part, eta less the steric sea level change, has no ocean mean.
    """
    setup = physics.setup
    mm = physics.scale_mm
    gmsl = setup.gmsl(physics.fingerprint)
    eta_mean = ocean_average_operator(
        physics.state, physics.field_space
    ) @ setup.space.projection(ETA)
    residual = (gmsl(model)[0] - setup.barystatic_gmsl()(model)[0]) * mm
    steric = setup.steric_gmsl(model)[0] * mm
    eta = eta_mean(model)[0] * mm
    return [
        f"SLE mass-balance check:  steric residual = {residual:.6f} mm | "
        f"steric direct = {steric:.6f} mm | diff = {residual - steric:.3e} mm",
        f"Constraint check:        <sterodynamic SL>_O = {eta:.6f} mm | "
        f"<DMSLC>_O = {eta - steric:.3e} mm (should be ~0)",
    ]


# ================================================================ #
#                      Inversion helpers                           #
# ================================================================ #


def surrogate_preconditioner(
    normal_operator,
    surrogate_forward,
    surrogate_prior,
    noise,
    /,
    *,
    max_jobs: Optional[int] = None,
    alpha: float = 0.1,
):
    """
    A preconditioner for the data-space normal operator A Q A* + R: the
    Woodbury inverse of a cheap surrogate problem,

        R^-1 - R^-1 A_s (Q_s^-1 + A_s* R^-1 A_s)^-1 A_s* R^-1,

    the inner operator factorised by LU, mixed with R^-1 as
    (1 - alpha) W + alpha R^-1.
    """
    woodbury = inf.WoodburyPreconditioner(
        surrogate_forward,
        surrogate_prior.covariance,
        noise.covariance,
        solver=inf.LUSolver(n_jobs=max_jobs),
        prior_inverse=surrogate_prior.precision,
        noise_inverse=noise.precision,
    )(normal_operator)
    return (1.0 - alpha) * woodbury + alpha * noise.precision


def solve_posterior(inversion, data, preconditioner, /, *, rtol, qoi_rtol):
    """
    The posterior of `inversion` for `data`, through conjugate-gradient
    solves of the data-space normal equations with `preconditioner`, at two
    tolerances.

    The posterior expectation, and the randomise-then-optimise samples drawn
    from `posterior`, use `rtol`. Their errors are then small beside the
    posterior's own spread, which is all a mean or a sample needs.

    A posterior variance is the prior variance less a correction computed
    through the solves, so its error relative to itself is about the
    tolerance divided by the ratio of the posterior variance to the prior.
    Where the data reduce a variance by 99 per cent, `rtol` leaves it wrong
    by 10-20 per cent. `qoi_posterior` therefore has the same expectation but
    applies its covariance through solves at `qoi_rtol`, with the same
    preconditioner; every push-forward of the posterior covariance (onto the
    quantities of interest, the estimator kernels and the covariance maps)
    goes through it, or through the Kalman operator of `qoi_inversion`.

    Returns (posterior, qoi_posterior, qoi_inversion).
    """
    callback = inf.ProgressCallback(report=print)
    solver = inf.CGSolver(callback=callback, rtol=rtol, preconditioner=preconditioner)
    posterior = inversion.with_solver(solver)(data)
    print(f"Solution reached in {callback.iterations} iterations.")
    qoi_inversion = inversion.with_solver(
        inf.CGSolver(
            callback=inf.ProgressCallback(report=print),
            rtol=qoi_rtol,
            preconditioner=preconditioner,
        )
    )
    qoi_posterior = inf.GaussianMeasure(
        posterior.domain,
        expectation=posterior.expectation,
        covariance=qoi_inversion.covariance,
    )
    return posterior, qoi_posterior, qoi_inversion


def dense(measure, operator=None, /, *, n_jobs: Optional[int] = None):
    """The push-forward of a measure (by `operator`, if given) with its
    covariance assembled, for low-dimensional quantities of interest."""
    pushed = measure if operator is None else measure.affine_map(operator)
    return pushed.with_dense_covariance(n_jobs=n_jobs)


def pointwise_std(measure, samples: int, /, *, rng, n_jobs=None) -> list:
    """The pointwise standard deviation of each field of a measure on the
    model space, estimated from `samples` draws about its expectation. The
    model space is a direct sum with no pointwise product of its own, so the
    fields are taken one by one."""
    space = measure.domain
    mean = measure.expectation
    variances = [sub.zero() for sub in space.subspaces]
    for sample in measure.samples(samples, rng=rng, n_jobs=n_jobs):
        for sub, x, m, variance in zip(space.subspaces, sample, mean, variances):
            deviation = sub.subtract(x, m)
            sub.axpy(1.0 / samples, sub.multiply(deviation, deviation), variance)
    return [sub.sqrt(v) for sub, v in zip(space.subspaces, variances)]


# ================================================================ #
#                       Tables and output                          #
# ================================================================ #

# The note under the title of every table of the regional decomposition.
REGIONAL_NOTE = (
    "(Components sum to the regional mean sea level change; DMSLC\n"
    " includes the GRD response to the ocean load, BMSLC that to\n"
    " the land-ice load.)"
)


def covariance_matrix(measure) -> np.ndarray:
    """The covariance of a measure with an assembled covariance."""
    return np.asarray(measure.covariance.matrix(form="components"))


def variances(measure) -> np.ndarray:
    return np.diag(covariance_matrix(measure))


def standard_deviations(measure) -> np.ndarray:
    return np.sqrt(variances(measure))


def correlation(covariance: np.ndarray) -> float:
    """The correlation of the two components of a 2 x 2 covariance."""
    return covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])


def reduction(prior_value: float, value: float) -> float:
    """The percentage reduction of a variance from its prior value; zero
    where the prior has none."""
    if prior_value <= 0.0:
        return 0.0
    return 100.0 * (1.0 - value / prior_value)


def variance_table(names, prior_variances, posterior_variances, width: int) -> str:
    """The prior and posterior variance of each component and the reduction."""
    rule = "-" * width
    lines = [
        rule,
        f"{'Component':<18} | {'Prior Var':<12} | {'Post Var':<12} | Reduction",
        rule,
    ]
    for name, prior_var, posterior_var in zip(
        names, prior_variances, posterior_variances
    ):
        lines.append(
            f"{name:<18} | {prior_var:<12.4f} | {posterior_var:<12.4f} | "
            f"{reduction(prior_var, posterior_var):>8.2f}%"
        )
    lines.append(rule)
    return "\n".join(lines) + "\n"


def covariance_summary(kl, prior_cov, posterior_cov, names, width: int = 65) -> str:
    """The KL divergence of a low-dimensional posterior from its prior, their
    total (trace) and generalised (determinant) variances, and the
    variance table of the components."""
    prior_trace, posterior_trace = np.trace(prior_cov), np.trace(posterior_cov)
    return (
        f"Joint KL Divergence:      {kl:.4f} nats\n"
        f"Prior Total Var (Trace):  {prior_trace:.4f} mm²\n"
        f"Post Total Var (Trace):   {posterior_trace:.4f} mm²\n"
        f"Overall Trace Reduction:  {reduction(prior_trace, posterior_trace):.2f}%\n"
        f"Prior Generalised Var:    {np.linalg.det(prior_cov):.4e}\n"
        f"Post Generalised Var:     {np.linalg.det(posterior_cov):.4e}\n"
        + variance_table(names, np.diag(prior_cov), np.diag(posterior_cov), width)
    )


def comparison_header(columns) -> str:
    """The header of a table comparing several posteriors with the prior:
    for each of `columns` (a column prefix) a variance and its reduction."""
    cells = [f"{'Component':<22}", f"{'Prior Var':<12}"]
    for column in columns:
        cells += [f"{column + ' Var':<12}", f"{column + ' Red%':<10}"]
    return " | ".join(cells)


def comparison_row(name: str, prior_value: float, values, fmt: str = ".4f") -> str:
    """A row of that table: the prior's value, then each posterior's with
    its reduction."""
    cells = [f"{name:<22}", f"{prior_value:<12{fmt}}"]
    for value in values:
        cells += [f"{value:<12{fmt}}", f"{reduction(prior_value, value):>9.2f}%"]
    return " | ".join(cells)


def start_run(
    output_dir: str,
    args: argparse.Namespace,
    /,
    *,
    metrics_file: Optional[str] = None,
    heading: str = "",
) -> None:
    """
    Starts a run's record. The output folder is emptied, so that it holds
    the outputs of one run and nothing older, and `settings.txt` there
    records what produced them: the script, the time, the library's version
    and commit, and every argument. A metrics file, if named, opens with the
    same record and then `heading`.
    """
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    record = run_record(args)
    with open(os.path.join(output_dir, "settings.txt"), "w", encoding="utf-8") as file:
        file.write(record)
    if metrics_file is not None:
        write_metrics(metrics_file, record + "\n" + heading, append=False)


def run_record(args: argparse.Namespace) -> str:
    """The script, time, library version and commit, and arguments of the run."""
    script = os.path.basename(sys.argv[0])
    when = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        version = importlib.metadata.version("pyslfp")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commit += " with local changes" if dirty else ""
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    settings = "\n".join(
        f"  --{name.replace('_', '-')} {value}"
        for name, value in sorted(vars(args).items())
    )
    return (
        f"Run: {script}, {when}, pyslfp {version} (commit {commit})\n"
        f"Settings:\n{settings}\n"
    )


def write_metrics(path: str, text: str, /, *, append: bool = True) -> None:
    """Writes `text` to the metrics file at `path`, after what it holds
    unless `append` is false; the directory is made if need be."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as file:
        file.write(text)


def save_figures(figures: dict, output_dir: str, /, *, dpi: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, figure in figures.items():
        path = os.path.join(output_dir, name)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"saved {path}")
