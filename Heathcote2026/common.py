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
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import splu

import pygeoinf as inf
import gia_ensemble
import grace_covariance
from pyslfp import EarthState, set_num_threads
from pyslfp.elastic import FingerPrintOperator
from pyslfp.observations import (
    WMBMethod,
    grace_observation_operator,
    ocean_altimetry_points,
    stokes_coefficient_operator,
)
from pyslfp.core import geocentric_sea_level_operator
from pyslfp.models import sea_level_change_to_load_operator
from pyslfp.core import sobolev_load_space
from pyslfp.core.spaces import underlying_space
from pyslfp.core import ocean_average_operator
from pyslfp.core import averaging_operator
from pyslfp.models import (
    GEOCENTRIC_SEA_LEVEL,
    OCEAN_MASS,
    ModelSetup,
    barystatic_gmsl_operator,
    ice_sheets,
    ocean_density,
    sterodynamic_sea_level,
)
from pyslfp.models import LoadComponent, Role

ICE, ETA, RHO = "ice_sheets", "sterodynamic_sea_level", "ocean_density"

# The fields of the three-field GRACE model beside the ice sheets: the water
# stored on the land that carries no ice sheet, and the ocean's own load, both
# as thicknesses of water.
LAND, OCEAN = "land_water", "ocean"

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
# The half-angle of the cap that stands in for a point where a field has no
# point values.
COVARIANCE_CAP_DEGREES = 2.0

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


def add_coefficient_error_arguments(parser: argparse.ArgumentParser) -> None:
    """The GRACE noise as the error covariance of the coefficients, for the
    GRACE experiments; `add_joint_grace_noise_arguments` has the same
    options with the prefix grace-."""
    parser.add_argument(
        "--coefficient-errors",
        type=int,
        nargs=2,
        default=None,
        metavar="YEAR",
        help="Give GRACE's coefficients the formal error covariance of "
        "ITSG-Grace2018 for the difference of the annual means of these two "
        "years (`grace_covariance.py`), in place of a spatial noise field.",
    )
    parser.add_argument(
        "--error-scale",
        type=float,
        default=1.0,
        help="A factor on the standard deviations of --coefficient-errors.",
    )


def add_preconditioner_arguments(parser: argparse.ArgumentParser) -> None:
    """The preconditioner of the ocean experiments' solves."""
    parser.add_argument(
        "--preconditioner",
        choices=("surrogate", "blocks"),
        default=None,
        help="The surrogate problem at --surrogate-degree "
        "(`surrogate_preconditioner`), or blocks assembled in the data space "
        "(`block_preconditioner`). The default is blocks with "
        "--grace-coefficient-errors and the surrogate otherwise.",
    )
    parser.add_argument(
        "--cutoff-km",
        type=float,
        default=1500.0,
        help="The distance beyond which the altimetry block takes two "
        "points as uncorrelated.",
    )
    parser.add_argument(
        "--low-rank",
        type=int,
        default=40,
        help="Modes kept of the large-scale correlated altimetry noise in "
        "the block preconditioner.",
    )


def add_grace_arguments(parser: argparse.ArgumentParser) -> None:
    """The resolution, prior and noise of the GRACE experiments."""
    parser.add_argument("--lmax", type=int, default=256)
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=None,
        help="The highest degree of GRACE's data: 100 by default, or 60 with "
        "--coefficient-errors, which allows at most 96.",
    )
    add_coefficient_error_arguments(parser)
    parser.add_argument(
        "--grace-convention",
        choices=("stokes", "orthonormal"),
        default="stokes",
        help="GRACE's data: the dimensionless Stokes coefficients of Level-2 "
        "products, or the potential's coefficients in pyslfp's own convention. "
        "The two differ by a factor per degree, so no result depends on it.",
    )
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
    parser.add_argument(
        "--land-water",
        action="store_true",
        help="Add a field for the water stored on the land outside the ice sheets.",
    )
    parser.add_argument(
        "--source-order",
        type=float,
        default=None,
        help="Sobolev order, below 1/2, of the space of the ice and land-water "
        "fields. If given, those fields have priors of their own "
        "(--source-prior-order and the options that follow it); if omitted, "
        "they share the space and the prior order of the other fields.",
    )
    parser.add_argument(
        "--source-prior-order",
        type=float,
        default=1.5,
        help="With --source-order: the order of the Sobolev spectrum, on the "
        "Lebesgue space, of the ice and land-water priors. It is a choice: it "
        "must exceed --source-order by more than one.",
    )
    parser.add_argument(
        "--ice-std-mm",
        type=float,
        default=80.0,
        help="With --source-order: the pointwise std of the ice load, mm EWT.",
    )
    parser.add_argument("--ice-scale-km", type=float, default=180.0)
    parser.add_argument(
        "--land-water-std-mm",
        type=float,
        default=65.0,
        help="The pointwise std of the land-water field, mm.",
    )
    parser.add_argument("--land-water-scale-km", type=float, default=340.0)
    parser.add_argument(
        "--source-reference-degree",
        type=int,
        default=180,
        help="With --source-order: the degree up to which the ice and "
        "land-water stds are the stated values, the resolution of the loads "
        "they were taken from.",
    )
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
    """The altimetry noise of `altimetry_noise`, for a change over one
    year. The defaults are derived from the error budgets of Ablain et al.
    (2019) and Prandi et al. (2021)."""
    parser.add_argument(
        f"--{prefix}noise-std-mm",
        type=float,
        default=6.0,
        help="Std of the local error, independent at the points, mm.",
    )
    parser.add_argument(
        f"--{prefix}noise-corr-std-mm",
        type=float,
        default=0.7,
        help="Pointwise std of the large-scale correlated error, mm; zero "
        "leaves it out.",
    )
    parser.add_argument(
        f"--{prefix}noise-corr-scale-km",
        type=float,
        default=2000.0,
        help="Scale of the large-scale correlated error, km.",
    )
    parser.add_argument(
        f"--{prefix}noise-std-map",
        default=None,
        help="netCDF file whose field gives the pattern of the local noise's "
        "std over the ocean; its amplitude is still set by the std in mm.",
    )
    parser.add_argument(f"--{prefix}noise-std-variable", default="uncertainty")
    parser.add_argument(
        f"--{prefix}noise-global-std-mm",
        type=float,
        default=1.0,
        help="Std of an error common to every point: a bias of the global mean.",
    )
    parser.add_argument(
        f"--{prefix}noise-frame-std-mm",
        type=float,
        default=0.33,
        help="Std of each component of an error in the origin of the "
        "reference frame, a degree-1 pattern in the geocentric height.",
    )


def altimetry_noise_options(args: argparse.Namespace, prefix: str = "") -> dict:
    """The keyword options of `altimetry_noise` from the arguments of
    `add_altimetry_noise_arguments` with the same prefix (with underscores,
    such as "alt_")."""
    return {
        "std_mm": getattr(args, f"{prefix}noise_std_mm"),
        "corr_std_mm": getattr(args, f"{prefix}noise_corr_std_mm"),
        "corr_scale_km": getattr(args, f"{prefix}noise_corr_scale_km"),
        "std_map": getattr(args, f"{prefix}noise_std_map"),
        "std_variable": getattr(args, f"{prefix}noise_std_variable"),
        "global_std_mm": getattr(args, f"{prefix}noise_global_std_mm"),
        "frame_std_mm": getattr(args, f"{prefix}noise_frame_std_mm"),
    }


def add_gia_arguments(parser: argparse.ArgumentParser) -> None:
    """The error of the GIA correction, from the ensemble of
    `gia_ensemble.py`."""
    parser.add_argument(
        "--gia-years",
        type=float,
        default=0.0,
        help="The interval the data span, in years: the error of the GIA "
        "correction is the ensemble's spread of rates times this. Zero "
        "leaves the error out.",
    )
    parser.add_argument(
        "--gia-scale",
        type=float,
        default=1.0,
        help="A factor on the std of the GIA correction's error.",
    )
    parser.add_argument(
        "--gia-lmax",
        type=int,
        default=None,
        help="The degree of the ensemble to read; the highest computed if omitted.",
    )


def add_grace_noise_model_arguments(parser: argparse.ArgumentParser) -> None:
    """The GRACE noise of `grace_noise_model`."""
    parser.add_argument(
        "--noise-std-mm",
        type=float,
        default=5.0,
        help="Root mean square over the sphere of the measurement error, mm EWT.",
    )
    parser.add_argument("--noise-scale-km", type=float, default=300.0)
    parser.add_argument("--noise-order", type=float, default=2.0)
    parser.add_argument(
        "--noise-std-map",
        default=None,
        help="netCDF file whose field gives the pattern of the measurement "
        "error's std, such as the uncertainty of a mascon solution; its "
        "amplitude is still set by --noise-std-mm.",
    )
    parser.add_argument("--noise-std-variable", default="uncertainty")
    parser.add_argument(
        "--noise-polar-ratio",
        type=float,
        default=1.0,
        help="Without a map: the measurement error's std at the poles over "
        "that at the equator, below one for a mission whose tracks converge "
        "at the poles.",
    )
    parser.add_argument(
        "--corrections-std-mm",
        type=float,
        default=0.0,
        help="Pointwise std of the error of the background corrections, mm EWT.",
    )
    parser.add_argument("--corrections-scale-km", type=float, default=2000.0)
    add_coefficient_error_arguments(parser)
    add_gia_arguments(parser)


def add_joint_grace_noise_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=None,
        help="The highest degree of GRACE's data. With "
        "--grace-coefficient-errors the default is 60, and 96, the degree of "
        "the ITSG-Grace2018 solutions, is the most it can be; otherwise the "
        "default is 100.",
    )
    parser.add_argument(
        "--grace-convention",
        choices=("stokes", "orthonormal"),
        default="stokes",
        help="GRACE's data: the dimensionless Stokes coefficients of Level-2 "
        "products, or the potential's coefficients in pyslfp's own convention. "
        "The two differ by a factor per degree, so no result depends on it.",
    )
    parser.add_argument("--grace-noise-scale-km", type=float, default=250.0)
    parser.add_argument(
        "--grace-noise-std-mm",
        type=float,
        default=None,
        help="The pointwise std of the GRACE noise field, mm EWT, stated "
        "outright; if omitted, the noise is solved against the ice load from "
        "the signal-to-noise options that follow.",
    )
    parser.add_argument("--grace-noise-order", type=float, default=2.0)
    parser.add_argument(
        "--grace-coefficient-errors",
        type=int,
        nargs=2,
        default=None,
        metavar="YEAR",
        help="Give GRACE's coefficients the formal error covariance of "
        "ITSG-Grace2018 for the difference of the annual means of these two "
        "years (`grace_covariance.py`), in place of a spatial noise field: a "
        "full covariance matrix, with the dependence on order and the "
        "correlations of a single pair of satellites.",
    )
    parser.add_argument(
        "--grace-error-scale",
        type=float,
        default=1.0,
        help="A factor on the standard deviations of --grace-coefficient-errors.",
    )
    parser.add_argument(
        "--grace-floor-std-mm",
        type=float,
        default=0.0,
        help="With --grace-coefficient-errors, a further error that is white "
        "as a load over the observed degrees, the same variance in every "
        "coefficient of the load, taken to GRACE's coefficients by the load "
        "Love numbers: its pointwise std, mm EWT. It stands for load at "
        "scales the model does not represent.",
    )
    parser.add_argument("--grace-snr-low-degree", type=float, default=10.0)
    parser.add_argument("--grace-snr-reference-degree", type=float, default=2.0)
    parser.add_argument("--grace-snr-crossover-degree", type=float, default=64.0)


def configure(args: argparse.Namespace) -> np.random.Generator:
    """Sets the thread count, settles the options whose defaults depend on
    others, and returns the random generator of the run."""
    for name in ("grace_coefficient_errors", "coefficient_errors"):
        if not hasattr(args, name):
            continue
        errors = getattr(args, name) is not None
        if args.obs_degree is None:
            args.obs_degree = 60 if errors else 100
        if errors and args.obs_degree > grace_covariance.DEGREE:
            raise SystemExit(
                f"--obs-degree {args.obs_degree}: the coefficient errors of "
                f"ITSG-Grace2018 stop at degree {grace_covariance.DEGREE}."
            )
    if hasattr(args, "preconditioner") and args.preconditioner is None:
        errors = getattr(args, "grace_coefficient_errors", None) is not None
        args.preconditioner = "blocks" if errors else "surrogate"
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


def grace_coefficients(
    response_space, model, obs_degree: int, convention: str = "orthonormal", /
):
    """GRACE's data read from a response: the coefficients of its potential
    change for degrees 2 to `obs_degree`, in pyslfp's convention
    ("orthonormal") or as the Stokes coefficients of Level-2 products
    ("stokes"), which are those times a factor per degree."""
    coefficients = grace_observation_operator(response_space, obs_degree)
    if convention == "stokes":
        return stokes_coefficient_operator(model, obs_degree) @ coefficients
    return coefficients


def wmb_method(state: EarthState, args: argparse.Namespace) -> WMBMethod:
    """The WMB method at --obs-degree in the data's convention."""
    return WMBMethod(state.model, args.obs_degree, convention=args.grace_convention)


def grace_forward_operator(physics: GracePhysics, args: argparse.Namespace):
    """GRACE's coefficients of the potential change of the response to the
    direct load. The fingerprint operator's response already includes the
    ocean load the direct load induces."""
    return (
        grace_coefficients(
            physics.response_space,
            physics.state.model,
            args.obs_degree,
            args.grace_convention,
        )
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


def wmb_error_operator(
    physics: GracePhysics, average, wmb_average, args: argparse.Namespace, /
):
    """
    WMB's estimate of the regional averages less their true values, as a
    function of the direct load alone: C A - B, with A the forward operator,
    B = `average` of the total load and C = `wmb_average`, WMB's averaging
    of the data.

    Written as it stands, C A - B solves the sea level equation twice per
    action, once for the data and once for the induced ocean load. Here the
    load is first paired with its response, and the data and the induced
    load are both read from that one response, so that an action costs one
    solve and an adjoint action one adjoint solve.
    """
    load_space, response_space = physics.load_space, physics.response_space
    sea_level = response_space.field_operator("relative_sea_level_change")
    to_load = sea_level_change_to_load_operator(
        physics.state, sea_level.codomain, load_space
    )
    observe = grace_coefficients(
        response_space, physics.state.model, args.obs_degree, args.grace_convention
    )
    paired = inf.ColumnLinearOperator(
        [inf.LinearOperator.identity(load_space), physics.fingerprint]
    )
    read = inf.RowLinearOperator(
        [-1.0 * average, wmb_average @ observe - average @ to_load @ sea_level]
    )
    return read @ paired


# The factor taking a field in each unit to metres of equivalent water
# thickness, a mass per unit area being divided by 1000 kg m^-3.
_WATER_THICKNESS_UNITS = {
    "m": 1.0,
    "cm": 1.0e-2,
    "mm": 1.0e-3,
    "kg m-2": 1.0e-3,
    "kg/m2": 1.0e-3,
    "kg/m^2": 1.0e-3,
}


def _netcdf_interpolators(path: str, variable: str, /):
    """The fields of `variable` in a netCDF file on a regular
    latitude-longitude grid, one with dimensions (lat, lon) or several with
    one leading dimension, as bilinear interpolators over (latitude,
    longitude in [0, 360)) that reach the poles and wrap in longitude.
    Missing values are read as zero. Returns (labels, units, interpolators),
    the labels being the leading coordinate's values."""
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    with xr.open_dataset(path) as dataset:
        data = dataset[variable]
        units_name = str(data.attrs.get("units", "")).strip()
        lead = [d for d in data.dims if d not in ("lat", "lon")]
        if len(lead) > 1:
            raise ValueError("The fields may have one leading dimension at most.")
        data = data.transpose(*lead, "lat", "lon").sortby("lat").sortby("lon")
        lats = data["lat"].values.astype(float)
        lons = data["lon"].values.astype(float) % 360.0
        values = np.nan_to_num(data.values.astype(float))
        labels = (
            [str(v) for v in np.atleast_1d(data[lead[0]].values)] if lead else ["field"]
        )
    if not lead:
        values = values[np.newaxis]
    order = np.argsort(lons)
    lons, values = lons[order], values[..., order]

    # Pad a column at each end in longitude, which is periodic, and a row
    # at each pole, so that every point of the sphere lies inside.
    lons_p = np.concatenate([[lons[-1] - 360.0], lons, [lons[0] + 360.0]])
    lats_p = np.concatenate([[-90.0], lats, [90.0]])
    interpolators = []
    for field in values:
        padded = np.concatenate([field[:, -1:], field, field[:, :1]], axis=1)
        padded = np.concatenate([padded[:1], padded, padded[-1:]], axis=0)
        interpolators.append(RegularGridInterpolator((lats_p, lons_p), padded))
    return labels, units_name, interpolators


def _grid_points(grid) -> np.ndarray:
    """The (latitude, longitude in [0, 360)) of a grid's points, row by row."""
    lat_mesh, lon_mesh = np.meshgrid(grid.lats(), grid.lons() % 360.0, indexing="ij")
    return np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])


def read_load_fields(
    physics: GracePhysics, path: str, /, *, variable: str = "lwe_thickness"
):
    """
    Loads read from a netCDF file, as fields of the load space.

    The file holds `variable` on a regular latitude-longitude grid, either
    one field with dimensions (lat, lon) or several with one leading
    dimension, such as a year. Its units attribute must be a length of
    equivalent water thickness (m, cm or mm) or a mass per unit area
    (kg m-2). Each field is interpolated bilinearly to the model's grid,
    truncated at the model's degree, and has its degree 0 removed, so that
    it conserves mass as a total load must; a field that does not vanish
    on average is reported.

    Returns (labels, fields): a label for each field, the leading
    coordinate's value, and the fields in the model's units of load.
    """
    model = physics.state.model
    p = model.parameters
    labels, units_name, interpolators = _netcdf_interpolators(path, variable)
    if units_name not in _WATER_THICKNESS_UNITS:
        raise ValueError(
            f"'{variable}' has units '{units_name}'; expected one of "
            f"{sorted(_WATER_THICKNESS_UNITS)}."
        )
    points = _grid_points(model.zero_grid())
    to_load = _WATER_THICKNESS_UNITS[units_name] * p.water_density / p.length_scale

    fields = []
    for label, interpolator in zip(labels, interpolators):
        sampled = model.zero_grid()
        sampled.data[:] = to_load * interpolator(points).reshape(sampled.data.shape)
        coefficients = model.expand_field(sampled)
        coefficients.coeffs[0, 0, 0] = 0.0
        load = model.expand_coefficient(coefficients)
        rms = np.sqrt(model.integrate(load * load) / model.integrate(1.0 + 0.0 * load))
        mean_mm = model.integrate(sampled) / model.integrate(1.0 + 0.0 * load)
        print(
            f"  {label}: rms {rms * physics.scale_mm:.1f} mm EWT, largest "
            f"{np.max(np.abs(load.data)) * physics.scale_mm:.0f} mm; a mean of "
            f"{mean_mm * physics.scale_mm:+.2f} mm was removed"
        )
        fields.append(load)
    return labels, fields


def direct_load_of_total(total_load, sigma, /, *, rtol=1.0e-10, max_iterations=200):
    """
    A direct load whose total load, itself plus the ocean load it induces
    through the sea level equation, is `sigma`.

    The total load operator is the identity plus the induced load, which is
    small beside it, so the iteration zeta <- zeta + (sigma - T zeta) from
    zeta = sigma converges. The operator has a one-dimensional kernel, a
    uniform layer over the ocean, which the sea level equation cancels
    exactly; that component of the direct load is left as the iteration
    finds it, and nothing computed from the response depends on it. `sigma`
    must conserve mass, as every total load does.

    Returns (zeta, residual): the direct load and the largest absolute
    residual of its total load relative to that of `sigma`.
    """
    scale = float(np.max(np.abs(sigma.data)))
    zeta = sigma.copy()
    residual = np.inf
    for _ in range(max_iterations):
        misfit = sigma - total_load(zeta)
        residual = float(np.max(np.abs(misfit.data))) / scale
        if residual < rtol:
            break
        zeta = zeta + misfit
    return zeta, residual


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
    solved spectrum. With --coefficient-errors there is no noise field:
    None is returned for it, and `grace_data_noise` builds the noise."""
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
    if args.coefficient_errors is not None:
        return initial, prior, None, {"signal_std": std}
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


def grace_data_noise(physics: GracePhysics, args: argparse.Namespace, noise_load, info):
    """
    The noise of the GRACE experiments' data, from what `grace_measures`
    returned: the spatial noise field taken to the coefficients by WMB's
    spectral map, or, with --coefficient-errors, the error covariance of
    `grace_coefficient_noise`, a full matrix. `info["std_factor"]`, which
    sets the tolerance of the solves, is then the best-determined
    coefficient's error, as the pointwise std of a white load, over the
    prior's.

    Returns (noise, matrix), the matrix being the covariance if it is not
    diagonal and None otherwise.
    """
    if args.coefficient_errors is None:
        wmb = wmb_method(physics.state, args)
        return wmb.load_measure_to_observation_measure(noise_load), None
    noise, coefficient_info = grace_coefficient_noise(
        physics.state,
        physics.load_space,
        args,
        years=args.coefficient_errors,
        scale=args.error_scale,
    )
    info["std_factor"] = coefficient_info["best_std"] / info["signal_std"]
    info["noise_std"] = coefficient_info["noise_std"]
    return noise, coefficient_info["covariance"]


def shifted(measure, shift: float, rng):
    """The measure translated by `shift` times one of its own samples."""
    if shift == 0.0:
        return measure
    return measure.translate(measure.domain.scale(shift, measure.sample(rng=rng)))


# ================================================================ #
#                  GRACE with land, ice and ocean                  #
# ================================================================ #


@dataclass
class BudgetPhysics:
    """
    The physics of the GRACE experiments with three fields: an ice thickness
    change over the ice sheets, a change in the water stored on the rest of
    the land, and the ocean's own load as a thickness of water. The fields
    are global, and the load carries their supports:

        zeta = rho_i (1 - C) chi_I h + rho_w chi_L w + rho_w C b,

    with C the ocean function and chi_I, chi_L the indicators of the ice
    sheets and of the land without them. A load of this form can jump at the
    coast and at the edge of an ice sheet, which a single smooth field
    cannot. The three supports cover the sphere, so that every load is of
    this form: the land field carries the glaciers and ice caps outside the
    two ice sheets along with the water, as a thickness of water.

    The fields, the loads and the response all lie in Lebesgue spaces:
    GRACE's data are coefficients, so nothing is evaluated at a point, and
    realistic loads are too rough for a Sobolev space of positive order.
    `scale_mm` takes a load to millimetres of equivalent water thickness.
    """

    state: EarthState
    setup: ModelSetup
    fingerprint: FingerPrintOperator
    scale_mm: float

    @property
    def load_space(self):
        return self.setup.load_space

    @property
    def model_space(self):
        return self.setup.space

    @property
    def response_space(self):
        return self.fingerprint.codomain

    @property
    def densities(self) -> dict:
        """The density taking each field, a thickness, to a load."""
        p = self.state.model.parameters
        return {ICE: p.ice_density, LAND: p.water_density, OCEAN: p.water_density}

    @property
    def supports(self) -> dict:
        """The indicator of each field's support, the grounded part for the
        ice sheets, by field name."""
        state = self.state
        return {
            ICE: state.ice_sheet_projection(value=0.0) * state.one_minus_ocean_function,
            LAND: _land_without_ice_sheets(state),
            OCEAN: state.ocean_projection(value=0.0),
        }

    def forward(self, args: argparse.Namespace) -> inf.LinearOperator:
        """GRACE's coefficients of the potential change, from the fields."""
        return (
            grace_coefficients(
                self.response_space,
                self.state.model,
                args.obs_degree,
                args.grace_convention,
            )
            @ self.fingerprint
            @ self.setup.load_operator
        )

    @property
    def load_to_total_load(self) -> inf.LinearOperator:
        """A direct load plus the ocean load it induces, on the load space."""
        sea_level = self.response_space.field_operator("relative_sea_level_change")
        to_load = sea_level_change_to_load_operator(
            self.state, sea_level.codomain, self.load_space
        )
        return inf.LinearOperator.identity(self.load_space) + (
            to_load @ sea_level @ self.fingerprint
        )

    @property
    def total_load(self) -> inf.LinearOperator:
        """The total load, from the fields."""
        return self.load_to_total_load @ self.setup.load_operator


def _land_without_ice_sheets(state: EarthState):
    """The indicator of the land outside the two ice sheets. Unlike
    `EarthState.land_projection(exclude_ice=True)`, it keeps the land under
    glaciers and ice caps, which would otherwise belong to no field."""
    return state.one_minus_ocean_function * (
        1.0 - state.ice_sheet_projection(value=0.0)
    )


def _land_water(state: EarthState, space, load_space=None) -> LoadComponent:
    """Mass on the land outside the ice sheets, as a thickness of water:
    stored water, and the glaciers and ice caps there. The field lies in
    `space` and its load in `load_space`, which is `space` if omitted."""
    mask = _land_without_ice_sheets(state)
    support = space.multiplication_operator(mask)
    density = state.model.parameters.water_density
    if load_space is None or load_space == space:
        to_load = density * support
    else:

        def mapping(u):
            return (density * mask) * u

        to_load = inf.LinearOperator.from_formal_adjoint(
            space,
            load_space,
            inf.LinearOperator.from_callables(
                underlying_space(space),
                underlying_space(load_space),
                mapping,
                adjoint=mapping,
            ),
        )
    return LoadComponent(
        name=LAND,
        space=space,
        to_load=to_load,
        support=support.with_traits(support.traits | inf.Traits.IDEMPOTENT),
        role=Role.MASS_SOURCE,
    )


def budget_physics(lmax: int) -> BudgetPhysics:
    state = EarthState.from_defaults(lmax=lmax)
    p = state.model.parameters
    fingerprint = FingerPrintOperator(state)
    load_space = fingerprint.domain
    setup = ModelSetup(
        state,
        [
            ice_sheets(state, load_space),
            _land_water(state, load_space),
            sterodynamic_sea_level(state, load_space, name=OCEAN),
        ],
    )
    return BudgetPhysics(
        state, setup, fingerprint, 1000.0 * p.length_scale / p.water_density
    )


def degree_variances(space, fields, /, *, lmax: Optional[int] = None) -> np.ndarray:
    """The mean square coefficient of a set of fields at each degree, from 0
    to `lmax` (the space's, if omitted), averaged over the orders and the
    fields. The coefficients are those of harmonics orthonormal on the unit
    sphere, so that a field's mean square over the sphere is the sum over
    degrees of (2 l + 1) v_l / (4 pi)."""
    lmax = space.lmax if lmax is None else lmax
    reader = (1.0 / space.radius) * space.coefficient_operator(lmax=lmax, lmin=0)
    squares = np.mean([reader(field) ** 2 for field in fields], axis=0)
    edges = np.arange(lmax + 2) ** 2
    return np.array([squares[edges[l] : edges[l + 1]].mean() for l in range(lmax + 1)])


def fit_spectrum(
    variances,
    radius: float,
    /,
    *,
    lmin: int = 2,
    min_order: float = 1.05,
    scales=None,
    order: Optional[float] = None,
):
    """
    Fits the degree variances of a field by the spectrum of a Sobolev
    measure, v_l = a (1 + s^2 l (l + 1) / r^2)^(-q), by least squares on
    their logarithms from degree `lmin`.

    The order q is kept above `min_order`: a Gaussian measure on a Lebesgue
    space needs q > 1. The scale s is kept within `scales`, (lower, upper)
    in the units of the radius, by default between a hundredth of the radius
    and half of it. With `order` given, the order is fixed at that value and
    only the amplitude and the scale are fitted: the fields the fit is made
    to stop at some degree and say little about the decay beyond it, so the
    order may be better chosen than fitted.

    Returns (order, scale).
    """
    from scipy.optimize import least_squares

    variances = np.asarray(variances, dtype=float)
    degrees = np.arange(len(variances), dtype=float)[lmin:]
    k = degrees * (degrees + 1.0) / radius**2
    target = np.log(variances[lmin:])
    lower, upper = (0.01 * radius, 0.5 * radius) if scales is None else scales

    def residual(x):
        return x[0] - x[2] * np.log1p(x[1] ** 2 * k) - target

    start = [target[0], np.sqrt(lower * upper)]
    if order is not None:
        fit = least_squares(
            lambda x: residual([x[0], x[1], order]),
            start,
            bounds=([-np.inf, lower], [np.inf, upper]),
        )
        return float(order), float(fit.x[1])
    fit = least_squares(
        residual,
        start + [1.5],
        bounds=([-np.inf, lower, min_order], [np.inf, upper, 8.0]),
    )
    return float(fit.x[2]), float(fit.x[1])


def _band_power(order: float, scale: float, radius: float, lmax: int) -> float:
    degrees = np.arange(lmax + 1, dtype=float)
    k = degrees * (degrees + 1.0) / radius**2
    return float(np.sum((2.0 * degrees + 1.0) * (1.0 + scale**2 * k) ** (-order)))


def field_loads(physics: BudgetPhysics, water, ice, ocean, /) -> dict:
    """
    The loads of the three fields that make up one change of the components
    of a load model, such as the H, I and O of the ESA Earth System Model,
    by field name. Each component is confined to where it belongs: the ice
    on the ice sheets to the ice field, the water and the remaining ice on
    the rest of the land to the land field, and the ocean, less its mean
    over the ocean, to the ocean field. The ocean's part then moves mass
    about without changing the ocean's total, which the sea level equation
    supplies. The three sum to a direct load.
    """
    state, model = physics.state, physics.state.model
    supports = physics.supports
    sea = state.ocean_function
    ocean_mean = model.integrate(sea * ocean) / model.integrate(sea)
    return {
        ICE: supports[ICE] * ice,
        LAND: supports[LAND] * (water + ice),
        OCEAN: sea * (ocean - ocean_mean * model.constant_grid(1.0)),
    }


def budget_prior(
    physics: BudgetPhysics,
    training: dict,
    /,
    *,
    mean: bool = False,
    modulate: Optional[str] = None,
    modulation_scale: float = 0.05,
    modulation_floor: float = 0.1,
    reference_degree: int = 180,
    min_order: float = 1.05,
    order: Optional[float] = None,
    fit_degree: Optional[int] = None,
):
    """
    A prior for the three-field GRACE model, estimated from training loads.

    `training` holds, by field name, a list of loads of that field on the
    model's grid, each confined to the field's support (`field_loads`). For
    each field:

    - with `mean`, the expectation is the mean of its training loads, and
      what follows is estimated from their deviations about it; otherwise
      the expectation is zero;
    - the spectrum is that of a Sobolev measure fitted to the degree
      variances of the training loads (`fit_spectrum`) up to `fit_degree`
      (`reference_degree`, if omitted). With `order` given, the order of
      the measure is that value and only its scale is fitted: the training
      loads stop at a finite degree, and how the spectrum decays beyond it
      is a choice, not something they determine;
    - the pointwise standard deviation is the root mean square of the
      training loads over the field's support, taken as the variance carried
      by degrees up to `reference_degree`, the resolution of the training
      loads. The spectrum continues past that degree by the fitted law, so
      the measure does not depend on the truncation of the model;
    - with `modulate`, the standard deviation varies over the support,
      scaled to leave the mean variance over the support unchanged. The
      field is then a positive function times an invariant field, which is
      a measure on the same space. With "pointwise" the function follows the
      root mean square of the training loads at each point, smoothed over
      `modulation_scale` (in the units of the radius) and kept above
      `modulation_floor` times its typical value; a few training loads say
      little about the variance at a point, which is why the pattern is
      smoothed. With "margin" only the ice sheets are modulated, and by one
      number: the standard deviation is 1 + a p, with p the nearness of a
      point to the edge of the ice (the fraction of ground that is not ice
      sheet within `modulation_scale` of it, nought in the interior and
      about a half at the edge) and a fitted to the training loads by least
      squares. Ice sheets change most at their margins, and this says so
      with a single parameter.

    The fields are independent, and the ocean is conditioned exactly on
    having no net mass, as the ocean's own load must.

    Returns (prior, fields, info): the prior; the invariant measures of the
    three fields, before any modulation, shift or conditioning, in the order
    of the model space; and a record of the fitted values by field.
    """
    setup, space = physics.setup, physics.load_space
    model = physics.state.model
    radius = space.radius
    supports, densities = physics.supports, physics.densities
    spectrum_degree = min(
        space.lmax, reference_degree if fit_degree is None else fit_degree
    )
    fit_degree = min(space.lmax, reference_degree)
    smoother = space.heat_measure(modulation_scale * radius).covariance
    invariant, measures, info = [], [], {}
    for name in setup.names:
        loads = training[name]
        count = len(loads)
        mask = supports[name]
        area = model.integrate(mask)
        centre = None
        if mean:
            centre = loads[0] * (1.0 / count)
            for load in loads[1:]:
                centre = centre + load * (1.0 / count)
            loads = [load - centre for load in loads]
        # With a mean estimated from them, the deviations understate the
        # variance by the usual factor.
        unbiased = count / (count - 1.0) if mean else 1.0
        fitted_order, scale = fit_spectrum(
            degree_variances(space, loads, lmax=spectrum_degree),
            radius,
            min_order=min_order,
            order=order,
        )
        variance = loads[0] * loads[0]
        for load in loads[1:]:
            variance = variance + load * load
        variance = variance * (unbiased / count)
        load_std = np.sqrt(model.integrate(mask * variance) / area)
        # The training loads stop at the reference degree, or at the model's
        # if that is lower; the measure is normalised at the model's degree.
        continued = np.sqrt(
            _band_power(fitted_order, scale, radius, space.lmax)
            / _band_power(fitted_order, scale, radius, fit_degree)
        )
        std = continued * load_std / densities[name]
        measure = space.sobolev_measure(fitted_order, scale, pointwise_std=std)
        invariant.append(measure)
        relative = None
        inside = mask.data > 0.5
        if modulate == "pointwise":
            # A smoothed variance over the support, as the ratio of the
            # smoothed masked variance to the smoothed mask.
            weight = smoother(mask)
            smoothed = smoother(mask * variance)
            relative = variance * 0.0
            inside = inside & (weight.data > 1.0e-6)
            relative.data[inside] = (
                np.maximum(smoothed.data[inside], 0.0) / weight.data[inside]
            )
            relative.data[:] = np.sqrt(relative.data) / load_std
            relative.data[inside] = np.maximum(relative.data[inside], modulation_floor)
        elif modulate == "margin" and name == ICE:
            # The nearness to the edge of the ice, and a straight-line fit of
            # the training loads' root mean square against it.
            sheets = physics.state.ice_sheet_projection(value=0.0)
            nearness = smoother(1.0 - sheets)
            # Area weights for the fit: the cosine of the latitude.
            cosine = np.cos(np.deg2rad(mask.lats()))
            weights = np.broadcast_to(cosine[:, None], mask.data.shape)[inside]
            weights = np.maximum(weights, 0.0)
            x = np.clip(nearness.data[inside], 0.0, 1.0)
            y = np.sqrt(np.maximum(variance.data[inside], 0.0))
            design = np.column_stack([np.ones_like(x), x]) * np.sqrt(weights)[:, None]
            intercept, slope = np.linalg.lstsq(
                design, y * np.sqrt(weights), rcond=None
            )[0]
            growth = max(slope / intercept, 0.0) if intercept > 0.0 else 0.0
            relative = variance * 0.0
            relative.data[inside] = 1.0 + growth * x
            info_growth = growth
        if relative is not None:
            relative.data[~inside] = 1.0
            relative = relative * (
                1.0 / np.sqrt(model.integrate(mask * relative * relative) / area)
            )
            relative.data[~inside] = 1.0
            measure = measure.affine_map(space.multiplication_operator(relative))
        if centre is not None:
            measure = measure.translate(centre * (1.0 / densities[name]))
        measures.append(measure)
        info[name] = {
            "order": fitted_order,
            "scale_km": scale * model.parameters.length_scale / 1000.0,
            "load_std_mm": load_std * physics.scale_mm,
            "thickness_std": std,
            "mean_rms_mm": 0.0
            if centre is None
            else np.sqrt(model.integrate(mask * centre * centre) / area)
            * physics.scale_mm,
            "margin_growth": info_growth
            if (modulate == "margin" and name == ICE)
            else None,
        }
    product = inf.GaussianMeasure.from_product(measures, domain=setup.space)
    constraint = setup.constraints[OCEAN_MASS]
    # A modulated field's covariance is M Q M with M multiplication by a
    # positive function, which is definite as Q is, though the algebra
    # cannot tell; the conditioning is told so.
    prior = product.condition(
        constraint, np.zeros(constraint.codomain.dim), definite=modulate is not None
    )
    return prior, invariant, info


# The three-field prior as stated numbers: each field's pointwise std as a
# load, in mm EWT, and its scale in km. They are representative of the
# year-on-year changes of the ESA Earth System Model resolved to degree 180
# (`esm_loads.py`), at the order `BUDGET_PRIOR_ORDER`.
BUDGET_PRIOR = {ICE: (80.0, 180.0), LAND: (65.0, 340.0), OCEAN: (8.0, 940.0)}
BUDGET_PRIOR_ORDER = 1.5


def stated_budget_prior(
    physics: BudgetPhysics,
    /,
    *,
    values: Optional[dict] = None,
    order: float = BUDGET_PRIOR_ORDER,
    reference_degree: int = 180,
):
    """
    A prior for the three-field GRACE model from stated numbers.

    Each field is an invariant Sobolev field of the given `order`, with the
    pointwise std (as a load, mm EWT) and the scale (km) that `values`
    gives it by name, `BUDGET_PRIOR` if omitted. The stds are those of
    loads resolved to `reference_degree`; the spectrum continues past that
    degree by its own law, so that the prior does not depend on the
    truncation of the model. The fields are independent, and the ocean is
    conditioned exactly on having no net mass.

    Returns (prior, fields, info) as `budget_prior` does.
    """
    values = BUDGET_PRIOR if values is None else values
    setup, space = physics.setup, physics.load_space
    p = physics.state.model.parameters
    densities = physics.densities
    invariant, info = [], {}
    for name in setup.names:
        std_mm, scale_km = values[name]
        scale = scale_km * 1000.0 / p.length_scale
        continued = np.sqrt(
            _band_power(order, scale, space.radius, space.lmax)
            / _band_power(order, scale, space.radius, reference_degree)
        )
        std = continued * std_mm / physics.scale_mm / densities[name]
        invariant.append(space.sobolev_measure(order, scale, pointwise_std=std))
        info[name] = {
            "order": order,
            "scale_km": scale_km,
            "load_std_mm": std_mm,
            "thickness_std": std,
            "mean_rms_mm": 0.0,
            "margin_growth": None,
        }
    product = inf.GaussianMeasure.from_product(invariant, domain=setup.space)
    constraint = setup.constraints[OCEAN_MASS]
    prior = product.condition(constraint, np.zeros(constraint.codomain.dim))
    return prior, invariant, info


def invariant_load_measure(physics: BudgetPhysics, fields, /):
    """
    An invariant measure on the load space that stands in for the three-field
    prior's law of the load, for the diagonal preconditioner. `fields` are
    the fields' measures, as `budget_prior` returns them: each one's spectrum
    is taken as a load and weighted by the fraction of the sphere its support
    covers. The shapes of the supports are ignored, which is what makes the
    result invariant.
    """
    model = physics.state.model
    space = physics.load_space
    sphere = model.integrate(1.0 + 0.0 * physics.supports[OCEAN])
    eigenvalues = 0.0
    for name, field in zip(physics.setup.names, fields):
        fraction = model.integrate(physics.supports[name]) / sphere
        eigenvalues = eigenvalues + (
            fraction * physics.densities[name] ** 2 * field.covariance.eigenvalues
        )
    return inf.GaussianMeasure(
        space, covariance=inf.DiagonalLinearOperator(space, eigenvalues)
    )


# ================================================================ #
#                       Ocean experiments                          #
# ================================================================ #


@dataclass
class OceanPhysics:
    """The ocean experiments' physics.

    `setup` is the set-up of the fields: the ice, with --land-water the
    water stored on the land, eta and delta-rho. Eta and delta-rho lie in
    H^`field_order` (`field_space`). The ice and the land water, the mass
    sources, lie in `source_space`: the same space, or with --source-order a
    space of that lower order, which rougher priors need. `fingerprint` maps
    the loads (in H^`load_order`) to the response (one order above);
    `geocentric_sea_level` is the continuous geocentric sea level change, in
    the response's field space, and `altimetry` its values at the track
    `points`. `field_point_evaluation` evaluates a field of the fields'
    space at the points, for noise fields. `scale_mm` takes a length to
    millimetres.
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
        return self.setup.component(ETA).space

    @property
    def source_space(self):
        return self.setup.component(ICE).space

    @property
    def sources(self) -> list:
        """The names of the mass sources: the ice and, if present, the land
        water."""
        return [name for name in self.setup.names if name in (ICE, LAND)]

    @property
    def rough_sources(self) -> bool:
        """Whether the mass sources have a space and priors of their own."""
        return self.source_space != self.field_space

    @property
    def load_space(self):
        return self.setup.load_space

    @property
    def display_scales(self) -> dict:
        """The factors taking each field to its display unit: mm for the ice
        thickness change, the land water and eta, g m^-3 for delta rho."""
        density = 1.0e3 * self.state.model.parameters.density_scale
        scales = {ICE: self.scale_mm, LAND: self.scale_mm, ETA: self.scale_mm}
        return scales | {RHO: density}

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
        on the supports and in the product of the two fields' display units.
        A field whose space is too rough for point values is averaged over a
        cap of `COVARIANCE_CAP_DEGREES` about the point instead."""
        space = self.setup.component(component).space
        if space.order > 1.0:
            dirac = space.dirac(point).representer
        else:
            dirac = space.spherical_cap_average(
                point, COVARIANCE_CAP_DEGREES
            ).representer
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

    def grace(self, args: argparse.Namespace) -> inf.LinearOperator:
        """GRACE's coefficients of the potential change, from the model, at
        --obs-degree and in the convention --grace-convention. A surrogate
        truncated below --obs-degree gives the coefficients up to its own
        degree and zeros above, in the same data space: the coefficients
        are ordered by degree, so the zeros follow the rest."""
        degree = min(args.obs_degree, self.fingerprint.codomain.field_space.lmax)
        operator = (
            grace_coefficients(
                self.fingerprint.codomain,
                self.state.model,
                degree,
                args.grace_convention,
            )
            @ self.fingerprint
            @ self._loads
        )
        if degree < args.obs_degree:
            data_space = inf.EuclideanSpace((args.obs_degree + 1) ** 2 - 4)
            read = data_space.coordinate_selection(np.arange(operator.codomain.dim))
            operator = read.adjoint @ operator
        return operator

    @property
    def _loads(self):
        return self.setup.load_operator


@dataclass
class SurrogateOceanPhysics(OceanPhysics):
    """The cheap surrogate of the preconditioners: the loads of the mass
    sources alone, one pass of the sea level equation, at a lower degree."""

    source_loads: inf.LinearOperator = None

    @property
    def _loads(self):
        return self.source_loads


def ocean_physics(
    args: argparse.Namespace,
    /,
    *,
    points: Optional[Sequence[tuple[float, float]]] = None,
    surrogate: bool = False,
    surrogate_loads: str = "sources",
    surrogate_iterations: Optional[int] = 1,
) -> OceanPhysics:
    """The physics of the ocean experiments at --lmax, or, for the surrogate
    of the preconditioners, at --surrogate-degree. The altimetry points are
    the regular grid at --spacing over the ice-free ocean within 66 degrees
    of the equator, unless given; the surrogate takes those of the exact
    physics, so that the two share the data space. The surrogate keeps the
    loads of the mass sources alone (`surrogate_loads="sources"`) or all of
    them ("all"), and makes `surrogate_iterations` passes of the sea level
    equation, or solves it if that is None."""
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
        max_iterations=surrogate_iterations if surrogate else None,
    )
    field_space = sobolev_load_space(state.model, args.field_order, scale)
    load_space = fingerprint.domain
    rough = args.source_order is not None
    if rough and not args.source_order < 0.5:
        raise ValueError("--source-order must be below 1/2.")
    source_space = (
        sobolev_load_space(state.model, args.source_order, scale)
        if rough
        else field_space
    )
    components = [
        ice_sheets(
            state,
            load_space,
            space=source_space,
            observable_space=None if rough else fingerprint.field_space,
        )
    ]
    if args.land_water:
        components.append(_land_water(state, source_space, load_space))
    components += [
        sterodynamic_sea_level(
            state,
            load_space,
            space=field_space,
            observable_space=fingerprint.field_space,
        ),
        ocean_density(state, load_space, space=field_space),
    ]
    setup = ModelSetup(state, components)
    loads = setup.load_operator
    if surrogate and surrogate_loads not in ("sources", "all"):
        raise ValueError("surrogate_loads is 'sources' or 'all'.")
    if surrogate and surrogate_loads == "sources":
        sources = None
        for name in (ICE, LAND):
            if name in setup.names:
                part = setup.space.inclusion(name) @ setup.space.projection(name)
                sources = part if sources is None else sources + part
        loads = loads @ sources
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
        return SurrogateOceanPhysics(**fields, source_loads=loads)
    return OceanPhysics(**fields)


def _std(measure, functional) -> float:
    """The standard deviation of a scalar functional under a measure."""
    return float(np.sqrt(measure.directional_variance(functional.adjoint(np.ones(1)))))


def _source_scale(physics: OceanPhysics, args: argparse.Namespace, name) -> float:
    """The scale of a mass source's own prior, in the model's units."""
    km = args.ice_scale_km if name == ICE else args.land_water_scale_km
    return km * 1000.0 / physics.state.model.parameters.length_scale


def _source_std(physics: OceanPhysics, args: argparse.Namespace, name) -> float:
    """The pointwise std of a mass source under its own prior, as a
    thickness in the model's units. The stated values are those of loads
    resolved to --source-reference-degree; the spectrum continues past that
    degree by its own law, so that the prior does not depend on the
    truncation of the model."""
    p = physics.state.model.parameters
    space = physics.source_space
    scale = _source_scale(physics, args, name)
    continued = np.sqrt(
        _band_power(args.source_prior_order, scale, space.radius, space.lmax)
        / _band_power(
            args.source_prior_order, scale, space.radius, args.source_reference_degree
        )
    )
    if name == ICE:
        # Given as the load in mm of water; the field is a thickness of ice.
        std = args.ice_std_mm * p.water_density / p.ice_density
    else:
        std = args.land_water_std_mm
    return continued * std / physics.scale_mm


def _field_measure(physics: OceanPhysics, args: argparse.Namespace, name, std):
    """The prior of one field alone, with pointwise std `std`.

    Eta and delta-rho, and the mass sources when they share their space,
    are Sobolev fields of order --prior-order on the fields' space. With
    --source-order the mass sources have instead the spectrum of a Sobolev
    measure on the Lebesgue space, (1 + s^2 k)^-t in the Laplacian's
    eigenvalue k, of order t = --source-prior-order and their own scale s.
    The eigenvalues of a covariance are relative to the inner product of the
    space it acts on, so the space's own weight is put back; the result is a
    measure on the sources' space because t exceeds its order by more than
    one."""
    if name in (ICE, LAND) and physics.rough_sources:
        space = physics.source_space
        order = args.source_prior_order
        if order - space.order <= 1.0:
            raise ValueError(
                "--source-prior-order must exceed --source-order by more than one."
            )
        scale = _source_scale(physics, args, name)

        def spectrum(k):
            return (1.0 + scale**2 * k) ** (-order) * (
                1.0 + space.length_scale**2 * k
            ) ** (space.order)

        return space.invariant_measure(spectrum, pointwise_std=std)
    length = physics.field_space.length_scale
    factors = {
        ICE: args.ice_scale_factor,
        LAND: args.ice_scale_factor,
        ETA: args.ocean_dyn_scale_factor,
        RHO: args.ocean_rho_scale_factor,
    }
    return physics.setup.component(name).space.sobolev_measure(
        args.prior_order, length * factors[name], pointwise_std=std
    )


def _stds_by_name(stds: dict) -> dict:
    """The pointwise stds of `calib["derived_stds"]` by field name."""
    by_name = {ICE: stds["ice_std"], ETA: stds["dyn_std"], RHO: stds["rho_std"]}
    if "land_std" in stds:
        by_name[LAND] = stds["land_std"]
    return by_name


def product_prior(
    physics: OceanPhysics,
    args: argparse.Namespace,
    stds: dict,
    /,
    *,
    reference: Optional[OceanPhysics] = None,
):
    """The uncorrelated, unconditioned product of the fields' priors at the
    pointwise stds `stds` (keyed as calib["derived_stds"]): the prior of the
    preconditioners' surrogate.

    A field's pointwise std is the sum of its variances over the degrees
    its space holds, so a field built with that std on a space truncated at
    a lower degree has too much variance at every degree it keeps: all the
    variance of the higher degrees is pushed onto them. With `reference`,
    the physics the surrogate stands in for, each field takes instead the
    variances of the reference's field at the degrees it shares with it, so
    that the surrogate's prior is the reference's, truncated."""
    by_name = _stds_by_name(stds)
    measures = []
    for name in physics.setup.names:
        if reference is None:
            measures.append(_field_measure(physics, args, name, by_name[name]))
            continue
        space = physics.setup.component(name).space
        spectrum = _field_measure(
            reference, args, name, by_name[name]
        ).covariance.eigenvalues
        # The components are ordered by degree, so the shared ones lead.
        measures.append(space.invariant_measure(spectrum[: space.dim]))
    return inf.GaussianMeasure.from_product(measures, domain=physics.setup.space)


def ocean_prior(physics: OceanPhysics, args: argparse.Namespace, rng):
    """
    The prior of the ocean experiments: the fields' own priors
    (`_field_measure`), eta and delta rho anti-correlated at long
    wavelengths, conditioned exactly on the conservation of the ocean's
    mass, then shifted. The fields are global and the prior is not masked.

    The amplitudes of eta and delta rho follow from one dimensioned input,
    `--ocean-dyn-std-mm`, the pointwise std of eta: the density std is
    `--steric-dyn-std-ratio` times it over the mean ocean depth. Those of the
    mass sources depend on how they are modelled.

    - Sharing the other fields' space, the ice std follows from
      `--gmsl-bary-steric-ratio`, the ratio of the ice's barystatic GMSL std
      to the realised steric GMSL std under the mass constraint. The ice is
      uncorrelated with the ocean fields and absent from the constraint, so
      the chain closes without iteration. The land water, if present, has
      the std `--land-water-std-mm`.
    - With `--source-order`, the ice and the land water have the stated
      stds `--ice-std-mm` and `--land-water-std-mm` and scales
      `--ice-scale-km` and `--land-water-scale-km`, values representative
      of year-on-year changes, and the ratio of the barystatic to the
      steric GMSL std is what results.

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
    names = list(setup.names)
    field_space = physics.field_space
    state = physics.state
    p = state.model.parameters
    rough = physics.rough_sources
    corr_scale = field_space.length_scale * args.ocean_corr_scale_factor

    def correlated(stds):
        measures = {n: _field_measure(physics, args, n, stds[n]) for n in names}
        if args.ocean_corr == 0.0:
            return inf.GaussianMeasure.from_product(
                [measures[n] for n in names], domain=setup.space
            )
        k = field_space.laplacian_eigenvalues
        r = -args.ocean_corr * np.exp(-(corr_scale**2) * k)
        if not rough:
            # The fields share a space: one measure, correlated by degree.
            i, j = names.index(ETA), names.index(RHO)
            matrices = np.tile(np.eye(len(names)), (len(k), 1, 1))
            matrices[:, i, j] = matrices[:, j, i] = r
            return field_space.correlated_measure_from_correlations(
                [measures[n].covariance.eigenvalues for n in names],
                matrices,
                labels=names,
            )
        # The mass sources lie in another space, so the same law is built
        # from independent fields: delta rho is a part that follows eta,
        # through an operator diagonal in the harmonics, plus a remainder.
        a = measures[ETA].covariance.eigenvalues
        b = measures[RHO].covariance.eigenvalues
        measures[RHO] = field_space.invariant_measure(b * (1.0 - r**2))
        follow = field_space.spectral_operator(r * np.sqrt(b / a))
        mixing = inf.LinearOperator.identity(setup.space) + (
            setup.space.inclusion(RHO) @ follow @ setup.space.projection(ETA)
        )
        independent = inf.GaussianMeasure.from_product(
            [measures[n] for n in names], domain=setup.space
        )
        return independent.affine_map(mixing)

    def conditioned(measure_):
        constraint = setup.constraints[OCEAN_MASS]
        # The mixed measure is positive definite as its factors are, though
        # the algebra cannot tell; the conditioning is told so.
        return measure_.condition(
            constraint, np.zeros(constraint.codomain.dim), definite=rough
        )

    dyn_std = args.ocean_dyn_std_mm / physics.scale_mm
    effective_steric_scale = state.mean_ocean_depth / p.water_density
    rho_std = args.steric_dyn_std_ratio * dyn_std / effective_steric_scale
    stds = {ETA: dyn_std, RHO: rho_std}
    if LAND in names:
        stds[LAND] = (
            _source_std(physics, args, LAND)
            if rough
            else args.land_water_std_mm / physics.scale_mm
        )

    # The realised steric GMSL std under the constraint. The ice std is a
    # placeholder, the ice being uncorrelated and absent from the constraint.
    trial = correlated(stds | {ICE: 1.0})
    sigma_s = _std(conditioned(trial), setup.steric_gmsl)
    eta_mean = ocean_average_operator(state, field_space) @ setup.space.projection(ETA)
    a = _std(trial, eta_mean)
    b = _std(trial, setup.steric_gmsl)
    d_x = eta_mean.adjoint(np.ones(1))
    d_y = setup.steric_gmsl.adjoint(np.ones(1))
    rho_xy = float(trial.directional_covariance(d_x, d_y)) / (a * b)

    # The barystatic GMSL of the ice per unit ice std, its support built
    # into the load.
    q_ice = _std(
        _field_measure(physics, args, ICE, 1.0),
        barystatic_gmsl_operator(state, setup.component(ICE)),
    )
    if rough:
        ice_std = _source_std(physics, args, ICE)
    else:
        ice_std = args.gmsl_bary_steric_ratio * sigma_s / q_ice
    stds[ICE] = ice_std

    prior = conditioned(correlated(stds))
    barystatic = _std(prior, setup.barystatic_gmsl())
    derived = {"ice_std": ice_std, "dyn_std": dyn_std, "rho_std": rho_std}
    if LAND in names:
        derived["land_std"] = stds[LAND]
    calib = {
        "constants": {
            "q_ice": q_ice,
            "u_eta": a / dyn_std,
            "u_rho": b / rho_std,
            "rho_xy": rho_xy,
            "effective_steric_scale": effective_steric_scale,
        },
        "gmsl": {
            "barystatic_std": barystatic,
            "ice_barystatic_std": q_ice * ice_std,
            "steric_std": sigma_s,
            "eta_mean_std_unconstrained": a,
            "bary_steric_ratio": barystatic / sigma_s,
            "steric_std_check": _std(prior, setup.steric_gmsl),
            "barystatic_std_check": barystatic,
        },
        "derived_stds": derived,
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
        + (f", land water std = {d['land_std'] * mm:.3f} mm" if "land_std" in d else "")
    )
    print(
        "  realised GMSL prior stds under the mass constraint: barystatic = "
        f"{g['barystatic_std'] * mm:.3f} mm (the ice alone "
        f"{g['ice_barystatic_std'] * mm:.3f}), steric = "
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
    args: argparse.Namespace, noise_corr_std_mm: float, noise_corr_scale_km: float
) -> None:
    """Prints the settings of the prior's correlation and of the correlated
    altimetry noise."""
    if args.ocean_corr > 0.0:
        print(
            f"Correlated (sterodynamic, density) prior: r0 = {args.ocean_corr:.2f}, "
            f"correlation scale factor = {args.ocean_corr_scale_factor:.2f}"
        )
    print(f"Sobolev priors: common order = {args.prior_order:.2f}")
    if noise_corr_std_mm > 0.0:
        print(
            f"Correlated altimetry error component: std = "
            f"{noise_corr_std_mm:.2f} mm, scale = {noise_corr_scale_km:.0f} km"
        )


def std_pattern(
    points: np.ndarray,
    /,
    *,
    path: Optional[str] = None,
    variable: str = "uncertainty",
    polar_ratio: float = 1.0,
    weights: Optional[np.ndarray] = None,
    floor: float = 0.1,
) -> Optional[np.ndarray]:
    """
    The pattern of a standard deviation at `points`, rows of (latitude,
    longitude): positive values of unit weighted mean square, so that a
    noise amplitude keeps its meaning as a root mean square. It is read from
    the field `variable` of the netCDF file `path`, kept above `floor` times
    its root mean square; or, without a file, falls or rises smoothly from
    the equator to `polar_ratio` at the poles. Returns None for a uniform
    standard deviation.
    """
    if path is None and polar_ratio == 1.0:
        return None
    if weights is None:
        weights = np.ones(len(points))
    if path is not None:
        _, _, interpolators = _netcdf_interpolators(path, variable)
        if len(interpolators) != 1:
            raise ValueError(f"'{variable}' of {path} must be a single field.")
        pattern = interpolators[0](points)
        if np.any(pattern < 0.0):
            raise ValueError(f"'{variable}' of {path} is negative in places.")
    else:
        if polar_ratio <= 0.0:
            raise ValueError("The polar ratio of a standard deviation is positive.")
        pattern = 1.0 + (polar_ratio - 1.0) * np.sin(np.deg2rad(points[:, 0])) ** 2

    def rms(values):
        return np.sqrt(np.sum(weights * values**2) / np.sum(weights))

    pattern = np.maximum(pattern, floor * rms(pattern))
    return pattern / rms(pattern)


def altimetry_noise(
    physics: OceanPhysics,
    prior_order: float,
    /,
    *,
    std_mm: float,
    corr_std_mm: float,
    corr_scale_km: float,
    calib: Optional[dict] = None,
    std_map: Optional[str] = None,
    std_variable: str = "uncertainty",
    global_std_mm: float = 0.0,
    frame_std_mm: float = 0.0,
    parts: Optional[dict] = None,
):
    """
    The altimetry noise, a sum of parts:

    - a local part, independent at the points with std `std_mm`, uniform or
      following the pattern of the field `std_variable` of the file
      `std_map`, scaled to that root mean square over the points;
    - if `corr_std_mm` is positive, a large-scale correlated part, a
      Sobolev field of the prior's order on the fields' space at the scale
      `corr_scale_km` with that pointwise std, evaluated at the points;
    - if `global_std_mm` is positive, an error common to every point, such
      as a bias between missions or of the wet troposphere correction;
    - if `frame_std_mm` is positive, an error in the origin of the reference
      frame the orbits are computed in: a shift of the origin by d changes
      the geocentric height in the direction r by d . r, a degree-1
      pattern, and each component of d has that std.

    All but the first barely average down, and so set a floor on the error
    of GMSL-type averages; that of the correlated field is recorded in
    `calib["noise"]`. The last two have rank four between them.

    Returns (noise, local): the noise and its local part, which is all the
    preconditioners' surrogate sees. If `parts` is given, the measures of
    the other parts are put in it by name ("correlated", "modes"), for a
    preconditioner that corrects for them (`low_rank_correction`).
    """
    data_space = inf.EuclideanSpace(len(physics.points))
    points = np.asarray(physics.points, dtype=float).reshape(-1, 2)
    points = np.column_stack([points[:, 0], points[:, 1] % 360.0])
    pattern = std_pattern(points, path=std_map, variable=std_variable)
    std = std_mm / physics.scale_mm
    if pattern is None:
        local = inf.GaussianMeasure.from_standard_deviation(data_space, std)
    else:
        local = inf.GaussianMeasure.from_standard_deviations(data_space, std * pattern)
    noise = local
    if corr_std_mm > 0.0:
        field_space = physics.field_space
        length_scale = physics.state.model.parameters.length_scale
        field = field_space.sobolev_measure(
            prior_order,
            corr_scale_km * 1000.0 / length_scale,
            pointwise_std=corr_std_mm / physics.scale_mm,
        )
        if calib is not None:
            mean = ocean_average_operator(physics.state, field_space)
            calib.setdefault("noise", {})["alt_corr_gmsl_floor"] = _std(field, mean)
        correlated = field.affine_map(physics.field_point_evaluation)
        if parts is not None:
            parts["correlated"] = correlated
        noise = noise + correlated
    modes = []
    if global_std_mm > 0.0:
        modes.append((global_std_mm / physics.scale_mm) * np.ones(len(points)))
    if frame_std_mm > 0.0:
        lat, lon = np.deg2rad(points[:, 0]), np.deg2rad(points[:, 1])
        for direction in (
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ):
            modes.append((frame_std_mm / physics.scale_mm) * direction)
    if modes:
        amplitudes = inf.EuclideanSpace(len(modes))
        spread = inf.LinearOperator.from_matrix(
            amplitudes, data_space, np.column_stack(modes), form="components"
        )
        common_modes = inf.GaussianMeasure.from_standard_deviation(
            amplitudes, 1.0
        ).affine_map(spread)
        if parts is not None:
            parts["modes"] = common_modes
        noise = noise + common_modes
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

    With --grace-noise-std-mm the noise is not solved against anything: it
    is a Sobolev field of order --grace-noise-order at that scale with that
    pointwise std, and `info["std_factor"]` is its ratio to the std of the
    ice load.

    With --grace-coefficient-errors the noise is that of
    `grace_coefficient_noise`, a full covariance on the coefficients, and
    `info["covariance"]` is its matrix.

    Returns (data_noise, info).
    """
    state = physics.state
    p = state.model.parameters
    field_space = physics.field_space
    if getattr(args, "grace_coefficient_errors", None) is not None:
        noise, info = grace_coefficient_noise(
            state,
            physics.load_space,
            args,
            years=args.grace_coefficient_errors,
            scale=args.grace_error_scale,
            floor_std_mm=args.grace_floor_std_mm,
        )
        # For the tolerance of the solves: the best-determined coefficient's
        # error, as the pointwise std of a white load, over the ice load's.
        info["std_factor"] = info["best_std"] / (p.ice_density * ice_std)
        return noise, info
    ice_scale = field_space.length_scale * args.ice_scale_factor
    noise_scale = args.grace_noise_scale_km * 1000.0 / p.length_scale
    if args.grace_noise_std_mm is not None:
        std = args.grace_noise_std_mm * p.water_density / 1000.0 / p.length_scale
        field = field_space.sobolev_measure(
            args.grace_noise_order, noise_scale, pointwise_std=std
        )
        noise = wmb_method(state, args).load_measure_to_observation_measure(field)
        info = {"noise_std": std, "std_factor": std / (p.ice_density * ice_std)}
        print(
            f"GRACE spatial noise: stated std = {args.grace_noise_std_mm:.2f} mm EWT "
            f"({info['std_factor']:.4f} x the ice load std), scale = "
            f"{args.grace_noise_scale_km:.0f} km, order = {args.grace_noise_order:.2f}"
        )
        return noise, info
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
    noise = wmb_method(state, args).load_measure_to_observation_measure(
        spatial_noise_measure(field_space, info, noise_scale)
    )
    return noise, info


def white_load_variance(
    state: EarthState, load_space, args: argparse.Namespace, std_mm: float = 1.0
) -> np.ndarray:
    """
    The variance of each of GRACE's coefficients under an error that is
    white as a load over the observed degrees: the same variance in every
    coefficient of the load, taken to GRACE's by the load Love numbers,
    with the pointwise std `std_mm` in mm EWT.
    """
    p = state.model.parameters
    # A unit variance in every coefficient of the load (orthonormal on the
    # unit sphere): the eigenvalues carry the metric of the space.
    unit = inf.GaussianMeasure(
        load_space,
        covariance=inf.DiagonalLinearOperator(load_space, load_space.metric_values),
    )
    shape = (
        wmb_method(state, args)
        .load_measure_to_observation_measure(unit)
        .covariance.diagonals()[0]
    )
    # Pointwise variance = the sum of the coefficients' variances / 4 pi.
    std = std_mm * p.water_density / 1000.0 / p.length_scale
    return shape * 4.0 * np.pi * std**2 / len(shape)


def grace_coefficient_noise(
    state: EarthState,
    load_space,
    args: argparse.Namespace,
    /,
    *,
    years: Sequence[int],
    scale: float = 1.0,
    floor_std_mm: float = 0.0,
):
    """
    The GRACE noise of --coefficient-errors and --grace-coefficient-errors:
    the formal error covariance of ITSG-Grace2018's coefficients for the
    difference of the annual means of `years`
    (`grace_covariance.coefficient_covariance`), its standard deviations
    multiplied by `scale`, in the data's convention; plus, if
    `floor_std_mm` is positive, an error that is white as a load over the
    observed degrees with that pointwise std, taken to the coefficients by
    the load Love numbers.

    The covariance is a full matrix on the data, of a few thousand
    coefficients. Returns (noise, info), with the matrix as
    `info["covariance"]` for the preconditioners that assemble GRACE's
    block.
    """
    p = state.model.parameters
    wmb = wmb_method(state, args)
    covariance = grace_covariance.coefficient_covariance(args.obs_degree, tuple(years))
    covariance *= scale**2
    if args.grace_convention != "stokes":
        factor = stokes_coefficient_operator(state.model, args.obs_degree).diagonals()[
            0
        ]
        covariance /= factor[:, None] * factor[None, :]
    # The pointwise std of the formal errors as a load: against a white load
    # of 1 mm, the mean over the coefficients of the ratio of variances.
    white = white_load_variance(state, load_space, args)
    relative = np.diag(covariance) / white
    formal_std, best_std = np.sqrt(np.mean(relative)), np.sqrt(np.min(relative))
    if floor_std_mm > 0.0:
        covariance[np.diag_indices_from(covariance)] += white * floor_std_mm**2
    domain = wmb.load_to_potential_coefficient_operator(load_space).codomain
    noise = inf.GaussianMeasure.from_covariance_matrix(
        domain, covariance, form="galerkin"
    )
    print(
        "GRACE coefficient errors: ITSG-Grace2018, annual means of "
        f"{years[0]} and {years[1]}, std x {scale:.2f}: pointwise "
        f"{formal_std:.1f} mm EWT to degree {args.obs_degree}, unfiltered; "
        f"white load floor {floor_std_mm:.2f} mm"
    )
    load_per_mm = p.water_density / 1000.0 / p.length_scale
    return noise, {
        "covariance": covariance,
        "noise_std": formal_std * load_per_mm,
        "best_std": best_std * load_per_mm,
    }


def grace_noise_model(
    state: EarthState, load_space, response_space, args: argparse.Namespace
):
    """
    The GRACE noise of `add_grace_noise_model_arguments`, a sum of parts:

    - the measurement error, a spatial field of errors in equivalent water
      thickness, correlated over --noise-scale-km, with a standard deviation
      that is uniform, follows the field of --noise-std-map, or varies with
      latitude by --noise-polar-ratio, and has the root mean square
      --noise-std-mm over the sphere. It is taken to GRACE's coefficients
      degree by degree through the load Love numbers, which is how the
      field of a mascon solution is defined from the potential: the field
      stands for the potential, and is a way of giving its errors a spatial
      structure, not an estimate of the load;
    - with --corrections-std-mm, the error of the background corrections
      made to the data, an invariant field of large scale, taken to the
      coefficients in the same way;
    - with --gia-years, the error of the GIA correction, from the ensemble
      of `gia_ensemble.py`.

    With --coefficient-errors the measurement error is instead that of
    `grace_coefficient_noise`, the error covariance of the coefficients
    themselves, a full matrix; the options of the spatial field are then
    not used.

    Returns (noise, invariant, parts): the noise; a stand-in for it with a
    diagonal covariance, for the preconditioner (the measurement error with
    a uniform standard deviation, or the diagonal of the coefficients'
    covariance, and the corrections); and the parts by name.
    """
    p = state.model.parameters
    km = 1000.0 / p.length_scale
    load_per_mm = p.water_density / 1000.0 / p.length_scale
    wmb = wmb_method(state, args)
    pattern = None
    if args.coefficient_errors is not None:
        measurement, info = grace_coefficient_noise(
            state,
            load_space,
            args,
            years=args.coefficient_errors,
            scale=args.error_scale,
        )
        invariant = inf.GaussianMeasure.from_standard_deviations(
            measurement.domain, np.sqrt(np.diag(info["covariance"]))
        )
        parts = {"measurement": measurement}
    else:
        field = load_space.sobolev_measure(
            args.noise_order,
            args.noise_scale_km * km,
            pointwise_std=args.noise_std_mm * load_per_mm,
        )
        invariant = wmb.load_measure_to_observation_measure(field)
        grid = state.model.zero_grid()
        points = _grid_points(grid)
        pattern = std_pattern(
            points,
            path=args.noise_std_map,
            variable=args.noise_std_variable,
            polar_ratio=args.noise_polar_ratio,
            weights=np.cos(np.deg2rad(points[:, 0])),
        )
        parts = {"measurement": invariant}
    if pattern is not None:
        grid.data[:] = pattern.reshape(grid.data.shape)
        to_data = wmb.load_to_potential_coefficient_operator(load_space)
        parts["measurement"] = field.affine_map(
            to_data @ load_space.multiplication_operator(grid)
        )
    if args.corrections_std_mm > 0.0:
        corrections = wmb.load_measure_to_observation_measure(
            load_space.sobolev_measure(
                args.noise_order,
                args.corrections_scale_km * km,
                pointwise_std=args.corrections_std_mm * load_per_mm,
            )
        )
        parts["corrections"] = corrections
        invariant = invariant + corrections
    if args.gia_years > 0.0:
        observe = grace_coefficients(
            response_space, state.model, args.obs_degree, args.grace_convention
        )
        parts["gia"], members = gia_ensemble.correction_error(
            observe, years=args.gia_years, scale=args.gia_scale, lmax=args.gia_lmax
        )
        print(f"GIA correction error from {len(members)} models of the ensemble")
    noise = None
    for part in parts.values():
        noise = part if noise is None else noise + part
    return noise, invariant, parts


def ocean_gia_error(
    physics: OceanPhysics,
    args: argparse.Namespace,
    /,
    *,
    altimetry: bool = True,
    grace: bool = False,
):
    """
    The error of the GIA correction of the ocean experiments' data, from the
    ensemble of `gia_ensemble.py` (--gia-years, --gia-scale, --gia-lmax): a
    measure on the altimetry data, on GRACE's, or, with both, on their
    direct sum in that order, where it is correlated between the two as one
    model's error must be. None if --gia-years is zero.
    """
    if args.gia_years <= 0.0:
        return None
    response_space = physics.fingerprint.codomain
    observations = []
    if altimetry:
        observations.append(
            physics.fingerprint.field_space.point_evaluation_operator(physics.points)
            @ geocentric_sea_level_operator(physics.state, response_space)
        )
    if grace:
        observations.append(
            grace_coefficients(
                response_space,
                physics.state.model,
                args.obs_degree,
                args.grace_convention,
            )
        )
    if not observations:
        raise ValueError("The GIA correction's error needs data to act on.")
    observe = (
        observations[0]
        if len(observations) == 1
        else inf.ColumnLinearOperator(observations)
    )
    measure, members = gia_ensemble.correction_error(
        observe, years=args.gia_years, scale=args.gia_scale, lmax=args.gia_lmax
    )
    print(f"GIA correction error from {len(members)} models of the ensemble")
    return measure


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


def gaspari_cohn(x: np.ndarray) -> np.ndarray:
    """The taper of Gaspari and Cohn (1999, their equation 4.10): a
    positive definite function of distance that falls smoothly from one at
    x = 0 to zero at x = 2 and beyond."""
    x = np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    near, far = x <= 1.0, (x > 1.0) & (x < 2.0)
    a = x[near]
    out[near] = -0.25 * a**5 + 0.5 * a**4 + 0.625 * a**3 - 5.0 / 3.0 * a**2 + 1.0
    b = x[far]
    out[far] = (
        b**5 / 12.0
        - 0.5 * b**4
        + 0.625 * b**3
        + 5.0 / 3.0 * b**2
        - 5.0 * b
        + 4.0
        - 2.0 / (3.0 * b)
    )
    return out


class SparseSolve:
    """A sparse symmetric positive definite matrix and its inverse, by a
    factorisation made on first use. It pickles as the matrix alone."""

    def __init__(self, matrix):
        self.matrix, self._factor = matrix, None

    def __getstate__(self) -> dict:
        return {"matrix": self.matrix}

    def __setstate__(self, state: dict) -> None:
        self.matrix, self._factor = state["matrix"], None

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.matrix @ x

    def solve(self, x: np.ndarray) -> np.ndarray:
        if self._factor is None:
            self._factor = splu(
                self.matrix,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
        return self._factor.solve(x)


def definite(operator):
    """The operator, claimed self-adjoint and positive definite."""
    return operator.with_traits(
        operator.traits | inf.Traits.SELF_ADJOINT | inf.Traits.POSITIVE_DEFINITE
    )


def covariance_matrix_of(measure, matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """The covariance of a measure on a Euclidean space as a matrix: the
    one given, the diagonal of a diagonal covariance, or the covariance
    applied to each basis vector."""
    if matrix is not None:
        return matrix
    covariance = measure.covariance
    if isinstance(covariance, inf.DiagonalLinearOperator):
        return np.diag(covariance.diagonals()[0])
    return covariance.matrix(form="components")


def exact_preconditioner(
    forward, prior, noise, /, *, noise_matrix=None, n_jobs: Optional[int] = None
):
    """
    The inverse of the data-space normal matrix A Q A* + R, assembled and
    factorised: a direct solver, for data few enough that the matrix can be
    held, as GRACE's coefficients are.

    It costs one application of A Q A* per datum, each independent of the
    others, and a dense factorisation. A conjugate-gradient solve
    preconditioned by it converges in an iteration or two, so every further
    solve (a quantity of interest, a row of an estimator kernel, a posterior
    sample) costs one application of the operator and no more.

    Returns the inverse as an operator on the data space.
    """
    space = forward.codomain
    signal = forward @ prior.covariance @ forward.adjoint
    matrix = signal.matrix(form="components", by="rows", n_jobs=n_jobs)
    matrix = 0.5 * (matrix + matrix.T) + covariance_matrix_of(noise, noise_matrix)
    inverse = cho_solve(cho_factor(matrix, lower=True), np.eye(len(matrix)))
    return definite(
        inf.LinearOperator.from_matrix(
            space, space, 0.5 * (inverse + inverse.T), form="components"
        )
    )


def altimetry_block(
    physics: OceanPhysics,
    args: argparse.Namespace,
    calib: dict,
    alt_local,
    cutoff_km: float,
):
    """
    The altimetry block of `block_preconditioner`, as an operator on the
    altimetry data: the inverse of the covariance of eta between the
    points under its prior, tapered to vanish beyond `cutoff_km` by the
    taper of Gaspari and Cohn, which keeps it positive definite, plus the
    local noise. The matrix is sparse and factorised once, so nothing dense
    is formed on the altimetry data, whose number is not bounded.
    """
    space = physics.field_space
    points = physics.points
    spectrum = _field_measure(
        physics, args, ETA, _stds_by_name(calib["derived_stds"])[ETA]
    ).covariance.eigenvalues
    cutoff = cutoff_km * 1000.0 / physics.state.model.parameters.length_scale
    eta = space.invariant_measure(spectrum, unsafe=True)
    rows, columns, distances = space.pairs_within_distance(
        points, cutoff, with_distances=True
    )
    values = space.covariance_function(eta, distances) * gaspari_cohn(
        distances / (0.5 * cutoff)
    )
    count = len(points)
    matrix = sparse.csc_matrix((values, (rows, columns)), shape=(count, count))
    matrix = matrix + sparse.diags(alt_local.covariance.diagonals()[0])
    print(
        f"  altimetry block: {matrix.nnz / count:.0f} entries per point within "
        f"{cutoff_km:.0f} km",
        flush=True,
    )
    solve = SparseSolve(matrix)
    domain = alt_local.domain
    return definite(
        inf.LinearOperator.from_callables(
            domain, domain, solve.solve, adjoint=solve.solve
        )
    )


def block_preconditioner(
    case: str,
    physics: OceanPhysics,
    args: argparse.Namespace,
    prior,
    calib: dict,
    /,
    *,
    alt_local=None,
    alt_parts: Optional[dict] = None,
    grace_noise=None,
    grace_matrix: Optional[np.ndarray] = None,
    gia=None,
    rng=None,
    cache: Optional[dict] = None,
):
    """
    A preconditioner for the data-space normal operator of the ocean
    experiments, assembled in the data space in blocks; `case` is
    "altimetry", "grace" or "joint".

    - GRACE's block is that of `exact_preconditioner`: the normal matrix of
      the true operator, sea level equation included, with GRACE's noise
      (`grace_noise`, with its covariance `grace_matrix` if that is not
      diagonal). GRACE's coefficients number a few thousand, so it can be
      held.
    - The altimetry's block is that of `altimetry_block`, sparse
      (`alt_local`, the local noise; --cutoff-km).
    - With both, the blocks are uncoupled: what the mass sources add to the
      altimetry through the sea level equation, and the covariance between
      the two sets of data, are left to the iteration.
    - The parts of the noise of low rank are then corrected for exactly
      (`low_rank_correction`): the common modes of the altimetry and the
      leading --low-rank modes of its large-scale correlated error
      (`alt_parts`, as `altimetry_noise` fills it), and the error of the GIA
      correction (`gia`, a measure on this case's data).

    The blocks are kept in `cache`, if one is given, so that the cases of
    one experiment share them: GRACE's is the costly one.
    """
    use_alt, use_grace = case != "grace", case != "altimetry"
    cache = {} if cache is None else cache
    blocks = []
    if use_alt:
        if "altimetry" not in cache:
            cache["altimetry"] = altimetry_block(
                physics, args, calib, alt_local, args.cutoff_km
            )
        blocks.append(cache["altimetry"])
    if use_grace:
        if "grace" not in cache:
            print(
                f"  GRACE block: {grace_noise.domain.dim} applications of the "
                "normal operator...",
                flush=True,
            )
            cache["grace"] = exact_preconditioner(
                physics.grace(args),
                prior,
                grace_noise,
                noise_matrix=grace_matrix,
                n_jobs=jobs(args),
            )
        blocks.append(cache["grace"])
    if len(blocks) == 1:
        preconditioner = blocks[0]
        space = preconditioner.domain
    else:
        space = inf.ColumnLinearOperator(
            [physics.altimetry, physics.grace(args)]
        ).codomain
        preconditioner = definite(
            inf.BlockDiagonalLinearOperator(blocks, domain=space, codomain=space)
        )
    factors = []
    if use_alt:
        for name, measure in (alt_parts or {}).items():
            if name == "correlated":
                measure = measure.low_rank_approximation(
                    rank=min(args.low_rank, measure.domain.dim - 1), rng=rng
                )
            factor = measure.covariance_factor
            if case == "joint":
                factor = space.inclusion(0) @ factor
            factors.append(factor)
    if gia is not None:
        factors.append(gia.covariance_factor)
    if factors:
        preconditioner = low_rank_correction(preconditioner, factors)
    return preconditioner


def unresolved_altimetry_variance(
    physics: OceanPhysics, args: argparse.Namespace, dyn_std: float, degree: int
) -> float:
    """
    The variance of eta at a point carried by the degrees above `degree`,
    under its own prior with pointwise std `dyn_std`.

    A surrogate truncated at `degree` cannot produce that part of the
    signal. In the altimetry data it varies over a scale not much greater
    than the spacing of the points, so in the surrogate problem it can be
    treated as further independent noise at each point, which tells the
    preconditioner how far those data can be fitted.
    """
    space = physics.field_space
    spectrum = _field_measure(physics, args, ETA, dyn_std).covariance.eigenvalues
    return float(space.pointwise_variance(spectrum * (space.degrees > degree)))


def unresolved_grace_variance(
    physics: OceanPhysics, args: argparse.Namespace, stds: dict, degree: int
) -> np.ndarray:
    """
    The variance of each of GRACE's coefficients above `degree` under the
    prior, as WMB would have it: the loads of the fields (the mass sources
    and eta) as invariant fields, each weighted by the fraction of the
    sphere its support covers, taken to the coefficients degree by degree.
    Zero at and below `degree`. `stds` are the fields' pointwise stds, keyed
    as calib["derived_stds"].

    A surrogate truncated at `degree` gives no signal in those
    coefficients; this is its stand-in for the signal's variance there, to
    be added to their noise in the surrogate problem.
    """
    state = physics.state
    model = state.model
    p = model.parameters
    load_space = physics.load_space
    sphere = model.integrate(model.constant_grid(1.0))
    ocean = state.ocean_projection(value=0.0)
    weights = {
        ICE: (
            p.ice_density,
            state.ice_sheet_projection(value=0.0) * state.one_minus_ocean_function,
        ),
        LAND: (p.water_density, _land_without_ice_sheets(state)),
        ETA: (p.water_density, ocean),
    }
    by_name = _stds_by_name(stds)
    eigenvalues = 0.0
    for name in physics.setup.names:
        if name not in weights:
            continue
        density, support = weights[name]
        space = physics.setup.component(name).space
        measure = _field_measure(physics, args, name, by_name[name])
        # A coefficient's variance is the eigenvalue over the space's metric;
        # the load space's metric takes it back to an eigenvalue there.
        coefficient = measure.covariance.eigenvalues / space.metric_values
        fraction = model.integrate(support) / sphere
        eigenvalues = eigenvalues + (
            fraction * density**2 * coefficient * load_space.metric_values
        )
    stand_in = inf.GaussianMeasure(
        load_space, covariance=inf.DiagonalLinearOperator(load_space, eigenvalues)
    )
    variance = (
        wmb_method(state, args)
        .load_measure_to_observation_measure(stand_in)
        .covariance.diagonals()[0]
    )
    degrees = load_space.degrees
    observed = degrees[(degrees >= 2) & (degrees <= args.obs_degree)]
    return np.where(observed > degree, variance, 0.0)


def with_added_variance(noise, variance):
    """A measure with a diagonal covariance: that of `noise`, itself
    diagonal, plus `variance` on the diagonal."""
    diagonal = noise.covariance.diagonals()[0] + variance
    return inf.GaussianMeasure.from_standard_deviations(noise.domain, np.sqrt(diagonal))


def low_rank_correction(preconditioner, factors: Sequence, /):
    """
    A preconditioner corrected for terms of low rank in the operator.

    `preconditioner` is M, an approximate inverse of an operator N on the
    data space, and each of `factors` an operator U_i into that space from
    a Euclidean space of low dimension. With U the factors side by side,
    the result is the Woodbury inverse of N + U U* with M standing for the
    inverse of N,

        M - M U (1 + U* M U)^-1 U* M,

    at the cost of one application of M per column of U, made once. It is
    for the parts of the noise that the surrogate problem leaves out and
    that have low rank: the error of the GIA correction, the common modes
    of the altimetry, and a truncation of a large-scale correlated field
    (`GaussianMeasure.low_rank_approximation`).
    """
    space = preconditioner.domain
    columns = []
    for factor in factors:
        source = factor.domain
        columns += [
            space.to_components(factor(source.from_components(e)))
            for e in np.eye(source.dim)
        ]
    count = len(columns)
    coefficients = inf.EuclideanSpace(count)
    spread = inf.LinearOperator.from_matrix(
        coefficients, space, np.column_stack(columns), form="components"
    )
    # M U, column by column, and the small matrix 1 + U* M U.
    applied = inf.LinearOperator.from_matrix(
        coefficients,
        space,
        np.column_stack(
            [
                space.to_components(preconditioner(space.from_components(column)))
                for column in columns
            ]
        ),
        form="components",
    )
    inner = np.eye(count) + np.column_stack(
        [spread.adjoint(applied(e)) for e in np.eye(count)]
    )
    solve = inf.LinearOperator.from_matrix(
        coefficients, coefficients, np.linalg.inv(inner), form="components"
    )
    # M is self-adjoint, so U* M is the adjoint of M U.
    return preconditioner - applied @ solve @ applied.adjoint


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
