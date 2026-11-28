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
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.sparse.linalg import splu

import pygeoinf as inf
import gia_ensemble
import gia_statistics
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
    STERIC_SEA_LEVEL,
    ModelSetup,
    barystatic_gmsl_operator,
    ice_sheets,
    ocean_density,
    sterodynamic_sea_level,
)
from pyslfp.models import LoadComponent, Role

ICE, ETA, RHO = "ice_sheets", "sterodynamic_sea_level", "ocean_density"
# With --ocean-prior steric the ocean's second field is the steric height
# itself, in place of the density.
STERIC = "steric_height"
# Names of fields that are shown and are not components of the model: the
# manometric sea level, the sterodynamic sea level less the steric height,
# and the barystatic sea level, the response of the sea level equation to
# the loads on the land (the fingerprint of those loads).
MANOMETRIC = "manometric_sea_level"
BARYSTATIC = "barystatic_sea_level"

# What a GRACE experiment takes the ocean's load in its data to be
# (--ocean-case). A model of the ocean is removed from GRACE's data before a
# gravity field is solved for, so the ocean term of the load is what that
# removal leaves, and how much it leaves at the scale of a year is not
# known: the published error series cover periods under a month alone. The
# cases bracket it. Each gives the std (mm of water) and the scale (km) of
# the prior's ocean term, the manometric sea level, and what the true loads
# of `grace_coverage.py` take for the ocean: a component of the ESA Earth
# System Model and the share of it that is left.
#
# - removed: the removal is taken as the error series has it (AOe07, which
#   holds no slow error): 0.2 mm, as that series' own year-on-year changes,
#   with the spectrum fitted to them through the ocean's mask
#   (`prior_fit.py`): `REMOVED_TAPERED` with the taper, and 209 km at
#   --manometric-order without it.
# - partly: a quarter of the ocean's load is left: 2 mm, about the unforced
#   variability of the ocean's mass from year to year (Carret et al., 2021),
#   which no model forced by the weather can follow; the scale is the
#   ocean load's own (--manometric-scale-km), which None stands for.
# - full: nothing is removed, as in the altimetry and joint experiments;
#   the prior's ocean term is as given (--manometric-std-mm and
#   --manometric-scale-km).
#
# With --steric-depth-scale-m the ocean's load has a second term, what the
# steric height of deep water leaves to the load over shallow water
# (`steric_depth_share`). The removal takes it as it takes the first: it is
# left in the ratio of the case's std to the full one (`configure`), so
# next to none of it with "removed" and a quarter with "partly".
OCEAN_CASES = {
    "removed": (0.2, 209.0, "AOe07", 1.0),
    "partly": (2.0, None, "O", 0.25),
    "full": (None, None, "O", 1.0),
}
# The scale (km) and order of the spectrum of what the removal leaves in
# the case "removed", as fitted to the error series with the taper
# (--prior-taper-degree); without the taper the scale above at
# --manometric-order.
REMOVED_TAPERED = (529.0, 1.28)

# The fields of the three-field GRACE model beside the ice sheets: the water
# stored on the land that carries no ice sheet, and the ocean's own load, both
# as thicknesses of water.
LAND, OCEAN = "land_water", "ocean"

# The regions of the GRACE experiments, by `EarthState.get_projection` names.
GRACE_REGIONS = {
    "Southern Greenland": ["SE", "SW"],
    "West Antarctica": ["F-G", "Ep-F", "J-Jpp", "G-H", "H-Hp"],
    "Amazon basin": "6030007000",
    "Ganges-Brahmaputra": "4030025450",
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
    ("Steric", STERIC, (30.0, -45.0), "North_Atlantic"),
    ("Manometric", MANOMETRIC, (30.0, -45.0), "North_Atlantic"),
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
        help="Threads of the main process, for the transforms and dense algebra.",
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
        "posterior covariance onto the quantities of interest, the "
        "prior-sensitivity tables and the covariance maps.",
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
        "--altimetry-neighbours",
        type=int,
        default=40,
        metavar="K",
        help="The altimetry block is a sparse approximate inverse, each point "
        "regressed on the K nearest of those before it in a coarse-to-fine "
        "order (`vecchia.py`). With K zero it is the covariance tapered "
        "beyond --cutoff-km, factorised.",
    )
    parser.add_argument(
        "--grace-block-rank",
        type=int,
        default=0,
        metavar="K",
        help="GRACE's block from K applications of the normal operator, a "
        "randomised sketch of its signal part in the metric of the noise, "
        "in place of one application a coefficient. Zero, or K at least the "
        "number of coefficients, for the exact block. The rank wanted is "
        "the number of directions in which the signal exceeds the noise.",
    )
    parser.add_argument(
        "--low-rank",
        type=int,
        default=40,
        help="Modes kept of the large-scale correlated altimetry noise in "
        "the block preconditioner.",
    )


def add_grace_arguments(parser: argparse.ArgumentParser) -> None:
    """The resolution, prior and noise of the GRACE experiments. Their
    unknown is one field, the direct load; its prior is either a Sobolev
    measure of its own or that which the fields of the ocean experiments
    give their load, so the options of those experiments are here too."""
    add_ocean_arguments(parser)
    # The load's order depends on its prior (`configure`), and the altimetry
    # points of the ocean experiments are not used.
    parser.set_defaults(load_order=None, spacing=8.0)
    parser.add_argument(
        "--load-prior",
        choices=("sobolev", "model"),
        default="sobolev",
        help="The prior of the direct load: a Sobolev measure on H^2 "
        "(--direct-scale-km, --direct-std-m), as in the submitted paper; or "
        "the image, under the load operator, of the prior of the fields of "
        "the ocean experiments (the ice, the land water, the sterodynamic "
        "sea level and the ocean density), from which the true load is then "
        "drawn. The inversion is for the load alone in both: it is not "
        "divided into its causes. The second needs --coefficient-errors.",
    )
    parser.add_argument(
        "--ocean-case",
        choices=tuple(OCEAN_CASES),
        default="partly",
        help="With --load-prior model: how much of the ocean's load is in the "
        "data, a model of it being removed before a GRACE gravity field is "
        "solved for. removed: next to none, the removal as the published "
        "error series has it. partly: a quarter, about 2 mm. full: all of "
        "it, nothing being removed. The first two set the prior's ocean term "
        "(--manometric-std-mm, --manometric-scale-km); see "
        "`common.OCEAN_CASES`.",
    )
    parser.add_argument(
        "--obs-degree",
        type=int,
        default=None,
        help="The highest degree of GRACE's data: 100 by default, or 96, the "
        "degree of the ITSG-Grace2018 solutions, with --coefficient-errors, "
        "which allows no more.",
    )
    add_coefficient_error_arguments(parser)
    add_gia_arguments(parser)
    parser.add_argument(
        "--grace-convention",
        choices=("stokes", "orthonormal"),
        default="stokes",
        help="GRACE's data: the dimensionless Stokes coefficients of Level-2 "
        "products, or the potential's coefficients in pyslfp's own convention. "
        "The two differ by a factor per degree, so no result depends on it.",
    )
    parser.add_argument(
        "--smoothing-scale-km",
        type=float,
        default=None,
        help="Smoothing of the region masks; --load-scale-km if omitted.",
    )
    parser.add_argument("--direct-scale-km", type=float, default=250.0)
    parser.add_argument("--direct-std-m", type=float, default=0.01)
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
    parser.add_argument("--lmax", type=int, default=255)
    if surrogate:
        parser.add_argument("--surrogate-degree", type=int, default=32)
    parser.add_argument(
        "--field-order",
        type=float,
        default=None,
        help="Sobolev order of the ocean's fields, above one for the point "
        "values of altimetry: 2 by default, or 1.25, that of the response, "
        "with --ocean-prior steric.",
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
        help="Sobolev order of the space of the ice and land-water fields, no "
        "higher than --field-order; equal to it, the four fields share one "
        "space. If given, those fields have priors of their own "
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
        "--ice-prior-order",
        type=float,
        default=None,
        help="The order of the ice prior's spectrum, if not --source-prior-order.",
    )
    parser.add_argument(
        "--land-water-prior-order",
        type=float,
        default=None,
        help="The same for the land water.",
    )
    parser.add_argument(
        "--prior-taper-degree",
        type=float,
        default=0.0,
        help="With --source-order: the priors of the ice, the land water and "
        "the manometric sea level follow their own spectra (1 + s^2 k)^-q, "
        "with q as fitted to their sources, as far as the sources go, and "
        "beyond this degree L decay faster, by the further factor (1 + k r^2 / "
        "L^2)^-(t - q) with t = --prior-taper-order: the decay a source shows "
        "to its own resolution says nothing of the decay beyond it, and the "
        "fields' space needs t. Zero for no taper.",
    )
    parser.add_argument(
        "--prior-taper-order",
        type=float,
        default=2.5,
        help="With --prior-taper-degree: the order of the spectra well beyond "
        "that degree; it must exceed the order of the fields' space by more "
        "than one.",
    )
    parser.add_argument(
        "--ice-margin-scale-km",
        type=float,
        default=None,
        help="With --ice-modulation margin: the scale at which the nearness to "
        "the edge of the grounded ice is taken; --ice-scale-km if omitted.",
    )
    parser.add_argument(
        "--ice-modulation",
        choices=("none", "margin"),
        default="none",
        help="With --source-order: the ice field's std varies over the ice "
        "sheets (`ice_modulation`). margin: an amplitude for each sheet, "
        "growing towards the edge of its grounded ice.",
    )
    parser.add_argument(
        "--ice-greenland-std-mm",
        type=float,
        default=155.0,
        help="With --ice-modulation margin: the root mean square over "
        "Greenland's grounded ice of the std of the ice load, mm EWT.",
    )
    parser.add_argument(
        "--ice-antarctica-std-mm",
        type=float,
        default=60.0,
        help="The same for Antarctica.",
    )
    parser.add_argument(
        "--ice-margin-growth",
        type=float,
        default=10.0,
        help="With --ice-modulation margin: a in the std's profile 1 + a p, "
        "p the nearness of a point to the edge of the grounded ice.",
    )
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
    parser.add_argument(
        "--ocean-prior",
        choices=("coupled", "steric"),
        default="coupled",
        help="The prior of the two ocean fields. coupled: the sterodynamic "
        "sea level and the density each with a Sobolev prior of its own, "
        "anti-correlated at long wavelengths (the options above). steric: "
        "two independent fields with stated sizes and scales, a steric "
        "height and a manometric sea level (the ocean's load as a height of "
        "water); the sterodynamic sea level is their sum, and the ocean's "
        "second field is the steric height itself, in place of the density.",
    )
    parser.add_argument(
        "--steric-std-mm",
        type=float,
        default=46.0,
        help="With --ocean-prior steric: the pointwise std of the steric "
        "height, mm (EN4: year-on-year changes, 1996 to 2006).",
    )
    parser.add_argument("--steric-scale-km", type=float, default=280.0)
    parser.add_argument(
        "--steric-order",
        type=float,
        default=3.0,
        help="With --ocean-prior steric: the order of the Sobolev spectrum, on "
        "the Lebesgue space, of the steric height; that of EN4's changes.",
    )
    parser.add_argument(
        "--steric-depth-scale-m",
        type=float,
        default=0.0,
        help="With --ocean-prior steric, optionally: the steric height of a "
        "column of depth D is 1 - exp(-D / this) of that of deep water, most "
        "of the change being in the upper ocean (about 500 m in EN4). Where it "
        "falls away over a shelf the sea level stays smooth, so the load takes "
        "up the difference (in the GRACE experiments, as much of it as "
        "--ocean-case leaves). Zero, the default, leaves the steric height "
        "independent of the depth; the paper's runs take 500.",
    )
    parser.add_argument(
        "--manometric-std-mm",
        type=float,
        default=8.0,
        help="With --ocean-prior steric: the pointwise std of the manometric "
        "sea level, mm of water (the ocean component of the ESA Earth System "
        "Model, 1995 to 2006).",
    )
    parser.add_argument(
        "--manometric-scale-km",
        type=float,
        default=330.0,
        help="Its scale, km: that fitted to the same loads at " "--manometric-order.",
    )
    parser.add_argument(
        "--manometric-order",
        type=float,
        default=2.5,
        help="With --ocean-prior steric: the order of the Sobolev spectrum, on "
        "the Lebesgue space, of the manometric sea level. The ocean's load "
        "has a redder spectrum than any order the fields' space allows, so "
        "this is the lowest that does: it must exceed --field-order by more "
        "than one.",
    )


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
    """The error of the GIA correction, from the published statistics of
    `gia_statistics.py` or the ensemble of `gia_ensemble.py`."""
    parser.add_argument(
        "--gia-years",
        type=float,
        default=1.0,
        help="The interval the data span, in years: the error of the GIA "
        "correction is that of its rates times this. One, the interval "
        "between the two annual means, by default; zero leaves the error "
        "out.",
    )
    parser.add_argument(
        "--gia-model",
        choices=["caron", "ensemble"],
        default="caron",
        help="Where the error of the rates comes from: the covariance of "
        "the Stokes coefficients' rates published by Caron et al. (2018), "
        "`gia_statistics.py`, or the spread of the ensemble of "
        "`gia_ensemble.py`.",
    )
    parser.add_argument(
        "--gia-rank",
        type=int,
        default=50,
        help="With --gia-model caron: the eigenvectors of the published "
        "covariance kept. Fifty hold all but a part in 100,000 of its trace.",
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
        "--grace-coefficient-errors the default is 96, the degree of the "
        "ITSG-Grace2018 solutions, and the most it can be; otherwise the "
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
            # No more than the model resolves, for runs at low degree.
            args.obs_degree = min(grace_covariance.DEGREE, args.lmax) if errors else 100
        if errors and args.obs_degree > grace_covariance.DEGREE:
            raise SystemExit(
                f"--obs-degree {args.obs_degree}: the coefficient errors of "
                f"ITSG-Grace2018 stop at degree {grace_covariance.DEGREE}."
            )
    if hasattr(args, "field_order") and args.field_order is None:
        args.field_order = 1.25 if _steric_prior(args) else 2.0
    if getattr(args, "load_prior", None) == "model":
        if not _steric_prior(args):
            # The cases are stated on the prior's manometric sea level.
            args.ocean_case = "full"
        std_mm, scale_km, _, _ = OCEAN_CASES[args.ocean_case]
        if std_mm is not None:
            # The load that the steric height leaves over shallow water is
            # removed as the rest of the ocean's load is.
            args.shelf_load_share = std_mm / args.manometric_std_mm
            args.manometric_std_mm = std_mm
            if scale_km is not None:
                args.manometric_scale_km = scale_km
                if prior_taper(args) is not None:
                    args.manometric_scale_km, args.manometric_order = REMOVED_TAPERED
    if hasattr(args, "load_prior") and args.load_order is None:
        # The masked loads of the fields are rough; a Sobolev load is not.
        args.load_order = 0.25 if args.load_prior == "model" else 2.0
    if hasattr(args, "preconditioner") and args.preconditioner is None:
        errors = getattr(args, "grace_coefficient_errors", None) is not None
        args.preconditioner = "blocks" if errors else "surrogate"
    # With --parallel too: the main process does the dense algebra and the
    # solves between the parallel loops, while the workers, which read their
    # thread count from the environment, are idle.
    if args.serial_threads is not None:
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
    a load to millimetres of equivalent water thickness.

    With --load-prior model the unknown is not the load but the fields of
    the ocean experiments, `fields` (an `OceanPhysics`), which the load
    operator carries to their direct load. The operators below then act on
    the fields, unless asked for as functions of a load."""

    state: EarthState
    load_space: object
    fingerprint: FingerPrintOperator
    scale_mm: float
    fields: Optional[object] = None

    @property
    def response_space(self):
        return self.fingerprint.codomain

    @property
    def load_operator(self):
        """The fields' direct load, or None where the load is the unknown."""
        return None if self.fields is None else self.fields.setup.load_operator

    def of_model(self, operator, of_load: bool = False):
        """An operator on the load space as one on the model space."""
        if self.fields is None or of_load:
            return operator
        return operator @ self.load_operator


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


def add_wmb_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of WMB's estimate as it is made in practice
    (`wmb_estimator`)."""
    parser.add_argument(
        "--wmb-filter",
        choices=("gaussian", "ddk"),
        default="ddk",
        help="The filter of WMB's estimate: DDK, the better of the two on "
        "these data, unless told otherwise.",
    )
    parser.add_argument(
        "--wmb-degree",
        type=int,
        default=None,
        help="The highest degree WMB's estimate uses, --obs-degree at most: "
        "60 by default with the Gaussian filter, and every degree with DDK.",
    )
    parser.add_argument(
        "--wmb-ddk-level",
        type=int,
        default=5,
        help="With --wmb-filter ddk: the filter, 1 (the strongest) to 8.",
    )
    parser.add_argument(
        "--wmb-radius-km",
        type=float,
        default=300.0,
        help="Half-width of the Gaussian filter of WMB's estimate; zero for none.",
    )
    parser.add_argument(
        "--wmb-rescale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale WMB's estimate of each region to be right for a load "
        "uniform over the region.",
    )


def gaussian_weights(degrees, radius: float, earth_radius: float):
    """The weights by degree of the Gaussian filter of half-width `radius`
    (Wahr et al., 1998, after Jekeli, 1981), in the closed form that is
    accurate where the weights are not negligible: exp(-(l r / a)^2 / (4 ln
    2))."""
    return np.exp(-((degrees * radius / earth_radius) ** 2) / (4.0 * np.log(2.0)))


def wmb_estimator(
    state: EarthState,
    load_space,
    args: argparse.Namespace,
    /,
    *,
    average=None,
    sharp_masks=None,
    scale: float = 1.0,
):
    """
    WMB's estimate as it is made in practice: the coefficients cut off at
    --wmb-degree and filtered, taken to a load by the load Love numbers,
    averaged over each region and rescaled.

    The filter (--wmb-filter) is the Gaussian one of Wahr et al. (1998), of
    half-width --wmb-radius-km, on the coefficients to degree 60; or the
    decorrelating filter DDK of Kusche et al. (2009) at the strength
    --wmb-ddk-level (`ddk_filter.py`), on all the coefficients. With
    --wmb-rescale each average is multiplied by the factor that makes it
    right for a load uniform over the region and absent elsewhere, the
    scaling of Velicogna and Wahr (2006): `sharp_masks` are the regions'
    own masks, and `scale` the value such a load of unit height should
    give, the factor that `average` carries.

    Returns (filtered_load, matrix, description): the operator from the
    data to the filtered load, a field, with no averaging or rescaling; the
    matrix of the estimator of the averages on the data's components (None
    without `average`); and a sentence saying what was done.
    """
    wmb = wmb_method(state, args)
    km = 1000.0 / state.model.parameters.length_scale
    wmb_load = wmb.potential_coefficient_to_load_operator(load_space)
    data_space = wmb_load.domain
    degrees, _ = wmb._observed_labels()
    ddk = args.wmb_filter == "ddk"
    wmb_degree = args.wmb_degree or (args.obs_degree if ddk else 60)
    wmb_degree = min(wmb_degree, args.obs_degree)
    weights = (degrees <= wmb_degree).astype(float)
    if not ddk and args.wmb_radius_km > 0.0:
        weights = weights * gaussian_weights(
            degrees, args.wmb_radius_km * km, load_space.radius
        )
    decorrelate = None
    if ddk:
        import ddk_filter

        decorrelate = ddk_filter.filter_matrix(args.wmb_ddk_level, args.obs_degree)

    def filtered(data):
        components = data_space.to_components(data)
        if decorrelate is not None:
            components = decorrelate @ components
        return wmb_load(data_space.from_components(weights * components))

    filtered_load = inf.LinearOperator.from_callables(
        data_space, wmb_load.codomain, filtered
    )
    matrix = None
    if average is not None:
        matrix = (average @ wmb_load).matrix(form="components", by="rows") * weights
        if decorrelate is not None:
            matrix = (decorrelate.T @ matrix.T).T
        if args.wmb_rescale:
            to_data = wmb.load_to_potential_coefficient_operator(load_space)
            uniform = np.array(
                [row @ to_data(mask) for row, mask in zip(matrix, sharp_masks)]
            )
            matrix *= (scale / uniform)[:, None]
    description = (
        f"WMB: degrees to {wmb_degree}, "
        + (
            f"filter DDK{args.wmb_ddk_level}, "
            if ddk
            else f"Gaussian filter of {args.wmb_radius_km:.0f} km, "
        )
        + ("rescaled" if args.wmb_rescale else "not rescaled")
        + "."
    )
    return filtered_load, matrix, description


def grace_forward_operator(
    physics: GracePhysics, args: argparse.Namespace, /, *, of_load: bool = False
):
    """GRACE's coefficients of the potential change of the response to the
    direct load. The fingerprint operator's response already includes the
    ocean load the direct load induces. A function of the model (the fields,
    where they are the unknown), or with `of_load` of a direct load."""
    return physics.of_model(
        grace_coefficients(
            physics.response_space,
            physics.state.model,
            args.obs_degree,
            args.grace_convention,
        )
        @ physics.fingerprint,
        of_load,
    )


def direct_load_operator(physics: GracePhysics):
    """The direct load of the model: the load operator of the fields, or
    the identity where the load is the unknown."""
    return physics.of_model(inf.LinearOperator.identity(physics.load_space))


def total_load_operator(physics: GracePhysics, /, *, of_load: bool = False):
    """The direct load plus the ocean load it induces, rho_w C xi, as a
    function of the model or, with `of_load`, of a direct load. Its values
    on the grid are the products with the masks as they stand, so a value
    at a point is the load there."""
    sea_level = physics.response_space.field_operator("relative_sea_level_change")
    to_load = sea_level_change_to_load_operator(
        physics.state, sea_level.codomain, physics.load_space
    )
    return physics.of_model(
        inf.LinearOperator.identity(physics.load_space)
        + (to_load @ sea_level @ physics.fingerprint),
        of_load,
    )


def load_point_covariance(physics: GracePhysics, measure, component: str, point):
    """With the fields as the unknown: the covariance under `measure` of
    the total load everywhere with the total load at `point`, a point of
    the support of the mass source `component` (the ice or the land water),
    where the total load is the field's value times its density. The
    fields have point values, so no averaging about the point is needed.

    Returns (field, variance): the covariance as a load field, and the
    variance of the total load at the point, in the load's units squared."""
    setup = physics.fields.setup
    p = physics.state.model.parameters
    density = p.ice_density if component == ICE else p.water_density
    space = setup.component(component).space
    test = setup.space.inclusion(component)(space.dirac(point).representer)
    image = measure.covariance(test)
    variance = density**2 * float(setup.space.inner_product(test, image))
    return density * total_load_operator(physics)(image), variance


def wmb_error_operator(
    physics: GracePhysics, average, wmb_average, args: argparse.Namespace, /
):
    """
    WMB's estimate of the regional averages less their true values, as a
    function of the model (the fields, or the direct load where that is the
    unknown): C A - B, with A the forward operator, B = `average` of the
    total load and C = `wmb_average`, WMB's averaging of the data.

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
    return physics.of_model(read @ paired)


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


def grace_region_masks(state: EarthState) -> list:
    """The masks of the GRACE regions (`GRACE_REGIONS`) themselves, sharp."""
    return [state.get_projection(names, value=0.0) for names in GRACE_REGIONS.values()]


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
    info["signal_std"] = std
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


def grace_experiment(args: argparse.Namespace, rng):
    """
    The physics, prior and noise of a GRACE experiment.

    With --load-prior sobolev the unknown is one field, the direct load;
    the prior is that of `grace_measures` and the noise that of
    `grace_data_noise`.

    With --load-prior model the unknown is the fields of the ocean
    experiments (`ocean_physics`) with their prior (`ocean_prior`): one
    model for every experiment. GRACE sees the fields through their load
    alone, so the results are read as loads (`direct_load_operator`,
    `total_load_operator`), and the steric height, which loads nothing, is
    left as the prior has it. The noise is that of
    `grace_coefficient_noise` and, with --gia-years, the error of the GIA
    correction (`gia_correction_error`), and `info["std_factor"]` the
    best-determined coefficient's error, as the pointwise std of a white
    load, over the std of the ice load.

    Returns (physics, initial_prior, prior, noise, noise_matrix, info): a
    `GracePhysics`; the unconditioned Sobolev prior, or None for the
    model's; the prior; the noise and, if it is not diagonal, its
    covariance matrix; and the noise's summary.
    """
    if args.load_prior == "sobolev":
        physics = grace_physics(args.lmax, args.load_order, args.load_scale_km)
        initial, prior, noise_load, info = grace_measures(physics, args, rng)
        noise, matrix = grace_data_noise(physics, args, noise_load, info)
        return physics, initial, prior, noise, matrix, info
    if args.coefficient_errors is None:
        raise SystemExit("--load-prior model needs --coefficient-errors.")
    if args.remove_degree_1:
        raise SystemExit("--remove-degree-1 is for --load-prior sobolev.")
    ocean = ocean_physics(args)
    fields_prior, calib = ocean_prior(ocean, args, rng)
    print_calibration_report(calib, ocean)
    p = ocean.state.model.parameters
    physics = GracePhysics(
        ocean.state,
        ocean.load_space,
        ocean.fingerprint,
        1000.0 * p.length_scale / p.water_density,
        fields=ocean,
    )
    prior = fields_prior
    noise, coefficient_info = grace_coefficient_noise(
        physics.state,
        physics.load_space,
        args,
        years=args.coefficient_errors,
        scale=args.error_scale,
    )
    covariance = coefficient_info["covariance"]
    if args.gia_years > 0.0:
        # The correction's error is of low rank on the coefficients; it is
        # added to their covariance, which is held as a matrix already.
        observe = grace_coefficients(
            physics.response_space,
            physics.state.model,
            args.obs_degree,
            args.grace_convention,
        )
        factor = gia_correction_error(
            physics.state.model, observe, args, grace=True
        ).covariance_factor
        columns = np.array(
            [
                factor(factor.domain.basis_vector(index))
                for index in range(factor.domain.dim)
            ]
        ).T
        covariance = covariance + columns @ columns.T
        noise = inf.GaussianMeasure.from_covariance_matrix(
            noise.domain, covariance, form="galerkin"
        )
    signal_std = p.ice_density * calib["derived_stds"]["ice_std"]
    info = {
        "signal_std": signal_std,
        "noise_std": coefficient_info["noise_std"],
        "std_factor": coefficient_info["best_std"] / signal_std,
        "independent": None,
    }
    if calib["independent"] is not None:
        # The independent fields the prior is built from.
        info["independent"] = calib["independent"]
    return physics, None, prior, noise, covariance, info


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


def tapered_spectrum(k, scale: float, order: float, radius: float, taper=None):
    """The spectrum (1 + s^2 k)^-q of a prior in the Laplacian's eigenvalue
    k, and with `taper` = (L, t) the further factor (1 + k r^2 / L^2)^-(t - q)
    that takes its order to t beyond the degree L (--prior-taper-degree),
    where q is below t."""
    values = (1.0 + scale**2 * k) ** (-order)
    if taper is not None and taper[0] > 0.0 and order < taper[1]:
        values = values * (1.0 + k * radius**2 / taper[0] ** 2) ** (order - taper[1])
    return values


def prior_taper(args: argparse.Namespace):
    """(L, t) of --prior-taper-degree and --prior-taper-order, or None."""
    degree = getattr(args, "prior_taper_degree", 0.0)
    return (degree, args.prior_taper_order) if degree > 0.0 else None


def _source_order(args: argparse.Namespace, name) -> float:
    """The order of a mass source's own spectrum."""
    own = args.ice_prior_order if name == ICE else args.land_water_prior_order
    return args.source_prior_order if own is None else own


def _band_power(
    order: float, scale: float, radius: float, lmax: int, taper=None
) -> float:
    degrees = np.arange(lmax + 1, dtype=float)
    k = degrees * (degrees + 1.0) / radius**2
    return float(
        np.sum(
            (2.0 * degrees + 1.0) * tapered_spectrum(k, scale, order, radius, taper)
        )
    )


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
    stated_sources: bool

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
        """Whether the mass sources have priors of their own, stated by
        amplitude and scale (--source-order). Their space is then of that
        order: the fields' own space if the orders agree, as in the paper,
        or a rougher one."""
        return self.stated_sources

    @property
    def load_space(self):
        return self.setup.load_space

    @property
    def display_scales(self) -> dict:
        """The factors taking each field to its display unit: mm for the ice
        thickness change, the land water and the heights of the ocean, g
        m^-3 for delta rho."""
        density = 1.0e3 * self.state.model.parameters.density_scale
        scales = {ICE: self.scale_mm, LAND: self.scale_mm, ETA: self.scale_mm}
        scales |= {MANOMETRIC: self.scale_mm, BARYSTATIC: self.scale_mm}
        return scales | {RHO: density, STERIC: self.scale_mm}

    @property
    def shows_manometric(self) -> bool:
        """Whether the ocean's fields are shown as the steric height and the
        manometric sea level: where the steric height is a field of the
        model (--ocean-prior steric), these are the two independent fields
        of the prior, and the sterodynamic sea level is their sum."""
        return STERIC in self.setup.names

    @property
    def outputs(self) -> inf.LinearOperator:
        """The model as it is shown: with `shows_manometric` the
        sterodynamic sea level gives up its place to the manometric sea
        level, eta less the steric height; the identity otherwise. The
        posterior is the same in either set of fields, so this is applied
        to what the inversion returns and the model is not re-posed."""
        space = self.setup.space
        if not self.shows_manometric:
            return inf.LinearOperator.identity(space)
        names = list(self.setup.names)
        i_eta, i_steric = names.index(ETA), names.index(STERIC)
        field_space = self.field_space

        def mapping(m):
            m = list(m)
            m[i_eta] = field_space.subtract(m[i_eta], m[i_steric])
            return tuple(m)

        def adjoint(y):
            y = list(y)
            y[i_steric] = field_space.subtract(y[i_steric], y[i_eta])
            return tuple(y)

        return inf.LinearOperator.from_callables(space, space, mapping, adjoint=adjoint)

    @property
    def output_names(self) -> list:
        """The names of the fields of `outputs`, in the model's order."""
        return [
            MANOMETRIC if name == ETA and self.shows_manometric else name
            for name in self.setup.names
        ]

    def display(self, model, /, *, outputs: bool = False) -> dict:
        """The fields of a model vector as they are shown (`outputs`), on
        their supports and in display units, by name. The fields are global,
        but only their restrictions to the supports are physical, so only
        those are shown. With `outputs` the vector is one of `outputs`
        already, such as a standard deviation taken there."""
        scales = self.display_scales
        shown = model if outputs else self.outputs(model)
        return {
            name: field * scales[name]
            for name, field in zip(self.output_names, self.setup.support(shown))
        }

    def point_covariance(self, measure, component: str, point) -> dict:
        """The covariance under `measure`, a measure on the model, of each
        field shown with the field `component` at `point` (latitude,
        longitude): the covariance of the outputs applied to the representer
        of that point evaluation. By name (`output_names`), on the supports
        and in the product of the two fields' display units. The fields have
        point values where their space's order exceeds one; a field of a
        rougher space is averaged over a cap of `COVARIANCE_CAP_DEGREES`
        about the point instead."""
        slot = self.setup.names[self.output_names.index(component)]
        space = self.setup.component(slot).space
        if space.order > 1.0:
            dirac = space.dirac(point).representer
        else:
            dirac = space.spherical_cap_average(
                point, COVARIANCE_CAP_DEGREES
            ).representer
        outputs = self.outputs
        test = outputs.adjoint(self.setup.space.inclusion(slot)(dirac))
        scale = self.display_scales[component]
        return {
            name: field * scale
            for name, field in self.display(measure.covariance(test)).items()
        }

    @property
    def barystatic_sea_level(self) -> inf.LinearOperator:
        """The relative sea level change that the sea level equation gives
        for the loads of the mass sources alone, the ice and the water on
        the land: their fingerprint, the barystatic part of the manometric
        sea level. From the model into the response's field space."""
        space = self.setup.space
        sources = None
        for name in self.sources:
            part = space.inclusion(name) @ space.projection(name)
            sources = part if sources is None else sources + part
        sea_level = self.fingerprint.codomain.field_operator("relative_sea_level_change")
        return sea_level @ self.fingerprint @ self.setup.load_operator @ sources

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
        return self._grace_observation(args) @ self.fingerprint @ self._loads

    def _grace_observation(self, args: argparse.Namespace) -> inf.LinearOperator:
        """GRACE's coefficients from the response of the sea level equation,
        padded as `grace` says."""
        degree = min(args.obs_degree, self.fingerprint.codomain.field_space.lmax)
        operator = grace_coefficients(
            self.fingerprint.codomain,
            self.state.model,
            degree,
            args.grace_convention,
        )
        if degree < args.obs_degree:
            data_space = inf.EuclideanSpace((args.obs_degree + 1) ** 2 - 4)
            read = data_space.coordinate_selection(np.arange(operator.codomain.dim))
            operator = read.adjoint @ operator
        return operator

    def joint(self, args: argparse.Namespace) -> inf.LinearOperator:
        """The altimetry and GRACE's coefficients together: the same
        operator as the column of `altimetry` over `grace`, on the same
        spaces, with the sea level equation solved once for the two and not
        once for each. Both sets of data are read from one response, so an
        application costs one solve and its adjoint one adjoint solve, which
        is most of what the column's second block costs."""
        column = inf.ColumnLinearOperator([self.altimetry, self.grace(args)])
        fingerprint = self.fingerprint
        respond = fingerprint @ self._loads
        geocentric = geocentric_sea_level_operator(self.state, fingerprint.codomain)
        direct = self.setup.observables[GEOCENTRIC_SEA_LEVEL]
        evaluate = fingerprint.field_space.point_evaluation_operator(self.points)
        observe = self._grace_observation(args)
        surface, response, model = (
            geocentric.codomain,
            fingerprint.codomain,
            column.domain,
        )

        def mapping(m):
            r = respond(m)
            return (evaluate(surface.add(geocentric(r), direct(m))), observe(r))

        def adjoint(d):
            y = evaluate.adjoint(d[0])
            s = response.add(geocentric.adjoint(y), observe.adjoint(d[1]))
            return model.add(respond.adjoint(s), direct.adjoint(y))

        return inf.LinearOperator.from_callables(
            column.domain, column.codomain, mapping, adjoint=adjoint
        )

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
    if rough and args.source_order > args.field_order:
        raise ValueError("--source-order must not exceed --field-order.")
    # With the orders equal the four fields share one space, and the edges
    # of the loads come from the load operator alone.
    source_space = (
        sobolev_load_space(state.model, args.source_order, scale)
        if rough and args.source_order != args.field_order
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
        (
            steric_height(state, load_space, space=field_space)
            if _steric_prior(args)
            else ocean_density(state, load_space, space=field_space)
        ),
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
        stated_sources=rough,
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
        _band_power(
            _source_order(args, name), scale, space.radius, space.lmax, prior_taper(args)
        )
        / _band_power(
            _source_order(args, name),
            scale,
            space.radius,
            args.source_reference_degree,
            prior_taper(args),
        )
    )
    if name == ICE:
        # Given as the load in mm of water; the field is a thickness of ice.
        std = args.ice_std_mm * p.water_density / p.ice_density
    else:
        std = args.land_water_std_mm
    return continued * std / physics.scale_mm


def _steric_prior(args: argparse.Namespace) -> bool:
    return getattr(args, "ocean_prior", "coupled") == "steric"


def sea_level_std_mm(args: argparse.Namespace) -> float:
    """The pointwise std of the sterodynamic sea level under its prior, mm:
    what the altimetry noise is set against in the tolerance of a solve."""
    if _steric_prior(args):
        return float(np.hypot(args.steric_std_mm, args.manometric_std_mm))
    return args.ocean_dyn_std_mm


def _steric_measures(physics: OceanPhysics, args: argparse.Namespace):
    """With --ocean-prior steric: the priors of the steric height of deep
    water and of the manometric sea level, as heights in the model's units.
    Each has the spectrum of a Sobolev measure on the Lebesgue space, (1 +
    s^2 k)^-t, of order t = --ocean-prior-order and its own scale s and
    pointwise std; the space's own weight is put back, as for the mass
    sources in `_field_measure`."""
    space = physics.field_space
    km = 1000.0 / physics.state.model.parameters.length_scale

    def measure(scale_km, std_mm, order, taper=None):
        final = order if taper is None else max(order, taper[1])
        if final - space.order <= 1.0:
            raise ValueError(
                "--steric-order and --manometric-order (or --prior-taper-order) "
                "must exceed --field-order by more than one."
            )
        scale = scale_km * km

        def spectrum(k):
            return tapered_spectrum(k, scale, order, space.radius, taper) * (
                1.0 + space.length_scale**2 * k
            ) ** (space.order)

        return space.invariant_measure(
            spectrum, pointwise_std=std_mm / physics.scale_mm
        )

    return (
        measure(args.steric_scale_km, args.steric_std_mm, args.steric_order),
        measure(
            args.manometric_scale_km,
            args.manometric_std_mm,
            args.manometric_order,
            prior_taper(args),
        ),
    )


def steric_height(state: EarthState, load_space, /, *, space) -> LoadComponent:
    """
    A steric height change s over the oceans, as a field of its own: the
    expansion of the water column, which raises the sea surface and moves
    no mass. It stands where pyslfp's `ocean_density` does, with the height
    in place of the density that gives it, so that the depth of the water
    does not enter. With a sterodynamic sea level eta the two load the
    ocean by rho_w C (eta - s), the part of the sea level change that the
    expansion does not explain, so the load here is -rho_w C s; the steric
    sea level is C s. The support is the ocean.
    """
    to_water = sea_level_change_to_load_operator(state, space, load_space)
    mask = space.multiplication_operator(state.ocean_projection(value=0.0))
    return LoadComponent(
        name=STERIC,
        space=space,
        to_load=-1.0 * to_water,
        support=mask.with_traits(mask.traits | inf.Traits.IDEMPOTENT),
        role=Role.OCEAN_INTERNAL,
        observables={
            STERIC_SEA_LEVEL: (1.0 / state.model.parameters.water_density) * to_water
        },
    )


def steric_depth_share(physics: OceanPhysics, args: argparse.Namespace):
    """With --ocean-prior steric: the share of deep water's steric height
    that a column of depth D has, w(D) = 1 - exp(-D / D_0), as an operator
    on the fields' space; the identity if --steric-depth-scale-m is zero.
    The product with w is truncated to the degrees of the space, as every
    field is.

    What the share takes from the steric height, 1 - w of deep water's, is
    left to the load. In the GRACE experiments in which a model of the
    ocean is removed from the data (`OCEAN_CASES`) only the part f of that
    load is left, f being `args.shelf_load_share` (set by `configure`; one
    where it is not set), and the share is then 1 - f (1 - w): the steric
    height itself is not seen by GRACE, so only the load matters there."""
    space = physics.field_space
    if args.steric_depth_scale_m <= 0.0:
        return inf.LinearOperator.identity(space)
    state = physics.state
    depth = (state.sea_level * state.ocean_function).data
    scale = args.steric_depth_scale_m / state.model.parameters.length_scale
    left = float(getattr(args, "shelf_load_share", 1.0))
    share = state.ocean_function.copy()
    share.data[:] = 1.0 - left * np.exp(-np.maximum(depth, 0.0) / scale)
    flat = underlying_space(space)

    def mapping(u):
        return flat.from_components(flat.to_components(share * u))

    return inf.LinearOperator.from_formal_adjoint(
        space,
        space,
        inf.LinearOperator.from_callables(flat, flat, mapping, adjoint=mapping),
    )


def ice_modulation(physics: OceanPhysics, args: argparse.Namespace):
    """
    With --ice-modulation margin: the positive function w by which the ice
    field's invariant prior is multiplied, and the multiplication as an
    operator on the ice's space; None without. The ice is then w u with u
    the invariant field of `_field_measure`, a Gaussian field on the same
    space whose pointwise std is w times u's, and whose correlation is u's.

    Ice sheets change most near their edges, and Greenland more than
    Antarctica. So over the grounded ice of each sheet the std follows the
    profile 1 + a p, with a = --ice-margin-growth and p the nearness of a
    point to the edge of the grounded ice: the fraction of the surface that
    is not grounded ice sheet within --ice-margin-scale-km of the point
    (the heat kernel at that scale applied to the complement; the ice
    prior's own scale, --ice-scale-km, if it is not given), nought
    deep in the interior and about a half at the edge. The profile is
    scaled so that its root mean square over the sheet's grounded ice is
    the sheet's own std, --ice-greenland-std-mm or --ice-antarctica-std-mm.

    w is one smooth function over the whole sphere, so that the product
    stays in the field's space: the profile, which is smooth everywhere,
    times an amplitude that is Antarctica's constant south of 30 S and
    Greenland's north of 30 N and passes smoothly from one to the other
    between. Off the grounded ice the load operator discards the field, so
    the values of w there have no effect.

    The numbers are those of the year-on-year changes of the land ice of the
    ESA Earth System Model, 1995 to 2006: root mean square 156 mm EWT over
    Greenland's grounded ice and 61 over Antarctica's, and the growth a from
    a least-squares fit of the profile to the root mean square at each
    point, made with the scale of the prior (`prior_fit.py`).

    The product with w is truncated to the degrees of the space, as every
    field is. Returns (w, operator).
    """
    if getattr(args, "ice_modulation", "none") == "none":
        return None
    if not physics.rough_sources:
        raise ValueError("--ice-modulation needs --source-order.")
    state, space = physics.state, physics.source_space
    model = state.model
    margin_km = getattr(args, "ice_margin_scale_km", None)
    margin = (
        _source_scale(physics, args, ICE)
        if margin_km is None
        else margin_km * 1000.0 / state.model.parameters.length_scale
    )
    smoother = space.heat_measure(margin).covariance
    grounded_all = state.ice_sheet_projection(value=0.0, exclude_ice_shelves=True)
    nearness = np.clip(smoother(1.0 - grounded_all).data, 0.0, 1.0)
    profile = 1.0 + args.ice_margin_growth * nearness
    weight = model.constant_grid(1.0)
    stds = {
        "greenland": args.ice_greenland_std_mm,
        "antarctic": args.ice_antarctica_std_mm,
    }
    factors = {}
    for sheet, std_mm in stds.items():
        grounded = state.ice_sheet_projection(
            value=0.0, sheets=sheet, exclude_ice_shelves=True
        )
        inside = grounded.data > 0.5
        factors[sheet] = 1.0
        if not inside.any():
            continue
        shape = grounded.copy()
        shape.data[:] = np.where(inside, profile, 0.0)
        rms = np.sqrt(model.integrate(shape * shape) / model.integrate(grounded))
        factors[sheet] = std_mm / args.ice_std_mm / rms
    # The amplitude: a smooth step in latitude, flat beyond 30 degrees.
    t = np.clip((np.asarray(weight.lats(), dtype=float) + 30.0) / 60.0, 0.0, 1.0)
    with np.errstate(divide="ignore", over="ignore"):
        rise, fall = np.exp(-1.0 / t), np.exp(-1.0 / (1.0 - t))
    step = rise / (rise + fall)
    amplitude = factors["antarctic"] + (
        factors["greenland"] - factors["antarctic"]
    ) * step
    weight.data[:] = amplitude[:, None] * profile
    flat = underlying_space(space)

    def mapping(u):
        return flat.from_components(flat.to_components(weight * u))

    operator = inf.LinearOperator.from_formal_adjoint(
        space,
        space,
        inf.LinearOperator.from_callables(flat, flat, mapping, adjoint=mapping),
    )
    return weight, operator


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
        order = _source_order(args, name)
        taper = prior_taper(args)
        if (order if taper is None else max(order, taper[1])) - space.order <= 1.0:
            raise ValueError(
                "--source-prior-order (or --prior-taper-order) must exceed "
                "--source-order by more than one."
            )
        scale = _source_scale(physics, args, name)

        def spectrum(k):
            return tapered_spectrum(k, scale, order, space.radius, taper) * (
                1.0 + space.length_scale**2 * k
            ) ** (space.order)

        return space.invariant_measure(spectrum, pointwise_std=std)
    if name in (ETA, STERIC) and _steric_prior(args):
        # The fields' own invariant parts, for the preconditioners: eta is
        # the sum of the two heights; `std` is not used.
        steric, manometric = _steric_measures(physics, args)
        if name == STERIC:
            return steric
        return physics.field_space.invariant_measure(
            steric.covariance.eigenvalues + manometric.covariance.eigenvalues
        )
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
    # The steric height's prior does not take its std from here.
    by_name[STERIC] = None
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


def ocean_prior(
    physics: OceanPhysics,
    args: argparse.Namespace,
    rng,
    /,
    *,
    reference: Optional[dict] = None,
):
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

    With `reference`, the covariance eigenvalues by name of the
    independent fields of the same prior on physics of a higher degree
    (the `measures` of its calib["independent"]), each independent field
    takes those at the degrees it holds and not its stated pointwise std:
    the prior is then the reference's truncated, where the stated std on
    a lower degree would push the variance of the lost degrees onto those
    kept. For --ocean-prior steric alone.

    Returns (prior, calib): the prior and the calibration record.
    """
    if reference is not None and not _steric_prior(args):
        raise NotImplementedError(
            "A reference prior is for --ocean-prior steric alone."
        )
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
    km = 1000.0 / p.length_scale
    # The independent fields the prior is built from, where it is built from
    # such, for `sensitivity_table`: set by `correlated`.
    parts = {}

    def correlated(stds):
        parts.clear()
        if _steric_prior(args):
            # Independent fields: the mass sources, the steric height of
            # deep water in eta's place and the manometric sea level in the
            # steric height's. Eta is then the two heights' sum, and the
            # steric height that of deep water times the share a column of
            # its depth has.
            if not rough:
                raise ValueError("--ocean-prior steric needs --source-order.")
            steric, manometric = _steric_measures(physics, args)
            measures = {
                n: _field_measure(physics, args, n, stds[n])
                for n in names
                if n not in (ETA, STERIC)
            } | {ETA: steric, STERIC: manometric}
            if reference is not None:
                # The components are ordered by degree: the shared lead.
                measures = {
                    n: setup.component(n).space.invariant_measure(
                        reference[n][: setup.component(n).space.dim]
                    )
                    for n in names
                }
            independent = inf.GaussianMeasure.from_product(
                [measures[n] for n in names], domain=setup.space
            )
            share = steric_depth_share(physics, args)
            whole = args.steric_depth_scale_m <= 0.0
            i_eta, i_steric = names.index(ETA), names.index(STERIC)

            # The mixing, written on the fields themselves: composed from
            # the inclusions and projections of the direct sum it is the
            # same operator at many times the cost, and it is applied twice
            # in every action of the prior covariance. The two fields lie in
            # one space, so its adjoint is the transpose with the share's
            # adjoint.
            def mix(u):
                m = list(u)
                m[i_eta] = field_space.add(u[i_eta], u[i_steric])
                m[i_steric] = u[i_eta] if whole else share(u[i_eta])
                return tuple(m)

            def mix_adjoint(y):
                u = list(y)
                shared = y[i_steric] if whole else share.adjoint(y[i_steric])
                u[i_eta] = field_space.add(y[i_eta], shared)
                u[i_steric] = y[i_eta]
                return tuple(u)

            mixing = inf.LinearOperator.from_callables(
                setup.space, setup.space, mix, adjoint=mix_adjoint
            )
            parts.update(
                measures=measures,
                mixing=mixing,
                product=independent,
                spectral=whole,
                labels={
                    ICE: "ice",
                    LAND: "land water",
                    ETA: "steric height",
                    STERIC: "manometric",
                },
                spectra={
                    n: (
                        _source_scale(physics, args, n),
                        _source_order(args, n),
                        prior_taper(args),
                    )
                    for n in names
                    if n in (ICE, LAND)
                }
                | {
                    ETA: (args.steric_scale_km * km, args.steric_order, None),
                    STERIC: (
                        args.manometric_scale_km * km,
                        args.manometric_order,
                        prior_taper(args),
                    ),
                },
            )
            return independent.affine_map(mixing)
        measures = {n: _field_measure(physics, args, n, stds[n]) for n in names}
        if args.ocean_corr == 0.0:
            parts.update(
                measures=measures, mixing=inf.LinearOperator.identity(setup.space)
            )
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
        parts.update(
            measures=measures,
            mixing=mixing,
            labels={RHO: "density remainder"},
        )
        return independent.affine_map(mixing)

    def conditioned(measure_):
        constraint = setup.constraints[OCEAN_MASS]
        # The mixed measure is positive definite as its factors are, though
        # the algebra cannot tell; the conditioning is told so.
        return measure_.condition(
            constraint, np.zeros(constraint.codomain.dim), definite=rough
        )

    effective_steric_scale = state.mean_ocean_depth / p.water_density
    if _steric_prior(args):
        # For the record alone: eta's std is that of the two heights' sum,
        # and the density's that of the steric height over the mean depth.
        dyn_std = np.hypot(args.steric_std_mm, args.manometric_std_mm)
        dyn_std /= physics.scale_mm
        rho_std = args.steric_std_mm / physics.scale_mm / effective_steric_scale
    else:
        dyn_std = args.ocean_dyn_std_mm / physics.scale_mm
        rho_std = args.steric_dyn_std_ratio * dyn_std / effective_steric_scale
    stds = {ETA: dyn_std, RHO: rho_std, STERIC: None}
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

    unconditioned = correlated(stds)
    record = _independent_fields(setup, names, parts, unconditioned)
    if record is not None and record["product"] is not None:
        # The same law as the conditioned measure, as the image of the
        # independent fields under the mixing and the projection: its
        # covariance then costs a quarter of what the conditioning's does.
        prior = record["product"].affine_map(record["map"])
        covariance = (
            _steric_covariance(setup, names, parts["measures"])
            if parts.get("spectral")
            else None
        )
        if covariance is not None:
            prior = inf.GaussianMeasure(
                setup.space,
                covariance=covariance,
                covariance_factor=prior.covariance_factor,
            )
    else:
        prior = conditioned(unconditioned)
    ice_barystatic = q_ice * ice_std
    modulation = ice_modulation(physics, args)
    if modulation is not None:
        # The ice is independent of the other fields and absent from the
        # constraint, so its modulation follows everything above: the prior
        # is the image of the unmodulated one under w on the ice alone.
        _, on_ice = modulation
        i_ice = names.index(ICE)

        def on_model(operator):
            def mapping(m):
                m = list(m)
                m[i_ice] = operator(m[i_ice])
                return tuple(m)

            return mapping

        modulate = inf.LinearOperator.from_callables(
            setup.space,
            setup.space,
            on_model(on_ice),
            adjoint=on_model(on_ice.adjoint),
        )
        constraint = setup.constraints[OCEAN_MASS]
        for index in range(constraint.codomain.dim):
            representer = constraint.adjoint(constraint.codomain.basis_vector(index))
            if np.any(representer[i_ice].data != 0.0):
                raise ValueError("--ice-modulation: the constraint involves the ice.")
        prior = prior.affine_map(modulate)
        if record is not None:
            record = record | {"map": modulate @ record["map"]}
        ice_barystatic = _std(
            _field_measure(physics, args, ICE, ice_std).affine_map(on_ice),
            barystatic_gmsl_operator(state, setup.component(ICE)),
        )
        print(
            f"Ice prior modulated: std {args.ice_greenland_std_mm:.0f} mm EWT over "
            f"Greenland's grounded ice and {args.ice_antarctica_std_mm:.0f} over "
            f"Antarctica's, growing towards the edge by 1 + "
            f"{args.ice_margin_growth:g} p"
        )
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
            "ice_barystatic_std": ice_barystatic,
            "steric_std": sigma_s,
            "eta_mean_std_unconstrained": a,
            "bary_steric_ratio": barystatic / sigma_s,
            "steric_std_check": _std(prior, setup.steric_gmsl),
            "barystatic_std_check": barystatic,
        },
        "derived_stds": derived,
        "independent": record,
    }
    return shifted(prior, args.prior_shift, rng), calib


def _steric_covariance(setup, names, measures):
    """The covariance of the steric prior with the whole column's share, P M
    Q0 M* P* in the notation of `_independent_fields`, applied in
    components; None where the constraint is not one number.

    Q0 is diagonal in the harmonics, the mixing M adds and exchanges two
    fields of one space, and the projection P is a change of rank one, so
    between the analysis of the argument and the synthesis of the result
    everything is arithmetic on the coefficients: one transform each way a
    field. Composed from the operators the same covariance makes six times
    as many, most of them in the two inner products of the projection, and
    it is applied in every iteration of every solve."""
    constraint = setup.constraints[OCEAN_MASS]
    if constraint.codomain.dim != 1:
        return None
    space = setup.space
    edges = np.concatenate(
        [[0], np.cumsum([setup.component(n).space.dim for n in names])]
    )
    eta, steric = (
        slice(edges[i], edges[i + 1]) for i in (names.index(ETA), names.index(STERIC))
    )
    spectrum = np.concatenate([measures[n].covariance.eigenvalues for n in names])

    def unconditioned(y):
        # M Q0 M*: the mixing and its adjoint are the same exchange.
        u = y.copy()
        u[eta] = y[eta] + y[steric]
        u[steric] = y[eta]
        u *= spectrum
        m = u.copy()
        m[eta] = u[eta] + u[steric]
        m[steric] = u[eta]
        return m

    representer = space.to_components(
        constraint.adjoint(constraint.codomain.basis_vector(0))
    )
    # Inner products with the representer and the gain, as dot products
    # with their Gram images.
    against_representer = space.apply_gram(representer)
    gain = unconditioned(representer)
    gain /= float(against_representer @ gain)
    against_gain = space.apply_gram(gain)

    def covariance(y):
        c = space.to_components(y)
        c = unconditioned(c - representer * float(against_gain @ c))
        return space.from_components(c - gain * float(against_representer @ c))

    operator = inf.LinearOperator.from_callables(
        space, space, covariance, adjoint=covariance
    )
    return operator.with_traits(
        operator.traits | inf.Traits.SELF_ADJOINT | inf.Traits.POSITIVE_SEMIDEFINITE
    )


def _independent_fields(setup, names, parts: dict, unconditioned) -> Optional[dict]:
    """The independent fields a prior of `ocean_prior` is built from, for
    `sensitivity_table`; None for a prior that is not built from such.

    The prior is the law of P M u: u the independent fields, each with an
    invariant measure of its own (`measures`, by name); M their mixing into
    the fields of the model; and P the projection that the conditioning on
    the conservation of the ocean's mass, C m = 0, amounts to, P = 1 - Q C*
    (C Q C*)^-1 C with Q the covariance of M u. `map` is P M. The constraint
    is one number, so P is a change of rank one, m - g (r, m) with r the
    constraint's representer and g = Q r / (r, Q r), and is applied as such.
    `product` is the product measure of the independent fields, where the
    prior is to be built as its image under the map, or None. `spectra`
    gives, where the measures are stated so, the scale and order of each
    one's spectrum (1 + s^2 k)^-t on the Lebesgue space, and `labels` the
    names of the independent fields, which need not be those of the slots
    they fill."""
    if not parts:
        return None
    constraint = setup.constraints[OCEAN_MASS]
    covariance = unconditioned.covariance
    space = setup.space
    if constraint.codomain.dim == 1:
        representer = constraint.adjoint(constraint.codomain.basis_vector(0))
        image = covariance(representer)
        gain = space.scale(1.0 / space.inner_product(representer, image), image)

        def project(m):
            return space.subtract(
                m, space.scale(space.inner_product(representer, m), gain)
            )

        def project_adjoint(y):
            return space.subtract(
                y, space.scale(space.inner_product(gain, y), representer)
            )

        projection = inf.LinearOperator.from_callables(
            space, space, project, adjoint=project_adjoint
        )
    else:
        gram = (constraint @ covariance @ constraint.adjoint).matrix(form="components")
        weights = inf.LinearOperator.from_matrix(
            constraint.codomain,
            constraint.codomain,
            np.linalg.inv(np.atleast_2d(gram)),
            form="components",
        )
        projection = inf.LinearOperator.identity(space) - (
            covariance @ constraint.adjoint @ weights @ constraint
        )
    return {
        "product": parts.get("product"),
        "names": list(names),
        "space": setup.space,
        "measures": dict(parts["measures"]),
        "labels": {n: n for n in names} | parts.get("labels", {}),
        "spectra": parts.get("spectra"),
        "map": projection @ parts["mixing"],
    }


def model_truth(physics: OceanPhysics, args: argparse.Namespace, years, /):
    """
    A true model that is not a sample of any prior: the change between the
    annual means of the two `years`, from models and analyses that the
    priors have not seen.

    - The ice thickness and the land water are the land ice and the
      terrestrial water of the ESA Earth System Model, version 3.0
      (`esm_loads`), as global fields: the load operator confines the ice
      to the ice sheets, and the water with the rest of the ice to the
      other land.
    - The steric height is that of the EN4 analyses (`en4_steric`). It is
      given over the ocean alone, and a field that stops at the coast would
      ring there when truncated; so it is continued over the land by the
      average of the ocean's values near each point (the heat kernel at
      --truth-fill-km, 300 by default, applied to the field over the ocean
      and divided by the same applied to the ocean function), which meets
      the ocean's values at the coast and decays inland. The land's values
      enter nothing.
    - The sterodynamic sea level is the steric height plus the manometric
      sea level, the ocean component of the Earth System Model, less its
      mean over the ocean, as a height of water. The ocean's load is then
      that component, with no net mass.

    Each field is truncated to the degrees of its space. With the density
    as the ocean's second field there is no such truth, since the depth of
    the water would divide the steric height: --ocean-prior steric.
    """
    import en4_steric
    import esm_loads

    if not _steric_prior(args):
        raise SystemExit("A truth from the models needs --ocean-prior steric.")
    if len(years) != 2 or years[1] != years[0] + 1:
        raise SystemExit("The truth is the change between two successive years.")
    setup, state = physics.setup, physics.state
    model = state.model
    p = model.parameters
    loads = {
        name: esm_loads.load_changes(model, name, years=years, version=3)[1][0]
        for name in ("H", "I", "O")
    }
    sea = state.ocean_function
    ocean_mean = model.integrate(sea * loads["O"]) / model.integrate(sea)
    manometric = (loads["O"] - ocean_mean * model.constant_grid(1.0)) * (
        1.0 / p.water_density
    )
    template = model.zero_grid()
    lats, lons = np.meshgrid(template.lats(), template.lons(), indexing="ij")
    steric = template.copy()
    steric.data[:] = en4_steric.changes_at(lats, lons, years)[1][0] / p.length_scale
    fill_km = getattr(args, "truth_fill_km", 300.0)
    if fill_km > 0.0:
        smoother = (
            underlying_space(physics.field_space)
            .heat_measure(fill_km * 1000.0 / p.length_scale)
            .covariance
        )
        wet = sea.data > 0.5
        near = smoother(sea * steric).data / np.maximum(smoother(sea).data, 1.0e-3)
        steric.data[:] = np.where(wet, steric.data, near)
    fields = {
        ICE: loads["I"] * (1.0 / p.ice_density),
        LAND: (loads["H"] + loads["I"]) * (1.0 / p.water_density),
        ETA: steric + manometric,
        STERIC: steric,
    }
    truth = []
    for name in setup.names:
        space = setup.component(name).space
        truth.append(space.from_components(space.to_components(fields[name])))
    return truth


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
    # WMB's map multiplies each coefficient of the load, orthonormal on the
    # unit sphere, by a factor of its degree: a unit variance in the load's
    # coefficient is the square of that factor in GRACE's.
    wmb = wmb_method(state, args)
    to_load = wmb.potential_coefficient_to_load_coefficient_operator()
    shape = 1.0 / to_load.diagonals()[0] ** 2
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
    - with --gia-years, the error of the GIA correction
      (`gia_correction_error`).

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
        parts["gia"] = gia_correction_error(state.model, observe, args, grace=True)
    noise = None
    for part in parts.values():
        noise = part if noise is None else noise + part
    return noise, invariant, parts


def gia_correction_error(
    model,
    observe,
    args: argparse.Namespace,
    /,
    *,
    grace: bool = False,
    field_space=None,
    points=None,
):
    """
    The error of the GIA correction on the data `observe` reads from a
    response, as a measure of low rank on its codomain: GRACE's
    coefficients (`grace`), the geocentric sea level at `points` of
    `field_space`, or both, on their direct sum in that order.

    With --gia-model caron it is that of the published covariance of the
    rates of the Stokes coefficients (`gia_statistics.py`): on GRACE's
    coefficients directly, and on the altimetry through the geoid. With
    --gia-model ensemble it is the spread of the ensemble of
    `gia_ensemble.py` (--gia-lmax), observed. Either is over --gia-years
    and has its std times --gia-scale.
    """
    if args.gia_model == "ensemble":
        measure, members = gia_ensemble.correction_error(
            observe, years=args.gia_years, scale=args.gia_scale, lmax=args.gia_lmax
        )
        print(f"GIA correction error from {len(members)} models of the ensemble")
        return measure
    altimetry_factor, grace_factor = gia_statistics.correction_error(
        model,
        # The altimetry's scripts have no options for GRACE's data.
        obs_degree=args.obs_degree if grace else None,
        convention=args.grace_convention if grace else "orthonormal",
        field_space=field_space,
        points=points,
        years=args.gia_years,
        rank=args.gia_rank,
    )
    factors = [
        None if factor is None else args.gia_scale * factor
        for factor in (altimetry_factor, grace_factor)
    ]
    print(
        f"GIA correction error from {args.gia_rank} eigenvectors of the "
        "covariance of Caron et al. (2018)"
    )
    return gia_statistics.measure(observe.codomain, *factors)


def ocean_gia_error(
    physics: OceanPhysics,
    args: argparse.Namespace,
    /,
    *,
    altimetry: bool = True,
    grace: bool = False,
):
    """
    The error of the GIA correction of the ocean experiments' data
    (`gia_correction_error`; --gia-years, --gia-model, --gia-scale): a
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
    return gia_correction_error(
        physics.state.model,
        observe,
        args,
        grace=grace,
        field_space=physics.fingerprint.field_space if altimetry else None,
        points=physics.points if altimetry else None,
    )


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
    forward,
    prior,
    noise,
    /,
    *,
    noise_matrix=None,
    n_jobs: Optional[int] = None,
    rank: int = 0,
):
    """
    The inverse of the data-space normal matrix A Q A* + R, assembled and
    factorised: a direct solver, for data few enough that the matrix can be
    held, as GRACE's coefficients are.

    It costs one application of A Q A* per datum, each independent of the
    others, and a dense factorisation. A conjugate-gradient solve
    preconditioned by it converges in an iteration or two, so every further
    solve (a quantity of interest, a row of an estimator, a posterior
    sample) costs one application of the operator and no more.

    With `rank` above zero and below the number of data the signal A Q A*
    is not assembled but sketched, from that many applications
    (`sketched_signal`): the build costs in proportion, and the solves take
    more iterations the further the rank falls below the number of
    directions in which the signal exceeds the noise.

    Returns the inverse as an operator on the data space.
    """
    space = forward.codomain
    signal = forward @ prior.covariance @ forward.adjoint
    covariance = covariance_matrix_of(noise, noise_matrix)
    if 0 < rank < space.dim:
        matrix = sketched_signal(signal, covariance, rank, n_jobs=n_jobs)
    else:
        matrix = _images(signal, np.eye(space.dim), n_jobs)
    matrix = 0.5 * (matrix + matrix.T) + covariance
    inverse = cho_solve(cho_factor(matrix, lower=True), np.eye(len(matrix)))
    return definite(
        inf.LinearOperator.from_matrix(
            space, space, 0.5 * (inverse + inverse.T), form="components"
        )
    )


def _images(operator, vectors: np.ndarray, n_jobs: Optional[int] = None) -> np.ndarray:
    """The images of the columns of `vectors` under an operator on a
    Euclidean data space, as columns: one application each."""
    space = operator.domain

    def columns(indices):
        return np.array(
            [
                space.to_components(
                    operator(space.from_components(vectors[:, int(index)]))
                )
                for index in indices
            ]
        )

    # Each worker is handed one share of the columns, so the operator is
    # carried to it once: a task per column carries it every time, which
    # costs far more than an application. Workers pay for themselves only
    # when the whole assembly is slow.
    count = vectors.shape[1]
    start = time.perf_counter()
    columns([0])
    serial_seconds = (time.perf_counter() - start) * count
    workers = 1 if n_jobs is None or serial_seconds < 60.0 else n_jobs
    shares = np.array_split(np.arange(count), workers)
    return np.vstack(inf.parallel.parallel_map(columns, shares, n_jobs=workers)).T


def sketched_signal(
    signal,
    noise_covariance: np.ndarray,
    rank: int,
    /,
    *,
    n_jobs: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    The matrix of a signal covariance S = A Q A* on the data from `rank`
    applications of it, in place of one a datum: the randomised Nystrom
    approximation (Frangella, Tropp and Udell, 2023) in the metric of the
    noise.

    With R = L L^T the noise, the signal in the noise's metric is
    M = L^-1 S L^-T, whose eigenvalues above one are the directions the
    data resolve; those below it barely change the normal matrix M + 1.
    The sketch is M G for G white, that is L^-1 S applied to the probes
    L^-T G. From it comes M ~ U E U^T of that rank, and the signal
    L U E U^T L^T. The probes are drawn from `seed`, so that a build is
    repeatable.

    Returned as a matrix on the data, the noise's covariance being one
    already. Nothing needs it formed: the inverse of its sum with the noise
    is L^-T (1 - U E (1 + E)^-1 U^T) L^-1.
    """
    dim = len(noise_covariance)
    lower = cholesky(noise_covariance, lower=True)
    white = np.random.default_rng(seed).standard_normal((dim, rank))
    probes = solve_triangular(lower, white, lower=True, trans="T")
    sketch = solve_triangular(lower, _images(signal, probes, n_jobs), lower=True)
    # The stable form: a small shift makes the core definite to rounding,
    # and is taken off the eigenvalues after.
    shift = np.sqrt(dim) * np.finfo(float).eps * np.linalg.norm(sketch, 2)
    sketch += shift * white
    core = white.T @ sketch
    factor = cholesky(0.5 * (core + core.T), lower=True)
    basis, singular, _ = np.linalg.svd(
        solve_triangular(factor, sketch.T, lower=True).T, full_matrices=False
    )
    half = lower @ (basis * np.sqrt(np.maximum(singular**2 - shift, 0.0)))
    return half @ half.T


def tabulated_kernel(space, measure):
    """The covariance of an invariant measure on a sphere as a function of
    the angle between two points, in radians. It is a sum over degrees; a
    nearest-neighbour block wants it between each point and its neighbours
    and between those, many values a point, so it is tabulated once, on two
    thousand nodes to the shortest wavelength, and interpolated linearly
    (to a few parts in a billion of its largest value)."""
    count = 2048 * (space.lmax + 1)
    step = np.pi / count
    table = space.covariance_function(
        measure, space.radius * np.linspace(0.0, np.pi, count + 1)
    )
    table = np.append(table, table[-1])

    def kernel(angle):
        position = np.asarray(angle) / step
        index = np.minimum(position.astype(np.int64), count)
        part = position - index
        return table[index] * (1.0 - part) + table[index + 1] * part

    return kernel


def altimetry_block(
    physics: OceanPhysics,
    args: argparse.Namespace,
    calib: dict,
    alt_local,
    cutoff_km: float,
    neighbours: int = 0,
):
    """
    The altimetry block of `block_preconditioner`, as an operator on the
    altimetry data: the inverse of the covariance of eta between the
    points under its prior, tapered to vanish beyond `cutoff_km` by the
    taper of Gaspari and Cohn, which keeps it positive definite, plus the
    local noise. The matrix is sparse and factorised once, so nothing dense
    is formed on the altimetry data, whose number is not bounded.

    With `neighbours` above zero the covariance is not tapered: its inverse
    with the local noise is approximated by the sparse factor that regresses
    each point on that many of the nearest points before it in a
    coarse-to-fine order (`vecchia.VecchiaInverse`). There is then no
    cut-off to choose, the result is positive definite by construction, and
    it has fewer entries and gives fewer iterations than the tapered block.
    """
    space = physics.field_space
    points = physics.points
    spectrum = _field_measure(
        physics, args, ETA, _stds_by_name(calib["derived_stds"])[ETA]
    ).covariance.eigenvalues
    cutoff = cutoff_km * 1000.0 / physics.state.model.parameters.length_scale
    eta = space.invariant_measure(spectrum, unsafe=True)
    if neighbours > 0:
        from vecchia import VecchiaInverse, unit_vectors

        approximate = VecchiaInverse(
            unit_vectors(points),
            tabulated_kernel(space, eta),
            alt_local.covariance.diagonals()[0],
            neighbours=neighbours,
        )
        print(
            f"  altimetry block: {approximate.entries_per_point:.0f} neighbours "
            "per point, no cut-off",
            flush=True,
        )
        domain = alt_local.domain
        return definite(
            inf.LinearOperator.from_callables(
                domain, domain, approximate.inverse, adjoint=approximate.inverse
            )
        )
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
      held. With --grace-block-rank its signal part is sketched from that
      many applications in place of one a coefficient.
    - The altimetry's block is that of `altimetry_block`, sparse
      (`alt_local`, the local noise; --cutoff-km, or --altimetry-neighbours).
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
                physics,
                args,
                calib,
                alt_local,
                args.cutoff_km,
                neighbours=getattr(args, "altimetry_neighbours", 0),
            )
        blocks.append(cache["altimetry"])
    if use_grace:
        if "grace" not in cache:
            rank = getattr(args, "grace_block_rank", 0)
            count = grace_noise.domain.dim
            print(
                f"  GRACE block: {rank if 0 < rank < count else count} "
                "applications of the normal operator...",
                flush=True,
            )
            cache["grace"] = exact_preconditioner(
                physics.grace(args),
                prior,
                grace_noise,
                noise_matrix=grace_matrix,
                n_jobs=jobs(args),
                rank=rank,
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


def noise_precision(
    case: str,
    space,
    /,
    *,
    alt_local=None,
    alt_parts: Optional[dict] = None,
    grace_matrix: Optional[np.ndarray] = None,
    gia=None,
    correlated_rank: int = 400,
    correlated_std: Optional[float] = None,
    rng=None,
):
    """
    The inverse of the noise covariance of a case of the ocean experiments
    ("altimetry", "grace" or "joint") as an operator on its data space
    `space`, with nothing dense formed on the altimetry data.

    The noise is a part that is easily inverted, the local noise of the
    altimetry (diagonal) and the covariance of GRACE's coefficients (a
    matrix on a few thousand data, factorised), plus parts of low rank: the
    common modes of the altimetry, the error of the GIA correction, and the
    large-scale correlated error of the altimetry. The last is a smooth
    field at the points and is kept to its `correlated_rank` leading modes
    (`GaussianMeasure.low_rank_approximation`); the share of its variance
    that this leaves out is returned (if `correlated_std`, its std at a
    point, is given), and is a statement about the noise model the
    precision stands for. The whole is the Woodbury inverse
    (`low_rank_correction`, exact for an exact base).

    Returns (precision, left_out).
    """
    use_alt, use_grace = case != "grace", case != "altimetry"
    blocks = []
    if use_alt:
        diagonal = alt_local.covariance.diagonals()[0]
        blocks.append(inf.DiagonalLinearOperator(alt_local.domain, 1.0 / diagonal))
    if use_grace:
        inverse = cho_solve(
            cho_factor(grace_matrix, lower=True), np.eye(len(grace_matrix))
        )
        grace_space = inf.EuclideanSpace(len(grace_matrix))
        blocks.append(
            inf.LinearOperator.from_matrix(
                grace_space, grace_space, 0.5 * (inverse + inverse.T), form="components"
            )
        )
    base = (
        blocks[0]
        if len(blocks) == 1
        else inf.BlockDiagonalLinearOperator(blocks, domain=space, codomain=space)
    )
    factors, left_out = [], 0.0
    if use_alt:
        for name, measure in (alt_parts or {}).items():
            if name == "correlated":
                rank = min(correlated_rank, measure.domain.dim - 1)
                measure = measure.low_rank_approximation(rank=rank, rng=rng)
                factor = measure.covariance_factor
                kept = sum(
                    float(np.sum(factor(factor.domain.basis_vector(i)) ** 2))
                    for i in range(factor.domain.dim)
                )
                if correlated_std is not None:
                    whole = measure.domain.dim * correlated_std**2
                    left_out = 1.0 - kept / whole
            factor = measure.covariance_factor
            if case == "joint":
                factor = space.inclusion(0) @ factor
            factors.append(factor)
    if gia is not None:
        factors.append(gia.covariance_factor)
    precision = low_rank_correction(base, factors) if factors else base
    return definite(precision), left_out


def low_rank_posterior(
    problem,
    prior,
    data,
    rank: int,
    /,
    *,
    n_jobs: Optional[int] = None,
    oversampling: int = 10,
    seed: int = 0,
):
    """
    The posterior of a linear problem as the prior corrected in the `rank`
    directions the data inform most (`inf.LowRankGaussianPosterior`), with
    no solve of the normal equations.

    The directions are the leading eigenvectors of the prior-preconditioned
    misfit Hessian H = L* A* R^-1 A L, with L the factor of the prior's
    covariance, on the factor's own (latent) space; `problem`'s error
    measure must have a precision (`noise_precision`). They are found by
    the randomised Nystrom method (Halko et al., 2011; Frangella et al.,
    2023), as `sketched_signal` finds GRACE's block: H is applied to
    rank + oversampling white vectors, the images are made orthonormal, and
    H is applied to those. Two applications a direction, each a forward
    and an adjoint action of the problem with nothing solved; they are
    shared among the worker processes a share each, so that the operator is
    carried to a worker once.

    Returns (posterior, eigenvalues): the eigenvalues kept, largest first.
    """
    forward = problem.forward_operator
    error = problem.error_measure
    if error.precision is None:
        raise ValueError("The error measure needs a precision.")
    factor = prior.covariance_factor
    latent = factor.domain
    hessian = factor.adjoint @ forward.adjoint @ error.precision @ forward @ factor
    # Coordinates in which the latent space's inner product is the dot
    # product: its components scaled by the root of its diagonal metric.
    if not latent.has_diagonal_metric:
        raise NotImplementedError("The latent space's metric is not diagonal.")
    weight = np.sqrt(np.asarray(latent.gram_diagonal()))

    def images(columns):
        def apply(indices):
            return np.array(
                [
                    weight
                    * latent.to_components(
                        hessian(latent.from_components(columns[:, int(i)] / weight))
                    )
                    for i in indices
                ]
            )

        count = columns.shape[1]
        workers = 1 if n_jobs is None else min(n_jobs, count)
        shares = np.array_split(np.arange(count), workers)
        return np.vstack(inf.parallel.parallel_map(apply, shares, n_jobs=workers)).T

    count = min(rank + oversampling, latent.dim, forward.codomain.dim)
    white = np.random.default_rng(seed).standard_normal((latent.dim, count))
    basis, _ = np.linalg.qr(images(white))
    sketch = images(basis)
    shift = np.sqrt(latent.dim) * np.finfo(float).eps * np.linalg.norm(sketch, 2)
    sketch += shift * basis
    core = basis.T @ sketch
    lower = cholesky(0.5 * (core + core.T), lower=True)
    vectors, singular, _ = np.linalg.svd(
        solve_triangular(lower, sketch.T, lower=True).T, full_matrices=False
    )
    keep = min(rank, count)
    eigenvalues = np.maximum(singular[:keep] ** 2 - shift, 0.0)
    vectors = vectors[:, :keep]
    reduced = inf.EuclideanSpace(keep)
    directions = inf.LinearOperator.from_callables(
        reduced,
        latent,
        lambda c: latent.from_components((vectors @ c) / weight),
        adjoint=lambda x: vectors.T @ (weight * latent.to_components(x)),
    )
    space, data_space = prior.domain, forward.codomain
    residual = data_space.subtract(data, forward(prior.expectation))
    residual = data_space.subtract(residual, error.expectation)
    pulled = factor.adjoint(forward.adjoint(error.precision(residual)))
    coefficients = directions.adjoint(pulled) / (1.0 + eigenvalues)
    mean = space.add(prior.expectation, factor(directions(coefficients)))
    posterior = inf.LowRankGaussianPosterior(
        prior,
        mean,
        eigenvalues,
        directions,
        relative_log_density=-problem.potential(data),
    )
    return posterior, eigenvalues


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
    solves of the data-space normal equations with `preconditioner`.

    The posterior expectation, and the randomise-then-optimise samples drawn
    from `posterior`, use `rtol`; `qoi_posterior` has the same expectation
    and applies its covariance through solves at `qoi_rtol`, with the same
    preconditioner. The scripts pass one tolerance for both (--qoi-rtol): a
    posterior variance is the prior variance less a correction computed
    through the solves, so its error relative to itself is about the
    tolerance divided by the ratio of the posterior variance to the prior,
    and a looser solve also biases the squared misfit printed below, and
    the sampled variances, low.

    Returns (posterior, qoi_posterior, qoi_inversion).
    """
    callback = inf.ProgressCallback(report=print)
    solver = inf.CGSolver(callback=callback, rtol=rtol, preconditioner=preconditioner)
    estimator = inversion.with_solver(solver)
    posterior = estimator(data)
    print(f"Solution reached in {callback.iterations} iterations.")
    # The data against what the prior and the noise predict for them: the
    # squared misfit is chi-squared on the number of data when the two
    # describe the data, so it tests the consistency of the experiment.
    count = estimator.data_space.dim
    misfit = float(estimator.mahalanobis(data))
    print(
        f"Squared misfit of the data to their prior prediction: {misfit:.0f} "
        f"for {count} data (expected {count} +/- {np.sqrt(2.0 * count):.0f})."
    )
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


def pointwise_stds_of(measure, operators: dict, samples: int, /, *, rng, n_jobs=None):
    """The pointwise standard deviations of several linear images of a
    measure on the model space, from the same `samples` draws about its
    expectation: for each name of `operators` the standard deviation of that
    operator's image, a field, or a tuple of fields for an operator into a
    direct sum. The draws are the cost (each is a solve for a posterior), so
    every image wanted is taken from each."""

    def parts(vector):
        return list(vector) if isinstance(vector, tuple) else [vector]

    means = {name: parts(op(measure.expectation)) for name, op in operators.items()}
    sums = {name: [np.zeros_like(f.data) for f in fs] for name, fs in means.items()}
    for sample in measure.samples(samples, rng=rng, n_jobs=n_jobs):
        for name, operator in operators.items():
            for total, x, m in zip(sums[name], parts(operator(sample)), means[name]):
                total += (x.data - m.data) ** 2
    out = {}
    for name, operator in operators.items():
        fields = []
        for total, m in zip(sums[name], means[name]):
            field = m.copy()
            field.data[:] = np.sqrt(total / samples)
            fields.append(field)
        image = operator(measure.expectation)
        out[name] = tuple(fields) if isinstance(image, tuple) else fields[0]
    return out


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


def sensitivity_table(
    kernels,
    weights,
    noise,
    independent: dict,
    labels,
    /,
    *,
    title: str,
    stated=None,
    model_hs: Optional[float] = None,
) -> str:
    """
    The sensitivity to the prior of the errors of linear estimates of some
    quantities, field by field of the prior.

    An estimate C d of the quantities B m, from data d = A m + z, has the
    error E m + C z with E = C A - B. For quantity i, `kernels[i]` is E*
    e_i, the kernel error k - t of the estimate, in the model space, and
    `weights[i]` is C* e_i, in the data space. If the models are drawn with
    covariance Q and the noise with R, the variance of the error is

        v = ((k - t), Q (k - t)) + (C* e, R C* e) ,

    a part the prior supplies and a part the noise does; and if the models
    are in truth drawn with Q + dQ, it is larger by ((k - t), dQ (k - t)).

    The prior is the law of T u, with u independent fields of covariances
    Q_f (`independent`, from `ocean_prior` or `grace_experiment`: T is its
    `map`). A change dQ_f of the covariance of field f, with kappa_f the
    part of T* (k - t) in that field, changes the variance by (kappa_f, dQ_f
    kappa_f), which is at most ||dQ_f|| ||kappa_f||^2 in the operator norm
    of dQ_f, and so too in its Hilbert-Schmidt norm, which is larger. As a
    fraction of the variance s^2 that the estimate states, and of the size
    of Q_f in the same norm:

        |change of variance| / s^2 <= (||dQ_f|| / ||Q_f||) F_f ,
        F_f = ||Q_f|| ||kappa_f||^2 / s^2 .

    The table gives, for each quantity: the stated std s, from the
    variances `stated`, or the std of v if they are omitted; the shares of
    s^2 that the prior and the noise supply, and their sum less one; then,
    for each field, its share (kappa_f, Q_f kappa_f) / s^2, which is also
    the exact change per unit of a relative change of that field's variance
    at every scale, and F_f in the two norms. For a Bayesian estimate s^2 is
    the posterior variance and equals v, so the sum less one checks the
    solves; for an estimate that states the noise's part alone it is the
    prior's share, the part of the error variance that is not stated.

    The norms are exact, from the spectra of the invariant measures. With
    `model_hs`, the Hilbert-Schmidt norm of the prior covariance on the
    model space, the factor of the bound stated on the model space itself,
    ||Q|| ||k - t||^2 / s^2, is given beside them.
    """
    names = independent["names"]
    measures, fields = independent["measures"], independent["space"]
    field_labels = [independent["labels"][n] for n in names]
    model_space, data_space = independent["map"].codomain, noise.domain
    hs = {n: measures[n].hilbert_schmidt_norm() for n in names}
    top = {n: float(np.max(measures[n].covariance.eigenvalues)) for n in names}
    # The changes of each field's covariance under the named changes of its
    # prior (`PRIOR_CHANGES`), as invariant operators: the error variance of
    # a fixed estimate is linear in the covariance of the truth, so its
    # change under each is (kappa_f, dQ_f kappa_f) exactly.
    changes = {}
    if independent.get("spectra"):
        for n in names:
            changed = changed_spectra(independent, n)
            # The change is not of one sign: its action is that of its
            # positive part less that of its negative part.
            changes[n] = {
                label: (
                    measures[n].domain.invariant_measure(
                        np.maximum(changed[label] - changed["prior"], 0.0), unsafe=True
                    ).covariance,
                    measures[n].domain.invariant_measure(
                        np.maximum(changed["prior"] - changed[label], 0.0), unsafe=True
                    ).covariance,
                )
                for label, *_ in PRIOR_CHANGES
            }
    exact = []

    rows = []
    for i, (kernel, weight) in enumerate(zip(kernels, weights)):
        noise_part = data_space.inner_product(noise.covariance(weight), weight)
        pulled = independent["map"].adjoint(kernel)
        norms, parts, moved = [], [], []
        for n in names:
            space = measures[n].domain
            kappa = fields.projection(n)(pulled)
            norms.append(space.squared_norm(kappa))
            parts.append(space.inner_product(measures[n].covariance(kappa), kappa))
            moved.append(
                {
                    label: space.inner_product(up(kappa), kappa)
                    - space.inner_product(down(kappa), kappa)
                    for label, (up, down) in changes.get(n, {}).items()
                }
            )
        variance = sum(parts) + noise_part if stated is None else stated[i]
        exact.append([{k: v / variance for k, v in m.items()} for m in moved])
        rows.append(
            (
                variance,
                noise_part / variance,
                [
                    (hs[n] * norm / variance, top[n] * norm / variance, part / variance)
                    for n, norm, part in zip(names, norms, parts)
                ],
                None
                if model_hs is None
                else model_hs * model_space.squared_norm(kernel) / variance,
            )
        )

    width = 16 + 38 + 12 * (model_hs is not None)
    lines = [
        "\n\n" + "=" * 110,
        f"PRIOR SENSITIVITY: {title}",
        "A share is a part of the stated variance. 'sum - 1' is the prior's "
        "and the noise's shares together,\nless one: zero for an estimate "
        "that states the whole of its error variance under the prior.",
        "-" * width,
        f"{'':<16}{'stated std':>12}{'prior':>8}{'noise':>8}{'sum - 1':>10}"
        + ("" if model_hs is None else f"{'model HS':>12}"),
        "-" * width,
    ]
    for label, (variance, noise_share, parts, written) in zip(labels, rows):
        inherited = sum(part[2] for part in parts)
        lines.append(
            f"{label[:15]:<16}{np.sqrt(variance):>12.3f}{inherited:>8.3f}"
            f"{noise_share:>8.3f}{inherited + noise_share - 1.0:>10.1e}"
            + ("" if written is None else f"{written:>12.3e}")
        )
    for heading, index in (
        ("Share of the stated variance from each field's prior", 2),
        ("Bound factor F, operator norm", 1),
        ("Bound factor F, Hilbert-Schmidt norm", 0),
    ):
        lines += [
            "",
            heading,
            f"{'':<16}" + "".join(f"{name[:15]:>16}" for name in field_labels),
        ]
        for label, row in zip(labels, rows):
            lines.append(
                f"{label[:15]:<16}"
                + "".join(f"{part[index]:>16.3e}" for part in row[2])
            )
    for change, *_ in PRIOR_CHANGES if changes else ():
        lines += [
            "",
            "Exact change of the error variance, as a fraction of the stated "
            f"variance, if one field's truth has its prior with: {change}",
            f"{'':<16}" + "".join(f"{name[:15]:>16}" for name in field_labels),
        ]
        for label, row in zip(labels, exact):
            lines.append(
                f"{label[:15]:<16}" + "".join(f"{m[change]:>+16.3e}" for m in row)
            )
    return "\n".join(lines) + "\n"


def prior_sensitivity(
    error,
    gain,
    noise,
    independent: dict,
    labels,
    /,
    *,
    title: str,
    stated=None,
    model_hs: Optional[float] = None,
) -> str:
    """
    The table of `sensitivity_table` for any linear estimate, from its
    error operator `error`, E = C A - B on the model space, and its `gain`
    C, which takes the data to the estimate: for WMB's regional averages,
    `wmb_error_operator` and WMB's averaging of the data. `stated` are the
    variances the estimate states, if it states any. Costs an adjoint
    action of each operator for each quantity.
    """
    units = [error.codomain.basis_vector(i) for i in range(error.codomain.dim)]
    return sensitivity_table(
        [error.adjoint(unit) for unit in units],
        [gain.adjoint(unit) for unit in units],
        noise,
        independent,
        labels,
        title=title,
        stated=stated,
        model_hs=model_hs,
    )


def posterior_sensitivity(
    target,
    inversion,
    independent: dict,
    labels,
    /,
    *,
    title: str,
    model_hs: Optional[float] = None,
) -> str:
    """
    The table of `sensitivity_table` for the posterior expectation of the
    quantities `target` (B) as their estimate: C = B K, with K the Kalman
    operator of `inversion`, and E = B (K A - 1). The stated variances are
    the posterior's, computed as the prior's less the data's reduction of
    them. Costs one solve of the normal equations for each quantity, which
    gives both K* B* e and the posterior variance.
    """
    problem = inversion.forward_problem
    forward, prior = problem.forward_operator, inversion.prior
    model_space, data_space = forward.domain, forward.codomain
    gain = target @ inversion.kalman_operator
    kernels, weights, stated = [], [], []
    for i in range(target.codomain.dim):
        unit = target.codomain.basis_vector(i)
        representer = target.adjoint(unit)
        weight = gain.adjoint(unit)
        image = prior.covariance(representer)
        kernels.append(model_space.subtract(forward.adjoint(weight), representer))
        weights.append(weight)
        stated.append(
            model_space.inner_product(image, representer)
            - data_space.inner_product(forward(image), weight)
        )
    return sensitivity_table(
        kernels,
        weights,
        problem.error_measure,
        independent,
        labels,
        title=title,
        stated=stated,
        model_hs=model_hs,
    )


def ocean_prior_sensitivity(
    physics: OceanPhysics,
    inversion,
    calib: dict,
    region: str,
    /,
    *,
    case: str = "",
    sizes: bool = True,
) -> str:
    """The tables of `posterior_sensitivity` for the quantities of the ocean
    experiments: the GMSL change, its barystatic and steric parts, and the
    three parts of the mean sea level change of `region`; and, with `sizes`,
    the table of `prior_perturbation_sizes`. `case` heads the tables. Empty for a prior
    that is not built from independent fields."""
    independent = calib["independent"]
    if independent is None:
        return ""
    mm = physics.scale_mm
    head = f"{case}: " if case else ""
    targets = (
        ("GMSL change (mm)", mm * physics.setup.gmsl(physics.fingerprint), ["GMSLR"]),
        (
            "barystatic and steric GMSL change (mm)",
            mm * physics.setup.gmsl_split,
            ["Barystatic", "Steric"],
        ),
        (
            f"parts of the {region} mean sea level change (mm)",
            physics.regional_decomposition(region),
            ["SSLC", "DMSLC", "BMSLC"],
        ),
    )
    text = "".join(
        posterior_sensitivity(
            target, inversion, independent, labels, title=head + title
        )
        for title, target, labels in targets
    )
    return text + (prior_perturbation_sizes(independent) if sizes else "")


PRIOR_CHANGES = (
    ("std x1.25", 1.0, 0.0, 1.25),
    ("scale x1.5", 1.5, 0.0, 1.0),
    ("scale /1.5", 1.0 / 1.5, 0.0, 1.0),
    ("order +0.5", 1.0, 0.5, 1.0),
)


def changed_spectra(independent: dict, name) -> dict:
    """The covariance eigenvalues of one independent field's prior under
    each of `PRIOR_CHANGES`, by the change's name, with those of the prior
    itself under "prior": the pointwise std 25 per cent larger, the scale
    longer and shorter by a factor 1.5, and the order higher by a half, the
    last three at the same pointwise std. For a prior whose spectra are
    stated as (1 + s^2 k)^-t."""
    measure = independent["measures"][name]
    space = measure.domain
    scale, order, taper = independent["spectra"][name]
    base = measure.covariance.eigenvalues
    k = space.laplacian_eigenvalues
    # A covariance's eigenvalues are relative to the space's own inner
    # product; the pointwise variance sums them with its weight taken out.
    weight = (1.0 + space.length_scale**2 * k) ** (-space.order)

    def spectrum(scale_, order_, amplitude):
        raw = tapered_spectrum(k, scale_, order_, space.radius, taper) / weight
        return raw * amplitude**2 * np.sum(base * weight) / np.sum(raw * weight)

    if not np.allclose(spectrum(scale, order, 1.0), base, rtol=1.0e-8):
        raise ValueError(f"The stated spectrum of {name} is not its measure's.")
    return {"prior": base} | {
        label: spectrum(scale * factor, order + step, amplitude)
        for label, factor, step, amplitude in PRIOR_CHANGES
    }


def prior_perturbation_sizes(independent: dict) -> str:
    """
    The sizes ||dQ_f|| / ||Q_f||, in the operator and the Hilbert-Schmidt
    norms, of some changes of one field's prior, to set against the factors
    of `sensitivity_table`: the pointwise std 25 per cent larger, the scale
    longer and shorter by a factor 1.5, and the order higher by a half, the
    last three at the same pointwise std. Empty for a prior whose spectra
    are not stated as (1 + s^2 k)^-t.
    """
    spectra = independent.get("spectra")
    if not spectra:
        return ""
    cases = PRIOR_CHANGES
    lines = [
        "\n\n" + "=" * 110,
        "PRIOR SENSITIVITY: the size ||dQ|| / ||Q|| of changes of one field's "
        "prior (operator norm, Hilbert-Schmidt norm)",
        "-" * (16 + 20 * len(cases)),
        f"{'':<16}" + "".join(f"{name:>20}" for name, *_ in cases),
        "-" * (16 + 20 * len(cases)),
    ]
    for n in independent["names"]:
        changed = changed_spectra(independent, n)
        base = changed["prior"]
        line = f"{independent['labels'][n][:15]:<16}"
        for label, *_ in cases:
            change = changed[label] - base
            line += (
                f"{np.max(np.abs(change)) / np.max(base):>11.2f}"
                f"{np.linalg.norm(change) / np.linalg.norm(base):>9.2f}"
            )
        lines.append(line)
    return "\n".join(lines) + "\n"


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
