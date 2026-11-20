"""
An ensemble of glacial isostatic adjustment models, for the uncertainty of
the GIA correction (Heathcote et al., 2026).

GRACE and altimetry data are corrected for glacial isostatic adjustment
with a model before they are inverted for modern changes, and the error of
that correction is part of the data error. It is not noise in the usual
sense: it has the pattern of the adjustment itself, concentrated under the
former ice sheets and around Antarctica, and it is common to every data set
the correction is applied to. Here it is represented by the spread of an
ensemble of models, each a forward run of the last deglaciation
(`pyslfp.viscoelastic`): an ice history, ICE-5G, ICE-6G or ICE-7G, on PREM
with an elastic lithosphere and a two-layer Maxwell mantle, the ice
histories crossed with the lithospheric thicknesses and the viscosities of
the upper and lower mantle. The ice histories were each built with a
viscosity model of their own, so crossing them with others gives members
that no longer fit the sea level data they were tuned to, and the ensemble
is not weighted by any fit: its spread is a generous measure of the
uncertainty, not a calibrated one. The models are spherically symmetric.

Run as a script, it computes the members that are missing and saves, for
each, the present-day rate of the response per year, from the last step of
the run: the coefficients of the sea level change, of the vertical
displacement and of the potential change, and the rate of the angular
velocity change. The files go to pyslfp's data directory, under
gia_ensemble/, one directory per truncation degree. A member takes a couple
of minutes at degree 128; with --parallel the members run side by side.

    python gia_ensemble.py --lmax 128 --parallel --max-jobs 16

Imported, it gives the members' rates as responses in any response space
(`rate_responses`), truncated or padded to its degree, and the Gaussian
measure of the correction's error on a data space (`correction_error`): the
empirical covariance of the members about their mean, of rank one less than
their number, scaled by the interval the data span. The error has zero
expectation, which takes the correction applied to the data to be the
ensemble's mean.
"""

import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import pygeoinf as inf

from pyslfp.data import DATADIR

DIRECTORY = DATADIR / "gia_ensemble"

# The first date of each ice history, in kiloyears before present.
START = {"ICE5G": 21.0, "ICE6G": 26.0, "ICE7G": 26.0}
STEP = 0.5

FIELDS = ("relative_sea_level_change", "displacement", "potential_change")


@dataclass(frozen=True)
class Member:
    """One model of the ensemble: an ice history, the thickness of the
    elastic lithosphere in km, and the viscosities of the upper and lower
    mantle in units of 10^21 Pa s."""

    ice: str
    lithosphere_km: float
    upper: float
    lower: float

    @property
    def name(self) -> str:
        return (
            f"{self.ice.lower()}_l{self.lithosphere_km:g}"
            f"_um{self.upper:g}_lm{self.lower:g}"
        )


def directory(lmax: int) -> Path:
    return DIRECTORY / f"lmax{lmax}"


def _coefficients(field_space, components: np.ndarray) -> np.ndarray:
    """A sphere space's components as a coefficient array (2, lmax + 1,
    lmax + 1), orthonormal and without the Condon-Shortley phase, as
    `GRDSpace.from_coefficients` takes them."""
    out = np.zeros((2, field_space.lmax + 1, field_space.lmax + 1))
    parts = (field_space.orders < 0).astype(int)
    out[parts, field_space.degrees, np.abs(field_space.orders)] = (
        components / field_space.radius
    )
    return out


def run(member: Member, lmax: int, /, *, shorelines: str = "frozen") -> dict:
    """The present-day rate of the response of one member, per year: the
    change over the last step of a run from the ice history's first date,
    divided by the step. Returns the arrays that are saved."""
    import pyslfp as sl
    from pyslfp.core import lebesgue_response_space
    from pyslfp.ice import IceNG
    from pyslfp.viscoelastic import (
        SeaLevelEvolution,
        ViscoelasticOperator,
        prem_maxwell,
        viscoelastic_earth,
    )

    planet = prem_maxwell(
        lithosphere=6371.0e3 - 1000.0 * member.lithosphere_km,
        upper_mantle=1.0e21 * member.upper,
        lower_mantle=1.0e21 * member.lower,
    )
    earth_model, rheology = viscoelastic_earth(planet, lmax)
    operator = ViscoelasticOperator(rheology, lmax)
    ice_model = IceNG(
        version=member.ice, length_scale=earth_model.parameters.length_scale
    )
    start = START[member.ice]
    state = sl.EarthState.from_ice_model(ice_model, start, earth_model)
    dates = np.arange(start, -1.0e-6, -STEP)
    space = lebesgue_response_space(earth_model)
    snapshots, _ = SeaLevelEvolution(operator, state).run_ice_history(
        dates, ice_model, space=space, shorelines=shorelines
    )
    years = 1000.0 * STEP
    last, previous = snapshots[-1].response, snapshots[-2].response
    field_space = space.field_space
    arrays = {
        name: _coefficients(
            field_space,
            field_space.to_components(last.field(name) - previous.field(name)) / years,
        )
        for name in FIELDS
    }
    arrays["angular_velocity_change"] = (
        last.angular_velocity_change - previous.angular_velocity_change
    ) / years
    return arrays


def _run_and_save(member: Member, lmax: int, shorelines: str) -> str:
    path = directory(lmax) / f"{member.name}.npz"
    if path.exists():
        return f"{member.name}: present"
    arrays = run(member, lmax, shorelines=shorelines)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part.npz")
    np.savez(
        partial,
        ice=member.ice,
        lithosphere_km=member.lithosphere_km,
        upper=member.upper,
        lower=member.lower,
        shorelines=shorelines,
        **arrays,
    )
    os.replace(partial, path)
    return f"{member.name}: done"


def saved_members(lmax: int) -> list[tuple[Member, Path]]:
    """The members computed at `lmax`, in a fixed order."""
    found = []
    for path in sorted(directory(lmax).glob("*.npz")):
        if path.name.endswith(".part.npz"):
            continue
        with np.load(path) as data:
            member = Member(
                str(data["ice"]),
                float(data["lithosphere_km"]),
                float(data["upper"]),
                float(data["lower"]),
            )
        found.append((member, path))
    return found


def _resized(coefficients: np.ndarray, lmax: int) -> np.ndarray:
    """A coefficient array truncated or padded with zeros to `lmax`."""
    out = np.zeros((2, lmax + 1, lmax + 1))
    n = min(lmax, coefficients.shape[1] - 1) + 1
    out[:, :n, :n] = coefficients[:, :n, :n]
    return out


def rate_responses(response_space, /, *, lmax: Optional[int] = None):
    """The members' present-day rates per year as responses of
    `response_space`, which must hold the angular velocity change and no
    horizontal displacement. They are read from the ensemble computed at
    `lmax` (the highest degree computed, if None) and truncated or padded to
    the space's degree. Returns (members, responses)."""
    if lmax is None:
        degrees = [
            int(path.name[4:])
            for path in DIRECTORY.glob("lmax*")
            if any(path.glob("*.npz"))
        ]
        if not degrees:
            raise FileNotFoundError(
                f"No ensemble under {DIRECTORY}; run gia_ensemble.py first."
            )
        lmax = max(degrees)
    saved = saved_members(lmax)
    if len(saved) < 2:
        raise FileNotFoundError(
            f"Fewer than two members under {directory(lmax)}; "
            "run gia_ensemble.py first."
        )
    target = response_space.field_space.lmax
    members, responses = [], []
    for member, path in saved:
        with np.load(path) as data:
            fields = [_resized(data[name], target) for name in FIELDS]
            omega = np.array(data["angular_velocity_change"])
        members.append(member)
        responses.append(
            response_space.from_coefficients(*fields, angular_velocity_change=omega)
        )
    return members, responses


def correction_error(
    observation: inf.LinearOperator,
    /,
    *,
    years: float = 1.0,
    scale: float = 1.0,
    lmax: Optional[int] = None,
):
    """
    The error of the GIA correction of the data `observation` reads from a
    response, over an interval of `years`: a Gaussian measure on the data
    space with zero expectation and the covariance of the ensemble's rates
    about their mean, times `years`, times a further factor `scale` on the
    standard deviation. For several data sets, give the operator onto their
    direct sum, and the error is correlated between them as it should be.

    Returns (measure, members).
    """
    members, responses = rate_responses(observation.domain, lmax=lmax)
    space = observation.codomain
    data = [observation(response) for response in responses]
    mean = space.sample_expectation(data)
    # The deviations sum to zero, so their empirical mean vanishes.
    deviations = [space.scale(years * scale, space.subtract(d, mean)) for d in data]
    return inf.GaussianMeasure.from_samples(space, deviations), members


def ensemble(args: argparse.Namespace) -> list[Member]:
    return [
        Member(ice, lithosphere, upper, lower)
        for ice, lithosphere, upper, lower in itertools.product(
            args.ice, args.lithosphere_km, args.upper_mantle, args.lower_mantle
        )
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute an ensemble of GIA models for the correction's error."
    )
    parser.add_argument("--lmax", type=int, default=128)
    parser.add_argument(
        "--ice",
        nargs="+",
        default=["ICE5G", "ICE6G", "ICE7G"],
        choices=sorted(START),
    )
    parser.add_argument(
        "--lithosphere-km",
        nargs="+",
        type=float,
        default=[71.0, 96.0, 120.0],
        help="Thicknesses of the elastic lithosphere.",
    )
    parser.add_argument(
        "--upper-mantle",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.8],
        help="Viscosities above 670 km, in 10^21 Pa s.",
    )
    parser.add_argument(
        "--lower-mantle",
        nargs="+",
        type=float,
        default=[2.0, 5.0, 10.0, 20.0],
        help="Viscosities below 670 km, in 10^21 Pa s.",
    )
    parser.add_argument(
        "--shorelines",
        choices=("frozen", "migrating"),
        default="frozen",
        help="Shorelines frozen at the first date, or migrating, at greater cost.",
    )
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()


def main():
    args = parse_arguments()
    members = ensemble(args)
    print(f"{len(members)} members at degree {args.lmax}, in {directory(args.lmax)}")
    if args.parallel:
        workers = min(args.max_jobs, len(members))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_and_save, member, args.lmax, args.shorelines)
                for member in members
            ]
            for future in futures:
                print(future.result(), flush=True)
    else:
        for member in members:
            print(_run_and_save(member, args.lmax, args.shorelines), flush=True)


if __name__ == "__main__":
    main()
