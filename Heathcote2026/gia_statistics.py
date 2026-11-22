"""
The error of the GIA correction from published statistics: the covariance
of the rates of the Stokes coefficients of Caron et al. (2018)
(Heathcote et al., 2026).

Caron et al. (2018) weight more than 100,000 forward models of glacial
isostatic adjustment, varying the ice history and a radially symmetric
viscosity structure, by their fit to relative sea level records and GPS
uplift rates, and publish the expectation and the covariance of the
present-day rates of the Stokes coefficients, degrees 2 to 89, with maps of
the expectation and standard deviation of the uplift, geoid and gravity
rates. The files are at https://vesl.jpl.nasa.gov/solid-earth/gia/ and are
fetched from there; no licence is stated, and the page asks for the paper
to be cited.

What the files hold, as read and checked (`python gia_statistics.py check`):

- the covariance is one symmetric matrix on the 8096 coefficients, given as
  its lower triangle in four files by degree band, in yr^-2;
- the coefficients are fully normalised (4 pi), the sine term S_lm at the
  order -m, as in GRACE's Level-2 products. With that reading, and by the
  route the experiments use, the geoid rate from the expectations has a
  correlation of 0.99997 with the published map and a slope of 1.003 on
  it, and its pointwise standard deviation from the covariance is 1.014
  times the published one at the median point (5 to 95 per cent of
  points: 1.00 to 1.03). The reference radius is not stated and is taken
  as that of GRACE's products; the check does not tell 6371 from 6378 km;
- the covariance has low rank to rounding, as an ensemble with few
  parameters must: its leading 10 eigenvectors hold 0.995 of its trace, 20
  hold 0.9994 and 50 hold 0.99999.

So the correction's error over an interval is a Gaussian of low rank on the
coefficients, exactly the form the ocean experiments' noise takes for it:
`correction_error` gives its factor on GRACE's data and, through the geoid,
on the altimetry's.

Against GRACE's noise (ITSG-Grace2018, the difference of two annual means)
the error of one year's correction is the larger at degrees 2 to 6, by a
factor of up to 1.7, and falls below a quarter of it by degree 15; in the
metric of that noise 15 of its eigenvalues exceed one and 43 exceed 0.01.
RANK modes, 50, are kept.

For the altimetry the error enters through the geoid alone. The sea
surface also moves by a uniform amount set by the uplift of the sea floor,
whose covariance with the coefficients is not published; over a year it is
of the order of 0.05 mm (Prandi et al., 2021), against the 1 mm of the
altimetry's own common error, and it is left out. The geoid part is itself
small there, 0.05 mm at the median point over a year, and is kept so that
the one model's error is correlated between the two sets of data as it
should be.

    python gia_statistics.py build      fetch the files and reduce them
    python gia_statistics.py check      the checks above, against the maps

References

Caron, L., Ivins, E. R., Larour, E., Adhikari, S., Nilsson, J. and Blewitt,
G. (2018). GIA model statistics for GRACE hydrology, cryosphere, and ocean
science. Geophys. Res. Lett. 45, 2203-2212. doi:10.1002/2017GL076644.

Prandi, P. et al. (2021). Local sea level trends, accelerations and
uncertainties over 1993-2019. Sci. Data 8, 1.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

import pygeoinf as inf

import grace_covariance
from pyslfp.observations.grace import stokes_coefficient_operator

URL = "https://vesl.jpl.nasa.gov/solid-earth/gia/downloads"
DATA_DIR = Path.home() / ".pyslfp_data" / "caron_gia"
RAW_DIR = DATA_DIR / "raw"
MODES_FILE = DATA_DIR / "reduced" / "modes_caron_2018.npz"
BANDS = ("deg2to60", "deg61to75", "deg76to83", "deg84to89")
EXPECTATION = "exp_Stokes_Caron_et_al_2018"
MAPS = "GIA_maps_Caron_et_al_2018"
MIN_DEGREE, MAX_DEGREE = 2, 89
# The eigenvectors kept by `build`, and those used by default.
STORED, RANK = 200, 50


def _band_file(band: str) -> str:
    return f"covStokes_{band}_Caron_et_al_2018"


def download(name: str, /) -> Path:
    """Fetches one file, unless it is already there, under a temporary name
    that is renamed when complete."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / name
    if target.exists():
        return target
    partial = target.with_name(name + ".part")
    command = ["curl", "-sS", "-L", "--fail", "-C", "-", "-o", str(partial)]
    command += ["--retry", "8", "--retry-delay", "15", "--retry-all-errors"]
    command.append(f"{URL}/{name}")
    subprocess.run(command, check=True)
    partial.rename(target)
    return target


def _labels() -> tuple[np.ndarray, np.ndarray]:
    """The degree and order of each coefficient in the files' order: by
    degree, and within one the orders from -l to l."""
    degrees = range(MIN_DEGREE, MAX_DEGREE + 1)
    return (
        np.concatenate([np.full(2 * l + 1, l) for l in degrees]),
        np.concatenate([np.arange(-l, l + 1) for l in degrees]),
    )


def read_covariance() -> np.ndarray:
    """The covariance of the rates of the coefficients, in the files'
    order and in yr^-2, from the four files of its lower triangle. The
    files are 1.3 GB of text: pandas reads them in seconds, and numpy,
    used if pandas is not installed, in some minutes."""
    try:
        import pandas
    except ImportError:  # pragma: no cover - depends on the install
        pandas = None

    def position(degree, order):
        return degree * degree - MIN_DEGREE**2 + degree + order

    size = (MAX_DEGREE + 1) ** 2 - MIN_DEGREE**2
    matrix = np.full((size, size), np.nan)
    for band in BANDS:
        path = download(_band_file(band))
        if pandas is None:
            table = np.loadtxt(path, comments="%")
        else:
            table = pandas.read_csv(
                path, comment="%", sep=r"\s+", header=None
            ).to_numpy()
        labels = table[:, :4].astype(np.int64)
        rows = position(labels[:, 0], labels[:, 1])
        columns = position(labels[:, 2], labels[:, 3])
        matrix[rows, columns] = table[:, 4]
    matrix = np.where(np.isnan(matrix), matrix.T, matrix)
    if np.isnan(matrix).any():
        raise ValueError("The files do not give every entry of the covariance.")
    return 0.5 * (matrix + matrix.T)


def build() -> Path:
    """Reduces the covariance to its leading eigenvectors and keeps them,
    with its trace, beside the files."""
    values, vectors = np.linalg.eigh(read_covariance())
    order = np.argsort(values)[::-1][:STORED]
    MODES_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODES_FILE,
        values=values[order],
        vectors=vectors[:, order],
        trace=float(values[values > 0.0].sum()),
    )
    return MODES_FILE


def modes(rank: int = RANK, /) -> tuple[np.ndarray, np.ndarray, float]:
    """The leading eigenvalues (yr^-2) and eigenvectors of the covariance,
    in the files' order, and the part of its trace they hold."""
    if not MODES_FILE.exists():
        raise FileNotFoundError(
            "No reduced covariance; run `python gia_statistics.py build`."
        )
    if not 1 <= rank <= STORED:
        raise ValueError(f"the rank is between 1 and {STORED}.")
    with np.load(MODES_FILE) as stored:
        values, vectors = stored["values"][:rank], stored["vectors"][:, :rank]
        return values, vectors, float(values.sum() / stored["trace"])


def stokes_members(*, years: float = 1.0, rank: int = RANK) -> np.ndarray:
    """
    The error of the correction over `years` as `rank` vectors of changes
    of the Stokes coefficients whose outer products sum to its covariance:
    each an eigenvector times the root of its eigenvalue. An array
    (coefficients, rank), degrees 2 to 89 in the order of pyslfp's
    coefficients.
    """
    values, vectors, _ = modes(rank)
    degrees, orders = _labels()
    position = grace_covariance.pyslfp_index(orders < 0, degrees, np.abs(orders))
    members = np.empty_like(vectors)
    members[position] = years * vectors * np.sqrt(values)
    return members


def correction_error(
    model,
    /,
    *,
    obs_degree: Optional[int] = None,
    convention: str = "orthonormal",
    field_space=None,
    points=None,
    years: float = 1.0,
    rank: int = RANK,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    The error of the GIA correction over `years` on the experiments' data,
    as arrays whose columns' outer products sum to its covariance.

    Args:
        model: pyslfp's Earth model, for the units and the radius.
        obs_degree: With it, the factor on GRACE's coefficients of degrees
            2 to `obs_degree` in `convention` ("stokes" or pyslfp's
            "orthonormal"); those above degree 89 have no error.
        field_space, points: With them, the factor on the geocentric sea
            level at the points, through the geoid: minus the potential
            change over g, in pyslfp's units. Degrees above the space's are
            left out.

    Returns:
        (altimetry factor, GRACE factor), each (data, rank) or None.
    """
    members = stokes_members(years=years, rank=rank)
    to_stokes = stokes_coefficient_operator(model, MAX_DEGREE)
    potential = members / to_stokes(np.ones(len(members)))[:, None]

    grace = None
    if obs_degree is not None:
        source = members if convention == "stokes" else potential
        count = (obs_degree + 1) ** 2 - MIN_DEGREE**2
        grace = np.zeros((count, rank))
        kept = min(count, len(source))
        grace[:kept] = source[:kept]

    altimetry = None
    if field_space is not None and points is not None:
        degree = min(field_space.lmax, MAX_DEGREE)
        kept = (degree + 1) ** 2 - MIN_DEGREE**2
        g = model.parameters.gravitational_acceleration
        evaluate = field_space.point_evaluation_operator(points)
        altimetry = np.empty((evaluate.codomain.dim, rank))
        components = np.zeros(field_space.dim)
        for index in range(rank):
            # The components of a field are the radius times its
            # coefficients; degrees 0 and 1 come first and are zero.
            components[MIN_DEGREE**2 : MIN_DEGREE**2 + kept] = (
                -field_space.radius / g
            ) * potential[:kept, index]
            altimetry[:, index] = evaluate(field_space.from_components(components))
    return altimetry, grace


def measure(space, altimetry, grace, /):
    """
    The Gaussian measure of zero mean on the data whose covariance is the
    sum of the outer products of the factors' columns: on the altimetry's
    data, on GRACE's, or, with both, on their direct sum in that order,
    where the columns pair. The data spaces are Euclidean.
    """
    parts = [part for part in (altimetry, grace) if part is not None]
    rank = parts[0].shape[1]
    single = len(parts) == 1

    def value(c):
        images = [part @ c for part in parts]
        return images[0] if single else tuple(images)

    def adjoint(d):
        vectors = [d] if single else d
        return sum(part.T @ vector for part, vector in zip(parts, vectors))

    factor = inf.LinearOperator.from_callables(
        inf.EuclideanSpace(rank), space, value, adjoint=adjoint
    )
    return inf.GaussianMeasure(space, covariance_factor=factor, unsafe=True)


# ================================================================ #
#                        Command line                              #
# ================================================================ #


def _check() -> None:
    """The conventions, against the published maps: the geoid rate and its
    pointwise standard deviation from the coefficients, by the route the
    experiments use, at points of the maps."""
    import pyslfp
    from pygeoinf.symmetric_space.sphere import Sobolev

    model = pyslfp.EarthModel(MAX_DEGREE + 1)
    radius = model.parameters.mean_sea_floor_radius
    # Any order at which fields have point values will do: the components
    # are the same.
    space = Sobolev(MAX_DEGREE + 1, 2.0, 0.1 * radius, radius=radius)
    maps = np.loadtxt(download(MAPS), comments="%")
    chosen = np.random.default_rng(0).choice(len(maps), 500, replace=False)
    points = [
        (90.0 - colatitude, longitude) for colatitude, longitude in maps[chosen, :2]
    ]
    mm = 1000.0 * model.parameters.length_scale
    values, _, share = modes(STORED)
    print(
        f"{STORED} modes hold {share:.8f} of the trace; the largest {values[0]:.3e} yr^-2"
    )
    for rank in (10, 20, RANK, STORED):
        altimetry, grace = correction_error(
            model,
            obs_degree=96,
            convention="stokes",
            field_space=space,
            points=points,
            rank=rank,
        )
        std = mm * np.sqrt((altimetry**2).sum(axis=1))
        ratio = std / maps[chosen, 5]
        print(
            f"rank {rank:>3}: geoid std over the published, median "
            f"{np.median(ratio):.4f} (5 to 95 per cent {np.percentile(ratio, 5):.3f} "
            f"to {np.percentile(ratio, 95):.3f}); trace on GRACE's data "
            f"{(grace**2).sum():.6e} yr^-2"
        )
    table = np.loadtxt(download(EXPECTATION), comments="%")
    table = table[table[:, 0] >= MIN_DEGREE]
    degrees, orders = table[:, 0].astype(int), table[:, 1].astype(int)
    position = grace_covariance.pyslfp_index(orders < 0, degrees, np.abs(orders))
    expectation = np.empty(len(table))
    expectation[position] = table[:, 2]
    potential = expectation / stokes_coefficient_operator(model, MAX_DEGREE)(
        np.ones(len(expectation))
    )
    components = np.zeros(space.dim)
    components[MIN_DEGREE**2 : MIN_DEGREE**2 + len(potential)] = (
        -space.radius / model.parameters.gravitational_acceleration
    ) * potential
    field = mm * space.point_evaluation_operator(points)(
        space.from_components(components)
    )
    slope = np.polyfit(maps[chosen, 4], field, 1)[0]
    print(
        f"expected geoid rate against the published map: slope {slope:.4f}, "
        f"correlation {np.corrcoef(maps[chosen, 4], field)[0, 1]:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The GIA correction's error from Caron et al. (2018)."
    )
    parser.add_argument("command", choices=["build", "check"])
    args = parser.parse_args()
    if args.command == "build":
        if MODES_FILE.exists():
            # The reduced file may have been copied without the downloads.
            print(f"{MODES_FILE} is there already; remove it to build again")
            return
        for name in [_band_file(band) for band in BANDS] + [EXPECTATION, MAPS]:
            download(name)
        print(f"saved {build()}")
    else:
        _check()


if __name__ == "__main__":
    main()
