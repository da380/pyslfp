"""
Realistic surface loads from the updated ESA Earth System Model
(Heathcote et al., 2026).

The updated ESA Earth System Model (Dobslaw et al., 2015) is a synthetic
but realistic model of the time-variable gravity field, built for
simulation studies of gravity missions and independent of any inversion of
GRACE data. It gives the Stokes coefficients of its components separately,
to degree 180 every six hours from 1995 to 2006: the atmosphere (A), the
ocean (O), terrestrially stored water (H), land ice (I) and the solid Earth
(S). The experiments use the H, I and O components as loads whose values
are known exactly: some to build a prior from, the rest held out to test
the estimators on.

This module fetches the yearly archives, reduces each to the monthly means
of its coefficients, and reads those back as loads on a model's grid. The
archives are large (about 0.5 GB per component and year) and the monthly
means small, so an archive is deleted once it has been reduced unless
--keep-raw is given. Everything is kept under ~/.pyslfp_data/esaesm, beside
the data pyslfp itself downloads.

    python esm_loads.py fetch                     H, I and O, 1995 to 2006
    python esm_loads.py fetch --components H --years 1995 1996
    python esm_loads.py list                      what has been reduced

The coefficients are those of the potential of each component's mass
together with the elastic response of the solid Earth to it, the surface
pressure P of the mass giving (Gruber et al., 2011, eq. 5)

    C_nm = a^2 (1 + k_n) / ((2n + 1) M g) * integral of P Y_nm dS,

with Y_nm the fully normalised harmonics, a, M and g the Earth's radius,
mass and surface gravity, and k_n the load Love numbers. The equivalent
water thickness of the mass therefore has the coefficients

    h_nm = (a rho_E / (3 rho_w)) (2n + 1) / (1 + k_n) * C_nm,

with rho_E the Earth's mean density: the relation of Wahr et al. (1998).
`water_thickness_coefficients` applies it with the load Love numbers of
the model in use, which differ from those of the Earth System Model
(Farrell, 1972) by a few per cent at most; at degree 1 the factor 1 + k_1
is taken as one, the centre of mass of the solid Earth being the origin
there. The loads serve as realistic fields, not as a reproduction of the
Earth System Model, so these differences do not matter.

References

Dobslaw, H. et al. (2015). The updated ESA Earth System Model for future
gravity mission simulation studies. J. Geodesy 89, 505-513. Data:
doi:10.5880/GFZ.1.3.2014.001.

Gruber, Th. et al. (2011). Simulation of the time-variable gravity field by
means of coupled geophysical models. Earth Syst. Sci. Data 3, 19-35.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import tarfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

URL = "https://isdc-data.gfz.de/future_missions/esaesm/ESAESM2/mtmshc"
DATA_DIR = Path.home() / ".pyslfp_data" / "esaesm"
RAW_DIR = DATA_DIR / "raw"
MONTHLY_DIR = DATA_DIR / "monthly"

COMPONENTS = ("A", "O", "H", "I", "S", "AOHIS")
YEARS = tuple(range(1995, 2007))
DEGREE = 180

# The constants of the Earth System Model's files, in SI.
REFERENCE_RADIUS = 6378137.0
GRAVITATIONAL_PARAMETER = 0.3986005e15
GRAVITATIONAL_CONSTANT = 6.67430e-11
WATER_DENSITY = 1000.0


def archive_name(component: str, year: int) -> str:
    return f"mtm_{year}_{component}.tar.gz"


def monthly_file(component: str, year: int) -> Path:
    return MONTHLY_DIR / f"mtm_{year}_{component}.npz"


# ================================================================ #
#                     Fetching and reducing                        #
# ================================================================ #


def download(component: str, year: int, /, *, limit_rate: Optional[str] = None) -> Path:
    """Downloads one yearly archive, unless it is already there. The file
    is written under a temporary name and renamed when complete, and an
    interrupted download is resumed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / archive_name(component, year)
    if target.exists():
        return target
    partial = target.with_suffix(target.suffix + ".part")
    # The server drops a connection now and then; curl retries and resumes.
    command = ["curl", "-sS", "-L", "--fail", "-C", "-", "-o", str(partial)]
    command += ["--retry", "8", "--retry-delay", "15", "--retry-all-errors"]
    if limit_rate:
        command += ["--limit-rate", limit_rate]
    command.append(f"{URL}/{archive_name(component, year)}")
    subprocess.run(command, check=True)
    partial.rename(target)
    return target


def _read_coefficients(raw: bytes):
    """The degrees, orders and coefficients of one file in the ICGEM
    format: a header closed by `end_of_head`, then lines `gfc n m C S`."""
    text = raw.decode("ascii")
    body = text[text.index("end_of_head") :].split("\n", 1)[1]
    table = np.loadtxt(io.StringIO(body), usecols=(1, 2, 3, 4))
    return table[:, 0].astype(int), table[:, 1].astype(int), table[:, 2], table[:, 3]


def reduce(component: str, year: int, /) -> Path:
    """Reduces a yearly archive to the monthly means of its coefficients,
    stored as arrays C and S of shape (12, 181, 181), indexed by month,
    degree and order, with the number of fields averaged in each month."""
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    target = monthly_file(component, year)
    if target.exists():
        return target
    c_sum = np.zeros((12, DEGREE + 1, DEGREE + 1))
    s_sum = np.zeros_like(c_sum)
    counts = np.zeros(12, dtype=int)
    with tarfile.open(RAW_DIR / archive_name(component, year), "r:gz") as archive:
        for member in archive:
            if not member.name.endswith(".gfc"):
                continue
            # The file is named mtmshc_<component>_<yyyymmdd>_<hh>.gfc.
            date = Path(member.name).stem.split("_")[-2]
            if int(date[:4]) != year:
                raise ValueError(f"{member.name} is not a file of {year}.")
            month = int(date[4:6]) - 1
            degrees, orders, c, s = _read_coefficients(
                archive.extractfile(member).read()
            )
            if degrees.max() != DEGREE:
                raise ValueError(f"{member.name} does not reach degree {DEGREE}.")
            c_sum[month, degrees, orders] += c
            s_sum[month, degrees, orders] += s
            counts[month] += 1
    if np.any(counts == 0):
        raise ValueError(f"{archive_name(component, year)} lacks a month: {counts}.")
    np.savez_compressed(
        target,
        C=c_sum / counts[:, None, None],
        S=s_sum / counts[:, None, None],
        counts=counts,
    )
    return target


def fetch(
    components: Sequence[str],
    years: Sequence[int],
    /,
    *,
    keep_raw: bool = False,
    limit_rate: Optional[str] = None,
) -> None:
    """Downloads and reduces each archive in turn, one at a time."""
    for year in years:
        for component in components:
            if monthly_file(component, year).exists():
                print(f"{archive_name(component, year)}: already reduced", flush=True)
                continue
            print(f"{archive_name(component, year)}: downloading...", flush=True)
            archive = download(component, year, limit_rate=limit_rate)
            print(f"{archive_name(component, year)}: reducing...", flush=True)
            reduce(component, year)
            if not keep_raw:
                archive.unlink()
    print("done", flush=True)


def available(component: str) -> list[int]:
    """The years of a component that have been reduced."""
    return [year for year in YEARS if monthly_file(component, year).exists()]


# ================================================================ #
#                         Reading back                             #
# ================================================================ #


def monthly_coefficients(component: str, years: Optional[Sequence[int]] = None):
    """The monthly mean Stokes coefficients of a component over `years`
    (all those reduced, if omitted), in order of time.

    Returns (months, C, S): the months as (year, month) pairs and the
    coefficients as arrays of shape (n_months, 181, 181)."""
    years = available(component) if years is None else list(years)
    if not years:
        raise FileNotFoundError(
            f"No reduced data for component {component}; run `esm_loads.py fetch`."
        )
    months, c, s = [], [], []
    for year in years:
        with np.load(monthly_file(component, year)) as data:
            c.append(data["C"])
            s.append(data["S"])
        months += [(year, month) for month in range(1, 13)]
    return months, np.concatenate(c), np.concatenate(s)


def water_thickness_coefficients(c, s, load_love_k, /):
    """
    Converts Stokes coefficients to those of the equivalent water thickness
    of the mass, in metres, by the relation at the head of this module.

    Args:
        c, s: The Stokes coefficients, arrays whose last two axes are
            degree and order.
        load_love_k: The conventional load Love numbers k_n by degree, from
            degree 0, reaching at least the degree of the coefficients. The
            values at degrees 0 and 1 are not used: the factor 1 + k_n is
            taken as one there.

    Returns:
        The coefficients of the thickness, fully normalised as the Stokes
        coefficients are.
    """
    c, s = np.asarray(c), np.asarray(s)
    lmax = c.shape[-2] - 1
    degrees = np.arange(lmax + 1)
    elastic = 1.0 + np.asarray(load_love_k, dtype=float)[: lmax + 1]
    elastic[:2] = 1.0
    earth_mass = GRAVITATIONAL_PARAMETER / GRAVITATIONAL_CONSTANT
    mean_density = earth_mass / (4.0 / 3.0 * np.pi * REFERENCE_RADIUS**3)
    factor = (
        REFERENCE_RADIUS
        * mean_density
        / (3.0 * WATER_DENSITY)
        * (2.0 * degrees + 1.0)
        / elastic
    )
    return c * factor[:, None], s * factor[:, None]


def interval_changes(values: np.ndarray, /, *, months: int = 12, step: int = 12):
    """
    The changes of a monthly series between the means of successive
    intervals: the mean over `months` months less the mean over the
    `months` before, the pair advancing by `step` months. With the
    defaults these are the differences of successive calendar-year means,
    which do not overlap.

    Returns (starts, changes): the index of the first month of each later
    interval, and the changes, stacked along the first axis.
    """
    starts = list(range(months, values.shape[0] - months + 1, step))
    changes = np.array(
        [
            values[k : k + months].mean(axis=0) - values[k - months : k].mean(axis=0)
            for k in starts
        ]
    )
    return starts, changes


def _on_model_grid(model, coefficients, lat_mesh, lon_mesh):
    """A field given by coefficients to degree 180, at its full degree on
    the model's grid.

    A model of degree 180 or more synthesises it directly. For one of lower
    degree on an equally spaced grid, the field is synthesised on a finer
    grid of the same kind whose points include the model's, and those are
    kept; on any other grid it is evaluated point by point, which is slow.
    """
    if model.lmax >= DEGREE:
        return model.expand_coefficient(coefficients.pad(model.lmax))
    field = model.zero_grid()
    if model.grid == "DH":
        # A grid of degree L has 2 (L + 1) latitudes from the north pole, so
        # one of degree m (L + 1) - 1 holds every point of it, m apart.
        ratio = -(-(DEGREE + 1) // (model.lmax + 1))
        fine = coefficients.pad(ratio * (model.lmax + 1) - 1).expand(
            grid="DH2" if model.sampling == 2 else "DH", extend=model.extend
        )
        field.data[:] = fine.data[::ratio, ::ratio]
    else:
        field.data[:] = coefficients.expand(
            lat=lat_mesh.ravel(), lon=lon_mesh.ravel()
        ).reshape(field.data.shape)
    return field


def load_changes(
    model,
    component: str,
    /,
    *,
    years: Optional[Sequence[int]] = None,
    months: int = 12,
    step: int = 12,
):
    """
    The changes in a component's load between the means of successive
    intervals (`interval_changes`), as fields on an Earth model's grid in
    the model's units of load.

    The coefficients are converted to equivalent water thickness with the
    model's own load Love numbers and evaluated on the model's grid at their
    full degree of 180, whatever the model's own degree. A model of lower
    degree then holds samples of the full field and not a truncation of it,
    which would ring about the sharp edges of the ice and the land beyond
    what the Earth System Model's own truncation at degree 180 already does.
    Nothing is removed: a component's mass need not be conserved, and its
    degrees 0 and 1 are kept.

    Args:
        model: The pyslfp EarthModel.
        component: "H", "I", "O" or another component that has been reduced.
        years: The years to use; all those reduced if omitted.
        months, step: As for `interval_changes`.

    Returns:
        (labels, fields): for each change the first month of its later
        interval, as "yyyy-mm", and the load as a field on the grid.
    """
    from pyshtools import SHCoeffs

    dates, c, s = monthly_coefficients(component, years)
    starts, dc = interval_changes(c, months=months, step=step)
    _, ds = interval_changes(s, months=months, step=step)
    k = model.love_numbers.conventional()["k"]
    if len(k) <= DEGREE:
        k = np.concatenate([k, np.full(DEGREE + 1 - len(k), k[-1])])
    hc, hs = water_thickness_coefficients(dc, ds, k)
    p = model.parameters
    to_load = p.water_density / p.length_scale
    fields = []
    template = model.zero_grid()
    lat_mesh, lon_mesh = np.meshgrid(template.lats(), template.lons(), indexing="ij")
    for cosine, sine in zip(hc, hs):
        coefficients = SHCoeffs.from_array(
            np.stack([cosine, sine]), normalization="4pi", csphase=1
        ).convert(normalization=model.normalization, csphase=model.csphase)
        fields.append(_on_model_grid(model, coefficients, lat_mesh, lon_mesh) * to_load)
    labels = [f"{dates[k][0]}-{dates[k][1]:02d}" for k in starts]
    return labels, fields


# ================================================================ #
#                          Command line                            #
# ================================================================ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and reduce the updated ESA Earth System Model."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetching = commands.add_parser("fetch", help="Download and reduce archives.")
    fetching.add_argument(
        "--components", nargs="+", default=["H", "I", "O"], choices=COMPONENTS
    )
    fetching.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    fetching.add_argument(
        "--keep-raw", action="store_true", help="Keep the archives once reduced."
    )
    fetching.add_argument(
        "--limit-rate", default=None, help="A download rate limit for curl, as 2M."
    )
    commands.add_parser("list", help="List what has been reduced.")
    args = parser.parse_args()

    if args.command == "fetch":
        fetch(
            args.components,
            args.years,
            keep_raw=args.keep_raw,
            limit_rate=args.limit_rate,
        )
    else:
        for component in COMPONENTS:
            years = available(component)
            if years:
                print(f"{component}: {', '.join(str(year) for year in years)}")


if __name__ == "__main__":
    main()
