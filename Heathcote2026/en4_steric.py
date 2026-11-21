"""
Steric height from the EN4 analyses of ocean temperature and salinity
(Heathcote et al., 2026).

EN4 (Good et al., 2013) is the Met Office Hadley Centre's objective
analysis of the ocean's temperature and salinity profiles: monthly fields
on a grid of one degree and 42 levels, from 1900. The experiments use it as
they use the ESA Earth System Model for the loads (`esm_loads.py`): for the
size and the scale of the year-on-year changes of steric height, which a
prior states, and for changes that no prior has seen, to test on.

The steric height of a column is its expansion relative to a reference,

    s = - integral of (rho(S, T, p) - rho_ref(p)) / rho_0 dz ,

with rho the in-situ density (TEOS-10, through the `gsw` package), rho_ref
that of water of salinity 35 at 0 degrees C at the same pressure, and rho_0
= 1027 kg m^-3. Only its changes are used, in which the reference cancels.
The integral is taken over the whole column and, separately, over its upper
700 m and 2000 m, the levels below the analysis's sea floor being left out.
The analysis is smooth, has no eddies, and relaxes to its climatology where
there are few profiles, which before the Argo floats (about 2005) is much
of the Southern Ocean and everything below 2000 m.

This module fetches the yearly archives (about 0.3 GB each), reduces each
to the annual mean of the steric height, and reads those back. Everything
is kept under ~/.pyslfp_data/en4. It needs the packages `gsw` and
`netCDF4`, which pyslfp does not.

    python en4_steric.py fetch                    1995 to 2020
    python en4_steric.py fetch --years 2007 2008
    python en4_steric.py list                     what has been reduced

The temperature of the files is taken to be potential temperature, as the
EN4 documentation states. Version EN.4.2.2 with the bias corrections of
Gouretski and Reseghetti (2010), the files labelled g10.

References

Good, S. A., Martin, M. J. and Rayner, N. A. (2013). EN4: quality
controlled ocean temperature and salinity profiles and monthly objective
analyses with uncertainty estimates. J. Geophys. Res. Oceans 118,
6704-6716.

Gouretski, V. and Reseghetti, F. (2010). On depth and temperature biases in
bathythermograph data. Deep-Sea Res. I 57, 812-833.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

URL = "https://www.metoffice.gov.uk/hadobs/en4/data/en4-2-1/EN.4.2.2"
VARIANT = "g10"
DATA_DIR = Path.home() / ".pyslfp_data" / "en4"
RAW_DIR = DATA_DIR / "raw"
ANNUAL_DIR = DATA_DIR / "annual"

YEARS = tuple(range(1995, 2021))
REFERENCE_DENSITY = 1027.0
# The depths to which the partial integrals are taken, in metres.
PARTIAL_DEPTHS = (700.0, 2000.0)
FIELDS = ("steric", "steric_700", "steric_2000")


def archive_name(year: int) -> str:
    return f"EN.4.2.2.analyses.{VARIANT}.{year}.zip"


def annual_file(year: int) -> Path:
    return ANNUAL_DIR / f"steric_{VARIANT}_{year}.npz"


# ================================================================ #
#                     Fetching and reducing                        #
# ================================================================ #


def download(year: int, /) -> Path:
    """Downloads one yearly archive, unless it is already there, under a
    temporary name that is renamed when complete."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / archive_name(year)
    if target.exists():
        return target
    partial = target.with_suffix(target.suffix + ".part")
    command = ["curl", "-sS", "-L", "--fail", "-C", "-", "-o", str(partial)]
    command += ["--retry", "8", "--retry-delay", "15", "--retry-all-errors"]
    command.append(f"{URL}/{archive_name(year)}")
    subprocess.run(command, check=True)
    partial.rename(target)
    return target


def _expansion_of_month(path: Path):
    """The expansion of each level of one monthly file, -(rho - rho_ref) /
    rho_0, with the levels that hold water, the grid and the levels'
    bounds."""
    import gsw
    from netCDF4 import Dataset

    with Dataset(path) as data:
        lat = np.asarray(data["lat"][:], dtype=float)
        lon = np.asarray(data["lon"][:], dtype=float)
        depth = np.asarray(data["depth"][:], dtype=float)
        bounds = np.asarray(data["depth_bnds"][:], dtype=float)
        temperature = np.ma.filled(data["temperature"][0].astype(float), np.nan)
        salinity = np.ma.filled(data["salinity"][0].astype(float), np.nan)
    wet = np.isfinite(temperature) & np.isfinite(salinity)
    pressure = gsw.p_from_z(-depth[:, None, None], lat[None, :, None])
    pressure = np.broadcast_to(pressure, temperature.shape)
    absolute = gsw.SA_from_SP(
        salinity, pressure, lon[None, None, :], lat[None, :, None]
    )
    conservative = gsw.CT_from_pt(absolute, temperature - 273.15)
    density = gsw.rho(absolute, conservative, pressure)
    reference = gsw.rho(35.16504, gsw.CT_from_pt(35.16504, 0.0), pressure)
    expansion = np.where(wet, -(density - reference) / REFERENCE_DENSITY, 0.0)
    return lat, lon, bounds, wet, expansion


def reduce(year: int, /) -> Path:
    """Reduces a yearly archive to the annual mean of the steric height:
    arrays of shape (173, 360) by latitude and longitude, in metres, NaN on
    land, with the latitudes, the longitudes and the depth of the sea
    floor. A level counts only where it holds water in all twelve months,
    so that the mean is of one column throughout."""
    ANNUAL_DIR.mkdir(parents=True, exist_ok=True)
    target = annual_file(year)
    if target.exists():
        return target
    # The archive is unpacked into a directory of its own, removed after.
    unpacked = DATA_DIR / f"unpacked_{year}"
    if unpacked.exists():
        shutil.rmtree(unpacked)
    unpacked.mkdir(parents=True)
    try:
        with zipfile.ZipFile(RAW_DIR / archive_name(year)) as archive:
            members = sorted(m for m in archive.namelist() if m.endswith(".nc"))
            if len(members) != 12 or any("/" in m or ".." in m for m in members):
                raise ValueError(f"{archive_name(year)}: unexpected contents.")
            archive.extractall(unpacked, members=members)
        total, common = None, None
        for member in members:
            lat, lon, bounds, wet, expansion = _expansion_of_month(unpacked / member)
            total = expansion if total is None else total + expansion
            common = wet if common is None else common & wet
    finally:
        shutil.rmtree(unpacked)
    mean = np.where(common, total / 12.0, 0.0)
    ocean = common[0]
    fields = {}
    for name, limit in zip(FIELDS, (np.inf,) + PARTIAL_DEPTHS):
        thickness = np.clip(np.minimum(bounds[:, 1], limit) - bounds[:, 0], 0.0, None)
        fields[name] = np.where(ocean, np.einsum("k,kij->ij", thickness, mean), np.nan)
    floor = np.einsum("k,kij->ij", bounds[:, 1] - bounds[:, 0], common.astype(float))
    np.savez_compressed(target, lat=lat, lon=lon, depth=floor, **fields)
    return target


def fetch(years: Sequence[int], /, *, keep_raw: bool = True) -> None:
    """Downloads and reduces each year in turn."""
    for year in years:
        if annual_file(year).exists():
            print(f"{year}: already reduced", flush=True)
            continue
        print(f"{year}: downloading...", flush=True)
        archive = download(year)
        print(f"{year}: reducing...", flush=True)
        reduce(year)
        if not keep_raw:
            archive.unlink()
    print("done", flush=True)


def available() -> list[int]:
    """The years that have been reduced."""
    return [year for year in YEARS if annual_file(year).exists()]


# ================================================================ #
#                         Reading back                             #
# ================================================================ #


def annual_means(years: Optional[Sequence[int]] = None, /, *, field: str = "steric"):
    """
    The annual means of the steric height over `years` (all those reduced,
    if omitted), on the grid of the analysis.

    Returns (years, lat, lon, depths, values): the latitudes and longitudes
    of the grid in degrees, each year's depth of the sea floor in metres
    (zero on land), and the steric height in metres, NaN on land; the last
    two of shape (n_years, 173, 360).
    """
    if field not in FIELDS:
        raise ValueError(f"the fields are {FIELDS}, not '{field}'.")
    years = available() if years is None else list(years)
    if not years:
        raise FileNotFoundError("No reduced data; run `en4_steric.py fetch`.")
    depths, values = [], []
    for year in years:
        with np.load(annual_file(year)) as data:
            lat, lon = data["lat"], data["lon"]
            depths.append(data["depth"])
            values.append(data[field])
    return years, lat, lon, np.array(depths), np.array(values)


def annual_changes(years: Optional[Sequence[int]] = None, /, *, field: str = "steric"):
    """
    The changes of the steric height between the means of successive years.
    A column whose sea floor differs between the two years, as a few of the
    analysis's do, has no change and is NaN.

    Returns (labels, lat, lon, depth, changes): for each change the later
    year; the depth of the sea floor, where it is the same in every year,
    and zero elsewhere; and the changes in metres, of shape (n_years - 1,
    173, 360), NaN on land.
    """
    years, lat, lon, depths, values = annual_means(years, field=field)
    if any(b - a != 1 for a, b in zip(years, years[1:])):
        raise ValueError("the years are not consecutive.")
    changes = np.where(depths[1:] == depths[:-1], np.diff(values, axis=0), np.nan)
    depth = np.where(np.all(depths == depths[0], axis=0), depths[0], 0.0)
    return years[1:], lat, lon, depth, changes


def changes_at(lats, lons, years: Sequence[int], /, *, field: str = "steric"):
    """
    The changes of `annual_changes` over `years` at the points of another
    grid, by linear interpolation: `lats` and `lons` in degrees, of any one
    shape. Land, and columns with no change, are zero.

    Returns (labels, changes): the changes in metres, one array of the
    shape of `lats` for each.
    """
    from scipy.interpolate import RegularGridInterpolator

    labels, lat, lon, _, changes = annual_changes(years, field=field)
    around = np.concatenate([[lon[-1] - 360.0], lon, [lon[0] + 360.0]])
    points = np.stack([np.asarray(lats, float), np.asarray(lons, float) % 360.0], -1)
    out = []
    for change in np.where(np.isfinite(changes), changes, 0.0):
        wrapped = np.concatenate([change[:, -1:], change, change[:, :1]], axis=1)
        interpolate = RegularGridInterpolator(
            (lat, around), wrapped, bounds_error=False, fill_value=0.0
        )
        out.append(interpolate(points))
    return labels, out


# ================================================================ #
#                          Command line                            #
# ================================================================ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annual means of steric height from the EN4 analyses."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetching = commands.add_parser("fetch", help="Download and reduce years.")
    fetching.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    fetching.add_argument(
        "--discard-raw", action="store_true", help="Delete each archive once reduced."
    )
    commands.add_parser("list", help="List the years reduced.")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.years, keep_raw=not args.discard_raw)
    else:
        print(" ".join(str(year) for year in available()))


if __name__ == "__main__":
    main()
