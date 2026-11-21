"""
The error covariance of GRACE's Stokes coefficients, from the normal
equations of ITSG-Grace2018 (Heathcote et al., 2026).

ITSG-Grace2018 (Mayer-Guerr et al., 2018; Kvas et al., 2019) publishes, for
each month from 2002-04 to 2017-06, the normal equations of its
unconstrained solution to degree 96 in the SINEX format: the normal matrix
N of the fully normalised coefficients C_lm and S_lm, degrees 2 to 96,
whose inverse is the formal covariance of that month's solution. The
errors are formal: they follow from the stochastic model of the
observations, which ITSG estimates from the residuals, and take no account
of errors in the background models.

The experiments' data are the change between two annual means. With the
months taken as independent, the covariance of the difference of the means
of two years is

    (1 / 144) * sum of the 24 monthly covariances,

each mean being that of 12 months. This module fetches the monthly files,
inverts each normal matrix, and stores that sum; `coefficient_covariance`
reads it back for the degrees a set of data keeps, in the order of
pyslfp's coefficients. The covariance of degrees 2 to L is the block of
the full one, which is that of those coefficients within the solution to
degree 96, the higher degrees being solved for with them and not set to
zero.

The files are large (about 0.44 GB a month) and are kept, under
~/.pyslfp_data/itsg_grace2018/normals_n96, unless --discard-raw is given;
the reduced covariance of a pair of years is one array of 0.7 GB.

    python grace_covariance.py build --years 2007 2008
    python grace_covariance.py list

References

Kvas, A. et al. (2019). ITSG-Grace2018: Overview and evaluation of a new
GRACE-only gravity field time series. J. Geophys. Res. Solid Earth 124,
9332-9344.

Mayer-Guerr, T. et al. (2018). ITSG-Grace2018 - Monthly, Daily and Static
Gravity Field Solutions from GRACE. GFZ Data Services.
doi:10.5880/ICGEM.2018.003.
"""

from __future__ import annotations

import argparse
import gzip
import io
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve

URL = (
    "https://ftp.tugraz.at/pub/ITSG/GRACE/ITSG-Grace2018/monthly/"
    "normals_SINEX/monthly_n96"
)
DATA_DIR = Path.home() / ".pyslfp_data" / "itsg_grace2018"
RAW_DIR = DATA_DIR / "normals_n96"
REDUCED_DIR = DATA_DIR / "reduced"
DEGREE = 96

# The constants of the files, in SI: those of pyslfp's Stokes convention.
REFERENCE_RADIUS = 6378136.3
GRAVITATIONAL_PARAMETER = 3.986004415e14


def file_name(year: int, month: int) -> str:
    return f"ITSG-Grace2018_n{DEGREE}_{year}-{month:02d}.snx.gz"


def _tag(years: Sequence[int]) -> str:
    return "_".join(str(year) for year in years)


def covariance_file(years: Sequence[int]) -> Path:
    return REDUCED_DIR / f"annual_difference_covariance_n{DEGREE}_{_tag(years)}.npy"


def table_file(years: Sequence[int]) -> Path:
    return REDUCED_DIR / f"annual_difference_table_n{DEGREE}_{_tag(years)}.npz"


# ================================================================ #
#                     Fetching and reducing                        #
# ================================================================ #


def download(year: int, month: int, /) -> Path:
    """Downloads one month's file, unless it is already there, under a
    temporary name that is renamed when complete."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / file_name(year, month)
    if target.exists():
        return target
    partial = target.with_suffix(target.suffix + ".part")
    command = ["curl", "-sS", "-L", "--fail", "-C", "-", "-o", str(partial)]
    command += ["--retry", "8", "--retry-delay", "15", "--retry-all-errors"]
    command.append(f"{URL}/{file_name(year, month)}")
    subprocess.run(command, check=True)
    partial.rename(target)
    return target


def read_normals(path: Path, /):
    """
    The normal matrix of a SINEX file and its parameters.

    The block SOLUTION/ESTIMATE lists the parameters in the order of the
    matrix: a type CN or SN, the degree, the order, the estimate and its
    formal standard deviation. The block SOLUTION/NORMAL_EQUATION_MATRIX U
    gives the upper triangle, each line a row, a first column, and up to
    three entries from that column on.

    Returns (N, is_sine, degrees, orders, stds).
    """
    import pandas as pd

    rows = []
    with gzip.open(path, "rt", errors="replace") as lines:
        for line in lines:
            if line.startswith("+SOLUTION/ESTIMATE"):
                break
        for line in lines:
            if line.startswith("-SOLUTION/ESTIMATE"):
                break
            if not line.startswith("*"):
                part = line.split()
                rows.append(
                    (part[1] == "SN", int(part[2]), int(part[4]), float(part[9]))
                )
        kind = None
        for line in lines:
            if line.startswith("+SOLUTION/NORMAL_EQUATION_MATRIX"):
                kind = line.split()[-1]
                break
        if kind != "U":
            raise ValueError(f"{path.name}: no upper-triangular normal matrix.")
        count = len(rows)
        normal = np.zeros((count, count))
        done = False
        while not done:
            chunk = lines.readlines(200_000_000)
            if not chunk:
                break
            for index, line in enumerate(chunk):
                if line[0] in "-%":
                    chunk, done = chunk[:index], True
                    break
            chunk = [line for line in chunk if line[0] != "*"]
            if not chunk:
                continue
            table = pd.read_csv(
                io.StringIO("".join(chunk)),
                sep=r"\s+",
                header=None,
                names=["row", "column", "a", "b", "c"],
            )
            row = table["row"].to_numpy() - 1
            column = table["column"].to_numpy() - 1
            for offset, name in enumerate("abc"):
                values = table[name].to_numpy()
                present = ~np.isnan(values)
                normal[row[present], column[present] + offset] = values[present]
    normal = np.triu(normal) + np.triu(normal, 1).T
    is_sine, degrees, orders, stds = map(np.array, zip(*rows))
    return normal, is_sine, degrees, orders, stds


def inverse(matrix: np.ndarray, /) -> np.ndarray:
    """The inverse of a symmetric positive definite matrix, by Cholesky
    factorisation of the matrix scaled to a unit diagonal."""
    scale = 1.0 / np.sqrt(np.diag(matrix))
    factor = cho_factor(matrix * scale[:, None] * scale[None, :], lower=True)
    return cho_solve(factor, np.eye(len(matrix))) * scale[:, None] * scale[None, :]


def build(years: Sequence[int], /, *, discard_raw: bool = False) -> Path:
    """Fetches the 12 months of each of two years and stores the covariance
    of the difference of their annual means, with the diagonal of each
    month's covariance beside it."""
    if len(years) != 2:
        raise ValueError("the difference is between the means of two years.")
    target = covariance_file(years)
    if target.exists():
        print(f"{target.name}: already built", flush=True)
        return target
    REDUCED_DIR.mkdir(parents=True, exist_ok=True)
    total, table, labels, diagonals = None, None, [], []
    for year in years:
        for month in range(1, 13):
            label = f"{year}-{month:02d}"
            print(f"{label}: downloading...", flush=True)
            path = download(year, month)
            print(f"{label}: inverting...", flush=True)
            normal, is_sine, degrees, orders, _ = read_normals(path)
            covariance = inverse(normal)
            del normal
            this = (is_sine, degrees, orders)
            if table is not None and not all(
                np.array_equal(a, b) for a, b in zip(this, table)
            ):
                raise ValueError(f"{label}: the parameters are in another order.")
            table = this
            labels.append(label)
            diagonals.append(np.diag(covariance).copy())
            total = covariance if total is None else total + covariance
            if discard_raw:
                path.unlink()
    total /= 144.0
    np.save(target, total)
    np.savez(
        table_file(years),
        is_s=table[0],
        l=table[1],
        m=table[2],
        months=np.array(labels),
        diagonals=np.array(diagonals),
    )
    print("done", flush=True)
    return target


def available() -> list[tuple[int, ...]]:
    """The pairs of years whose covariance has been built."""
    prefix = f"annual_difference_covariance_n{DEGREE}_"
    return sorted(
        tuple(int(year) for year in path.stem[len(prefix) :].split("_"))
        for path in REDUCED_DIR.glob(prefix + "*.npy")
    )


# ================================================================ #
#                         Reading back                             #
# ================================================================ #


def pyslfp_index(is_sine, degrees, orders, /, *, min_degree: int = 2) -> np.ndarray:
    """The position of each parameter among pyslfp's coefficients from
    `min_degree` on: ordered by degree and, within a degree, the cosine
    terms of orders 0 to l and then the sine terms of orders 1 to l."""
    return (
        degrees * degrees
        - min_degree * min_degree
        + orders
        + np.where(is_sine, degrees, 0)
    )


def parameter_table(years: Sequence[int], /):
    """The parameters of a built covariance, in the files' order, and the
    diagonal of each month's covariance: (is_sine, degrees, orders, months,
    diagonals)."""
    with np.load(table_file(years)) as table:
        return tuple(table[key] for key in ("is_s", "l", "m", "months", "diagonals"))


def coefficient_covariance(
    obs_degree: int, years: Sequence[int] = (2007, 2008), /
) -> np.ndarray:
    """
    The covariance of the errors of the Stokes coefficients of degrees 2 to
    `obs_degree`, for the difference of the annual means of `years`, in the
    order of pyslfp's coefficients.
    """
    if not covariance_file(years).exists():
        raise FileNotFoundError(
            f"No covariance for {_tag(years)}; run "
            f"`python grace_covariance.py build --years {years[0]} {years[1]}`."
        )
    if not 2 <= obs_degree <= DEGREE:
        raise ValueError(f"the covariance covers degrees 2 to {DEGREE}.")
    is_sine, degrees, orders, _, _ = parameter_table(years)
    keep = np.flatnonzero(degrees <= obs_degree)
    full = np.load(covariance_file(years), mmap_mode="r")
    block = np.array(full[np.ix_(keep, keep)])
    position = pyslfp_index(is_sine[keep], degrees[keep], orders[keep])
    result = np.empty_like(block)
    result[np.ix_(position, position)] = block
    return 0.5 * (result + result.T)


# ================================================================ #
#                        Command line                              #
# ================================================================ #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The error covariance of GRACE's coefficients from the "
        "normal equations of ITSG-Grace2018."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    building = commands.add_parser(
        "build", help="Fetch two years and store the covariance."
    )
    building.add_argument("--years", type=int, nargs=2, default=[2007, 2008])
    building.add_argument(
        "--discard-raw", action="store_true", help="Delete each file once inverted."
    )
    commands.add_parser("list", help="List the covariances built.")
    args = parser.parse_args()
    if args.command == "build":
        build(args.years, discard_raw=args.discard_raw)
    else:
        for years in available():
            print(_tag(years))


if __name__ == "__main__":
    main()
