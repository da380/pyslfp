"""
The DDK filters of GRACE's Stokes coefficients (Heathcote et al., 2026).

The DDK filters (Kusche, 2007; Kusche et al., 2009) are the decorrelating
filters most widely applied to unconstrained GRACE solutions. Each is a
fixed matrix W, and the filtered coefficients are W times the unfiltered
ones. It is built from an approximation of the normal matrix N of one
month, block diagonal by order, and a signal covariance that falls as a
power of the degree, with inverse M:

    W = (N + a M)^-1 N ,

which is the estimate a Gaussian prior of that covariance would give with
that noise, the weight a setting how strongly it smooths. The filters are
numbered DDK1 (the strongest) to DDK8.

The matrices are those published by Roelof Rietbroek with the software of
Kusche et al. (2009), at https://github.com/strawpants/GRACE-filter (MIT
licence), for degrees 2 to 120. Each is block diagonal: a block for the
cosine coefficients of order 0, then for each order a block for the cosine
and one for the sine coefficients, each over the degrees from the larger of
2 and the order. This module fetches a matrix, reads its binary format, and
gives the filter of the coefficients of degrees 2 to L, L at most 120, in
the order of pyslfp's coefficients: the leading part of each block, as the
published routine takes it. The files are kept under ~/.pyslfp_data/ddk.

    python ddk_filter.py fetch --levels 3 5
    python ddk_filter.py check          the published test case, DDK2

References

Kusche, J. (2007). Approximate decorrelation and non-isotropic smoothing of
time-variable GRACE-type gravity field models. J. Geodesy 81, 733-749.

Kusche, J., Schmidt, R., Petrovic, S. and Rietbroek, R. (2009).
Decorrelated GRACE time-variable gravity solutions by GFZ, and their
validation using a hydrological model. J. Geodesy 83, 903-913.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
from pathlib import Path

import numpy as np
from scipy import sparse

URL = "https://raw.githubusercontent.com/strawpants/GRACE-filter/master"
DATA_DIR = Path.home() / ".pyslfp_data" / "ddk"
MAX_DEGREE, MIN_DEGREE = 120, 2

# The weight of each filter, as its file is named.
WEIGHTS = {
    1: "1d14",
    2: "1d13",
    3: "1d12",
    4: "5d11",
    5: "1d11",
    6: "5d10",
    7: "1d10",
    8: "5d9",
}
TEST_FILES = (
    "GSM-2_2008122-2008153_0030_EIGEN_G---_0004in",
    "GSM-2_2008122-2008153_0030_EIGEN_G---_0004out",
)


def file_name(level: int) -> str:
    if level not in WEIGHTS:
        raise ValueError(f"the filters are DDK1 to DDK8, not DDK{level}.")
    return f"Wbd_2-120.a_{WEIGHTS[level]}p_4"


def _download(remote: str, target: Path) -> Path:
    """Downloads a file of the repository, unless it is already there,
    under a temporary name that is renamed when complete."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    command = ["curl", "-sS", "-L", "--fail", "-o", str(partial)]
    command += ["--retry", "5", "--retry-delay", "10", f"{URL}/{remote}"]
    subprocess.run(command, check=True)
    partial.rename(target)
    return target


def fetch(level: int, /) -> Path:
    return _download(f"data/DDK/{file_name(level)}", DATA_DIR / file_name(level))


# ================================================================ #
#                            Reading                               #
# ================================================================ #


def read_blocks(path: Path, /) -> list[np.ndarray]:
    """
    The blocks of a block-diagonal matrix in the binary format of the
    published filters, after its reader `read_BIN.m`: a header, the ends of
    the blocks along the side, and the blocks in full, column by column.
    """
    raw = path.read_bytes()
    order = "<" if struct.unpack_from("<H", raw, 0)[0] == 18754 else ">"
    position = 2

    def take(code: str, count: int = 1):
        nonlocal position
        values = struct.unpack_from(f"{order}{count}{code}", raw, position)
        position += struct.calcsize(f"{order}{count}{code}")
        return values

    def text(count: int) -> str:
        nonlocal position
        chunk = raw[position : position + count]
        position += count
        return chunk.decode("ascii", errors="replace")

    version = float(text(6)[2:])
    kind = text(8)
    text(80)
    n_ints, n_doubles, side, side_2 = take("I", 4)
    packed, packed_2 = take("I", 2) if version < 2.4 else take("Q", 2)
    if kind != "BDFULLV0":
        raise ValueError(f"{path.name}: a matrix of type {kind}, not BDFULLV0.")
    if version <= 2.1:
        # The early versions hold no vectors or notes, and one packed array.
        n_vectors, n_readme, packed_2, side_2 = 0, 0, 1, side
    else:
        n_vectors, n_readme = take("i", 2)
    (n_blocks,) = take("i")
    text(80 * n_readme)
    text(24 * n_ints)
    take("i" if version <= 2.4 else "q", n_ints)
    text(24 * n_doubles)
    take("d", n_doubles)
    text(24 * side)
    ends = take("i", n_blocks)
    if version > 2.2:
        text(24 * side_2)
    position += 8 * n_vectors * side
    values = np.frombuffer(
        raw, dtype=f"{order}f8", count=packed * packed_2, offset=position
    )
    blocks, start, last = [], 0, 0
    for end in ends:
        size = end - last
        blocks.append(values[start : start + size * size].reshape(size, size).T)
        start, last = start + size * size, end
    if start != len(values) or last != side:
        raise ValueError(f"{path.name}: the blocks do not fill the matrix.")
    return blocks


def filter_matrix(level: int, obs_degree: int, /) -> sparse.csr_matrix:
    """
    The filter DDK`level` of the Stokes coefficients of degrees 2 to
    `obs_degree`, as a sparse matrix in the order of pyslfp's coefficients:
    by degree and, within a degree, the cosine terms of orders 0 to l and
    then the sine terms of orders 1 to l.
    """
    if not MIN_DEGREE <= obs_degree <= MAX_DEGREE:
        raise ValueError(f"the filters cover degrees {MIN_DEGREE} to {MAX_DEGREE}.")
    blocks = read_blocks(fetch(level))
    if len(blocks) != 2 * MAX_DEGREE + 1:
        raise ValueError("the matrix has not a block for each order and phase.")
    rows, columns, values = [], [], []
    for index, block in enumerate(blocks):
        # Block 0 is the cosine of order 0; then cosine and sine by order.
        order = (index + 1) // 2
        sine = index > 0 and index % 2 == 0
        first = max(MIN_DEGREE, order)
        if len(block) != MAX_DEGREE - first + 1:
            raise ValueError(f"block {index} has not the size of order {order}.")
        if first > obs_degree:
            continue
        degrees = np.arange(first, obs_degree + 1)
        place = degrees * degrees - MIN_DEGREE**2 + order + (degrees if sine else 0)
        kept = block[: len(degrees), : len(degrees)]
        rows.append(np.repeat(place, len(degrees)))
        columns.append(np.tile(place, len(degrees)))
        values.append(kept.ravel())
    count = (obs_degree + 1) ** 2 - MIN_DEGREE**2
    return sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(count, count),
    )


# ================================================================ #
#                     The published test case                      #
# ================================================================ #


def _read_gsm(path: Path, degree: int) -> np.ndarray:
    """The coefficients of degrees 2 to `degree` of a test file, lines
    `l m C S ...` after a header line, in the order of pyslfp's
    coefficients."""
    out = np.zeros((degree + 1) ** 2 - MIN_DEGREE**2)
    for line in path.read_text(errors="replace").splitlines()[1:]:
        part = line.split()
        if len(part) < 4:
            continue
        l, m = int(part[0]), int(part[1])
        if MIN_DEGREE <= l <= degree:
            base = l * l - MIN_DEGREE**2
            out[base + m] = float(part[2])
            if m > 0:
                out[base + l + m] = float(part[3])
    return out


def check(degree: int = 120) -> float:
    """Applies DDK2 to the published test input and returns the largest
    difference from the published output, relative to its largest
    coefficient."""
    source, expected = (
        _read_gsm(_download(f"tests/{name}", DATA_DIR / "tests" / name), degree)
        for name in TEST_FILES
    )
    filtered = filter_matrix(2, degree) @ source
    return float(np.max(np.abs(filtered - expected)) / np.max(np.abs(expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description="The DDK filters of GRACE.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetching = commands.add_parser("fetch", help="Download filter matrices.")
    fetching.add_argument("--levels", nargs="+", type=int, default=[3, 5])
    commands.add_parser("check", help="Run the published test case.")
    args = parser.parse_args()
    if args.command == "fetch":
        for level in args.levels:
            print(fetch(level))
    else:
        print(f"largest relative difference from the published output: {check():.1e}")


if __name__ == "__main__":
    main()
