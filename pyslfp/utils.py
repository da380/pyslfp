"""
Module collecting some small helper functions or classes.
"""

from collections import defaultdict
from typing import Tuple, List


from . import DATADIR


def read_gloss_tide_gauge_data() -> Tuple[List[str], List[float], List[float]]:
    """
    Reads and parses the GLOSS tide gauge data file.

    The function opens the `gloss.txt` file located in the package's
    data directory, reads each line, and parses it to extract the station
    name, latitude, and longitude.

    Returns:
        A tuple containing three lists:
        - A list of station names (str).
        - A list of latitudes (float).
        - A list of longitudes (float).
    """
    file_path = DATADIR + "/tide_gauge/gloss_full.txt"
    # names = []
    lats = []
    lons = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                #          names.append(parts[0])
                lats.append(float(parts[0]))
                lons.append(float(parts[1]))

    # return names, lats, lons
    return lats, lons


def partition_points_by_grid(
    points: List[Tuple[float, float]], grid_size: float
) -> List[List[int]]:
    """
    Partitions a list of (latitude, longitude) points into geographic bins.

    This is highly efficient for grouping dense observations (like satellite
    altimetry tracks) into localized blocks. The resulting blocks can be
    passed to the diagonal_normal_preconditioner in pygeoinf to drastically
    reduce the number of forward model evaluations.

    Args:
        points: A list of (latitude, longitude) tuples in degrees.
        grid_size: The size of the grid cells in degrees.

    Returns:
        A list of lists, where each sub-list contains the original indices
        of the points that fall into the same grid cell.
    """
    bins = defaultdict(list)

    for i, (lat, lon) in enumerate(points):
        # Normalize longitude to [0, 360) to avoid boundary wrapping issues
        lon_norm = lon % 360.0

        # Determine discrete grid cell index using floor division
        lat_idx = int(lat // grid_size)
        lon_idx = int(lon_norm // grid_size)

        bins[(lat_idx, lon_idx)].append(i)

    return list(bins.values())


def get_spherical_harmonic_degree_blocks(
    lmax_obs: int, min_degree: int = 0
) -> list[list[int]]:
    """
    Generates index blocks grouping spherical harmonic coefficients by degree l.
    Assumes standard ordering where degree l coefficients start at index l^2.
    """
    blocks = []
    for l in range(min_degree, lmax_obs + 1):
        start_idx = l**2 - min_degree**2
        end_idx = (l + 1) ** 2 - min_degree**2
        blocks.append(list(range(start_idx, end_idx)))
    return blocks
