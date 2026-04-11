"""
Data management for Tide Gauge networks.
"""

from typing import Tuple, List
from .downloader import ensure_data


def read_gloss_tide_gauge_data() -> Tuple[List[float], List[float]]:
    """
    Reads and parses the GLOSS tide gauge network coordinates.

    Automatically fetches the GLOSS dataset from Zenodo if not present locally.

    Returns:
        Tuple[List[float], List[float]]: Two lists containing the
            latitudes and longitudes of the tide gauge stations.
    """
    # Use our V2 robust downloader!
    data_dir = ensure_data("TIDE_GAUGE")
    file_path = data_dir / "gloss_full.txt"

    lats = []
    lons = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                lats.append(float(parts[0]))
                lons.append(float(parts[1]))

    return lats, lons
