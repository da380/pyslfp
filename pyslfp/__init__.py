"""
Unified public imports for the library
"""

# Import shared constants first
from .config import DATADIR

from .ice_ng import IceNG, IceModel
from .physical_parameters import EarthModelParameters
from .finger_print import FingerPrint

from .operators import (
    tide_gauge_operator,
    grace_operator,
    averaging_operator,
    WMBMethod,
    ice_thickness_change_to_load_operator,
    ice_projection_operator,
    ocean_projection_operator,
    land_projection_operator,
    spatial_mutliplication_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    remove_ocean_average_operator,
    ocean_altimetry_operator,
    altimetry_averaging_operator,
    ice_sheet_averaging_operator,
    ice_sheet_basis_operator,
    get_ice_sheet_masks_and_labels,
    standard_ice_groupings,
)


from .plotting import plot, create_map_figure

from .utils import (
    read_gloss_tide_gauge_data,
    partition_points_by_grid,
    get_spherical_harmonic_degree_blocks,
)


def where_is_my_data():
    """Returns the absolute path to the current data directory."""
    return str(DATADIR.absolute())


__all__ = [
    "DATADIR",
    "IceNG",
    "IceModel",
    "EarthModelParameters",
    "FingerPrint",
    "tide_gauge_operator",
    "grace_operator",
    "averaging_operator",
    "WMBMethod",
    "ice_thickness_change_to_load_operator",
    "ice_projection_operator",
    "ocean_projection_operator",
    "land_projection_operator",
    "spatial_mutliplication_operator",
    "plot",
    "plot_corner_distributions",
    "create_map_figure",
    "read_gloss_tide_gauge_data",
    "partition_points_by_grid",
    "get_spherical_harmonic_degree_blocks",
    "sea_level_change_to_load_operator",
    "sea_surface_height_operator",
    "remove_ocean_average_operator",
    "ocean_altimetry_operator",
    "altimetry_averaging_operator",
    "ice_sheet_averaging_operator",
    "ice_sheet_basis_operator",
    "get_ice_sheet_masks_and_labels",
    "standard_ice_groupings",
]
