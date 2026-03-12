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
)


from .plotting import plot

from .utils import SHVectorConverter, read_gloss_tide_gauge_data


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
    "read_gloss_tide_gauge_data",
    "SHVectorConverter",
    "sea_level_change_to_load_operator",
    "sea_surface_height_operator",
    "remove_ocean_average_operator",
]
