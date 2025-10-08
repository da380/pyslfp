"""
Unified public imports for the library
"""

# Import shared constants first
from .config import DATADIR

from pyslfp.ice_ng import IceNG, IceModel
from pyslfp.physical_parameters import EarthModelParameters
from pyslfp.finger_print import FingerPrint

from pyslfp.operators import (
    tide_gauge_operator,
    grace_operator,
    field_to_sh_coefficient_operator,
    sh_coefficient_to_field_operator,
    averaging_operator,
    WMBMethod,
    ice_thickness_change_to_load_operator,
    ice_projection_operator,
    ocean_projection_operator,
    land_projection_operator,
    spatial_mutliplication_operator,
)


from pyslfp.plotting import plot

from pyslfp.utils import SHVectorConverter, read_gloss_tide_gauge_data


__all__ = [
    "DATADIR",
    "IceNG",
    "IceModel",
    "EarthModelParameters",
    "FingerPrint",
    "tide_gauge_operator",
    "grace_operator",
    "field_to_sh_coefficient_operator",
    "sh_coefficient_to_field_operator",
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
]
