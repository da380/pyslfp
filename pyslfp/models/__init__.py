"""
Model spaces for sea level inversions, built from named load components.

`components` holds the vocabulary: the `LoadComponent` record, its two
roles, and the factories for ice sheets, land water storage, firn,
dynamic topography and ocean density. `setup` holds the assembler,
`ModelSetup`, which derives a model space and its operators from a list of
components.
"""

from .components import (
    ICE_SURFACE_HEIGHT,
    SEA_SURFACE_HEIGHT,
    STERIC_SEA_LEVEL,
    LoadComponent,
    Role,
    barystatic_gmsl_operator,
    dynamic_topography,
    firn,
    ice_sheets,
    land_water_storage,
    ocean_density,
)
from .setup import OCEAN_MASS, ModelSetup

__all__ = [
    "LoadComponent",
    "Role",
    "SEA_SURFACE_HEIGHT",
    "ICE_SURFACE_HEIGHT",
    "STERIC_SEA_LEVEL",
    "ice_sheets",
    "land_water_storage",
    "firn",
    "dynamic_topography",
    "ocean_density",
    "barystatic_gmsl_operator",
    "ModelSetup",
    "OCEAN_MASS",
]
