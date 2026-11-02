"""
Model spaces for sea level inversions, built from named load components.

`components` holds the vocabulary: the `LoadComponent` record, its two
roles, and the factories for ice sheets, glaciers, land water storage,
firn, sterodynamic sea level and ocean density. `setup` holds the assembler,
`ModelSetup`, which derives a model space and its operators from a list of
components, and `setups` the set-ups in common use.
"""

from .components import (
    ICE_SURFACE_HEIGHT,
    GEOCENTRIC_SEA_LEVEL,
    STERIC_SEA_LEVEL,
    LoadComponent,
    Role,
    barystatic_gmsl_operator,
    sterodynamic_sea_level,
    firn,
    glaciers,
    ice_sheets,
    land_water_storage,
    ocean_density,
)
from .setup import OCEAN_MASS, ModelSetup, RegionalDecomposition
from .setups import ice_and_firn, ice_and_ocean, ice_ocean_and_density, sea_level_budget

__all__ = [
    "LoadComponent",
    "Role",
    "GEOCENTRIC_SEA_LEVEL",
    "ICE_SURFACE_HEIGHT",
    "STERIC_SEA_LEVEL",
    "ice_sheets",
    "glaciers",
    "land_water_storage",
    "firn",
    "sterodynamic_sea_level",
    "ocean_density",
    "barystatic_gmsl_operator",
    "ModelSetup",
    "OCEAN_MASS",
    "RegionalDecomposition",
    "ice_and_ocean",
    "ice_ocean_and_density",
    "sea_level_budget",
    "ice_and_firn",
]
