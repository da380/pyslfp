"""
Ready-made model set-ups: the component lists in common use.

Each is a `ModelSetup` over the vocabulary of `pyslfp.models.components`,
and doubles as a worked example of building one. All take the Earth state,
the load space and, optionally, the components' space and the space of their
pointwise observables, as the factories do; for a set-up feeding a
fingerprint operator, the load space is the operator's domain and the
observable space its field space (`FingerPrintOperator.field_space`).

- `ice_and_ocean`: the ice sheets and dynamic topography, the joint ice-ocean
  space of satellite altimetry.
- `ice_ocean_and_density`: those and a vertically averaged ocean density
  change, which separates the steric sea level change from the manometric.
- `sea_level_budget`: the ice sheets, the glaciers, land water storage,
  dynamic topography and ocean density, the terms of the modern sea level
  budget.
- `ice_and_firn`: the ice sheets and their firn, observed together by
  ice-sheet altimetry but loading with different densities.
"""

from __future__ import annotations

from typing import Optional

from pyslfp.core import EarthState
from pyslfp.models.components import (
    Space,
    dynamic_topography,
    firn,
    glaciers,
    ice_sheets,
    land_water_storage,
    ocean_density,
)
from pyslfp.models.setup import ModelSetup

__all__ = ["ice_and_ocean", "ice_ocean_and_density", "sea_level_budget", "ice_and_firn"]


def ice_and_ocean(
    state: EarthState,
    load_space: Space,
    /,
    *,
    space: Optional[Space] = None,
    observable_space: Optional[Space] = None,
) -> ModelSetup:
    """The ice sheets and dynamic topography."""
    return ModelSetup(
        state,
        [
            ice_sheets(
                state, load_space, space=space, observable_space=observable_space
            ),
            dynamic_topography(
                state, load_space, space=space, observable_space=observable_space
            ),
        ],
    )


def ice_ocean_and_density(
    state: EarthState,
    load_space: Space,
    /,
    *,
    space: Optional[Space] = None,
    observable_space: Optional[Space] = None,
) -> ModelSetup:
    """The ice sheets, dynamic topography and a vertically averaged ocean
    density change. The density change carries the steric sea level change,
    so the set-up splits the global mean into barystatic and steric parts and
    the regional means into steric, dynamic manometric and barystatic
    manometric parts."""
    return ModelSetup(
        state,
        [
            ice_sheets(
                state, load_space, space=space, observable_space=observable_space
            ),
            dynamic_topography(
                state, load_space, space=space, observable_space=observable_space
            ),
            ocean_density(state, load_space, space=space),
        ],
    )


def sea_level_budget(
    state: EarthState,
    load_space: Space,
    /,
    *,
    space: Optional[Space] = None,
    observable_space: Optional[Space] = None,
) -> ModelSetup:
    """The terms of the modern sea level budget: the ice sheets, the glaciers,
    land water storage, dynamic topography and ocean density."""
    return ModelSetup(
        state,
        [
            ice_sheets(
                state, load_space, space=space, observable_space=observable_space
            ),
            glaciers(state, load_space, space=space),
            land_water_storage(state, load_space, space=space),
            dynamic_topography(
                state, load_space, space=space, observable_space=observable_space
            ),
            ocean_density(state, load_space, space=space),
        ],
    )


def ice_and_firn(
    state: EarthState,
    load_space: Space,
    /,
    *,
    firn_density: float,
    space: Optional[Space] = None,
    observable_space: Optional[Space] = None,
) -> ModelSetup:
    """The ice sheets and their firn, whose thickness changes ice-sheet
    altimetry sees as one surface height change while they load with
    different densities.

    Args:
        firn_density: The firn density, in the model's units; a modelling
            choice, so it has no default.
    """
    return ModelSetup(
        state,
        [
            ice_sheets(
                state, load_space, space=space, observable_space=observable_space
            ),
            firn(
                state,
                load_space,
                density=firn_density,
                space=space,
                observable_space=observable_space,
            ),
        ],
    )
