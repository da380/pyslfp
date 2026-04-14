"""
Mass load conversions and composite domain operators for the pyslfp library.

This module provides pygeoinf LinearOperators that convert physical fields
(like ice thickness or ocean dynamic topography) into equivalent surface
mass loads. It also includes factories for combining disjoint parameter spaces
into unified loads for joint inversions.
"""

from __future__ import annotations
from typing import Union

from pyshtools import SHGrid
from pygeoinf import (
    LinearOperator,
    RowLinearOperator,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.state import EarthState
from pyslfp.linear_operators.utils import underlying_space, check_load_space


# ================================================================ #
#                Mass-to-Load Conversion Operators                 #
# ================================================================ #


def ice_thickness_change_to_load_operator(
    state: EarthState,
    ice_space: Union[Lebesgue, Sobolev],
    load_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """
    Maps an ice thickness change to the corresponding surface mass load.
    Automatically masks out the oceans.
    """
    check_load_space(ice_space)
    check_load_space(load_space)

    multiplier = state.model.parameters.ice_density * state.one_minus_ocean_function

    def mapping(ice_thickness_change: SHGrid) -> SHGrid:
        return multiplier * ice_thickness_change

    l2_ice_space = underlying_space(ice_space)
    l2_load_space = underlying_space(load_space)

    l2_operator = LinearOperator(
        l2_ice_space, l2_load_space, mapping, adjoint_mapping=mapping
    )
    return LinearOperator.from_formal_adjoint(ice_space, load_space, l2_operator)


def sea_level_change_to_load_operator(
    state: EarthState,
    sea_level_space: Union[Lebesgue, Sobolev],
    load_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """
    Maps a sea level change (or ocean dynamic topography) to the corresponding
    surface mass load. Automatically masks out the land.
    """
    check_load_space(sea_level_space)
    check_load_space(load_space)

    multiplier = state.model.parameters.water_density * state.ocean_function

    def mapping(sea_level_change: SHGrid) -> SHGrid:
        return multiplier * sea_level_change

    l2_sea_level_space = underlying_space(sea_level_space)
    l2_load_space = underlying_space(load_space)

    l2_operator = LinearOperator(
        l2_sea_level_space, l2_load_space, mapping, adjoint_mapping=mapping
    )
    return LinearOperator.from_formal_adjoint(sea_level_space, load_space, l2_operator)


def ocean_density_change_to_load_operator(
    state: EarthState,
    density_space: Union[Lebesgue, Sobolev],
    load_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """
    Maps an ocean density change to the corresponding surface mass load.
    """
    check_load_space(density_space)
    check_load_space(load_space)

    multiplier = state.sea_level * state.ocean_function

    def mapping(density_change: SHGrid) -> SHGrid:
        return multiplier * density_change

    l2_density_space = underlying_space(density_space)
    l2_load_space = underlying_space(load_space)

    l2_operator = LinearOperator(
        l2_density_space, l2_load_space, mapping, adjoint_mapping=mapping
    )
    return LinearOperator.from_formal_adjoint(density_space, load_space, l2_operator)


# ================================================================ #
#                 Composite Joint Load Operators                   #
# ================================================================ #


def joint_ice_ocean_to_load_operator(
    state: EarthState,
    ice_space: Union[Lebesgue, Sobolev],
    ocean_space: Union[Lebesgue, Sobolev],
    load_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """
    Constructs an operator that maps a joint parameter space of
    [Ice Thickness Change, Ocean Topography] into a single combined Direct Load.

    Args:
        state: The EarthState providing geometry and physical parameters.
        ice_space: The Hilbert space defining the ice thickness prior.
        ocean_space: The Hilbert space defining the ocean dynamic topography prior.
        load_space: The target Hilbert space for the SLE Fingerprint solver.

    Returns:
        A LinearOperator mapping from a HilbertSpaceDirectSum([ice, ocean])
        to the combined load_space.
    """
    ice_to_load = ice_thickness_change_to_load_operator(state, ice_space, load_space)
    ocean_to_load = sea_level_change_to_load_operator(state, ocean_space, load_space)

    return RowLinearOperator([ice_to_load, ocean_to_load])
