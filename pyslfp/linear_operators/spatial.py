"""
Spatial observation and masking operators for the pyslfp library.

This module provides standalone pygeoinf LinearOperators for spatial
manipulations—such as masking oceans, regional basin averaging, and
mass conversions—decoupled from the physical Sea Level Equation.
"""

from __future__ import annotations
from typing import Union, List, Optional

from pyshtools import SHGrid
from pygeoinf import LinearOperator, CholeskySolver
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.state import EarthState
from pyslfp.linear_operators.utils import (
    underlying_space,
    spatial_multiplication_operator,
    averaging_operator,
    check_load_space,
)


# ================================================================ #
#                    Spatial Projection Operators                  #
# ================================================================ #


def ocean_projection_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
) -> LinearOperator:
    """Returns a LinearOperator that projects a field onto the oceans."""
    check_load_space(space)
    projection_field = state.ocean_projection(
        value=0.0, exclude_ice_shelves=exclude_ice_shelves
    )
    return spatial_multiplication_operator(space, projection_field)


def ice_projection_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
) -> LinearOperator:
    """Returns a LinearOperator that projects a field onto the ice sheets."""
    check_load_space(space)
    projection_field = state.ice_projection(
        value=0.0, exclude_ice_shelves=exclude_ice_shelves
    )
    return spatial_multiplication_operator(space, projection_field)


def land_projection_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice: bool = True,
) -> LinearOperator:
    """Returns a LinearOperator that projects a field onto background land."""
    check_load_space(space)
    projection_field = state.land_projection(value=0.0, exclude_ice=exclude_ice)
    return spatial_multiplication_operator(space, projection_field)


# ================================================================ #
#                     Spatial Averaging Operators                  #
# ================================================================ #


def ocean_average_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
) -> LinearOperator:
    """Returns a LinearOperator that computes the spatial average over the oceans."""
    check_load_space(space)
    projection_field = state.ocean_projection(
        value=0.0, exclude_ice_shelves=exclude_ice_shelves
    )
    return averaging_operator(state, space, [projection_field])


def ice_average_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
) -> LinearOperator:
    """Returns a LinearOperator that computes the spatial average over ice sheets."""
    check_load_space(space)
    projection_field = state.ice_projection(
        value=0.0, exclude_ice_shelves=exclude_ice_shelves
    )
    return averaging_operator(state, space, [projection_field])


def land_average_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice: bool = True,
) -> LinearOperator:
    """Returns a LinearOperator that computes the spatial average over land."""
    check_load_space(space)
    projection_field = state.land_projection(value=0.0, exclude_ice=exclude_ice)
    return averaging_operator(state, space, [projection_field])


# ================================================================ #
#                Mass-to-Load Conversion Operators                 #
# ================================================================ #


def ice_thickness_change_to_load_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """Maps an ice thickness change to the corresponding surface mass load."""
    check_load_space(space)
    multiplier = state.model.parameters.ice_density * state.one_minus_ocean_function
    return spatial_multiplication_operator(space, multiplier)


def sea_level_change_to_load_operator(
    state: EarthState,
    sea_level_space: Union[Lebesgue, Sobolev],
    load_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """Maps a sea level change to the corresponding surface mass load."""
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
    space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """Maps an ocean density change to the corresponding surface mass load."""
    check_load_space(space)
    multiplier = state.sea_level * state.ocean_function
    return spatial_multiplication_operator(space, multiplier)


# ================================================================ #
#                     Mass Conservation Operators                  #
# ================================================================ #


def remove_ocean_average_operator(
    state: EarthState, space: Union[Lebesgue, Sobolev]
) -> LinearOperator:
    """
    Returns a LinearOperator that adjusts a scalar surface function such that
    its integral over the oceans is strictly zero.
    """
    check_load_space(space)
    l2_space = underlying_space(space)

    ocean_function = state.ocean_function
    ocean_area = state.ocean_area

    def mapping(field: SHGrid) -> SHGrid:
        ocean_average = state.model.integrate(ocean_function * field) / ocean_area
        new_field = field.copy()
        new_field.data -= ocean_average
        return new_field

    def adjoint_mapping(field: SHGrid) -> SHGrid:
        average = state.model.integrate(field)
        return field - average * ocean_function / ocean_area

    l2_operator = LinearOperator(
        l2_space, l2_space, mapping, adjoint_mapping=adjoint_mapping
    )

    return LinearOperator.from_formal_adjoint(space, space, l2_operator)


# ================================================================ #
#                 Regional Basin Averaging                         #
# ================================================================ #


def ice_sheet_averaging_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    groupings: Optional[Union[str, List[List[str]]]] = None,
) -> LinearOperator:
    """Maps a Global Field -> [N x 1] Averages for specified basin groupings."""
    check_load_space(space)

    masks, _ = state.grouped_ice_projections(groupings=groupings)
    return averaging_operator(state, space, masks)


def ice_sheet_basis_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    groupings: Optional[Union[str, List[List[str]]]] = None,
) -> LinearOperator:
    """
    Maps [N x 1] coefficients -> Global Field.
    Acts as a strict right-inverse to the averaging operator.
    """
    check_load_space(space)
    B = ice_sheet_averaging_operator(state, space, groupings=groupings)
    M = B @ B.adjoint
    M_inv = CholeskySolver()(M)
    return B.adjoint @ M_inv
