"""
General mathematical utilities for linear operators.

This module provides the core machinery for resolving Hilbert space types,
validating domain/codomain structures, and performing general spatial or
spectral manipulations.
"""

from __future__ import annotations
from typing import Union


from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.state import EarthState
from pyslfp.linear_operators.utils import (
    check_load_space,
    spatial_multiplication_operator,
)


def ice_projection_operator(
    state: EarthState,
    load_space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
):
    """
    Returns a LinearOpeator multiplies a load by a function that is one
    over the background ice sheets and zero elsewhere.

    Args:
        state: The EarthState object.
        load_space: The Hilbert space for the load.
        exclude_ice_shelves: If True, the function is set to zero in ice-shelved regions.

    Returns:
        A LinearOperator object.

    """

    check_load_space(load_space)

    projection_field = state.ice_projection(
        value=0, exclude_ice_shelves=exclude_ice_shelves
    )
    return spatial_multiplication_operator(projection_field, load_space)


def ocean_projection_operator(
    state: EarthState,
    load_space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
):
    """
    Returns a LinearOpeator multiplies a load by a function that is one
    over the background oceans and zero elsewhere.

    Args:
        state: The EarthState object.
        load_space: The Hilbert space for the load.
        exclude_ice_shelves: If True, the function is set to zero in ice-shelved regions.

    Returns:
        A LinearOperator object.

    """
    check_load_space(load_space)
    projection_field = state.ocean_projection(
        value=0, exclude_ice_shelves=exclude_ice_shelves
    )
    return spatial_multiplication_operator(projection_field, load_space)


def land_projection_operator(
    state: EarthState,
    load_space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice: bool = True,
):
    """
    Returns a LinearOpeator multiplies a load by a function that is one
    over the background land and zero elsewhere.

    Args:
        state: The EarthState object.
        load_space: The Hilbert space for the load.
        exclude_ice: If True, the function is set to zero in ice-covered regions.

    Returns:
        A LinearOperator object.

    """
    check_load_space(load_space)
    projection_field = state.land_projection(value=0, exclude_ice=exclude_ice)
    return spatial_multiplication_operator(projection_field, load_space)
