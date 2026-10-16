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

from pyslfp.core import EarthState
from pyslfp.linear_operators.spatial import ocean_average_operator
from pyslfp.linear_operators.utils import underlying_space, check_load_space


# ================================================================ #
#                Mass-to-Load Conversion Operators                 #
# ================================================================ #


def ice_thickness_change_to_load_operator(
    state: EarthState,
    ice_space: Union[Lebesgue, Sobolev],
    load_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """Maps an ice thickness change to its surface mass load,
    rho_ice (1 - C) delta I: grounded ice only.

    Args:
        state: The Earth state supplying the ocean function.
        ice_space: The operator's domain, Lebesgue or Sobolev.
        load_space: The operator's codomain.
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
    """Maps a sea level change (or ocean dynamic topography) to its
    surface mass load, rho_w C delta xi: ocean only.

    Args:
        state: The Earth state supplying the ocean function.
        sea_level_space: The operator's domain, Lebesgue or Sobolev.
        load_space: The operator's codomain.
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
    """Maps an ocean density change to its surface mass load,
    C SL delta rho: the water column's depth times the change.

    Args:
        state: The Earth state supplying the ocean function and sea level.
        density_space: The operator's domain, Lebesgue or Sobolev.
        load_space: The operator's codomain.
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
#              Steric Response and Mass Conservation               #
# ================================================================ #


def steric_sea_level_operator(
    state: EarthState,
    density_space: Union[Lebesgue, Sobolev],
    sea_level_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """Maps a vertically averaged ocean density change to the steric
    sea level change it produces,

        eta_s = -(D / rho_w) C delta rho,

    with D the local ocean depth and C the ocean function. The scaling
    leaves the mass of every water column pointwise unchanged
    (rho_w eta_s + D delta rho = 0), so the steric part of a sea
    surface height change loads nothing and has no induced
    gravitational-rotational-deformation response. Used to split an
    (eta, delta rho) model state into the standard steric /
    ocean-dynamic decomposition.

    Args:
        state: The Earth state supplying the ocean function and depth.
        density_space: The operator's domain, Lebesgue or Sobolev.
        sea_level_space: The operator's codomain.
    """
    return (
        -1.0 / state.model.parameters.water_density
    ) * ocean_density_change_to_load_operator(state, density_space, sea_level_space)


def ocean_mass_conservation_operator(
    state: EarthState,
    sea_level_space: Union[Lebesgue, Sobolev],
    density_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """The ocean mass change of an (eta, delta rho) pair, as a linear
    operator whose kernel is the mass-conservation constraint

        < rho_w C eta + D C delta rho >_O = 0.

    A sea surface height change and a density change that carry no net
    water mass must satisfy this; conditioning a prior on the kernel
    (for example through an affine subspace, with the solver of the
    caller's choosing) enforces it. The operator only states the
    constraint; nothing about the conditioning is chosen here.

    Args:
        state: The Earth state supplying the ocean function and depth.
        sea_level_space: The eta component's space, Lebesgue or Sobolev.
        density_space: The delta rho component's space.
    """
    average = ocean_average_operator(state, sea_level_space)
    return RowLinearOperator(
        [
            average @ sea_level_change_to_load_operator(
                state, sea_level_space, sea_level_space
            ),
            average @ ocean_density_change_to_load_operator(
                state, density_space, sea_level_space
            ),
        ]
    )


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
