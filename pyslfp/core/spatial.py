"""
Spatial projection and averaging operators over a background state.

These are pygeoinf LinearOperators on the load spaces: projections onto the
ocean, the ice, the ice sheets, the land and named regions, and averages over
them, built from the state's ocean function and ice. They involve no sea
level equation, and serve the physics, the observations and the models
alike.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Union
import numpy as np

from pyshtools import SHGrid
from pygeoinf import LinearOperator, CholeskySolver, Traits
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pygeoinf.symmetric_space.sphere import Sphere

from .spaces import check_load_space, underlying_space
from .state import EarthState


# ================================================================ #
#                    Spatial Projection Operators                  #
# ================================================================ #


def ocean_projection_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
) -> LinearOperator[SHGrid, SHGrid]:
    """Returns a LinearOperator that projects a field onto the oceans."""
    check_load_space(space)
    projection_field = state.ocean_projection(
        value=0.0, exclude_ice_shelves=exclude_ice_shelves
    )
    return space.multiplication_operator(projection_field)


def ice_projection_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
) -> LinearOperator[SHGrid, SHGrid]:
    """Returns a LinearOperator that projects a field onto the state's ice
    (`EarthState.ice_projection`)."""
    check_load_space(space)
    projection_field = state.ice_projection(
        value=0.0, exclude_ice_shelves=exclude_ice_shelves
    )
    return space.multiplication_operator(projection_field)


def ice_sheet_projection_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    sheets: Optional[Union[str, Sequence[str]]] = None,
    exclude_ice_shelves: bool = False,
) -> LinearOperator[SHGrid, SHGrid]:
    """Returns a LinearOperator that projects a field onto the ice sheets
    (`EarthState.ice_sheet_projection`)."""
    check_load_space(space)
    projection_field = state.ice_sheet_projection(
        value=0.0, sheets=sheets, exclude_ice_shelves=exclude_ice_shelves
    )
    return space.multiplication_operator(projection_field)


def land_projection_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice: bool = True,
) -> LinearOperator[SHGrid, SHGrid]:
    """Returns a LinearOperator that projects a field onto background land."""
    check_load_space(space)
    projection_field = state.land_projection(value=0.0, exclude_ice=exclude_ice)
    return space.multiplication_operator(projection_field)


# ================================================================ #
#                     Spatial Averaging Operators                  #
# ================================================================ #


def ocean_average_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
) -> LinearOperator[SHGrid, np.ndarray]:
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
) -> LinearOperator[SHGrid, np.ndarray]:
    """Returns a LinearOperator that computes the spatial average over the
    state's ice (`EarthState.ice_projection`)."""
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
) -> LinearOperator[SHGrid, np.ndarray]:
    """Returns a LinearOperator that computes the spatial average over land."""
    check_load_space(space)
    projection_field = state.land_projection(value=0.0, exclude_ice=exclude_ice)
    return averaging_operator(state, space, [projection_field])


# ================================================================ #
#                     Mass Conservation Operators                  #
# ================================================================ #


def remove_ocean_average_operator(
    state: EarthState, space: Union[Lebesgue, Sobolev]
) -> LinearOperator[SHGrid, SHGrid]:
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

    l2_operator = LinearOperator.from_callables(
        l2_space, l2_space, mapping, adjoint=adjoint_mapping
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
) -> LinearOperator[SHGrid, np.ndarray]:
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
) -> LinearOperator[np.ndarray, SHGrid]:
    """
    Maps [N x 1] coefficients -> Global Field.
    Acts as a strict right-inverse to the averaging operator.
    """
    check_load_space(space)
    B = ice_sheet_averaging_operator(state, space, groupings=groupings)
    # B B* is only semidefinite in general. The basins of a grouping are
    # cells of the IMBIE and Mouginot partitions, so groups that share no
    # basin have disjoint masks, B has full row rank and B B* is positive
    # definite. A grouping that repeats a basin breaks this, and the
    # factorisation then fails rather than returning a wrong inverse.
    M = (B @ B.adjoint).with_traits(Traits.POSITIVE_DEFINITE)
    M_inv = CholeskySolver()(M)
    return B.adjoint @ M_inv


# ================================================================ #
#              Universal Region Routing Operators                  #
# ================================================================ #


def _parse_averaging_regions(
    state: EarthState, regions: Union[str, List[str], List[List[str]]]
) -> List[SHGrid]:
    """
    Internal helper to resolve varied region inputs into a list of SHGrid masks
    specifically for the region_average_operator.
    """
    if isinstance(regions, str):
        groupings = [[regions]]
    elif isinstance(regions, list):
        if len(regions) > 0 and isinstance(regions[0], str):
            # It's a flat list of strings (a single composite group)
            groupings = [regions]
        else:
            # It's already a list of lists
            groupings = regions
    else:
        raise TypeError("Regions must be a string, list of strings, or list of lists.")

    return [state.get_projection(group, value=0.0) for group in groupings]


def region_average_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    regions: Union[str, List[str], List[List[str]]],
) -> LinearOperator[SHGrid, np.ndarray]:
    """
    Maps a global field to a Euclidean vector of regional spatial averages.

    Args:
        state: The EarthState used for routing and area calculations.
        space: The Hilbert space defining the input field.
        regions: A single name, a flat list (forming one composite average),
            or a list of lists (forming multiple independent averages).
    """
    check_load_space(space)
    masks = _parse_averaging_regions(state, regions)
    return averaging_operator(state, space, masks)


def region_projection_operator(
    state: EarthState,
    space: Union[Lebesgue, Sobolev],
    regions: Union[str, List[str]],
) -> LinearOperator[SHGrid, SHGrid]:
    """
    Projects a global field onto a single (potentially composite) region.
    Maps Space -> Space.

    Args:
        state: The EarthState used for region routing.
        space: The Hilbert space defining the input field.
        regions: A single region name or a flat list of region names.
    """
    check_load_space(space)

    # get_projection automatically flattens a List[str] into a single composite mask
    mask = state.get_projection(regions, value=0.0)

    return space.multiplication_operator(mask)


def averaging_operator(
    state: EarthState,
    load_space: Union[Lebesgue, Sobolev],
    weighting_functions: List[SHGrid],
    /,
) -> LinearOperator[SHGrid, np.ndarray]:
    """
    Creates an operator that computes the true spatial average over given regions.

    The action on function `u` returns a vector `d` where `d_i` is the
    integral of `u * w_i` divided by the integral (area) of `w_i`. For a
    weighting function of unit integral this coincides with the plain L2
    product, ``load_space.l2_products_operator([w_i])``; the normalisation
    only matters for weights with non-unit integral.

    Args:
        state: The EarthState object used for integration (area calculation).
        load_space: The input domain space.
        weighting_functions: SHGrid masks representing the regions to average over.

    Returns:
        LinearOperator: Mapping from load_space to EuclideanSpace(N_weights).
    """
    if not isinstance(load_space, Sphere):
        raise TypeError("load_space must be a Lebesgue or Sobolev space.")

    # Pre-calculate the physical areas for normalization using the EarthState
    areas = [state.model.integrate(w_i) for w_i in weighting_functions]

    # Normalize the weighting functions so the inner product acts as an average
    normalized_weights = [w_i / area for w_i, area in zip(weighting_functions, areas)]

    # Delegate to the pygeoinf-native construction
    return load_space.l2_products_operator(normalized_weights)
