"""
General mathematical utilities for linear operators.

This module provides the core machinery for resolving Hilbert space types
and validating domain/codomain structures.

The generic operator constructions previously defined here (L2 products
against weighting functions and spatial multiplication) are now
implemented directly on the pygeoinf symmetric-space classes as
``space.l2_products_operator(...)`` and
``space.spatial_multiplication_operator(...)``. The free functions of the
same names below are retained as deprecated wrappers that delegate to
those methods; ``averaging_operator`` remains the supported normalising
convenience built on top of them.
"""

from __future__ import annotations

import warnings
from typing import List, Union

from pyshtools import SHGrid

from pygeoinf import (
    LinearOperator,
    HilbertSpace,
    EuclideanSpace,
    HilbertSpaceDirectSum,
    MassWeightedHilbertSpace,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.state import EarthState


def underlying_space(space: HilbertSpace, /) -> HilbertSpace:
    """
    Recursively resolves the underlying unweighted L2 space of a Hilbert space.

    If the space is not mass-weighted, the original space is returned. If the
    space is a direct sum, the method is applied to each subspace recursively
    to resolve the full L2 base.

    Args:
        space (HilbertSpace): The space to resolve.

    Returns:
        HilbertSpace: The resolved unweighted L2 space.
    """
    if isinstance(space, MassWeightedHilbertSpace):
        return space.underlying_space
    elif isinstance(space, HilbertSpaceDirectSum):
        return HilbertSpaceDirectSum(
            [underlying_space(subspace) for subspace in space.subspaces]
        )
    else:
        return space


def check_load_space(
    load_space: HilbertSpace, /, *, point_values: bool = False
) -> bool:
    """
    Validates that a load space is of a suitable mathematical form.

    Args:
        load_space (HilbertSpace): The space to validate.
        point_values (bool): If True, verifies the space supports point evaluation
            (i.e., is a Sobolev space of order > 1).

    Returns:
        bool: True if the space is valid.

    Raises:
        ValueError: If the space does not meet the requirements.
    """
    if not isinstance(load_space, (Lebesgue, Sobolev)):
        raise ValueError("Load space must be a Lebesgue or Sobolev space.")

    if point_values:
        if not isinstance(load_space, Sobolev) or not (load_space.order > 1.0):
            raise ValueError(
                "Load space must be a Sobolev space of order > 1 for point evaluation."
            )

    return True


def check_response_space(
    response_space: HilbertSpace, /, *, point_values: bool = False
) -> None:
    """
    Validates that a response space is correctly structured for the SLE response.

    The response space must be a 4-component direct sum [SLC, Disp, GPC, AVC].

    Args:
        response_space (HilbertSpace): The composite space to validate.
        point_values (bool): If True, verifies field components support point evaluation.

    Raises:
        ValueError: If the space is malformed.
    """
    if not isinstance(response_space, HilbertSpaceDirectSum):
        raise ValueError("Response space must be a HilbertSpaceDirectSum.")

    if not response_space.number_of_subspaces == 4:
        raise ValueError("Response space must have exactly 4 subspaces.")

    field_space = response_space.subspace(0)

    if not isinstance(field_space, (Lebesgue, Sobolev)):
        raise ValueError("Subspace 0 must be a Lebesgue or Sobolev space.")

    if not all(subspace == field_space for subspace in response_space.subspaces[1:3]):
        raise ValueError("Subspaces 1 and 2 must match the field type of subspace 0.")

    rot_space = response_space.subspace(3)
    if not isinstance(rot_space, EuclideanSpace) or not (rot_space.dim == 2):
        raise ValueError("Subspace 3 must be a 2D Euclidean space for rotation.")

    if point_values:
        if not isinstance(field_space, Sobolev) or not (field_space.order > 1.0):
            raise ValueError(
                "Subspace 0 must be a Sobolev space of order > 1 for point evaluation."
            )


def l2_products_operator(
    load_space: Union[Lebesgue, Sobolev], weighting_functions: List[SHGrid], /
) -> LinearOperator:
    """
    Deprecated: use ``load_space.l2_products_operator(weighting_functions)``.

    The construction now lives on the pygeoinf symmetric-space classes;
    this wrapper delegates to it and will be removed in a future release.
    The action on function `u` returns a vector `d` where `d_i = <u, w_i>_L2`,
    with the standard L2 integral used even on a Sobolev space.
    """
    warnings.warn(
        "pyslfp.linear_operators.l2_products_operator is deprecated; use "
        "the pygeoinf space method load_space.l2_products_operator(...) "
        "instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(load_space, (Lebesgue, Sobolev)):
        raise TypeError("load_space must be a Lebesgue or Sobolev space.")

    return load_space.l2_products_operator(weighting_functions)


def averaging_operator(
    state: EarthState,
    load_space: Union[Lebesgue, Sobolev],
    weighting_functions: List[SHGrid],
    /,
) -> LinearOperator:
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
    if not isinstance(load_space, (Lebesgue, Sobolev)):
        raise TypeError("load_space must be a Lebesgue or Sobolev space.")

    # Pre-calculate the physical areas for normalization using the EarthState
    areas = [state.model.integrate(w_i) for w_i in weighting_functions]

    # Normalize the weighting functions so the inner product acts as an average
    normalized_weights = [w_i / area for w_i, area in zip(weighting_functions, areas)]

    # Delegate to the pygeoinf-native construction
    return load_space.l2_products_operator(normalized_weights)


def spatial_multiplication_operator(
    space: Union[Lebesgue, Sobolev], v: SHGrid, /
) -> LinearOperator:
    """
    Deprecated: use ``space.spatial_multiplication_operator(v)``.

    The construction now lives on the pygeoinf symmetric-space classes;
    this wrapper delegates to it and will be removed in a future release.
    Returns the linear operator u -> v * u on the given space.
    """
    warnings.warn(
        "pyslfp.linear_operators.spatial_multiplication_operator is "
        "deprecated; use the pygeoinf space method "
        "space.spatial_multiplication_operator(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return space.spatial_multiplication_operator(v)
