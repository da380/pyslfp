"""
General mathematical utilities for linear operators.

This module provides the core machinery for resolving Hilbert space types,
validating domain/codomain structures, and performing general spatial or
spectral manipulations.
"""

from __future__ import annotations
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
    Creates an operator that computes a vector of L2 inner products.

    The action on function `u` returns a vector `d` where `d_i = <u, w_i>_L2`.
    The inner product is always the standard L2 integral, explicitly bypassing
    the Sobolev inner product if a Sobolev space is provided.

    Args:
        load_space: The input domain space.
        weighting_functions: SHGrid masks used for integration.

    Returns:
        LinearOperator: Mapping from load_space to EuclideanSpace(N_weights).
    """
    if not isinstance(load_space, (Lebesgue, Sobolev)):
        raise TypeError("load_space must be a Lebesgue or Sobolev space.")

    # 1. Resolve the underlying L2 space to avoid Sobolev gradient penalties
    l2_space = underlying_space(load_space)
    codomain = EuclideanSpace(len(weighting_functions))

    # 2. Build the operator natively using pygeoinf's vector factory
    l2_operator = LinearOperator.from_vectors(l2_space, weighting_functions)

    # 3. Formally lift the operator back to the target domain (handles Sobolev)
    return LinearOperator.from_formal_adjoint(load_space, codomain, l2_operator)


def averaging_operator(
    state: EarthState,
    load_space: Union[Lebesgue, Sobolev],
    weighting_functions: List[SHGrid],
    /,
) -> LinearOperator:
    """
    Creates an operator that computes the true spatial average over given regions.

    The action on function `u` returns a vector `d` where `d_i` is the
    integral of `u * w_i` divided by the integral (area) of `w_i`.

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

    # Delegate the heavy lifting to the L2 products operator
    return l2_products_operator(load_space, normalized_weights)


def spatial_multiplication_operator(
    space: Union[Lebesgue, Sobolev], v: SHGrid, /
) -> LinearOperator:
    """
    Returns a linear operator that multiplies a field by another field.

    Args:
        v (SHGrid): The scalar field to multiply by.
        space: The Hilbert space for the field

    Returns:
        LinearOperator: Mapping from load_space to itself.
    """

    def mapping(u: SHGrid) -> SHGrid:
        return v * u

    l2_space = underlying_space(space)
    l2_operator = LinearOperator.self_adjoint(l2_space, mapping)
    return LinearOperator.from_formally_self_adjoint(space, l2_operator)
