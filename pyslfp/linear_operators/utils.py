"""
General mathematical utilities for linear operators.

This module provides the core machinery for resolving Hilbert space types,
validating domain/codomain structures, and performing general spatial or
spectral manipulations.
"""

from __future__ import annotations
from typing import List, Union

import numpy as np
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


def averaging_operator(
    load_space: Union[Lebesgue, Sobolev], weighting_functions: List[SHGrid], /
) -> LinearOperator:
    """
    Creates an operator that computes a vector of L2 inner products.

    The action on function `u` returns a vector `d` where `d_i = <u, w_i>_L2`.
    The inner product is always the L2 integral, even if the load_space is Sobolev.

    Args:
        load_space: The input domain space.
        weighting_functions: SHGrid masks used for integration.

    Returns:
        LinearOperator: Mapping from load_space to EuclideanSpace(N_weights).
    """
    if not isinstance(load_space, (Lebesgue, Sobolev)):
        raise TypeError("load_space must be a Lebesgue or Sobolev space.")

    is_sobolev = isinstance(load_space, Sobolev)
    l2_space = underlying_space(load_space)

    n_weights = len(weighting_functions)
    codomain = EuclideanSpace(n_weights)

    def mapping(u: SHGrid) -> np.ndarray:
        results = np.zeros(n_weights)
        for i, w_i in enumerate(weighting_functions):
            results[i] = l2_space.inner_product(u, w_i)
        return results

    def adjoint_mapping(d: np.ndarray) -> SHGrid:
        result_grid = l2_space.zero
        for i, w_i in enumerate(weighting_functions):
            l2_space.axpy(d[i], w_i, result_grid)
        return result_grid

    l2_operator = LinearOperator(
        l2_space, codomain, mapping, adjoint_mapping=adjoint_mapping
    )

    if is_sobolev:
        return LinearOperator.from_formal_adjoint(load_space, codomain, l2_operator)
    return l2_operator


def spatial_multiplication_operator(
    projection_field: SHGrid, load_space: Union[Lebesgue, Sobolev], /
) -> LinearOperator:
    """
    Returns a linear operator that multiplies a load by a spatial field.

    Args:
        projection_field (SHGrid): The scalar field to multiply by.
        load_space: The Hilbert space for the load.

    Returns:
        LinearOperator: Mapping from load_space to itself.
    """

    def mapping(load: SHGrid) -> SHGrid:
        return projection_field * load

    l2_load_space = underlying_space(load_space)
    l2_operator = LinearOperator.self_adjoint(l2_load_space, mapping)
    return LinearOperator.from_formally_self_adjoint(load_space, l2_operator)


def remove_ocean_average_operator(
    state: EarthState, load_space: Union[Lebesgue, Sobolev], /
) -> LinearOperator:
    """
    Adjusts a scalar function so that its integral over the oceans is zero.

    Args:
        state: An instance of EarthState that defined the ocean.
        load_space: The input domain space.
    """
    l2_load_space = underlying_space(load_space)
    ocean_func = state.ocean_function
    ocean_area = state.ocean_area

    def mapping(load: SHGrid) -> SHGrid:
        ocean_avg = state.integrate(ocean_func * load) / ocean_area
        new_load = load.copy()
        new_load.data -= ocean_avg
        return new_load

    def adjoint_mapping(load: SHGrid) -> SHGrid:
        total_avg = state.integrate(load)
        return load - total_avg * ocean_func / ocean_area

    l2_operator = LinearOperator(
        l2_load_space, l2_load_space, mapping, adjoint_mapping=adjoint_mapping
    )
    return LinearOperator.from_formal_adjoint(load_space, load_space, l2_operator)
