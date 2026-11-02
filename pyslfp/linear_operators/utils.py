"""
General mathematical utilities for linear operators.

This module provides the core machinery for resolving Hilbert space types
and validating domain/codomain structures.

The free functions ``l2_products_operator`` and
``spatial_multiplication_operator`` are deprecated wrappers around the
pygeoinf symmetric-space methods of the same names;
``averaging_operator`` is the supported normalising convenience built
on them.
"""

from __future__ import annotations

import warnings
from typing import List, Union

from pyshtools import SHGrid

from pygeoinf import (
    LinearOperator,
    HilbertSpace,
    DirectSum,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev, Sphere

from pyslfp.core import EarthState
from pyslfp.core.response import GRDSpace


def underlying_space(space: HilbertSpace, /) -> HilbertSpace:
    """
    Recursively resolves the L2 space underlying a Hilbert space.

    A Sobolev space on the sphere and the Lebesgue space of the same
    truncation and grid share their vectors and components, and differ only
    in the metric. An operator whose action and adjoint are easy to write on
    the Lebesgue space is written there and carried over to the Sobolev
    space by ``LinearOperator.from_formal_adjoint``; this function supplies
    the Lebesgue side of that pairing.

    A space on the sphere is mapped to the Lebesgue space with the same
    truncation, radius and grid. A direct sum is resolved summand by
    summand. Any other space, a Euclidean one for instance, is returned as
    it is.

    Args:
        space (HilbertSpace): The space to resolve.

    Returns:
        HilbertSpace: The resolved L2 space.
    """
    if isinstance(space, Sphere):
        return space.with_order(0.0)
    elif isinstance(space, DirectSum):
        return DirectSum([underlying_space(subspace) for subspace in space.subspaces])
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
    if not isinstance(load_space, Sphere):
        raise ValueError("Load space must be a Lebesgue or Sobolev space.")

    if point_values:
        if not (load_space.order > 1.0):
            raise ValueError(
                "Load space must be a Sobolev space of order > 1 for point evaluation."
            )

    return True


def check_response_space(
    response_space: HilbertSpace, /, *, point_values: bool = False
) -> None:
    """
    Validates that a space is a response space, a `GRDSpace` over a sphere
    space.

    Args:
        response_space (HilbertSpace): The space to validate.
        point_values (bool): If True, verifies that the fields support point
            evaluation, a Sobolev order above 1.

    Raises:
        ValueError: If the space is not a response space, or its fields do
            not support point evaluation when asked.
    """
    if not isinstance(response_space, GRDSpace):
        raise ValueError("The response space must be a GRDSpace.")
    field_space = response_space.field_space
    if not isinstance(field_space, Sphere):
        raise ValueError("The response fields must be a Lebesgue or Sobolev space.")
    if point_values and not field_space.order > 1.0:
        raise ValueError(
            "The response fields must be a Sobolev space of order > 1 for point "
            "evaluation."
        )


def l2_products_operator(
    load_space: Union[Lebesgue, Sobolev], weighting_functions: List[SHGrid], /
) -> LinearOperator:
    """
    Deprecated: use ``load_space.l2_products_operator(weighting_functions)``.

    The construction lives on the pygeoinf symmetric-space classes;
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
    if not isinstance(load_space, Sphere):
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
    if not isinstance(load_space, Sphere):
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
    Deprecated: use ``space.multiplication_operator(v)``.

    The construction lives on the pygeoinf symmetric-space classes;
    this wrapper delegates to it and will be removed in a future release.
    Returns the linear operator u -> v * u on the given space.
    """
    warnings.warn(
        "pyslfp.linear_operators.spatial_multiplication_operator is "
        "deprecated; use the pygeoinf space method "
        "space.multiplication_operator(...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return space.multiplication_operator(v)
