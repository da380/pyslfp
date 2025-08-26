"""
Module for defining some operators related to the sea level problem.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Union

import inspect

import numpy as np

from pyshtools import SHGrid, SHCoeffs

from pygeoinf import (
    LinearOperator,
    HilbertSpace,
    EuclideanSpace,
    HilbertSpaceDirectSum,
    RowLinearOperator,
    MassWeightedHilbertSpace,
)

from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from .utils import SHVectorConverter


def underlying_space(space: HilbertSpace):
    """
    Returns the underlying space of a HilbertSpace object. The the space
    is not mass weighted, the original space is returned. If the space is
    a direct sum, the method is applied to each subspace recursively.
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
    Checks that the load space is of a suitable form.
    """

    if not isinstance(load_space, Lebesgue) and not isinstance(load_space, Sobolev):
        raise ValueError("Load space must be a Lebesgue or Sobolev space.")

    if point_values:
        if not isinstance(load_space, Sobolev) and not load_space.order > 1:
            raise ValueError("Load space must be a Sobolev space of order > 1.")

    return True


def check_response_space(
    response_space: HilbertSpace, /, *, point_values: bool = False
) -> None:
    """
    Checks that the response space is of a suitable form.

    Args:
        response_space: The response space.
        point_values: If True, the field spaces must be Sobolev spaces
            for which point-evaluation is defined.
    """

    if not isinstance(response_space, HilbertSpaceDirectSum):
        raise ValueError("Response space must be a HilbertSpaceDirectSum.")

    if not response_space.number_of_subspaces == 4:
        raise ValueError("Response space must have 4 subspaces.")

    field_space = response_space.subspace(0)

    if not isinstance(field_space, Lebesgue) and not isinstance(field_space, Sobolev):
        raise ValueError("Subspace 0 must be a Lebesgue or Sobolev space.")

    if not all(subspace == field_space for subspace in response_space.subspaces[1:3]):
        raise ValueError("Subspaces 1 and 2 must equal subspace 0.")

    angular_velocity_space = response_space.subspace(3)
    if (
        not isinstance(angular_velocity_space, EuclideanSpace)
        or not angular_velocity_space.dim == 2
    ):
        raise ValueError("Subspace 3 must be a 2D Euclidean space.")

    if point_values:
        if not isinstance(field_space, Sobolev) and not field_space.order > 1:
            raise ValueError("Subspace 0 must be a Sobolev space of order > 1.")


def tide_gauge_operator(
    response_space: HilbertSpaceDirectSum, points
) -> LinearOperator:
    """
    Maps the response fields to a vector of sea level change values at
    a discrete set of locations.

    Args:
        response_space: The response space, which is a HilbertSpaceDirectSum
            whose elements are lists of three SHGrid objects: the sea level
            change, displacement, gravitational potential change fields, and
            a numpy array for the angular velocity change.
        points: A list of (latitude, longitude) points in degrees
            where the sea level change is to be evaluated.

    Returns:
        A LinearOperator object.
    """

    check_response_space(response_space, point_values=True)

    field_space = response_space.subspace(0)
    euclidean_space = response_space.subspace(3)
    point_evaluation_operator = field_space.point_evaluation_operator(points)
    codomain = point_evaluation_operator.codomain

    return RowLinearOperator(
        [
            point_evaluation_operator,
            field_space.zero_operator(codomain=codomain),
            field_space.zero_operator(codomain=codomain),
            euclidean_space.zero_operator(codomain=codomain),
        ]
    )


def grace_operator(
    response_space: HilbertSpaceDirectSum,
    observation_degree: int,
) -> LinearOperator:
    """
    Maps the response fields to a vector of spherical harmonic coefficients
    of the gravitational potential change, for degrees  2 <= l <= observation_degree.

    The output coefficients are fully normalised and include the Condon-Shortley
    phase factor.

    Args:
        response_space: The response space, which is a HilbertSpaceDirectSum.
        observation_degree: The max degree of the SH coefficient observations.
    Returns:
        A LinearOperator object.
    """

    check_response_space(response_space, point_values=False)
    sobolev = isinstance(response_space.subspace(0), Sobolev)

    converter = SHVectorConverter(lmax=observation_degree, lmin=2)

    l2_response_space = underlying_space(response_space)
    field_space = l2_response_space.subspace(0)
    euclidean_space = l2_response_space.subspace(3)
    codomain = EuclideanSpace(converter.vector_size)

    def mapping(u: SHGrid) -> np.ndarray:
        ulm = field_space.to_coefficient(u)
        return converter.to_vector(ulm.coeffs)

    def adjoint_mapping(data: np.ndarray) -> SHGrid:
        coeffs = converter.from_vector(data, output_lmax=field_space.lmax)
        adjoint_load_lm = SHCoeffs.from_array(
            coeffs,
            normalization=field_space.normalization,
            csphase=field_space.csphase,
        )
        adjoint_load = (
            field_space.from_coefficient(adjoint_load_lm) / field_space.radius**2
        )
        return adjoint_load

    l2_partial_operator = LinearOperator(
        field_space, codomain, mapping, adjoint_mapping=adjoint_mapping
    )

    l2_operator = RowLinearOperator(
        [
            field_space.zero_operator(codomain=codomain),
            field_space.zero_operator(codomain=codomain),
            l2_partial_operator,
            euclidean_space.zero_operator(codomain=codomain),
        ]
    )

    return (
        LinearOperator.from_formal_adjoint(response_space, codomain, l2_operator)
        if sobolev
        else l2_operator
    )
