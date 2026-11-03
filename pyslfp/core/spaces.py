"""
The Hilbert spaces of loads and responses over an Earth model's sphere.

A load space is a pygeoinf sphere space on the model's grid and radius, so
that its vectors are the grids the solvers take; a response space is a
`GRDSpace` over one. The Lebesgue spaces carry the L2 inner product, the
surface integral; the Sobolev spaces weight degree l by
(1 + (scale / radius)^2 l (l + 1))^order, so that their inner products also
measure smoothness. The components are the same in every space over the
sphere, so the choice sets an inner product and nothing else.

`underlying_space` gives the L2 space beneath a space, where an operator
whose adjoint is easy to write in L2 is written before
`LinearOperator.from_formal_adjoint` carries it over; `check_load_space`
and `check_response_space` validate the spaces operators are handed.
"""

from __future__ import annotations

from pygeoinf import DirectSum, HilbertSpace
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev, Sphere

from .model import EarthModel
from .response import GRDSpace

__all__ = [
    "lebesgue_load_space",
    "sobolev_load_space",
    "lebesgue_response_space",
    "sobolev_response_space",
    "underlying_space",
    "check_load_space",
    "check_response_space",
]


def lebesgue_load_space(earth_model: EarthModel, /) -> Lebesgue:
    """
    Defines the L2 mathematical space for square-integrable surface loads.

    Args:
        earth_model: The EarthModel relative to which the space is defined

    Returns:
        Lebesgue: The mathematical space for surface loads.
    """
    return Lebesgue(
        earth_model.lmax,
        radius=earth_model.parameters.mean_sea_floor_radius,
        grid=earth_model.grid,
        sampling=earth_model.sampling,
        extend=earth_model.extend,
    )


def lebesgue_response_space(
    earth_model: EarthModel, /, *, horizontal: bool = False, rotation: bool = True
) -> GRDSpace:
    """
    The L2 response space: a `GRDSpace` over the Lebesgue field space.

    Args:
        earth_model: The EarthModel relative to which the space is defined.
        horizontal: Whether the displacement holds its horizontal part.
        rotation: Whether the angular velocity change is held.
    """
    return GRDSpace(
        lebesgue_load_space(earth_model), horizontal=horizontal, rotation=rotation
    )


def sobolev_load_space(
    earth_model: EarthModel, order: float, scale: float, /
) -> Sobolev:
    """
    Defines a Sobolev space for surface mass loads with smoothness constraints.

    Args:
        earth_model: The EarthModel relative to which the space is defined.
        order (float): The Sobolev order (smoothness constraint).
        scale (float): The characteristic length scale.

    Returns:
        Sobolev: The regularized mathematical space for surface loads.
    """
    return Sobolev(
        earth_model.lmax,
        order,
        scale,
        radius=earth_model.parameters.mean_sea_floor_radius,
        grid=earth_model.grid,
        sampling=earth_model.sampling,
        extend=earth_model.extend,
    )


def sobolev_response_space(
    earth_model: EarthModel,
    order: float,
    scale: float,
    /,
    *,
    horizontal: bool = False,
    rotation: bool = True,
) -> GRDSpace:
    """
    A Sobolev response space: a `GRDSpace` over the Sobolev field space of
    the given order and scale. The sea level equation gains one order, so the
    response's order is at most the load's plus one.

    Args:
        earth_model: The EarthModel relative to which the space is defined.
        order (float): The Sobolev order of the response fields.
        scale (float): The characteristic length scale.
        horizontal: Whether the displacement holds its horizontal part.
        rotation: Whether the angular velocity change is held.
    """
    return GRDSpace(
        sobolev_load_space(earth_model, order, scale),
        horizontal=horizontal,
        rotation=rotation,
    )


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
