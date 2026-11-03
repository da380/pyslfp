"""
The Hilbert spaces of loads and responses over an Earth model's sphere.

A load space is a pygeoinf sphere space on the model's grid and radius, so
that its vectors are the grids the solvers take; a response space is a
`GRDSpace` over one. The Lebesgue spaces carry the L2 inner product, the
surface integral; the Sobolev spaces weight degree l by
(1 + (scale / radius)^2 l (l + 1))^order, so that their inner products also
measure smoothness. The components are the same in every space over the
sphere, so the choice sets an inner product and nothing else.
"""

from __future__ import annotations

from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from .model import EarthModel
from .response import GRDSpace

__all__ = [
    "lebesgue_load_space",
    "sobolev_load_space",
    "lebesgue_response_space",
    "sobolev_response_space",
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
