"""
Physically named load components, the vocabulary of model spaces.

A model space for a sea level inversion is a direct sum of fields: an ice
thickness change, a dynamic topography, an ocean density change, and so
on. Each field is a `LoadComponent`, a frozen record of what the field is
and how it enters the physics:

- its Hilbert space;
- the operator taking it to a surface mass load, the input of the sea
  level equation;
- the projection onto its support, the region where it can be non-zero;
- its direct contributions to named observables, the parts of an
  observable that bypass the sea level equation, such as dynamic
  topography's own term in the sea surface height;
- its role, which says how it changes the ocean's mass.

There are two roles. A mass source lives on land and adds mass to the
ocean or takes it away, so it has a barystatic global mean sea level
change (`barystatic_gmsl_operator`). An ocean-internal component
rearranges the ocean's mass without changing it, and is what the ocean
mass-conservation constraint couples.

The factories below build the components in use; glaciers will join them
once the library has a glacier inventory. Every component's space
must share the load space's grid, so that its field and its load are
fields on the same grid; its Sobolev order may differ. The load operators
are written on the Lebesgue spaces, where their adjoints are plain
multiplications, and carried to the given spaces by
`LinearOperator.from_formal_adjoint`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Union

from pyshtools import SHGrid

from pygeoinf import LinearOperator, Traits
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.core import EarthState
from pyslfp.linear_operators.utils import check_load_space, underlying_space
from pyslfp.linear_operators.loads import (
    ice_thickness_change_to_load_operator,
    ocean_density_change_to_load_operator,
    sea_level_change_to_load_operator,
    steric_sea_level_operator,
)

Space = Union[Lebesgue, Sobolev]

# The observables a component can contribute to directly. A contribution
# maps the component's space into the load space, which is where the
# contributions of different components are added.
SEA_SURFACE_HEIGHT = "sea_surface_height"
ICE_SURFACE_HEIGHT = "ice_surface_height"
STERIC_SEA_LEVEL = "steric_sea_level"


class Role(Enum):
    """How a component changes the ocean's mass."""

    MASS_SOURCE = "mass source"
    OCEAN_INTERNAL = "ocean internal"


@dataclass(frozen=True)
class LoadComponent:
    """
    One field of a model space and how it enters the physics.

    Attributes:
        name: The component's name, used as its label in a model space.
        space: The Hilbert space of the field.
        to_load: The operator from `space` to the load space, or None for
            a component that loads nothing.
        projection: The projection onto the field's support, an operator
            on `space`.
        role: Whether the component is a mass source or ocean-internal.
        observables: The component's direct contributions to named
            observables, each an operator from `space` into the load
            space. Read-only.
    """

    name: str
    space: Space
    to_load: Optional[LinearOperator]
    projection: LinearOperator
    role: Role
    observables: Mapping[str, LinearOperator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A component needs a non-empty name.")
        if not isinstance(self.role, Role):
            raise TypeError(f"The role must be a Role, not {self.role!r}.")
        if self.to_load is not None and self.to_load.domain != self.space:
            raise ValueError(
                f"The load operator of '{self.name}' must act on its space."
            )
        if self.projection.domain != self.space or self.projection.codomain != (
            self.space
        ):
            raise ValueError(
                f"The projection of '{self.name}' must map its space to itself."
            )
        for observable, operator in self.observables.items():
            if operator.domain != self.space:
                raise ValueError(
                    f"The contribution of '{self.name}' to '{observable}' must "
                    "act on its space."
                )
        object.__setattr__(
            self, "observables", MappingProxyType(dict(self.observables))
        )

    @property
    def load_space(self) -> Optional[Space]:
        """The space of the loads, or None for a component that loads nothing."""
        return None if self.to_load is None else self.to_load.codomain


# ================================================================ #
#                        Mass sources                              #
# ================================================================ #


def ice_sheets(
    state: EarthState,
    load_space: Space,
    /,
    *,
    space: Optional[Space] = None,
    name: str = "ice_sheets",
) -> LoadComponent:
    """
    An ice thickness change over the ice sheets.

    The load is rho_i (1 - C) delta I, the grounded part only: floating
    ice displaces its own weight of water and loads nothing. The support
    is the ice the state attributes to the ice sheets,
    `EarthState.ice_projection`, shelves included, so the component follows
    the library's definition of the ice sheets. The thickness change
    contributes directly to the ice surface height.

    Args:
        state: The Earth state supplying the ocean function and the masks.
        load_space: The space of the loads.
        space: The component's space, on the load space's grid. The load
            space if omitted.
        name: The component's name.
    """
    space = _component_space(space, load_space)
    return LoadComponent(
        name=name,
        space=space,
        to_load=ice_thickness_change_to_load_operator(state, space, load_space),
        projection=_mask_projection(space, state.ice_projection(value=0.0)),
        role=Role.MASS_SOURCE,
        observables={ICE_SURFACE_HEIGHT: _inclusion(space, load_space)},
    )


def land_water_storage(
    state: EarthState,
    load_space: Space,
    /,
    *,
    space: Optional[Space] = None,
    name: str = "land_water",
) -> LoadComponent:
    """
    A change in land water storage, as a water-equivalent thickness.

    The load is rho_w w on the land that carries no ice, the support of the
    component; the ice-covered land belongs to the ice components.

    Args:
        state: The Earth state supplying the masks.
        load_space: The space of the loads.
        space: The component's space, on the load space's grid. The load
            space if omitted.
        name: The component's name.
    """
    space = _component_space(space, load_space)
    land = state.land_projection(value=0.0, exclude_ice=True)
    return LoadComponent(
        name=name,
        space=space,
        to_load=_multiplication_into(
            space, load_space, state.model.parameters.water_density * land
        ),
        projection=_mask_projection(space, land),
        role=Role.MASS_SOURCE,
    )


def firn(
    state: EarthState,
    load_space: Space,
    /,
    *,
    density: float,
    space: Optional[Space] = None,
    name: str = "firn",
) -> LoadComponent:
    """
    A change in firn thickness over the ice sheets.

    Firn and ice are observed together, as one change in surface height,
    but load differently, which is why firn is a component of its own.
    The load is rho_f (1 - C) delta F with the given firn density, on the
    ice sheets' support.

    Args:
        state: The Earth state supplying the ocean function and the masks.
        load_space: The space of the loads.
        density: The firn density, in the model's units, as for
            `parameters.ice_density`. A modelling choice, so it has no
            default.
        space: The component's space, on the load space's grid. The load
            space if omitted.
        name: The component's name.
    """
    space = _component_space(space, load_space)
    return LoadComponent(
        name=name,
        space=space,
        to_load=_multiplication_into(
            space, load_space, density * state.one_minus_ocean_function
        ),
        projection=_mask_projection(space, state.ice_projection(value=0.0)),
        role=Role.MASS_SOURCE,
        observables={ICE_SURFACE_HEIGHT: _inclusion(space, load_space)},
    )


# ================================================================ #
#                     Ocean-internal components                    #
# ================================================================ #


def dynamic_topography(
    state: EarthState,
    load_space: Space,
    /,
    *,
    space: Optional[Space] = None,
    name: str = "dynamic_topography",
) -> LoadComponent:
    """
    A dynamic sea surface height change eta over the oceans.

    The load is the water it moves, rho_w C eta. It contributes directly to
    the sea surface height, a term that bypasses the sea level equation.

    Args:
        state: The Earth state supplying the ocean function.
        load_space: The space of the loads.
        space: The component's space, on the load space's grid. The load
            space if omitted.
        name: The component's name.
    """
    space = _component_space(space, load_space)
    return LoadComponent(
        name=name,
        space=space,
        to_load=sea_level_change_to_load_operator(state, space, load_space),
        projection=_mask_projection(space, state.ocean_projection(value=0.0)),
        role=Role.OCEAN_INTERNAL,
        observables={SEA_SURFACE_HEIGHT: _inclusion(space, load_space)},
    )


def ocean_density(
    state: EarthState,
    load_space: Space,
    /,
    *,
    space: Optional[Space] = None,
    name: str = "ocean_density",
) -> LoadComponent:
    """
    A vertically averaged ocean density change delta rho.

    The load is C D delta rho, with D the water depth. Its steric sea level
    change, -(D / rho_w) C delta rho, is its direct contribution to the
    steric sea level, from which a set-up derives the steric global mean.

    Args:
        state: The Earth state supplying the ocean function and the depth.
        load_space: The space of the loads.
        space: The component's space, on the load space's grid. The load
            space if omitted.
        name: The component's name.
    """
    space = _component_space(space, load_space)
    return LoadComponent(
        name=name,
        space=space,
        to_load=ocean_density_change_to_load_operator(state, space, load_space),
        projection=_mask_projection(space, state.ocean_projection(value=0.0)),
        role=Role.OCEAN_INTERNAL,
        observables={
            STERIC_SEA_LEVEL: steric_sea_level_operator(state, space, load_space)
        },
    )


# ================================================================ #
#                       Derived functionals                        #
# ================================================================ #


def barystatic_gmsl_operator(
    state: EarthState, component: LoadComponent, /
) -> LinearOperator:
    """
    The barystatic global mean sea level change a mass source implies.

    A load sigma added on land takes its mass from the ocean, and global
    mass conservation spreads the loss as a uniform layer over the ocean
    area A_O, so the global mean sea level changes by

        -(1 / (rho_w A_O)) * integral of sigma over the sphere.

    The integral is taken of the component's load, so the rule applies to
    any mass source, whatever its density or support.

    Args:
        state: The Earth state supplying the ocean area.
        component: A mass source.

    Returns:
        An operator from the component's space into EuclideanSpace(1).

    Raises:
        ValueError: If the component is not a mass source, or loads nothing.
    """
    if component.role is not Role.MASS_SOURCE or component.to_load is None:
        raise ValueError(
            f"'{component.name}' is not a mass source with a load, so it has no "
            "barystatic sea level change."
        )
    ones = state.model.zero_grid()
    ones.data[:] = 1.0
    total_mass = component.load_space.l2_products_operator([ones])
    rho_w = state.model.parameters.water_density
    return (-1.0 / (rho_w * state.ocean_area)) * total_mass @ component.to_load


# ================================================================ #
#                         Private helpers                          #
# ================================================================ #


def _component_space(space: Optional[Space], load_space: Space) -> Space:
    """The component's space, checked to share the load space's grid."""
    check_load_space(load_space)
    if space is None:
        return load_space
    check_load_space(space)
    if underlying_space(space) != underlying_space(load_space):
        raise ValueError(
            "A component's space must be on the load space's grid: the same "
            "truncation, radius, grid type, sampling and wrap column."
        )
    return space


def _mask_projection(space: Space, mask: SHGrid) -> LinearOperator:
    """Multiplication by a 0/1 mask, which is idempotent on any space."""
    projection = space.multiplication_operator(mask)
    return projection.with_traits(projection.traits | Traits.IDEMPOTENT)


def _inclusion(space: Space, target: Space) -> LinearOperator:
    """The identity from a component's space into the load space."""
    if space == target:
        return LinearOperator.identity(space)
    return space.order_inclusion_operator(target)


def _multiplication_into(
    space: Space, load_space: Space, multiplier: SHGrid
) -> LinearOperator:
    """u -> multiplier * u from a component's space into the load space,
    written on the Lebesgue spaces and lifted to the given ones."""

    def mapping(u: SHGrid) -> SHGrid:
        return multiplier * u

    l2_operator = LinearOperator.from_callables(
        underlying_space(space),
        underlying_space(load_space),
        mapping,
        adjoint=mapping,
    )
    return LinearOperator.from_formal_adjoint(space, load_space, l2_operator)
