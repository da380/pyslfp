"""
Physically named load components, the vocabulary of model spaces.

A model space for a sea level inversion is a direct sum of fields: an ice
thickness change, a sterodynamic sea level change, an ocean density change, and so
on. Each field is a `LoadComponent`, a frozen record of what the field is
and how it enters the physics:

- its Hilbert space;
- the operator taking it to a surface mass load, the input of the sea
  level equation;
- its support, the region where the field has physical meaning;
- its direct contributions to named observables, the parts of an
  observable that bypass the sea level equation, such as the sterodynamic
  sea level change's own term in the geocentric sea level;
- its role, which says how it changes the ocean's mass.

**The fields are defined globally, and the supports are part of the
physics.** Each component's space holds functions on the whole sphere,
and a prior is a measure on that space as it stands. Masking the prior
instead, by pushing it forward through a support projection, gives
samples that jump at the support's edge; that is not a measure on a
Sobolev space of order 1/2 or more, and it is only semidefinite. The
support enters through the operators, in the way their regularity
allows:

- **Loads carry the support.** Multiplication by the indicator of a
  region with a Lipschitz boundary is bounded on H^s for -1/2 < s < 1/2 and
  on no H^s with s >= 1/2. Every load carries the ocean function or a
  support's indicator, so it lies in H^s only for s < 1/2, and the load
  space should have an order in [0, 1/2). A glacier's load carries the
  indicator of its outline in the same way; glaciers are far smaller than
  the grid's resolution, so the grid holds each as a point mass. The sea level equation gains
  one order, so a load space of order above 0 puts the response above
  order 1, where evaluation at points is bounded.
- **Pointwise observables leave the support to their points.** The sea
  and ice surface heights are sampled at points, and a masked field is
  too rough for that. Their direct contributions pass the field through
  unmasked, into an observable space of order above 1: the field space
  of the response they are added to. At a point inside the support a
  field and its masked version agree, so the observation points carry
  the support.
- **Integrated diagnostics carry the support.** The steric sea level
  carries the ocean function and the depth and is rough like a load, so
  it lands in the load space; it serves the ocean averages a set-up
  derives from it, which are bounded on L^2.

Loads, integrals and averages see a field only on its support, and point
observations do when their points lie in it, so all of them have the same
posterior as under a masked prior. The `support` field serves display and
diagnostics: plot `support(u)` to show a field where it is physical.

There are two roles. A mass source lives on land and adds mass to the
ocean or takes it away, so it has a barystatic global mean sea level
change (`barystatic_gmsl_operator`). An ocean-internal component
rearranges the ocean's mass without changing it, and is what the ocean
mass-conservation constraint couples.

The factories below build the components in use. Every component's space, and
every observable space, must share the load space's grid, so that a
field, its load and its contributions are fields on the same grid; their
Sobolev orders differ. The load operators are written on the Lebesgue
spaces, where their adjoints are plain multiplications, and carried to
the given spaces by `LinearOperator.from_formal_adjoint`.

Each load is one pointwise product, the support's indicator folded into
its multiplier, not a support projection followed by a product. A
product is left on the grid, untruncated, and the next product in a
composition acts on all of it, while a lifted operator's adjoint passes
through the space's components, which keep only the part up to the
truncation degree. So the computed adjoint of g (f u) is the exact
adjoint of g T(f u), with T the truncation, and the composition fails
the adjoint identity by the pairing of the two products' parts above the
truncation degree. That is a discretisation error which falls with the
degree, to about 1e-5 of the norms at degree 256 for a support's
indicator; a single product satisfies the identity to rounding at every
degree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Union

from pyshtools import SHGrid

from pygeoinf import LinearOperator, Traits
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.core import EarthState
from pyslfp.linear_operators.utils import check_load_space, underlying_space
from pyslfp.linear_operators.loads import (
    ocean_density_change_to_load_operator,
    sea_level_change_to_load_operator,
    steric_sea_level_operator,
)

Space = Union[Lebesgue, Sobolev]

# The observables a component can contribute to directly. The geocentric sea
# level and the ice surface height are sampled at points, so a contribution to them is the
# field itself, unmasked, in an observable space of order above 1; the
# steric sea level is rough like a load and lands in the load space.
# Contributions to one observable share its space, where they are added.
GEOCENTRIC_SEA_LEVEL = "geocentric_sea_level"
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
        support: Multiplication by the indicator of the field's support,
            an idempotent operator on `space`. The load accounts for it,
            and the pointwise contributions leave it to their observation
            points; it is for display and diagnostics, not for masking a
            prior.
        role: Whether the component is a mass source or ocean-internal.
        observables: The component's direct contributions to named
            observables, each an operator from `space` into that
            observable's space. Read-only.
    """

    name: str
    space: Space
    to_load: Optional[LinearOperator]
    support: LinearOperator
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
        if self.support.domain != self.space or self.support.codomain != self.space:
            raise ValueError(
                f"The support of '{self.name}' must map its space to itself."
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
    observable_space: Optional[Space] = None,
    name: str = "ice_sheets",
) -> LoadComponent:
    """
    An ice thickness change over the ice sheets.

    The support is the ice sheets as the library defines them,
    `EarthState.ice_sheet_projection`: the Antarctic and Greenland ice,
    shelves and margins included. On it the load is
    rho_i (1 - C) delta I, the grounded part only: floating ice displaces
    its own weight of water and loads nothing. The thickness change
    contributes directly, unmasked, to the ice surface height, which ice
    altimetry samples at points on the ice sheets.

    Args:
        state: The Earth state supplying the ocean function and the masks.
        load_space: The space of the loads.
        space: The component's space, on the load space's grid. The load
            space if omitted.
        observable_space: The space of the ice surface height, on the same
            grid, of order no higher than the component's and above 1 if
            it is to be sampled at points. The component's space if
            omitted.
        name: The component's name.
    """
    space = _component_space(space, load_space)
    ice = state.ice_sheet_projection(value=0.0)
    rho_i = state.model.parameters.ice_density
    return LoadComponent(
        name=name,
        space=space,
        to_load=_multiplication_into(
            space, load_space, rho_i * state.one_minus_ocean_function * ice
        ),
        support=_mask(space, ice),
        role=Role.MASS_SOURCE,
        observables={
            ICE_SURFACE_HEIGHT: _inclusion(space, observable_space, load_space)
        },
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

    The support is the land that carries no ice; the ice-covered land
    belongs to the ice components. The load is rho_w w on the support, the
    mask entering as part of the multiplier. Glaciers on ice-free land lie
    on the support too: the glaciers component loads their ice, this one
    the water stored around them.

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
        support=_mask(space, land),
        role=Role.MASS_SOURCE,
    )


def firn(
    state: EarthState,
    load_space: Space,
    /,
    *,
    density: float,
    space: Optional[Space] = None,
    observable_space: Optional[Space] = None,
    name: str = "firn",
) -> LoadComponent:
    """
    A change in firn thickness over the ice sheets.

    Firn and ice are observed together, as one change in surface height,
    but load differently, which is why firn is a component of its own.
    The support is the ice sheets'. On it the load is rho_f (1 - C) delta F
    with the given firn density. The thickness change contributes
    directly, unmasked, to the ice surface height, as the ice thickness
    change does.

    Args:
        state: The Earth state supplying the ocean function and the masks.
        load_space: The space of the loads.
        density: The firn density, in the model's units, as for
            `parameters.ice_density`. A modelling choice, so it has no
            default.
        space: The component's space, on the load space's grid. The load
            space if omitted.
        observable_space: The space of the ice surface height, as for
            `ice_sheets`. The component's space if omitted.
        name: The component's name.
    """
    space = _component_space(space, load_space)
    ice = state.ice_sheet_projection(value=0.0)
    return LoadComponent(
        name=name,
        space=space,
        to_load=_multiplication_into(
            space, load_space, density * state.one_minus_ocean_function * ice
        ),
        support=_mask(space, ice),
        role=Role.MASS_SOURCE,
        observables={
            ICE_SURFACE_HEIGHT: _inclusion(space, observable_space, load_space)
        },
    )


def glaciers(
    state: EarthState,
    load_space: Space,
    /,
    *,
    space: Optional[Space] = None,
    regions: Optional[Sequence[Union[int, str]]] = None,
    name: str = "glaciers",
) -> LoadComponent:
    """
    A thickness change of the glaciers of the Randolph Glacier Inventory 7.0.

    The field h is a regional thickness change, which each glacier takes at
    its own position. Glaciers are far smaller than the grid's resolution,
    and sea level is wanted well away from them, so each is a point mass at
    its grid node (`EarthState.glacier_area_density`), and the load is
    rho_i a h, with a the glacier area density. The whole glacier area
    loads, with no factor (1 - C): the glaciers are grounded, and a node the
    ocean function counts as ocean, often a small island or a coast the grid
    does not resolve, says nothing about them. Glaciers at the nodes of the
    ice sheets belong to the ice sheets. The support is the nodes holding
    glaciers. There is no direct observable: contributions to one pointwise
    observable share a support, so glacier altimetry would need an
    observable of its own.

    Args:
        state: The Earth state supplying the glaciers and the ice sheets.
        load_space: The space of the loads.
        space: The component's space, on the load space's grid. The load
            space if omitted.
        regions: GTN-G first-order regions to keep, by number or name; all
            if None. One component per region gives regional budgets.
        name: The component's name.
    """
    space = _component_space(space, load_space)
    density = state.glacier_area_density(regions=regions)
    rho_i = state.model.parameters.ice_density
    support = state.model.zero_grid()
    support.data[:] = (density.data > 0).astype(float)
    return LoadComponent(
        name=name,
        space=space,
        to_load=_multiplication_into(space, load_space, rho_i * density),
        support=_mask(space, support),
        role=Role.MASS_SOURCE,
    )


# ================================================================ #
#                     Ocean-internal components                    #
# ================================================================ #


def sterodynamic_sea_level(
    state: EarthState,
    load_space: Space,
    /,
    *,
    space: Optional[Space] = None,
    observable_space: Optional[Space] = None,
    name: str = "sterodynamic_sea_level",
) -> LoadComponent:
    """
    A sterodynamic sea level change eta over the oceans, after Gregory et al.
    (2019): the ocean dynamic sea level change plus any global-mean steric
    change. With an ocean density component the latter is the density's
    steric global mean; without one, the ocean mass constraint makes the
    ocean mean of eta vanish.

    The support is the ocean. The load is the water the change moves,
    rho_w C eta, the ocean function C being the support. The change
    contributes directly, unmasked, to the geocentric sea level, a term that
    bypasses the sea level equation; altimetry samples it at points in the
    ocean.

    Args:
        state: The Earth state supplying the ocean function.
        load_space: The space of the loads.
        space: The component's space, on the load space's grid. The load
            space if omitted.
        observable_space: The space of the geocentric sea level, on the same
            grid, of order no higher than the component's and above 1 if
            it is to be sampled at points; the field space of the response
            it is added to. The component's space if omitted.
        name: The component's name.
    """
    space = _component_space(space, load_space)
    return LoadComponent(
        name=name,
        space=space,
        to_load=sea_level_change_to_load_operator(state, space, load_space),
        support=_mask(space, state.ocean_projection(value=0.0)),
        role=Role.OCEAN_INTERNAL,
        observables={
            GEOCENTRIC_SEA_LEVEL: _inclusion(space, observable_space, load_space)
        },
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

    The support is the ocean. The load is C D delta rho, with D the water
    depth, and the steric sea level change, -(D / rho_w) C delta rho, is
    the direct contribution to the steric sea level, from which a set-up
    derives the steric global mean. The ocean function C in both is the
    support; with the depth it makes the steric sea level rough like a
    load, so it lands in the load space.

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
        support=_mask(space, state.ocean_projection(value=0.0)),
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


def _inclusion(
    space: Space, observable_space: Optional[Space], load_space: Space
) -> LinearOperator:
    """The field itself, read from its space into an observable space on
    the load space's grid; the identity on its own space if none is given."""
    if observable_space is None or observable_space == space:
        return LinearOperator.identity(space)
    check_load_space(observable_space)
    if underlying_space(observable_space) != underlying_space(load_space):
        raise ValueError(
            "An observable space must be on the load space's grid: the same "
            "truncation, radius, grid type, sampling and wrap column."
        )
    return space.order_inclusion_operator(observable_space)


def _mask(space: Space, indicator: SHGrid) -> LinearOperator:
    """Multiplication by a 0/1 indicator, which is idempotent on any space."""
    mask = space.multiplication_operator(indicator)
    return mask.with_traits(mask.traits | Traits.IDEMPOTENT)


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
