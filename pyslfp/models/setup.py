"""
The assembler: a model space and its derived objects from load components.

`ModelSetup` takes the components of a model space and derives, from their
roles and direct contributions alone, everything an inversion built on the
space needs from the physics:

- the joint space, a direct sum labelled by the components' names;
- the load operator, the sum of the components' loads, which feeds the sea
  level equation;
- the supports, block by block, for display and diagnostics;
- the direct operator of each observable, adding up the components'
  contributions in the observable's space;
- the barystatic global mean sea level change of each mass source and in
  total, the steric change from the steric sea level contributions, and the
  two together as the barystatic/steric split;
- the ocean mass constraint over the ocean-internal components.

Adding a component never means changing the assembler. A set-up is passive:
it holds spaces and operators, and leaves priors, conditioning and solvers to
the caller. Scalar diagnostics map into Euclidean spaces, so that they
compose with push-forwards and dense covariances.

The joint space holds globally defined fields, so a prior is a measure on
it as it stands, and is not masked. The spaces follow the regularity of
the physics (see `pyslfp.models.components`):

    component spaces, H^t  --load-->  load space, H^s with 0 <= s < 1/2
                           --sea level equation-->  response, H^(s+1)

with the pointwise observables in a space of order above 1, normally the
response's field space, and t no lower than that order. The loads carry
the supports; the pointwise observables leave them to their observation
points.
"""

from __future__ import annotations

from functools import reduce
from operator import add
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple

from pygeoinf import (
    BlockDiagonalLinearOperator,
    ColumnLinearOperator,
    DirectSum,
    EuclideanSpace,
    LinearOperator,
    RowLinearOperator,
    Traits,
)

from pyslfp.core import EarthState
from pyslfp.linear_operators.spatial import ocean_average_operator
from pyslfp.models.components import (
    STERIC_SEA_LEVEL,
    LoadComponent,
    Role,
    Space,
    barystatic_gmsl_operator,
)

OCEAN_MASS = "ocean_mass"


class ModelSetup:
    """
    A model space assembled from load components, with its derived operators.

    All operators act on `space`, the direct sum of the components' spaces
    labelled by their names, so `space.projection(name)` and
    `space.inclusion(name)` reach a component's block.
    """

    def __init__(
        self, state: EarthState, components: Iterable[LoadComponent], /
    ) -> None:
        """
        Args:
            state: The Earth state the components were built on, supplying the
                ocean area and the ocean averages.
            components: The components, in the order of the joint space.

        Raises:
            ValueError: If there are no components, two share a name, the
                components load different spaces, or their contributions to
                an observable lie in different spaces.
        """
        components = tuple(components)
        if not components:
            raise ValueError("A model set-up needs at least one component.")
        names = [component.name for component in components]
        if len(set(names)) != len(names):
            raise ValueError(f"Component names must be distinct, got {names}.")
        load_spaces = {
            component.load_space
            for component in components
            if component.load_space is not None
        }
        if len(load_spaces) != 1:
            raise ValueError(
                "The components must load one and the same space; "
                f"{len(load_spaces)} were found."
            )

        self._state = state
        self._components = components
        self._load_space = load_spaces.pop()
        self._space = DirectSum([c.space for c in components], labels=names)

        self._load_operator = self._row(
            {c.name: c.to_load for c in components if c.to_load is not None},
            self._load_space,
        )
        self._support = BlockDiagonalLinearOperator(
            [c.support for c in components],
            domain=self._space,
            codomain=self._space,
        )

        observable_names = []
        for component in components:
            for name in component.observables:
                if name not in observable_names:
                    observable_names.append(name)
        self._observables = MappingProxyType(
            {name: self._observable(name) for name in observable_names}
        )

        self._barystatic = MappingProxyType(
            {
                c.name: barystatic_gmsl_operator(state, c)
                @ self._space.projection(c.name)
                for c in components
                if c.role is Role.MASS_SOURCE and c.to_load is not None
            }
        )
        self._total_barystatic = (
            reduce(add, self._barystatic.values()) if self._barystatic else None
        )

        if STERIC_SEA_LEVEL in self._observables:
            steric = self._observables[STERIC_SEA_LEVEL]
            self._steric = ocean_average_operator(state, steric.codomain) @ steric
        else:
            self._steric = None
        if self._total_barystatic is not None and self._steric is not None:
            split = ColumnLinearOperator([self._total_barystatic, self._steric])
            self._split = split.codomain.coordinate_projection() @ split
        else:
            self._split = None

        ocean_average = ocean_average_operator(state, self._load_space)
        ocean_internal = {
            c.name: ocean_average @ c.to_load
            for c in components
            if c.role is Role.OCEAN_INTERNAL and c.to_load is not None
        }
        constraints = {}
        if ocean_internal:
            constraint = self._row(ocean_internal, EuclideanSpace(1))
            constraints[OCEAN_MASS] = constraint.with_traits(
                constraint.traits | Traits.SURJECTIVE
            )
        self._constraints = MappingProxyType(constraints)

    # ---------------------------------------------------------#
    #                         Structure                        #
    # ---------------------------------------------------------#

    @property
    def state(self) -> EarthState:
        """The Earth state the components were built on."""
        return self._state

    @property
    def components(self) -> Tuple[LoadComponent, ...]:
        """The components, in the order of the joint space."""
        return self._components

    @property
    def names(self) -> Tuple[str, ...]:
        """The components' names, which label the joint space."""
        return tuple(c.name for c in self._components)

    def component(self, name: str, /) -> LoadComponent:
        """The component with the given name."""
        for component in self._components:
            if component.name == name:
                return component
        raise KeyError(f"No component is named '{name}'; there are {self.names}.")

    @property
    def space(self) -> DirectSum:
        """The joint space, labelled by the components' names."""
        return self._space

    @property
    def load_space(self) -> Space:
        """The space of the loads."""
        return self._load_space

    # ---------------------------------------------------------#
    #                         Operators                        #
    # ---------------------------------------------------------#

    @property
    def load_operator(self) -> LinearOperator:
        """The total load, the sum of the components' loads."""
        return self._load_operator

    @property
    def support(self) -> LinearOperator:
        """The components' supports, block by block: multiplication by each
        support's indicator. The operators of the set-up account for the
        supports already; this is for display and diagnostics, such as
        plotting a field where it is physical, not for masking a prior."""
        return self._support

    @property
    def observables(self) -> Mapping[str, LinearOperator]:
        """The direct operator of each observable, into the observable's
        space, summing the components' contributions. Read-only."""
        return self._observables

    @property
    def constraints(self) -> Mapping[str, LinearOperator]:
        """
        The constraints the components' roles imply, each into a Euclidean
        space and claiming surjectivity, as exact conditioning needs. Read-only.

        `ocean_mass`, present when there are ocean-internal components, is
        the ocean average of their loads, zero when they move water without
        changing the ocean's mass. A prior on the joint space's global
        fields is positive definite, so it can be conditioned on the
        constraint exactly as it stands.
        """
        return self._constraints

    # ---------------------------------------------------------#
    #                 Global mean sea level                    #
    # ---------------------------------------------------------#

    def barystatic_gmsl(self, name: Optional[str] = None, /) -> LinearOperator:
        """
        The barystatic global mean sea level change, into EuclideanSpace(1).

        Args:
            name: A mass source, for its contribution alone; all the mass
                sources together if omitted.

        Raises:
            ValueError: If the set-up has no mass source, or the named
                component is not one.
        """
        if name is None:
            if self._total_barystatic is None:
                raise ValueError("This set-up has no mass source.")
            return self._total_barystatic
        if name not in self._barystatic:
            raise ValueError(f"'{name}' is not a mass source of this set-up.")
        return self._barystatic[name]

    @property
    def steric_gmsl(self) -> Optional[LinearOperator]:
        """The steric global mean sea level change, the ocean average of the
        steric sea level, into EuclideanSpace(1); None without a steric
        contribution."""
        return self._steric

    @property
    def gmsl_split(self) -> Optional[LinearOperator]:
        """The barystatic and steric global mean sea level changes together,
        into EuclideanSpace(2) in that order; None unless the set-up has
        both."""
        return self._split

    # ---------------------------------------------------------#
    #                     Private helpers                      #
    # ---------------------------------------------------------#

    def _observable(self, name: str) -> LinearOperator:
        """The row of the components' contributions to an observable, which
        must share its space."""
        blocks = {
            c.name: c.observables[name]
            for c in self._components
            if name in c.observables
        }
        codomains = {operator.codomain for operator in blocks.values()}
        if len(codomains) != 1:
            raise ValueError(
                f"The contributions to '{name}' lie in {len(codomains)} different "
                "spaces; give their components one observable space."
            )
        return self._row(blocks, codomains.pop())

    def _row(
        self, blocks: Mapping[str, LinearOperator], codomain: Space
    ) -> LinearOperator:
        """A row over the joint space, the given block for each named
        component and zero for the rest."""
        row = RowLinearOperator(
            [
                blocks.get(c.name, LinearOperator.zero(c.space, codomain=codomain))
                for c in self._components
            ]
        )
        return row
