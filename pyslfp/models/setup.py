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
- the ocean mass constraint over the ocean-internal components;
- through the sea level equation, given a fingerprint operator: the relative
  sea level change, its global mean, and its regional decomposition into
  steric, dynamic manometric and barystatic manometric parts after Gregory et
  al. (2019).

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
from typing import Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from pygeoinf import (
    BlockDiagonalLinearOperator,
    ColumnLinearOperator,
    DirectSum,
    EuclideanSpace,
    LinearOperator,
    RowLinearOperator,
    Traits,
)

from pyshtools import SHGrid

from pyslfp.core import EarthState
from pyslfp.core.observables import geocentric_sea_level_operator
from pyslfp.core.response import RELATIVE_SEA_LEVEL_CHANGE, GRDSpace
from pyslfp.elastic import FingerPrintOperator
from pyslfp.core import (
    ocean_average_operator,
    region_average_operator,
)
from pyslfp.core import averaging_operator
from pyslfp.models.components import (
    GEOCENTRIC_SEA_LEVEL,
    STERIC_SEA_LEVEL,
    LoadComponent,
    Role,
    Space,
    barystatic_gmsl_operator,
)

OCEAN_MASS = "ocean_mass"

# Regions as `region_average_operator` takes them (a name, a flat list for one
# composite region, a list of lists for several), or as masks.
Regions = Union[str, List[str], List[List[str]], Sequence[SHGrid]]


class RegionalDecomposition(NamedTuple):
    """The regional averages of the relative sea level change, split after
    Gregory et al. (2019); each an operator from the joint space into
    EuclideanSpace(number of regions), the three summing to the average of
    `ModelSetup.relative_sea_level_change`.

    Attributes:
        steric: The steric sea level change, which loads nothing.
        dynamic_manometric: The rest of the direct contributions to the
            geocentric sea level (the sterodynamic sea level change less its
            steric part), plus
            the sea level equation's response to the ocean-internal loads.
        barystatic_manometric: The sea level equation's response to the
            mass sources' loads, their gravitational, rotational and
            deformational fingerprint.
    """

    steric: LinearOperator
    dynamic_manometric: LinearOperator
    barystatic_manometric: LinearOperator


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
            self._steric = self._ocean_mean(steric.codomain) @ steric
        else:
            self._steric = None
        if self._total_barystatic is not None and self._steric is not None:
            split = ColumnLinearOperator([self._total_barystatic, self._steric])
            self._split = split.codomain.coordinate_projection() @ split
        else:
            self._split = None

        ocean_average = self._ocean_mean(self._load_space)
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
        contribution. The steric sea level carries the ocean function, so its
        ocean average is its integral over the sphere over the ocean area,
        taken by the grid's quadrature, as the sea level equation takes the
        mass of a load."""
        return self._steric

    @property
    def gmsl_split(self) -> Optional[LinearOperator]:
        """The barystatic and steric global mean sea level changes together,
        into EuclideanSpace(2) in that order; None unless the set-up has
        both."""
        return self._split

    # ---------------------------------------------------------#
    #            Through the sea level equation                #
    # ---------------------------------------------------------#

    def relative_sea_level_change(
        self, fingerprint: FingerPrintOperator, /
    ) -> LinearOperator:
        """
        The relative sea level change, into the fingerprint operator's field
        space: the sea level equation's response to the total load, plus the
        direct contributions to the geocentric sea level (the sterodynamic
        sea level change's), which raise the sea surface over an unmoved solid
        surface.

        Args:
            fingerprint: The fingerprint operator, on the set-up's load space;
                the geocentric sea level contributions must lie in its field
                space (the components' `observable_space`).

        Raises:
            ValueError: If the spaces do not match.
        """
        response = self._response_space(fingerprint)
        change = (
            response.field_operator(RELATIVE_SEA_LEVEL_CHANGE)
            @ fingerprint
            @ (self._load_operator)
        )
        return self._with_geocentric_contributions(change, response)

    def geocentric_sea_level(
        self,
        fingerprint: FingerPrintOperator,
        /,
        *,
        remove_rotational_contribution: bool = True,
    ) -> LinearOperator:
        """
        The geocentric sea level change, into the fingerprint operator's field
        space, the observable of altimetry: the response's geocentric sea level
        (`pyslfp.core.geocentric_sea_level_operator`, the relative sea level
        change plus the vertical displacement) for the total load, plus the
        direct contributions to the geocentric sea level, as in
        `relative_sea_level_change`. Point evaluation of it gives altimetry
        data.

        Args:
            fingerprint: The fingerprint operator, on the set-up's load space.
            remove_rotational_contribution: Whether the centrifugal
                potential's height is added, as for
                `geocentric_sea_level_operator`.

        Raises:
            ValueError: If the spaces do not match.
        """
        response = self._response_space(fingerprint)
        change = (
            geocentric_sea_level_operator(
                self._state,
                response,
                remove_rotational_contribution=remove_rotational_contribution,
            )
            @ fingerprint
            @ self._load_operator
        )
        return self._with_geocentric_contributions(change, response)

    def _with_geocentric_contributions(
        self, change: LinearOperator, response: GRDSpace
    ) -> LinearOperator:
        """A sea level operator with the components' direct contributions to
        the geocentric sea level added, which must lie in the response's
        field space."""
        if GEOCENTRIC_SEA_LEVEL not in self._observables:
            return change
        direct = self._observables[GEOCENTRIC_SEA_LEVEL]
        if direct.codomain != response.field_space:
            raise ValueError(
                "the geocentric sea level contributions must lie in the "
                "fingerprint operator's field space"
            )
        return change + direct

    def gmsl(self, fingerprint: FingerPrintOperator, /) -> LinearOperator:
        """
        The global mean sea level change, the ocean average of
        `relative_sea_level_change`, into EuclideanSpace(1). Mass conservation makes it
        the barystatic change plus the steric change of `gmsl_split`, the
        ocean-internal loads' own response cancelling the mean of the
        sterodynamic sea level change less its steric part; the difference measures the sea level equation's mass
        balance.
        """
        field_space = self._response_space(fingerprint).field_space
        return ocean_average_operator(self._state, field_space) @ (
            self.relative_sea_level_change(fingerprint)
        )

    def regional_decomposition(
        self, fingerprint: FingerPrintOperator, regions: Regions, /
    ) -> RegionalDecomposition:
        """
        The regional averages of the relative sea level change, split into
        steric, dynamic manometric and barystatic manometric parts after
        Gregory et al. (2019), through the components' roles: the response to
        the mass sources' loads is barystatic, the response to the
        ocean-internal loads and the direct contributions to the geocentric sea
        level
        less the steric sea level are dynamic, and the steric sea level is
        steric. The three sum to the average of `relative_sea_level_change`.

        Args:
            fingerprint: The fingerprint operator, as for `relative_sea_level_change`.
            regions: Region names, as `region_average_operator` takes them, or
                a sequence of masks.
        """
        response = self._response_space(fingerprint)
        field_space = response.field_space
        on_fields = self._averaging(field_space, regions)
        on_loads = self._averaging(self._load_space, regions)
        codomain = on_fields.codomain
        sea_level = response.field_operator(RELATIVE_SEA_LEVEL_CHANGE) @ fingerprint

        def of_role(role: Role) -> LinearOperator:
            return self._row(
                {
                    c.name: c.to_load
                    for c in self._components
                    if c.role is role and c.to_load is not None
                },
                self._load_space,
            )

        barystatic = on_fields @ sea_level @ of_role(Role.MASS_SOURCE)
        dynamic = on_fields @ sea_level @ of_role(Role.OCEAN_INTERNAL)
        if GEOCENTRIC_SEA_LEVEL in self._observables:
            dynamic = dynamic + on_fields @ self._observables[GEOCENTRIC_SEA_LEVEL]
        if STERIC_SEA_LEVEL in self._observables:
            steric = on_loads @ self._observables[STERIC_SEA_LEVEL]
            dynamic = dynamic - steric
        else:
            steric = LinearOperator.zero(self._space, codomain=codomain)
        return RegionalDecomposition(steric, dynamic, barystatic)

    # ---------------------------------------------------------#
    #                     Private helpers                      #
    # ---------------------------------------------------------#

    def _ocean_mean(self, space: Space) -> LinearOperator:
        """A field's integral over the sphere over the ocean area, into
        EuclideanSpace(1): the ocean average of a field the ocean function
        already confines to the ocean, such as an ocean-internal load or the
        steric sea level. The weight is the constant 1, which is band-limited,
        so the pairing is the grid's quadrature of the field itself, exactly
        the integral the sea level equation's mass conservation takes."""
        ones = self._state.model.zero_grid()
        ones.data[:] = 1.0
        return (1.0 / self._state.ocean_area) * space.l2_products_operator([ones])

    def _response_space(self, fingerprint: FingerPrintOperator):
        """The fingerprint operator's response space, checked against the
        set-up's load space."""
        if fingerprint.domain != self._load_space:
            raise ValueError(
                "the fingerprint operator must act on the set-up's load space"
            )
        return fingerprint.codomain

    def _averaging(self, space: Space, regions: Regions) -> LinearOperator:
        """Regional averages on a space, from names or masks."""
        if (
            isinstance(regions, (list, tuple))
            and regions
            and all(isinstance(r, SHGrid) for r in regions)
        ):
            return averaging_operator(self._state, space, list(regions))
        return region_average_operator(self._state, space, regions)

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
