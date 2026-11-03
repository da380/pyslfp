"""
A model set-up through time.

A `ModelSetup` describes one epoch. `HistorySetup` is the same vocabulary
through time: the set-up's joint space tensored with a time space, so that
every component is a history and every derived operator acts at each time.
The product metric is pygeoinf's tensor product's; a lift applies one of the
set-up's operators at every time, at one application per time mode, and a
contraction with a functional on the time factor samples a history
(`pyslfp.core.history`). The time coordinate is the model's non-dimensional
time, as everywhere in the library; `EarthModel.times_from_dates` converts
dates in kiloyears before present.

The scalar diagnostics, the GMSL functionals and the constraints, become
functions of time in the time space's basis. The viscoelastic coupling is
composition: a component's history, `history(name)`, lands on the history
space the evolution operators take. A history set-up is passive, like a
set-up.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Optional

import numpy as np

from pygeoinf import (
    CoordinateSpace,
    EuclideanSpace,
    LinearOperator,
    TensorProduct,
    Traits,
)

from ..core.history import FieldHistorySpace, at_time
from ..elastic.operator import FingerPrintOperator
from .components import LoadComponent
from .setup import ModelSetup

__all__ = ["HistorySetup"]


class HistorySetup:
    """
    A model set-up through time: `space` is the tensor product of the time
    space and the set-up's joint space, whose vectors are (K, N) arrays, K
    the time modes and N the set-up's dimension, and every derived operator
    is the set-up's lifted to act at each time.
    """

    def __init__(self, setup: ModelSetup, time_space: CoordinateSpace, /) -> None:
        """
        Args:
            setup: The set-up, which supplies the components and their
                operators.
            time_space: A pygeoinf interval space over the interval of time,
                in the model's non-dimensional time.

        Raises:
            TypeError: If the time space provides no coordinates.
        """
        if not isinstance(time_space, CoordinateSpace):
            raise TypeError("the time space must provide coordinates")
        self._setup = setup
        self._time_space = time_space
        self._space = TensorProduct([time_space, setup.space])
        self._observables = MappingProxyType(
            {name: self.lift(operator) for name, operator in setup.observables.items()}
        )
        self._constraints = MappingProxyType(
            {
                name: self.through_time(operator)
                for name, operator in setup.constraints.items()
            }
        )

    # -- structure ---------------------------------------------------------

    @property
    def setup(self) -> ModelSetup:
        """The set-up at one epoch."""
        return self._setup

    @property
    def time_space(self) -> CoordinateSpace:
        """The interval space of the time factor."""
        return self._time_space

    @property
    def space(self) -> TensorProduct:
        """The space of histories of the set-up: TensorProduct([time space,
        setup.space])."""
        return self._space

    @property
    def names(self) -> tuple[str, ...]:
        """The components' names, the labels of the set-up's space."""
        return self._setup.names

    @property
    def components(self) -> tuple[LoadComponent, ...]:
        return self._setup.components

    # -- lifting -----------------------------------------------------------

    def lift(self, operator: LinearOperator, /) -> LinearOperator:
        """
        An operator on the set-up's space, applied at every time: the tensor
        product of the identity on the time space with it, onto the product
        of the time space and its codomain. A field-valued codomain lands on
        the `FieldHistorySpace` over it.
        """
        if operator.domain != self._setup.space:
            raise ValueError("the operator must act on the set-up's space")
        return self._space.lift(operator)

    def through_time(self, functional: LinearOperator, /) -> LinearOperator:
        """
        A scalar diagnostic of the set-up, an operator into EuclideanSpace(1),
        applied at every time, as a function of time in the time space: its
        lift followed by the isometry from TensorProduct([time space,
        EuclideanSpace(1)]) onto the time space.
        """
        if operator_dim(functional) != 1:
            raise ValueError("the functional must map into EuclideanSpace(1)")
        lifted = self.lift(functional)
        time_space = self._time_space

        def squeeze(x: np.ndarray):
            return time_space.from_components(np.asarray(x).reshape(-1))

        def expand(y) -> np.ndarray:
            return time_space.to_components(y).reshape(-1, 1)

        isometry = LinearOperator.from_callables(
            lifted.codomain,
            time_space,
            squeeze,
            adjoint=expand,
            traits=Traits.ISOMETRY | Traits.SURJECTIVE,
        )
        composed = isometry @ lifted
        # an isometry onto the time space keeps a surjective functional's
        # surjectivity, which the constraints claim
        return composed.with_traits(
            composed.traits | (lifted.traits & Traits.SURJECTIVE)
        )

    def history(self, name: str, /) -> LinearOperator:
        """The history of one component, from the space onto the
        `FieldHistorySpace` over the component's space: the lift of the
        set-up's projection."""
        return self.lift(self._setup.space.projection(name))

    def at(self, time: float, /) -> LinearOperator:
        """The set-up's vector at a time of the interval, from the space onto
        the set-up's space."""
        return at_time(self._space, time)

    # -- the derived operators ---------------------------------------------

    @property
    def load_space(self) -> FieldHistorySpace:
        """The space of load histories over the set-up's load space."""
        return FieldHistorySpace(self._time_space, self._setup.load_space)

    @property
    def load_operator(self) -> LinearOperator:
        """From the space to the load history."""
        return self.lift(self._setup.load_operator)

    @property
    def support(self) -> LinearOperator:
        """The set-up's support at every time."""
        return self.lift(self._setup.support)

    @property
    def observables(self) -> Mapping[str, LinearOperator]:
        """The set-up's direct observables, each a history of its field."""
        return self._observables

    @property
    def constraints(self) -> Mapping[str, LinearOperator]:
        """The set-up's constraints at every time, each into the time space;
        the surjectivity of a constraint survives the lift."""
        return self._constraints

    def barystatic_gmsl(self, name: Optional[str] = None, /) -> LinearOperator:
        """The barystatic GMSL of a mass source, or of all of them, as a
        function of time."""
        return self.through_time(self._setup.barystatic_gmsl(name))

    @property
    def steric_gmsl(self) -> Optional[LinearOperator]:
        """The steric GMSL as a function of time, or None without a density
        component."""
        steric = self._setup.steric_gmsl
        return None if steric is None else self.through_time(steric)

    @property
    def gmsl_split(self) -> Optional[LinearOperator]:
        """The barystatic and steric GMSL at every time, the set-up's split
        lifted onto TensorProduct([time space, EuclideanSpace(2)]), a pair of
        functions of time held as a (K, 2) array; None without both."""
        split = self._setup.gmsl_split
        return None if split is None else self.lift(split)

    def response(self, fingerprint: FingerPrintOperator, /) -> LinearOperator:
        """The response to the set-up's load at every time: the elastic
        physics at each epoch, one fingerprint solve per time mode, onto the
        product of the time space and the response space."""
        return self.lift(self._setup.response(fingerprint))

    def relative_sea_level_change(
        self, fingerprint: FingerPrintOperator, /
    ) -> LinearOperator:
        """The relative sea level change at every time, direct terms
        included, onto the history of the fingerprint's field space."""
        return self.lift(self._setup.relative_sea_level_change(fingerprint))

    def geocentric_sea_level(
        self,
        fingerprint: FingerPrintOperator,
        /,
        *,
        remove_rotational_contribution: bool = True,
    ) -> LinearOperator:
        """The geocentric sea level change at every time, as for the
        set-up's."""
        return self.lift(
            self._setup.geocentric_sea_level(
                fingerprint,
                remove_rotational_contribution=remove_rotational_contribution,
            )
        )

    def gmsl(self, fingerprint: FingerPrintOperator, /) -> LinearOperator:
        """The global mean sea level change through the sea level equation,
        as a function of time."""
        return self.through_time(self._setup.gmsl(fingerprint))

    def __repr__(self) -> str:
        return f"HistorySetup({self._setup!r}, {self._time_space!r})"


def operator_dim(operator: LinearOperator) -> Optional[int]:
    """The dimension of an operator's Euclidean codomain, or None."""
    codomain = operator.codomain
    return codomain.dim if isinstance(codomain, EuclideanSpace) else None
