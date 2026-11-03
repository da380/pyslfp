"""
The viscoelastic evolution as an operator on the ice history, the initial
sea level and the log viscosity, with its derivatives.

`SeaLevelEvolutionOperator` maps (I, S_0, chi), an ice thickness history of a
`FieldHistorySpace`, an initial sea level of a sphere space and the log
viscosity of every branch of a `LogViscositySpace`, to the responses at a
set of observation dates, under frozen or migrating shorelines. Its value is
a forward run; its derivative's forward action is the tangent-linear run and
its adjoint the adjoint sweep, by the discrete or the continuous method, one
run each and all three kernels from one sweep. Under frozen shorelines the
initial sea level contributes nothing to a change, and the map is linear in
the ice history; `fixing` holds arguments fixed by composition with an affine
inclusion, and with the shorelines frozen and the other two fixed returns the
linear `IceHistoryOperator`.

The derivative holds the forward trajectory's ocean functions, which is exact
in the continuum and, on the grid, the derivative of the affine piece the
trajectory lies on; see `SeaLevelEvolution.tangent_linear_ice_history`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
from pyshtools import SHGrid

from pygeoinf import (
    AffineOperator,
    DirectSum,
    HilbertSpace,
    Linearization,
    LinearOperator,
    Operator,
)

from ..core import EarthState
from ..core.history import FieldHistorySpace
from ..core.response import GRDResponse, GRDSpace
from ..core.spaces import check_load_space, lebesgue_response_space
from .evolution import SeaLevelEvolution, SeaLevelSnapshot
from .ice_history import IceHistoryOperator, _NodalIceModel
from .operator import ViscoelasticOperator
from .viscosity import LogViscositySpace

__all__ = ["SeaLevelEvolutionOperator"]

ARGUMENTS = ("ice_history", "initial_sea_level", "log_viscosity")
_METHODS = ("discrete", "continuous")
_SHORELINES = ("frozen", "migrating")


class SeaLevelEvolutionOperator(Operator):
    """
    The evolution as an operator: from DirectSum([history space, sea level
    space, log-viscosity space]), labelled `ice_history`,
    `initial_sea_level` and `log_viscosity`, to a `DirectSum` of one
    response space per observation date.

    The value at (I, S_0, chi) runs the ice history I from the state of
    sea level S_0 at the first date, with the rheology chi names, under the
    shorelines chosen: frozen at the evolution's background ocean function,
    or migrating by flotation from S_0 and I at the first date. The run is
    recorded, so that the linearisation at a point costs one run and its
    derivative is applied without another.
    """

    def __init__(
        self,
        evolution: SeaLevelEvolution,
        dates: Sequence[float],
        history_space: FieldHistorySpace,
        sea_level_space: HilbertSpace,
        viscosity_space: LogViscositySpace,
        observation_dates: Sequence[float],
        /,
        *,
        shorelines: str = "frozen",
        method: str = "discrete",
        space: Optional[GRDSpace] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
    ) -> None:
        """
        Args:
            evolution: The evolution, which supplies the Earth model, the
                scheme, the operator's material and, under frozen
                shorelines, the ocean function of its background state; the
                state's `land` is kept.
            dates: The run's dates, in kiloyears before present, decreasing,
                inside the history's interval of time.
            history_space: The space of ice thickness histories, its time
                coordinate the date and its field space over the model's
                sphere.
            sea_level_space: A pygeoinf sphere space over the model's sphere,
                for the initial sea level.
            viscosity_space: The log-viscosity space, on the evolution's
                material.
            observation_dates: The dates of the responses, on the run's
                dates.
            shorelines: "frozen" or "migrating".
            method: "discrete", the forward and reverse modes of the
                time-stepped map, exact transposes of each other; or
                "continuous", the discretised continuous tangent-linear and
                adjoint problems, transposes to the order of the scheme.
            space: The response space at each observation date, as for
                `SeaLevelEvolution.run`; None gives the L2 space without the
                horizontal part.
            rotational_feedbacks, rtol: As for `run_ice_history`.

        Raises:
            ValueError: If a space is not over the model's sphere or the
                material, a date lies outside the history's interval, an
                observation date is not on the run's dates or is repeated,
                or the shorelines or the method are unknown.
        """
        if shorelines not in _SHORELINES:
            raise ValueError(
                f"shorelines must be one of {_SHORELINES}, got {shorelines!r}"
            )
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        dates = np.asarray(dates, dtype=float)
        if dates.ndim != 1 or dates.size < 2 or np.any(np.diff(dates) >= 0.0):
            raise ValueError(
                "dates must be decreasing (kiloyears before present, the past "
                "first), with at least two entries"
            )
        if not isinstance(history_space, FieldHistorySpace):
            raise ValueError("the history space must be a FieldHistorySpace")
        if not isinstance(viscosity_space, LogViscositySpace):
            raise ValueError("the viscosity space must be a LogViscositySpace")
        model = evolution.model
        b = model.parameters.mean_sea_floor_radius
        for name, field_space in (
            ("history", history_space.field_space),
            ("sea level", sea_level_space),
        ):
            check_load_space(field_space)
            model.check_field(field_space.zero())
            if not np.isclose(field_space.radius, b, rtol=1e-12, atol=0.0):
                raise ValueError(f"the {name} space is not over the model's sphere")
        if viscosity_space.mesh is not evolution.operator.material.mesh:
            raise ValueError("the viscosity space is not on the evolution's material")
        if shorelines == "migrating" and evolution.state.has_prescribed_ocean_function:
            raise ValueError(
                "migrating shorelines follow the flotation condition, so the "
                "evolution's state must not prescribe its ocean function"
            )

        observation_dates = [float(d) for d in observation_dates]
        if len(set(observation_dates)) != len(observation_dates):
            raise ValueError("an observation date is repeated")
        nodes = []
        for date in observation_dates:
            matches = np.flatnonzero(np.isclose(dates, date))
            if matches.size != 1:
                raise ValueError(f"observation date {date} is not on the run's dates")
            nodes.append(int(matches[0]))
        if space is None:
            space = lebesgue_response_space(
                model, rotation=rotational_feedbacks is not False
            )

        self._evolution = evolution
        self._dates = dates
        self._observation_dates = observation_dates
        self._observation_nodes = nodes
        self._response_space = space
        self._shorelines = shorelines
        self._method = method
        self._rotational_feedbacks = rotational_feedbacks
        self._rtol = rtol
        self._basis = history_space.basis_matrix(dates)
        domain = DirectSum(
            [history_space, sea_level_space, viscosity_space], labels=ARGUMENTS
        )
        super().__init__(domain, DirectSum([space] * len(observation_dates)))

    # -- structure ---------------------------------------------------------

    @property
    def evolution(self) -> SeaLevelEvolution:
        """The evolution the operator is built on."""
        return self._evolution

    @property
    def dates(self) -> np.ndarray:
        """The run's dates, in kiloyears before present."""
        return self._dates.copy()

    @property
    def observation_dates(self) -> list[float]:
        """The dates of the responses, in the codomain's order."""
        return list(self._observation_dates)

    @property
    def response_space(self) -> GRDSpace:
        """The response space at each observation date."""
        return self._response_space

    @property
    def history_space(self) -> FieldHistorySpace:
        return self.domain.subspaces[0]

    @property
    def sea_level_space(self) -> HilbertSpace:
        return self.domain.subspaces[1]

    @property
    def viscosity_space(self) -> LogViscositySpace:
        return self.domain.subspaces[2]

    @property
    def shorelines(self) -> str:
        """ "frozen" or "migrating"."""
        return self._shorelines

    @property
    def method(self) -> str:
        """ "discrete" or "continuous"."""
        return self._method

    # -- the runs ----------------------------------------------------------

    def thickness_at_dates(self, history: np.ndarray, /) -> list[SHGrid]:
        """A history's thickness at the run's dates."""
        field_space = self.history_space.field_space
        return [field_space.from_components(row) for row in self._basis @ history]

    def _state(self, ice: SHGrid, sea_level) -> EarthState:
        """The state at the first date: frozen at the background ocean
        function, or by flotation."""
        background = self._evolution.state
        return EarthState(
            ice,
            sea_level,
            self._evolution.model,
            land=background.land,
            ocean_function=(
                background.ocean_function if self._shorelines == "frozen" else None
            ),
        )

    def _evolution_at(self, state: EarthState, chi) -> SeaLevelEvolution:
        """The evolution of the rheology chi names, from a state."""
        template = self._evolution.operator
        operator = ViscoelasticOperator(
            self.viscosity_space.rheology(chi),
            template.lmax,
            eps=template.eps,
            threads=template.threads,
        )
        return SeaLevelEvolution(operator, state, scheme=self._evolution.scheme)

    def _run(self, x) -> tuple[SeaLevelEvolution, list[SeaLevelSnapshot]]:
        history, sea_level, chi = x
        thickness = self.thickness_at_dates(history)
        evolution = self._evolution_at(self._state(thickness[0], sea_level), chi)
        snapshots, _ = evolution.run_ice_history(
            self._dates,
            _NodalIceModel(
                self._dates,
                thickness,
                length_scale=self._evolution.model.parameters.length_scale,
            ),
            shorelines=self._shorelines,
            space=self._response_space,
            rotational_feedbacks=self._rotational_feedbacks,
            rtol=self._rtol,
            record=True,
        )
        return evolution, snapshots

    def _responses(self, snapshots: Sequence[SeaLevelSnapshot]) -> tuple:
        return tuple(snapshots[node].response for node in self._observation_nodes)

    def _value(self, x) -> tuple:
        _, snapshots = self._run(x)
        return self._responses(snapshots)

    def _derivative(self, x) -> LinearOperator:
        return self._linearize(x).derivative

    def _linearize(self, x) -> Linearization:
        evolution, snapshots = self._run(x)
        return Linearization(
            x, self._responses(snapshots), self._derivative_about(evolution, snapshots)
        )

    def _derivative_about(
        self, evolution: SeaLevelEvolution, snapshots: Sequence[SeaLevelSnapshot]
    ) -> LinearOperator:
        """The derivative at the point a recorded run was made at: the
        tangent-linear run forwards, the adjoint sweep backwards, both
        holding the run's ocean functions."""
        domain = self.domain
        history_space = self.history_space
        sea_level_space = self.sea_level_space
        viscosity_space = self.viscosity_space
        rheology = evolution.operator.rheology
        ocean_functions = (
            [snap.ocean_function for snap in snapshots]
            if self._shorelines == "migrating"
            else None
        )

        def forward(v) -> tuple:
            delta_history, delta_sea_level, delta_chi = v
            nodal = viscosity_space.nodal(delta_chi)
            tangent, _ = evolution.tangent_linear_ice_history(
                self._dates,
                snapshots,
                ice_thickness=self.thickness_at_dates(delta_history),
                initial_sea_level=delta_sea_level,
                itau=[-itau * values for itau, values in zip(rheology.itau, nodal)],
                method=self._method,
                space=self._response_space,
                rotational_feedbacks=self._rotational_feedbacks,
                rtol=self._rtol,
            )
            return self._responses(tangent)

        def adjoint(duals: Sequence[GRDResponse]) -> tuple:
            sweep = (
                evolution.adjoint_ice_history
                if self._method == "discrete"
                else evolution.continuous_adjoint_ice_history
            )
            kernels = sweep(
                self._dates,
                dict(zip(self._observation_dates, duals)),
                ocean_functions=ocean_functions,
                internal_states=snapshots,
                rotational_feedbacks=self._rotational_feedbacks,
                rtol=self._rtol,
            )
            field_space = history_space.field_space
            thickness = np.stack(
                [field_space.to_components(d) for d in kernels.thickness]
            )
            history = history_space.from_components(
                history_space.solve_gram((self._basis.T @ thickness).reshape(-1))
            )
            sea_level = sea_level_space.from_components(
                sea_level_space.solve_gram(
                    sea_level_space.to_components(kernels.initial_sea_level)
                )
            )
            chi = viscosity_space.gradient(
                [-itau * k for itau, k in zip(rheology.itau, kernels.itau)]
            )
            return (history, sea_level, chi)

        return LinearOperator.from_callables(
            domain, self.codomain, forward, adjoint=adjoint
        )

    # -- fixing arguments --------------------------------------------------

    def fixing(
        self,
        *,
        ice_history: Optional[np.ndarray] = None,
        initial_sea_level: Optional[SHGrid] = None,
        log_viscosity: Optional[tuple] = None,
    ) -> Operator:
        """
        The operator on the arguments not given, the others held at the
        values given: the composition with the affine inclusion of the
        free arguments, whose derivatives follow by the chain rule. With
        one argument free its domain is that argument's space; with two, the
        direct sum of theirs in the order of `ARGUMENTS`.

        Under frozen shorelines with the initial sea level and the log
        viscosity fixed and the discrete method, the map is linear in the
        ice history and the result is an `IceHistoryOperator`, a
        `LinearOperator` whose adjoint is exact.

        Raises:
            ValueError: If every argument is given.
        """
        fixed = (ice_history, initial_sea_level, log_viscosity)
        free = [i for i, value in enumerate(fixed) if value is None]
        if not free:
            raise ValueError("at least one argument must be left free")
        if len(free) == 3:
            return self
        if free == [0] and self._shorelines == "frozen" and self._method == "discrete":
            background = self._evolution.state
            state = self._state(background.ice_thickness, initial_sea_level)
            return IceHistoryOperator(
                self._evolution_at(state, log_viscosity),
                self._dates,
                self.history_space,
                self._observation_dates,
                space=self._response_space,
                rotational_feedbacks=self._rotational_feedbacks,
                rtol=self._rtol,
            )
        spaces = self.domain.subspaces
        if len(free) == 1:
            free_domain = spaces[free[0]]

            def spread(y) -> tuple:
                return (y,)

            def gather(parts) -> object:
                return parts[0]

        else:
            free_domain = DirectSum(
                [spaces[i] for i in free], labels=[ARGUMENTS[i] for i in free]
            )

            def spread(y) -> tuple:
                return tuple(y)

            def gather(parts) -> tuple:
                return tuple(parts)

        def inclusion(y) -> tuple:
            parts = spread(y)
            return tuple(
                parts[free.index(i)] if i in free else space.zero()
                for i, space in enumerate(spaces)
            )

        def projection(x) -> object:
            return gather([x[i] for i in free])

        linear = LinearOperator.from_callables(
            free_domain, self.domain, inclusion, adjoint=projection
        )
        translation = tuple(
            space.zero() if i in free else fixed[i] for i, space in enumerate(spaces)
        )
        return self @ AffineOperator(linear, translation)

    def __repr__(self) -> str:
        return (
            f"SeaLevelEvolutionOperator({len(self._dates)} dates, "
            f"{len(self._observation_dates)} observed, shorelines={self._shorelines!r}, "
            f"method={self._method!r})"
        )
