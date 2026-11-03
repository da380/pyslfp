"""
The ice history as the argument of a linear operator, and its Sobolev kernels.

`IceHistoryOperator` maps an ice thickness history, a vector of a
`FieldHistorySpace`, to the responses at a set of observation dates, under
frozen shorelines, where the map is linear: it is `SeaLevelEvolutionOperator`
with the initial sea level and the viscosity fixed, as a `LinearOperator`
whose adjoint is exact. The forward reads the history at
the run's dates and runs `SeaLevelEvolution.run_ice_history`. The adjoint
passes the duals of the responses, as measurements, to the discrete adjoint
`SeaLevelEvolution.adjoint_ice_history`, whose thickness duals d_n satisfy

    delta J = sum_n int d_n delta I(t_n) dS

exactly for the discrete map. With phi_k(t) the time eigenfunctions and D
the array whose rows are the components of the d_n, the derivative's
components are Phi^T D, and the adjoint, the Riesz representer of the
derivative in the history space, divides them by the metric mu_k nu_j.

In a Sobolev history space of time order above one half this representer is
the Sobolev kernel. The delta in time that an observation contributes becomes
the Green's function of the time operator, continuous in time, so the kernel
converges as time modes are added, where an L2 representer grows without
bound, and as the date grid is refined, where the node duals at an
observation date do not converge pointwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
from pyshtools import SHGrid

from pygeoinf import DirectSum, LinearOperator

from ..core.history import FieldHistorySpace
from ..core.response import GRDResponse, GRDSpace
from ..core.spaces import lebesgue_response_space
from ..ice import BaseIceModel
from .evolution import SeaLevelEvolution

__all__ = ["IceHistoryOperator"]


class _NodalIceModel(BaseIceModel):
    """An ice history given by its thickness at dates and interpolated
    linearly between them, as the `BaseIceModel` contract asks; a run whose
    dates are the nodes reads the nodes themselves."""

    def __init__(
        self, dates: np.ndarray, thicknesses: Sequence[SHGrid], /, *, length_scale
    ) -> None:
        super().__init__(length_scale=length_scale)
        order = np.argsort(dates)
        self._dates = np.asarray(dates, dtype=float)[order]
        self._values = [thicknesses[i].data for i in order]
        self._template = thicknesses[0]

    def get_ice_thickness(
        self, date, lmax, /, *, grid="DH", sampling=1, extend=True
    ) -> SHGrid:
        template = self._template
        if lmax != template.lmax or extend != template.extend:
            raise ValueError("the history is held on the operator's grid alone")
        dates = self._dates
        j = int(np.clip(np.searchsorted(dates, date) - 1, 0, dates.size - 2))
        weight = (date - dates[j]) / (dates[j + 1] - dates[j])
        if weight < -1e-9 or weight > 1.0 + 1e-9:
            raise ValueError(f"date {date} lies outside the history's nodes")
        values = (1.0 - weight) * self._values[j] + weight * self._values[j + 1]
        return SHGrid.from_array(values, grid=template.grid)


class IceHistoryOperator(LinearOperator[np.ndarray, list[GRDResponse]]):
    """
    The map from an ice thickness history to the responses at observation
    dates, under frozen shorelines: a pygeoinf `LinearOperator` from a
    `FieldHistorySpace` into a `DirectSum` of one response space per
    observation date.

    The history's time coordinate is the model's non-dimensional time,
    increasing from zero at the run's first date, as for
    `SeaLevelEvolution.run`; `EarthModel.times_from_dates` converts dates.
    The direct load at each date is the grounded thickness change since the
    first, over the background state's ocean function, so a history constant
    in time loads nothing. The adjoint is the discrete adjoint's thickness
    duals carried onto the history space; in a Sobolev history space it is
    the Sobolev kernel of the measurement it is applied to.
    """

    def __init__(
        self,
        evolution: SeaLevelEvolution,
        dates: Sequence[float],
        domain: FieldHistorySpace,
        observation_dates: Sequence[float],
        /,
        *,
        space: Optional[GRDSpace] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
    ) -> None:
        """
        Args:
            evolution: The evolution, whose background state gives the
                frozen ocean function; its state when the operator is
                applied is the one used.
            dates: The run's dates, in kiloyears before present, decreasing;
                their times from the first, in the model's unit, must lie
                inside the history's interval.
            domain: The history space, its time coordinate the model's time
                from the run's first date and its field space over the
                evolution's model sphere.
            observation_dates: The dates of the responses, on the run's
                dates.
            space: The response space at each observation date, as for
                `SeaLevelEvolution.run`; None gives the L2 space without the
                horizontal part.
            rotational_feedbacks, rtol: As for `run_ice_history`.

        Raises:
            ValueError: If the domain is not over the model's sphere, a date
                lies outside its interval, or an observation date is not on
                the run's dates or is repeated.
        """
        dates = np.asarray(dates, dtype=float)
        if dates.ndim != 1 or dates.size < 2 or np.any(np.diff(dates) >= 0.0):
            raise ValueError(
                "dates must be decreasing (kiloyears before present, the past "
                "first), with at least two entries"
            )
        if not isinstance(domain, FieldHistorySpace):
            raise ValueError("the domain must be a FieldHistorySpace")
        model = evolution.model
        field_space = domain.field_space
        model.check_field(field_space.zero())
        if not np.isclose(
            field_space.radius,
            model.parameters.mean_sea_floor_radius,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError("the history's field space is not over the model's sphere")

        observation_dates = [float(d) for d in observation_dates]
        if len(set(observation_dates)) != len(observation_dates):
            raise ValueError("an observation date is repeated")
        nodes = []
        for date in observation_dates:
            matches = np.flatnonzero(np.isclose(dates, date))
            if matches.size != 1:
                raise ValueError(f"observation date {date} is not on the run's dates")
            nodes.append(int(matches[0]))
        self._observation_nodes = nodes

        if space is None:
            space = lebesgue_response_space(
                model, rotation=rotational_feedbacks is not False
            )

        self._evolution = evolution
        self._dates = dates
        self._observation_dates = observation_dates
        self._response_space = space
        self._rotational_feedbacks = rotational_feedbacks
        self._rtol = rtol
        # the time eigenfunctions at the run's dates, refusing a date outside
        # the history's interval
        self._times = model.times_from_dates(dates, dates[0])
        self._basis = domain.basis_matrix(self._times)
        codomain = DirectSum([space] * len(observation_dates))
        super().__init__(domain, codomain)

    # -- structure ---------------------------------------------------------

    @property
    def evolution(self) -> SeaLevelEvolution:
        """The evolution the operator runs."""
        return self._evolution

    @property
    def dates(self) -> np.ndarray:
        """The run's dates, in kiloyears before present."""
        return self._dates.copy()

    @property
    def times(self) -> np.ndarray:
        """The run's dates as the history's times, from zero at the first."""
        return self._times.copy()

    @property
    def observation_dates(self) -> list[float]:
        """The dates of the responses, in the codomain's order."""
        return list(self._observation_dates)

    @property
    def response_space(self) -> GRDSpace:
        """The response space at each observation date."""
        return self._response_space

    def thickness_at_dates(self, history: np.ndarray, /) -> list[SHGrid]:
        """The history's thickness at the run's dates."""
        field_space = self.domain.field_space
        values = self._basis @ history
        return [field_space.from_components(row) for row in values]

    # -- the maps ----------------------------------------------------------

    def _value(self, history: np.ndarray) -> list[GRDResponse]:
        ice_model = _NodalIceModel(
            self._dates,
            self.thickness_at_dates(history),
            length_scale=self._evolution.model.parameters.length_scale,
        )
        snapshots, _ = self._evolution.run_ice_history(
            self._dates,
            ice_model,
            space=self._response_space,
            rotational_feedbacks=self._rotational_feedbacks,
            rtol=self._rtol,
        )
        # the snapshots come in the order of the run's dates
        return [snapshots[node].response for node in self._observation_nodes]

    def _adjoint_value(self, duals: list[GRDResponse]) -> np.ndarray:
        measurement = dict(zip(self._observation_dates, duals))
        kernels = self._evolution.adjoint_ice_history(
            self._dates,
            measurement,
            rotational_feedbacks=self._rotational_feedbacks,
            rtol=self._rtol,
        )
        domain = self.domain
        field_space = domain.field_space
        thickness = np.stack([field_space.to_components(d) for d in kernels.thickness])
        derivative = self._basis.T @ thickness
        return domain.from_components(domain.solve_gram(derivative.reshape(-1)))
