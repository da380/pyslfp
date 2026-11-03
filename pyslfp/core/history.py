"""
Fields on the sphere through time, and their Hilbert spaces.

A `FieldHistory` is a field on the sphere at every time of an interval,
u(t, x). Its space, `FieldHistorySpace`, is built from a pygeoinf interval
space for the time and a pygeoinf sphere space for the field, both coordinate
spaces with diagonal metrics. A history is held by its components c_kj on the
products phi_k(t) Y_j(x) of the time space's eigenfunctions and the field
space's basis, as a K x N array, and the metric is the product mu_k nu_j of
the two spaces' metrics.

With Sobolev factors of orders s and r the norm is that of
H^s(T) (x) H^r(S^2), whose two orders are independent: for s > 1/2 the
history at a time is a field of the field space, and evaluation at a time is
bounded. An isotropic space-time norm on the same basis would lose half a
derivative in space at each time. The time basis is smooth, as the histories
the library models are continuous in time, and the time space's mesh is
padded at both ends, so that no end condition marks a history inside its
interval.

A forward map that acts at each time alone commutes with the time basis, so
it costs one application per time mode, whatever the number of times
observed.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from pyshtools import SHGrid

from pygeoinf import DiagonalMetricSpace, HilbertSpace, LinearOperator

__all__ = ["FieldHistory", "FieldHistorySpace"]


class FieldHistory:
    """
    A field on the sphere through an interval of time, a vector of a
    `FieldHistorySpace`.

    `components` is the K x N array of its components, K the time space's
    dimension and N the field space's: row k holds the field space's
    components of the history's part along the k-th time eigenfunction.
    """

    __slots__ = ("_space", "components")

    def __init__(self, space: FieldHistorySpace, components: np.ndarray, /) -> None:
        self._space = space
        self.components = np.asarray(components, dtype=float)

    @property
    def space(self) -> FieldHistorySpace:
        """The space the history belongs to."""
        return self._space

    def at(self, time: float, /) -> SHGrid:
        """The field at a time of the interval, a vector of the field space."""
        row = self._space.basis_matrix([time])[0]
        return self._space.field_space.from_components(row @ self.components)

    def __repr__(self) -> str:
        lower, upper = self._space.bounds
        k, _ = self._space.shape
        return (
            f"FieldHistory([{lower}, {upper}], {k} time modes, "
            f"lmax={self._space.field_space.lmax})"
        )


class FieldHistorySpace(DiagonalMetricSpace[FieldHistory]):
    """
    The Hilbert space of fields on the sphere through an interval of time: the
    product of a time space and a field space, with the product metric.

    The components are the K x N array of a history read row by row; the
    metric of component (k, j) is mu_k nu_j, mu the time space's and nu the
    field space's. A rank-one history a(t) f(x) therefore has as its norm
    the product of the norms of a and f, and the inner product of two is the
    product of theirs.
    """

    def __init__(self, time_space: HilbertSpace, field_space: HilbertSpace, /) -> None:
        """
        Args:
            time_space: A pygeoinf interval space (Lebesgue or Sobolev) over the
                interval of time, in the time coordinate the history is read
                in; its padding and length scale set the time basis.
            field_space: A pygeoinf sphere space (Lebesgue or Sobolev), whose
                grid, radius, components and metric the fields take.

        Raises:
            ValueError: If either space lacks a diagonal metric.
        """
        for name, space in (("time", time_space), ("field", field_space)):
            if not getattr(space, "has_diagonal_metric", False):
                raise ValueError(f"the {name} space must have a diagonal metric")
        self._time_space = time_space
        self._field_space = field_space
        self._shape = (time_space.dim, field_space.dim)
        super().__init__(
            np.outer(time_space.metric_values, field_space.metric_values).ravel()
        )

    # -- structure ---------------------------------------------------------

    @property
    def time_space(self) -> HilbertSpace:
        """The interval space of the time factor."""
        return self._time_space

    @property
    def field_space(self) -> HilbertSpace:
        """The sphere space of the fields."""
        return self._field_space

    @property
    def shape(self) -> tuple[int, int]:
        """(K, N): the dimensions of the time space and the field space."""
        return self._shape

    @property
    def bounds(self) -> tuple[float, float]:
        """The interval of time, its padding excluded."""
        return self._time_space.bounds

    def basis_matrix(self, times: Sequence[float], /) -> np.ndarray:
        """The time eigenfunctions at the given times, an array (len(times),
        K): row n holds phi_k(t_n).

        Raises:
            ValueError: If a time lies outside the interval.
        """
        return self._time_space.basis_matrix(times)

    def _key(self) -> Hashable:
        return ("FieldHistorySpace", self._time_space, self._field_space)

    # -- vectors -----------------------------------------------------------

    def to_components(self, x: FieldHistory) -> np.ndarray:
        return np.array(x.components, dtype=float).reshape(-1)

    def from_components(self, c: np.ndarray) -> FieldHistory:
        c = np.asarray(c, dtype=float)
        if c.shape != (self.dim,):
            raise ValueError(f"Expected {self.dim} components, got {c.shape}.")
        return FieldHistory(self, c.reshape(self._shape).copy())

    def copy(self, x: FieldHistory) -> FieldHistory:
        return FieldHistory(self, np.array(x.components, dtype=float))

    def axpy(self, a: float, x: FieldHistory, y: FieldHistory) -> FieldHistory:
        y.components += a * x.components
        return y

    def scale_inplace(self, a: float, x: FieldHistory) -> FieldHistory:
        x.components *= a
        return x

    def from_rank_one(self, profile: np.ndarray, field: SHGrid, /) -> FieldHistory:
        """The history a(t) f(x), from a vector `profile` of the time space
        and a field `field` of the field space."""
        return FieldHistory(
            self,
            np.outer(
                self._time_space.to_components(profile),
                self._field_space.to_components(field),
            ),
        )

    # -- operators ---------------------------------------------------------

    def evaluation_operator(
        self, time: float, /
    ) -> LinearOperator[FieldHistory, SHGrid]:
        """
        The map from a history to its field at a time of the interval, into
        the field space.

        With phi_k(t) the time eigenfunctions there, the field's components
        are sum_k phi_k(t) c_kj, and the adjoint of a field y is the history
        of components phi_k(t) y_j / mu_k: the field's metric cancels, and
        the time metric divides. For a time space of order above one half
        the adjoint is bounded and continuous in t.

        Raises:
            ValueError: If the time lies outside the interval.
        """
        row = self.basis_matrix([time])[0]
        weights = row / self._time_space.metric_values
        field_space = self._field_space

        def mapping(u: FieldHistory) -> SHGrid:
            return field_space.from_components(row @ u.components)

        def adjoint(y: SHGrid) -> FieldHistory:
            return FieldHistory(self, np.outer(weights, field_space.to_components(y)))

        return LinearOperator.from_callables(
            self, field_space, mapping, adjoint=adjoint
        )
