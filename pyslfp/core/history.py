"""
Fields on the sphere through time: their Hilbert space.

A history is a field on the sphere at every time of an interval, u(t, x). Its
space, `FieldHistorySpace`, is the pygeoinf tensor product of an interval
space for the time and a sphere space for the field. A history is the K x N
array of its components c_kj on the products phi_k(t) Y_j(x) of the time
space's eigenfunctions and the field space's basis, and the metric is the
product mu_k nu_j of the two spaces' metrics, applied factor by factor.

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
observed; `lift` builds it from the map on the field space.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pyshtools import SHGrid

from pygeoinf import CoordinateSpace, LinearFunctional, LinearOperator, TensorProduct

__all__ = ["FieldHistorySpace"]


class FieldHistorySpace(TensorProduct):
    """
    The Hilbert space of fields on the sphere through an interval of time: the
    tensor product of a time space and a field space.

    A history is a (K, N) array, K the time space's dimension and N the field
    space's: row k holds the field space's components of the history's part
    along the k-th time eigenfunction. A rank-one history a(t) f(x) is
    `rank_one(a, f)`, and has as its norm the product of the norms of a and f.
    """

    def __init__(self, time_space: CoordinateSpace, field_space: CoordinateSpace, /):
        """
        Args:
            time_space: A pygeoinf interval space (Lebesgue or Sobolev) over the
                interval of time, in the time coordinate the history is read
                in; its padding and length scale set the time basis.
            field_space: A pygeoinf sphere space (Lebesgue or Sobolev), whose
                grid, radius, components and metric the fields take.

        Raises:
            TypeError: If either space provides no coordinates.
        """
        super().__init__([time_space, field_space])

    # -- identity ----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Equal to any tensor product of the same factors, which is what this
        is: a lift or a contraction on the product builds a plain one, and
        has to land here."""
        if self is other:
            return True
        if not isinstance(other, TensorProduct):
            return NotImplemented
        return self._key() == other._key()

    # Defining __eq__ discards the inherited hash; the product's is the right one.
    def __hash__(self) -> int:
        return hash((TensorProduct, self._key()))

    # -- structure ---------------------------------------------------------

    @property
    def time_space(self) -> CoordinateSpace:
        """The interval space of the time factor."""
        return self.factors[0]

    @property
    def field_space(self) -> CoordinateSpace:
        """The sphere space of the fields."""
        return self.factors[1]

    @property
    def bounds(self) -> tuple[float, float]:
        """The interval of time, its padding excluded."""
        return self.time_space.bounds

    def basis_matrix(self, times: Sequence[float], /) -> np.ndarray:
        """The time eigenfunctions at the given times, an array (len(times),
        K): row n holds phi_k(t_n).

        Raises:
            ValueError: If a time lies outside the interval.
        """
        return self.time_space.basis_matrix(times)

    # -- operators ---------------------------------------------------------

    def evaluation_operator(self, time: float, /) -> LinearOperator[np.ndarray, SHGrid]:
        """
        The map from a history to its field at a time of the interval, into
        the field space: the contraction with the time space's Dirac
        functional there.

        With phi_k(t) the time eigenfunctions, the field's components are
        sum_k phi_k(t) c_kj, and the adjoint of a field y is the history
        v (x) y with v the functional's representer: the field's metric
        cancels, and the time metric is applied once. For a time space of
        order above one half the adjoint is bounded and continuous in t.

        Raises:
            ValueError: If the time lies outside the interval.
        """
        dirac = LinearFunctional.from_derivative_components(
            self.time_space, self.basis_matrix([time])[0]
        )
        return self.contraction(dirac)

    def at(self, history: np.ndarray, time: float, /) -> SHGrid:
        """The history's field at a time of the interval."""
        return self.evaluation_operator(time)(history)
