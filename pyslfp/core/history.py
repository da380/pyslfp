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
from typing import Any

import numpy as np
from pyshtools import SHGrid

from pygeoinf import CoordinateSpace, LinearFunctional, LinearOperator, TensorProduct

__all__ = [
    "FieldHistorySpace",
    "at_time",
    "time_average",
    "difference_quotient",
    "sample_at_times",
]


# ================================================================ #
#                        Sampling in time                          #
# ================================================================ #


def _time_factor(space: TensorProduct) -> CoordinateSpace:
    """The time factor of a product, its first, which must be an interval
    space with a basis at points."""
    if not isinstance(space, TensorProduct) or not hasattr(
        space.factors[0], "basis_matrix"
    ):
        raise ValueError("the space must be a tensor product with a time factor first")
    return space.factors[0]


def _window_components(time_space, lower: float, upper: float) -> np.ndarray:
    """The integrals of the time basis over a window, by Gauss quadrature on
    each piece between the mesh's nodes inside it, exact for the polynomials
    the eigenfunctions are on the elements."""
    lower, upper = float(lower), float(upper)
    if not upper > lower:
        raise ValueError("the window must have positive length")
    start, end = time_space.bounds
    if lower < start or upper > end:
        raise ValueError("the window must lie inside the time interval")
    nodes = np.asarray(time_space.nodes, dtype=float)
    inside = nodes[(nodes > lower) & (nodes < upper)]
    breaks = np.concatenate(([lower], np.unique(inside), [upper]))
    points, weights = np.polynomial.legendre.leggauss(4)
    out = np.zeros(time_space.dim)
    for a, b in zip(breaks[:-1], breaks[1:]):
        half = 0.5 * (b - a)
        positions = a + half * (points + 1.0)
        out += (half * weights) @ time_space.basis_matrix(positions)
    return out


def at_time(space: TensorProduct, time: float, /) -> LinearOperator[np.ndarray, Any]:
    """
    The value of a history at a time: the contraction of a product whose
    first factor is the time space with the Dirac functional there, onto the
    product of the remaining factors, bounded for a time order above one
    half.

    Raises:
        ValueError: If the time lies outside the interval.
    """
    time_space = _time_factor(space)
    return space.contraction(
        LinearFunctional.from_derivative_components(
            time_space, time_space.basis_matrix([float(time)])[0]
        )
    )


def time_average(
    space: TensorProduct, lower: float, upper: float, /
) -> LinearOperator[np.ndarray, Any]:
    """
    The mean of a history over a window of time, the contraction with the
    functional (1 / (upper - lower)) int_lower^upper dt, bounded on a time
    space of any order: what an observation averaged over an interval, an
    epoch of a satellite solution say, sees.

    Raises:
        ValueError: If the window is empty or lies outside the interval.
    """
    time_space = _time_factor(space)
    components = _window_components(time_space, lower, upper) / (
        float(upper) - float(lower)
    )
    return space.contraction(
        LinearFunctional.from_derivative_components(time_space, components)
    )


def difference_quotient(
    space: TensorProduct, lower: float, upper: float, /
) -> LinearOperator[np.ndarray, Any]:
    """
    The change of a history over a window per unit time, the contraction
    with (delta_upper - delta_lower) / (upper - lower): the rate an
    observation over a span measures, a velocity or a trend, which needs no
    more of the time space than evaluation does.

    Raises:
        ValueError: If the window is empty or an end lies outside the interval.
    """
    time_space = _time_factor(space)
    lower, upper = float(lower), float(upper)
    if not upper > lower:
        raise ValueError("the window must have positive length")
    rows = time_space.basis_matrix([lower, upper])
    return space.contraction(
        LinearFunctional.from_derivative_components(
            time_space, (rows[1] - rows[0]) / (upper - lower)
        )
    )


def sample_at_times(
    space: TensorProduct, times: Sequence[float], /
) -> LinearOperator[np.ndarray, tuple]:
    """
    A history's values at several times, one `at_time` per time stacked
    into a `DirectSum` of the remaining factors' product, in order: a series
    of epochs.
    """
    from pygeoinf import ColumnLinearOperator

    return ColumnLinearOperator([at_time(space, float(t)) for t in times])


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

    @classmethod
    def from_product(cls, space: TensorProduct, /) -> FieldHistorySpace:
        """The history space equal to a tensor product of a time space and a
        field space, as a lift or a contraction on a larger product leaves
        it; a `FieldHistorySpace` is returned as it is.

        Raises:
            ValueError: If the product has other than two factors.
        """
        if isinstance(space, FieldHistorySpace):
            return space
        if not isinstance(space, TensorProduct) or len(space.factors) != 2:
            raise ValueError("a history space is a tensor product of two factors")
        return cls(*space.factors)

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
        return at_time(self, time)

    def at(self, history: np.ndarray, time: float, /) -> SHGrid:
        """The history's field at a time of the interval."""
        return self.evaluation_operator(time)(history)
