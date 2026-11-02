"""
The response of the sea level equation to a load, held spectrally, and its
Hilbert spaces.

A `GRDResponse` is the gravitational, rotational and deformational response to
a surface load: the sea level change xi, the surface displacement, the
gravitational potential change phi and, with rotational feedback, the angular
velocity change omega. A `SurfaceDisplacement` is one field::

    u = U r_hat + grad_1 V,

spheroidal, since a load drives no toroidal motion; there is no rigid
rotation, degree 1 is in the centre-of-mass frame, and V, the potential of the
horizontal part, has no degree 0. grad_1 is the gradient on the unit sphere,
so V and U are both lengths.

Every member is band-limited: U, V and phi are Love-number multiples of the
load's coefficients, and xi is a synthesis of coefficients up to the
truncation degree plus a uniform shift, the ocean function entering only the
load. So the members are held as coefficients, in pygeoinf's component
convention for its sphere spaces: the radius times the orthonormal
coefficients, in the order of the sphere space's `degrees` and `orders`. A
scalar member's space is then `ComponentView` of a pygeoinf sphere space, and
`field_space.from_components` carries it into that space as a field. Grids
are made on demand, and the displacement is evaluated at points, horizontal
part included, by ducc0's spin-weighted synthesis, with its exact adjoint.

The spaces are diagonal-metric coordinate spaces (`DiagonalMetricSpace`), so
pygeoinf's measures, solvers and operator checks apply as they stand. The
displacement space carries the vector-field norm::

    sum_l lambda_l (U_lm^2 + l(l + 1) V_lm^2),

with lambda_l the field space's metric, so U and V are weighted as parts of
one field: in L2 it is the surface integral of u . u. The response space is
built from its members' spaces, with projections and inclusions by name.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Optional

import ducc0
import numpy as np
from pyshtools import SHCoeffs, SHGrid

from pygeoinf import (
    DiagonalMetricSpace,
    EuclideanSpace,
    HilbertSpace,
    LinearOperator,
)
from pygeoinf.algebra.spaces import ComponentView

__all__ = [
    "SurfaceDisplacement",
    "SurfaceDisplacementSpace",
    "GRDResponse",
    "GRDSpace",
    "SEA_LEVEL_CHANGE",
    "DISPLACEMENT",
    "POTENTIAL_CHANGE",
    "ANGULAR_VELOCITY_CHANGE",
]

SEA_LEVEL_CHANGE = "sea_level_change"
DISPLACEMENT = "displacement"
POTENTIAL_CHANGE = "potential_change"
ANGULAR_VELOCITY_CHANGE = "angular_velocity_change"

# The components of a displacement at a point, as `at` and the point
# evaluation operator name them.
_DIRECTIONS = ("east", "north", "up")

_EPSILON = 1e-13


# ================================================================ #
#                          The displacement                        #
# ================================================================ #


class SurfaceDisplacement:
    """
    A surface displacement u = U r_hat + grad_1 V, a vector of a
    `SurfaceDisplacementSpace`.

    `vertical` holds the components of U and `horizontal` those of V over the
    degrees from 1, or None in a space without the horizontal part.
    """

    __slots__ = ("_space", "vertical", "horizontal")

    def __init__(
        self,
        space: SurfaceDisplacementSpace,
        vertical: np.ndarray,
        horizontal: Optional[np.ndarray] = None,
        /,
    ) -> None:
        self._space = space
        self.vertical = np.asarray(vertical, dtype=float)
        self.horizontal = None if horizontal is None else np.asarray(horizontal, float)

    @property
    def space(self) -> SurfaceDisplacementSpace:
        """The space the displacement belongs to."""
        return self._space

    def vertical_field(self) -> SHGrid:
        """U on the field space's grid."""
        return self._space.field_space.from_components(self.vertical)

    def horizontal_fields(self) -> tuple[SHGrid, SHGrid]:
        """The east and north components of grad_1 V on the field space's grid."""
        return self._space.horizontal_fields(self)

    def at(self, points: Sequence[tuple[float, float]], /) -> np.ndarray:
        """The east, north and up components at points (latitude, longitude
        in degrees), one row per point; east and north are zero without the
        horizontal part."""
        operator = self._space.point_evaluation_operator(points)
        return operator(self).reshape(len(points), 3)

    def __repr__(self) -> str:
        part = "" if self.horizontal is None else ", with its horizontal part"
        return f"SurfaceDisplacement(lmax={self._space.field_space.lmax}{part})"


class SurfaceDisplacementSpace(DiagonalMetricSpace):
    """
    The Hilbert space of surface displacements over a pygeoinf sphere space.

    The components are those of U followed, with the horizontal part, by those
    of V over the degrees from 1; the metric is the field space's lambda_l for
    U and lambda_l l(l + 1) for V, the vector-field norm.
    """

    def __init__(self, field_space, /, *, horizontal: bool = False) -> None:
        """
        Args:
            field_space: A pygeoinf sphere space (Lebesgue or Sobolev), whose
                grid, radius, components and metric the displacement takes.
            horizontal: Whether the horizontal part is held.
        """
        self._field_space = field_space
        self._horizontal = bool(horizontal)
        degrees = field_space.degrees
        self._v_mask = degrees >= 1
        metric = field_space.metric_values
        if self._horizontal:
            ell = degrees[self._v_mask]
            metric = np.concatenate([metric, metric[self._v_mask] * ell * (ell + 1.0)])
        super().__init__(metric)
        self._n_u = field_space.dim

    # -- structure ---------------------------------------------------------

    @property
    def field_space(self):
        """The sphere space of U, whose grid and radius the space takes."""
        return self._field_space

    @property
    def has_horizontal(self) -> bool:
        """Whether the horizontal part is held."""
        return self._horizontal

    @property
    def horizontal_degrees(self) -> np.ndarray:
        """The degree of each component of V."""
        return self._field_space.degrees[self._v_mask]

    @property
    def horizontal_orders(self) -> np.ndarray:
        """The signed order of each component of V, negative for sine."""
        return self._field_space.orders[self._v_mask]

    def _key(self) -> Hashable:
        return ("SurfaceDisplacementSpace", self._field_space, self._horizontal)

    # -- vectors -----------------------------------------------------------

    def to_components(self, u: SurfaceDisplacement) -> np.ndarray:
        if u.horizontal is None or not self._horizontal:
            return np.array(u.vertical)
        return np.concatenate([u.vertical, u.horizontal])

    def from_components(self, c: np.ndarray) -> SurfaceDisplacement:
        c = np.asarray(c, dtype=float)
        if c.shape != (self.dim,):
            raise ValueError(f"Expected {self.dim} components, got {c.shape}.")
        if self._horizontal:
            return SurfaceDisplacement(
                self, c[: self._n_u].copy(), c[self._n_u :].copy()
            )
        return SurfaceDisplacement(self, c.copy())

    def copy(self, u: SurfaceDisplacement) -> SurfaceDisplacement:
        return self.from_components(self.to_components(u))

    def axpy(self, a: float, x: SurfaceDisplacement, y: SurfaceDisplacement):
        y.vertical += a * x.vertical
        if self._horizontal:
            y.horizontal += a * x.horizontal
        return y

    def scale_inplace(self, a: float, x: SurfaceDisplacement) -> SurfaceDisplacement:
        x.vertical *= a
        if self._horizontal:
            x.horizontal *= a
        return x

    def from_coefficients(
        self, vertical: np.ndarray, horizontal: Optional[np.ndarray] = None, /
    ) -> SurfaceDisplacement:
        """The displacement with the given coefficient arrays of U and V,
        shaped (2, lmax + 1, lmax + 1) in the orthonormal convention without
        the Condon-Shortley phase."""
        u = _components(vertical, self._field_space)
        if not self._horizontal:
            return SurfaceDisplacement(self, u)
        if horizontal is None:
            raise ValueError("this space holds the horizontal part; give V too")
        v = _components(horizontal, self._field_space)[self._v_mask]
        return SurfaceDisplacement(self, u, v)

    # -- operators ---------------------------------------------------------

    def vertical_projection(self) -> LinearOperator:
        """u -> U, into `ComponentView(field_space)`."""
        target = ComponentView(self._field_space)

        def adjoint(c: np.ndarray) -> SurfaceDisplacement:
            out = self.zero()
            out.vertical = np.array(c, dtype=float)
            return out

        return LinearOperator.from_callables(
            self, target, lambda u: np.array(u.vertical), adjoint=adjoint
        )

    def horizontal_fields(self, u: SurfaceDisplacement, /) -> tuple[SHGrid, SHGrid]:
        """The east and north components of grad_1 V on the field space's
        grid, zero without the horizontal part."""
        space = self._field_space
        grid = "DH2" if (space.grid == "DH" and space.sampling == 2) else space.grid
        if not self._horizontal:
            zero = space.from_components(np.zeros(space.dim))
            return zero, zero.copy()
        full = np.zeros(space.dim)
        full[self._v_mask] = u.horizontal
        coefficients = SHCoeffs.from_array(
            _coefficients(full, space), normalization="ortho", csphase=1
        )
        gradient = coefficients.gradient(grid=grid, extend=space.extend, radius=1.0)
        return gradient.phi, gradient.theta * -1.0

    def point_evaluation_operator(
        self,
        points: Sequence[tuple[float, float]],
        /,
        *,
        directions: Sequence[str] = _DIRECTIONS,
    ) -> LinearOperator:
        """
        The displacement at points, into a Euclidean space: for each point in
        turn, the components in `directions`, from "east", "north" and "up".

        East and north are the components of grad_1 V, by ducc0's spin-1
        synthesis, and zero in a space without the horizontal part; they are
        undefined at the poles, which are refused. The adjoint is exact, by
        ducc0's adjoint synthesis.

        Args:
            points: (latitude, longitude) pairs in degrees.
            directions: The components wanted at each point, in order.
        """
        directions = tuple(directions)
        unknown = set(directions) - set(_DIRECTIONS)
        if unknown or not directions:
            raise ValueError(
                f"directions are drawn from {_DIRECTIONS}, got {directions}"
            )
        loc = _locations(points)
        horizontal = any(d in ("east", "north") for d in directions)
        if horizontal and np.any(np.isclose(np.abs(loc[:, 0] - np.pi / 2), np.pi / 2)):
            raise ValueError("east and north are undefined at the poles")
        n_points = loc.shape[0]
        codomain = EuclideanSpace(n_points * len(directions))
        space = self._field_space
        lmax = space.lmax
        u_alm = _AlmMap(space.degrees, space.orders, lmax, space.radius)
        v_alm = _AlmMap(
            self.horizontal_degrees, self.horizontal_orders, lmax, space.radius
        )
        v_factor = np.sqrt(v_alm.degrees * (v_alm.degrees + 1.0))
        index = {d: i for i, d in enumerate(directions)}

        def mapping(u: SurfaceDisplacement) -> np.ndarray:
            out = np.zeros((n_points, len(directions)))
            if "up" in index:
                alm = u_alm.forward(u.vertical)
                values = ducc0.sht.synthesis_general(
                    alm=alm[None, :], spin=0, lmax=lmax, loc=loc, epsilon=_EPSILON
                )[0]
                out[:, index["up"]] = values
            if horizontal and self._horizontal:
                e = v_alm.forward(v_factor * u.horizontal)
                theta, phi = ducc0.sht.synthesis_general(
                    alm=np.stack([e, np.zeros_like(e)]),
                    spin=1,
                    lmax=lmax,
                    loc=loc,
                    epsilon=_EPSILON,
                )
                if "east" in index:
                    out[:, index["east"]] = phi
                if "north" in index:
                    out[:, index["north"]] = -theta
            return out.ravel()

        def transpose(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float).reshape(n_points, len(directions))
            vertical = np.zeros(self._n_u)
            if "up" in index:
                back = ducc0.sht.adjoint_synthesis_general(
                    map=np.ascontiguousarray(y[:, index["up"]])[None, :],
                    spin=0,
                    lmax=lmax,
                    loc=loc,
                    epsilon=_EPSILON,
                )[0]
                vertical = u_alm.transpose(back)
            if not self._horizontal:
                return vertical
            horizontal_c = np.zeros(self.dim - self._n_u)
            if horizontal:
                theta = (
                    -y[:, index["north"]] if "north" in index else np.zeros(n_points)
                )
                phi = y[:, index["east"]] if "east" in index else np.zeros(n_points)
                back = ducc0.sht.adjoint_synthesis_general(
                    map=np.stack([theta, phi]),
                    spin=1,
                    lmax=lmax,
                    loc=loc,
                    epsilon=_EPSILON,
                )
                horizontal_c = v_factor * v_alm.transpose(back[0])
            return np.concatenate([vertical, horizontal_c])

        def adjoint(y: np.ndarray) -> SurfaceDisplacement:
            return self.from_components(self.solve_gram(transpose(y)))

        return LinearOperator.from_callables(self, codomain, mapping, adjoint=adjoint)


# ================================================================ #
#                            The response                          #
# ================================================================ #


class GRDResponse:
    """
    The response to a load, a vector of a `GRDSpace`: the components of the
    sea level change and of the potential change, the `SurfaceDisplacement`,
    and the angular velocity change [omega_x, omega_y, omega_z], None without
    rotational feedback.
    """

    __slots__ = (
        "_space",
        "sea_level_change",
        "displacement",
        "potential_change",
        "angular_velocity_change",
    )

    def __init__(
        self,
        space: GRDSpace,
        sea_level_change: np.ndarray,
        displacement: SurfaceDisplacement,
        potential_change: np.ndarray,
        angular_velocity_change: Optional[np.ndarray] = None,
        /,
    ) -> None:
        self._space = space
        self.sea_level_change = np.asarray(sea_level_change, dtype=float)
        self.displacement = displacement
        self.potential_change = np.asarray(potential_change, dtype=float)
        self.angular_velocity_change = (
            None
            if angular_velocity_change is None
            else np.asarray(angular_velocity_change, dtype=float)
        )

    @property
    def space(self) -> GRDSpace:
        """The space the response belongs to."""
        return self._space

    def field(self, name: str, /) -> SHGrid:
        """A scalar member, "sea_level_change" or "potential_change", or the
        vertical displacement, "displacement", on the field space's grid."""
        space = self._space.field_space
        if name == SEA_LEVEL_CHANGE:
            return space.from_components(self.sea_level_change)
        if name == POTENTIAL_CHANGE:
            return space.from_components(self.potential_change)
        if name == DISPLACEMENT:
            return self.displacement.vertical_field()
        raise ValueError(f"'{name}' is not a field of the response")

    def __repr__(self) -> str:
        return f"GRDResponse({', '.join(self._space.names)})"


class GRDSpace(DiagonalMetricSpace):
    """
    The Hilbert space of responses to a load, over a pygeoinf sphere space.

    Its members, in order, are the sea level change and the displacement, the
    potential change, and the angular velocity change with rotational
    feedback; the components are the members' in that order, and the metric is
    theirs, so a member's projection and inclusion are each other's adjoints.
    """

    def __init__(
        self, field_space, /, *, horizontal: bool = False, rotation: bool = True
    ) -> None:
        """
        Args:
            field_space: A pygeoinf sphere space (Lebesgue or Sobolev) for the
                scalar fields; its grid, radius, components and metric are the
                response's.
            horizontal: Whether the displacement holds its horizontal part.
            rotation: Whether the angular velocity change is held.
        """
        self._field_space = field_space
        self._displacement_space = SurfaceDisplacementSpace(
            field_space, horizontal=horizontal
        )
        self._rotation = bool(rotation)
        self._scalar_space = ComponentView(field_space)
        n = field_space.dim
        self._sizes = {
            SEA_LEVEL_CHANGE: n,
            DISPLACEMENT: self._displacement_space.dim,
            POTENTIAL_CHANGE: n,
        }
        metrics = [
            field_space.metric_values,
            self._displacement_space.metric_values,
            field_space.metric_values,
        ]
        if self._rotation:
            self._sizes[ANGULAR_VELOCITY_CHANGE] = 3
            metrics.append(np.ones(3))
        offsets = np.cumsum([0] + list(self._sizes.values()))
        self._slices = {
            name: slice(int(offsets[i]), int(offsets[i + 1]))
            for i, name in enumerate(self._sizes)
        }
        super().__init__(np.concatenate(metrics))

    # -- structure ---------------------------------------------------------

    @property
    def field_space(self):
        """The sphere space of the scalar fields."""
        return self._field_space

    @property
    def names(self) -> tuple[str, ...]:
        """The members, in order."""
        return tuple(self._sizes)

    @property
    def has_horizontal(self) -> bool:
        """Whether the displacement holds its horizontal part."""
        return self._displacement_space.has_horizontal

    @property
    def has_rotation(self) -> bool:
        """Whether the angular velocity change is held."""
        return self._rotation

    def member_space(self, name: str, /) -> HilbertSpace:
        """The space of a member: `ComponentView(field_space)` for the scalar
        fields, the `SurfaceDisplacementSpace`, and EuclideanSpace(3)."""
        if name in (SEA_LEVEL_CHANGE, POTENTIAL_CHANGE):
            return self._scalar_space
        if name == DISPLACEMENT:
            return self._displacement_space
        if name == ANGULAR_VELOCITY_CHANGE and self._rotation:
            return EuclideanSpace(3)
        raise ValueError(f"'{name}' is not a member of this space; it has {self.names}")

    def _key(self) -> Hashable:
        return (
            "GRDSpace",
            self._field_space,
            self.has_horizontal,
            self._rotation,
        )

    # -- vectors -----------------------------------------------------------

    def to_components(self, x: GRDResponse) -> np.ndarray:
        parts = [
            x.sea_level_change,
            self._displacement_space.to_components(x.displacement),
            x.potential_change,
        ]
        if self._rotation:
            parts.append(x.angular_velocity_change)
        return np.concatenate(parts)

    def from_components(self, c: np.ndarray) -> GRDResponse:
        c = np.asarray(c, dtype=float)
        if c.shape != (self.dim,):
            raise ValueError(f"Expected {self.dim} components, got {c.shape}.")
        part = {name: c[s].copy() for name, s in self._slices.items()}
        return GRDResponse(
            self,
            part[SEA_LEVEL_CHANGE],
            self._displacement_space.from_components(part[DISPLACEMENT]),
            part[POTENTIAL_CHANGE],
            part.get(ANGULAR_VELOCITY_CHANGE),
        )

    def copy(self, x: GRDResponse) -> GRDResponse:
        return self.from_components(self.to_components(x))

    def axpy(self, a: float, x: GRDResponse, y: GRDResponse) -> GRDResponse:
        y.sea_level_change += a * x.sea_level_change
        self._displacement_space.axpy(a, x.displacement, y.displacement)
        y.potential_change += a * x.potential_change
        if self._rotation:
            y.angular_velocity_change += a * x.angular_velocity_change
        return y

    def scale_inplace(self, a: float, x: GRDResponse) -> GRDResponse:
        x.sea_level_change *= a
        self._displacement_space.scale_inplace(a, x.displacement)
        x.potential_change *= a
        if self._rotation:
            x.angular_velocity_change *= a
        return x

    def from_coefficients(
        self,
        sea_level_change: np.ndarray,
        displacement: np.ndarray,
        potential_change: np.ndarray,
        /,
        *,
        horizontal: Optional[np.ndarray] = None,
        angular_velocity_change: Optional[np.ndarray] = None,
    ) -> GRDResponse:
        """The response with the given coefficient arrays, shaped
        (2, lmax + 1, lmax + 1) in the orthonormal convention without the
        Condon-Shortley phase, as the sea level solver holds them."""
        if self._rotation and angular_velocity_change is None:
            raise ValueError("this space holds the angular velocity change; give it")
        return GRDResponse(
            self,
            _components(sea_level_change, self._field_space),
            self._displacement_space.from_coefficients(displacement, horizontal),
            _components(potential_change, self._field_space),
            angular_velocity_change if self._rotation else None,
        )

    # -- operators ---------------------------------------------------------

    def projection(self, name: str, /) -> LinearOperator:
        """The map from a response to one member, into `member_space(name)`."""
        target = self.member_space(name)
        s = self._slices[name]

        def mapping(x: GRDResponse):
            return target.from_components(self.to_components(x)[s])

        def adjoint(y):
            c = np.zeros(self.dim)
            c[s] = target.to_components(y)
            return self.from_components(c)

        return LinearOperator.from_callables(self, target, mapping, adjoint=adjoint)

    def inclusion(self, name: str, /) -> LinearOperator:
        """The map from one member to the response holding it alone."""
        return self.projection(name).adjoint


# ================================================================ #
#                         Private helpers                          #
# ================================================================ #


def _components(coefficients: np.ndarray, space) -> np.ndarray:
    """A coefficient array (2, lmax + 1, lmax + 1) as a sphere space's
    components: the radius times the coefficients, in its order."""
    parts = (space.orders < 0).astype(int)
    return space.radius * coefficients[parts, space.degrees, np.abs(space.orders)]


def _coefficients(components: np.ndarray, space) -> np.ndarray:
    """The coefficient array of a sphere space's components."""
    out = np.zeros((2, space.lmax + 1, space.lmax + 1))
    parts = (space.orders < 0).astype(int)
    out[parts, space.degrees, np.abs(space.orders)] = components / space.radius
    return out


def _locations(points: Sequence[tuple[float, float]]) -> np.ndarray:
    """(colatitude, longitude) in radians, as ducc0 takes them."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.ascontiguousarray(
        np.stack([np.radians(90.0 - p[:, 0]), np.radians(np.mod(p[:, 1], 360.0))], -1)
    )


class _AlmMap:
    """Real components in pygeoinf's convention to ducc0's complex a_lm
    (healpy order, orthonormal with the Condon-Shortley phase), and the
    transpose of that map for the real pairing that ducc0's adjoint
    synthesis gives, in which an order m > 0 counts twice."""

    def __init__(self, degrees, orders, lmax: int, radius: float) -> None:
        self.degrees = np.asarray(degrees)
        m = np.abs(np.asarray(orders))
        self._index = m * (2 * lmax + 3 - m) // 2 + self.degrees - m
        sign = np.where(m % 2 == 0, 1.0, -1.0)
        root_half = np.where(m > 0, 1.0 / np.sqrt(2.0), 1.0)
        # a_lm = (-1)^m (C - i S) / sqrt(2) for m > 0, a_l0 = C
        self._weight = (
            np.where(np.asarray(orders) < 0, -1j, 1.0) * sign * root_half / radius
        )
        # d/dC of sum_m w_m Re(conj(a) b), w = 2 for m > 0
        self._transpose = np.where(m > 0, 2.0, 1.0) * np.conj(self._weight)
        self._size = (lmax + 1) * (lmax + 2) // 2

    def forward(self, components: np.ndarray) -> np.ndarray:
        alm = np.zeros(self._size, dtype=np.complex128)
        np.add.at(alm, self._index, self._weight * components)
        return alm

    def transpose(self, b: np.ndarray) -> np.ndarray:
        return np.real(self._transpose * b[self._index])
