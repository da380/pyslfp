"""
GNSS: the surface displacement at stations, in the frame of the data.

A response's displacement is in the centre-of-mass frame. A GNSS solution's
frame differs from it by a Helmert transformation, a translation T, a small
rotation R and a change of scale s, or their rates for velocities; at a
station x = b r on the sphere of radius b the transformation adds::

    delta u = T + s x + R x x,

with R x x the cross product. In east, north and up this is::

    east:   T . e  + b R . n
    north:  T . n  - b R . e
    up:     T . r  + b s

so the translation is a degree-1 field whose horizontal potential equals its
vertical part, the scale a degree-0 vertical field, and the rotation a rigid
toroidal motion, which no response holds. `HelmertTransformation` is this
design, from the parameters to the station data.

`GnssObservationModel` samples a displacement observable at the stations and
takes the frame in one of three ways: as nuisance parameters joined to the
model space, by reducing the data to the combinations the frame cannot change,
or not at all, the data then being in the centre-of-mass frame.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg

from pygeoinf import (
    EuclideanSpace,
    GaussianMeasure,
    LinearOperator,
    MatrixLinearOperator,
    RowLinearOperator,
)

from pyslfp.core.response import DISPLACEMENT, SurfaceDisplacementSpace
from pyslfp.core.spaces import check_load_space
from pyslfp.elastic.operator import FingerPrintOperator
from pyslfp.observations.base import Noise, ObservationModel, noise_measure

__all__ = [
    "GNSS_DIRECTIONS",
    "HELMERT_PARAMETERS",
    "HelmertTransformation",
    "GnssObservationModel",
]

#: The components of the displacement at a station, in the order of the data.
GNSS_DIRECTIONS = ("east", "north", "up")

#: The groups of Helmert parameters, in their default order.
HELMERT_PARAMETERS = ("translation", "rotation", "scale")

_PARAMETER_NAMES = {
    "translation": ("translation_x", "translation_y", "translation_z"),
    "rotation": ("rotation_x", "rotation_y", "rotation_z"),
    "scale": ("scale",),
}

_FRAMES = (None, "nuisance", "invariant")


# ================================================================ #
#                     The Helmert transformation                   #
# ================================================================ #


class HelmertTransformation(MatrixLinearOperator):
    """
    The Helmert transformation at stations, from its parameters,
    EuclideanSpace(k), to the station data, in the order of the displacement's
    point evaluation: for each station in turn, the components in
    `directions`.

    The parameters are the translation (T_x, T_y, T_z), a length; the rotation
    (R_x, R_y, R_z), in radians; and the scale s, a ratio; in the Earth-fixed
    axes, x towards latitude and longitude zero and z towards the north pole.
    The groups appear in the order given. For velocities the parameters are
    the rates of these.
    """

    def __init__(
        self,
        points: Sequence[Tuple[float, float]],
        radius: float,
        /,
        *,
        parameters: Sequence[str] = HELMERT_PARAMETERS,
        directions: Sequence[str] = GNSS_DIRECTIONS,
    ) -> None:
        """
        Args:
            points: The stations, (latitude, longitude) in degrees.
            radius: The radius b of the sphere the stations lie on.
            parameters: The groups of parameters, from "translation",
                "rotation" and "scale", each at most once.
            directions: The components at each station, from "east", "north"
                and "up", in the order of the data.

        Raises:
            ValueError: If a group or direction is unknown or repeated, or
                east or north is asked for at a pole.
        """
        parameters = _distinct(parameters, HELMERT_PARAMETERS, "parameters")
        directions = _distinct(directions, GNSS_DIRECTIONS, "directions")
        self._points = [tuple(p) for p in points]
        self._radius = float(radius)
        self._parameters = parameters
        self._directions = directions
        matrix = _design(self._points, self._radius, parameters, directions)
        super().__init__(
            EuclideanSpace(matrix.shape[1]),
            EuclideanSpace(matrix.shape[0]),
            matrix,
            form="components",
        )

    @property
    def points(self) -> List[Tuple[float, float]]:
        """The stations, (latitude, longitude) in degrees."""
        return list(self._points)

    @property
    def radius(self) -> float:
        """The radius of the sphere the stations lie on."""
        return self._radius

    @property
    def parameters(self) -> Tuple[str, ...]:
        """The groups of parameters, in order."""
        return self._parameters

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """The name of each parameter, in the order of the components."""
        return tuple(name for p in self._parameters for name in _PARAMETER_NAMES[p])

    @property
    def directions(self) -> Tuple[str, ...]:
        """The components at each station, in order."""
        return self._directions

    def frame_reduction(self) -> LinearOperator:
        """
        Z^T, from the station data to the combinations the transformation
        cannot change: Z is an orthonormal basis of the complement of the
        design's range, so Z^T A = 0 and Z^T Z = I, and the reduced data have
        dimension the number of data less the number of parameters.

        Z is held as the Householder reflectors of the design's QR
        factorisation, each application costing one pass over the data per
        parameter.

        Raises:
            ValueError: If the station data cannot separate the parameters:
                there are no more data than parameters, or the design's
                columns are dependent, as the rotation's are with the vertical
                component alone.
        """
        design = self.stored_matrix
        m, k = design.shape
        if m <= k:
            raise ValueError(
                "the station data cannot separate the Helmert parameters: "
                f"{m} data for {k} parameters"
            )
        norms = np.linalg.norm(design, axis=0)
        if np.any(norms == 0.0):
            raise ValueError(
                "the station data cannot separate the Helmert parameters: "
                "a parameter changes none of them"
            )
        # The range does not depend on the columns' scales, which differ by
        # the radius between the translation and the rotation and scale.
        (h, tau), r = scipy.linalg.qr(design / norms, mode="raw")
        if np.min(np.abs(np.diag(r))) < 1e-8:
            raise ValueError(
                "the station data cannot separate the Helmert parameters: "
                "the design's columns are dependent"
            )
        # The reflectors H_j = I - tau_j v_j v_j^T, v_j zero above j and one
        # at j, with Q = H_1 ... H_k.
        vectors = np.tril(h, -1)[:, :k].T.copy()
        vectors[np.arange(k), np.arange(k)] = 1.0

        def mapping(d: np.ndarray) -> np.ndarray:
            c = np.array(d, dtype=float)
            for v, t in zip(vectors, tau):
                c -= t * (v @ c) * v
            return c[k:]

        def adjoint(y: np.ndarray) -> np.ndarray:
            c = np.concatenate([np.zeros(k), np.asarray(y, dtype=float)])
            for v, t in zip(vectors[::-1], tau[::-1]):
                c -= t * (v @ c) * v
            return c

        return LinearOperator.from_callables(
            EuclideanSpace(m), EuclideanSpace(m - k), mapping, adjoint=adjoint
        )

    def __repr__(self) -> str:
        return (
            f"HelmertTransformation({len(self._points)} stations, "
            f"parameters={self._parameters}, directions={self._directions})"
        )


def _distinct(given: Sequence[str], allowed: Tuple[str, ...], what: str, /):
    given = tuple(given)
    if not given or set(given) - set(allowed) or len(set(given)) != len(given):
        raise ValueError(f"{what} are distinct members of {allowed}, got {given}")
    return given


def _design(points, radius: float, parameters, directions) -> np.ndarray:
    """The design matrix, one row per datum and one column per parameter."""
    p = np.radians(np.asarray(points, dtype=float).reshape(-1, 2))
    lat, lon = p[:, 0], p[:, 1]
    if any(d in ("east", "north") for d in directions) and np.any(
        np.isclose(np.abs(lat), np.pi / 2)
    ):
        raise ValueError("east and north are undefined at the poles")
    n = lat.size
    zero = np.zeros(n)
    east = np.stack([-np.sin(lon), np.cos(lon), zero], -1)
    north = np.stack(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], -1
    )
    up = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1
    )
    none = np.zeros((n, 1))
    # (R x x) . e = b R . (r x e) = b R . n, and (R x x) . n = -b R . e.
    blocks = {
        "east": {"translation": east, "rotation": radius * north, "scale": none},
        "north": {"translation": north, "rotation": -radius * east, "scale": none},
        "up": {
            "translation": up,
            "rotation": np.zeros((n, 3)),
            "scale": radius * np.ones((n, 1)),
        },
    }
    rows = np.stack(
        [
            np.concatenate([blocks[d][q] for q in parameters], axis=1)
            for d in directions
        ],
        axis=1,
    )
    return rows.reshape(n * len(directions), -1)


# ================================================================ #
#                         The observation model                    #
# ================================================================ #


class GnssObservationModel(ObservationModel):
    """
    GNSS: a surface displacement, or its rate, sampled at the stations in the
    frame of the data.

    The displacement's field space must allow point evaluation, a Sobolev
    order above 1, and hold the horizontal part if east or north is observed.
    The frame is taken in one of three ways:

    - None: the data are in the centre-of-mass frame of the response.
    - "nuisance": the Helmert parameters h join the model space, the forward
      operator acting on DirectSum([model space, EuclideanSpace(k)]) as
      F u + A h; a prior on h, joined by `joint_prior`, states how well the
      frame is known.
    - "invariant": the data are reduced to Z^T d, the combinations the frame
      cannot change (`HelmertTransformation.frame_reduction`), so that the
      data space has dimension N d - k, with N stations and d directions,
      the forward operator is Z^T F and the noise Z^T C Z. This is the nuisance form with no prior
      information on the frame, without a singular covariance. Station data
      enter as `frame_reduction(d)`.

    The noise is given for the station data in every frame.
    """

    def __init__(
        self,
        observable: LinearOperator,
        points: Sequence[Tuple[float, float]],
        /,
        *,
        names: Optional[Sequence[str]] = None,
        directions: Sequence[str] = GNSS_DIRECTIONS,
        frame: Optional[str] = None,
        parameters: Sequence[str] = HELMERT_PARAMETERS,
        noise: Noise = None,
    ) -> None:
        """
        Args:
            observable: The displacement, or its rate, from the model space
                into a `SurfaceDisplacementSpace`, such as a set-up's
                `response(fingerprint)` followed by the response space's
                projection onto the displacement.
            points: The stations, (latitude, longitude) in degrees.
            names: The stations' names; numbered by default.
            directions: The components at each station, from "east", "north"
                and "up", in the order of the data.
            frame: None, "nuisance" or "invariant", as above.
            parameters: The groups of Helmert parameters in a frame, from
                "translation", "rotation" and "scale".
            noise: For the station data: a Gaussian measure on their space,
                or standard deviations, one for every datum or one per datum.

        Raises:
            ValueError: If the observable is not a displacement that the
                stations can sample, the frame is unknown, or, in the
                invariant frame, the data cannot separate the parameters.
        """
        space = observable.codomain
        if not isinstance(space, SurfaceDisplacementSpace):
            raise ValueError("the observable must be a surface displacement")
        check_load_space(space.field_space, point_values=True)
        directions = _distinct(directions, GNSS_DIRECTIONS, "directions")
        if not space.has_horizontal and any(d in ("east", "north") for d in directions):
            raise ValueError("east and north need the displacement's horizontal part")
        if frame not in _FRAMES:
            raise ValueError(f"the frame is one of {_FRAMES}, got {frame!r}")
        self._points = [tuple(p) for p in points]
        self._names = (
            list(names)
            if names is not None
            else [f"Station_{i}" for i in range(len(self._points))]
        )
        self._directions = directions
        self._frame = frame
        evaluation = space.point_evaluation_operator(
            self._points, directions=directions
        )
        self._station_space = evaluation.codomain
        self._helmert = (
            None
            if frame is None
            else HelmertTransformation(
                self._points,
                space.field_space.radius,
                parameters=parameters,
                directions=directions,
            )
        )
        self._reduction = (
            self._helmert.frame_reduction() if frame == "invariant" else None
        )
        sampling = (
            evaluation if self._reduction is None else self._reduction @ evaluation
        )
        super().__init__(observable, sampling, noise=noise)
        if frame == "nuisance":
            self._forward = RowLinearOperator([self._forward, self._helmert])

    def _as_noise(self, noise: Noise) -> Optional[GaussianMeasure]:
        if isinstance(noise, GaussianMeasure) and noise.domain != self._station_space:
            raise ValueError("the noise must be a measure on the station data")
        measure = noise_measure(self._station_space, noise)
        if measure is None or self._reduction is None:
            return measure
        return self._reduction @ measure

    @classmethod
    def from_fingerprint(
        cls,
        fingerprint: FingerPrintOperator,
        points: Sequence[Tuple[float, float]],
        /,
        **options,
    ) -> GnssObservationModel:
        """GNSS observing the displacement of a fingerprint operator's
        response to its load; the options are the constructor's. The operator
        must hold the horizontal part if east or north is observed."""
        observable = fingerprint.codomain.projection(DISPLACEMENT) @ fingerprint
        return cls(observable, points, **options)

    def joint_prior(self, prior: GaussianMeasure, frame: Noise, /) -> GaussianMeasure:
        """
        In the nuisance frame, the prior on the model space: `prior` on the
        observable's domain, independent of the prior on the Helmert
        parameters.

        Args:
            prior: A Gaussian measure on the observable's domain.
            frame: The prior on the Helmert parameters: a Gaussian measure on
                EuclideanSpace(k), or standard deviations, one for every
                parameter or one per parameter, in the order of
                `helmert_transformation.parameter_names`.

        Raises:
            ValueError: Outside the nuisance frame, or if either measure is on
                the wrong space.
        """
        if self._frame != "nuisance":
            raise ValueError("only the nuisance frame has frame parameters")
        if prior.domain != self._observable.domain:
            raise ValueError("the prior must be a measure on the observable's domain")
        if frame is None:
            raise ValueError("the prior on the frame parameters is needed")
        if isinstance(frame, GaussianMeasure) and frame.domain != self._helmert.domain:
            raise ValueError("the frame prior must be a measure on the parameters")
        frame_measure = noise_measure(self._helmert.domain, frame)
        return GaussianMeasure.from_product(
            [prior, frame_measure], domain=self.model_space
        )

    @property
    def points(self) -> List[Tuple[float, float]]:
        """The stations, (latitude, longitude) in degrees."""
        return list(self._points)

    @property
    def names(self) -> List[str]:
        """The stations' names."""
        return list(self._names)

    @property
    def directions(self) -> Tuple[str, ...]:
        """The components at each station, in the order of the data."""
        return self._directions

    @property
    def frame(self) -> Optional[str]:
        """None, "nuisance" or "invariant"."""
        return self._frame

    @property
    def helmert_transformation(self) -> Optional[HelmertTransformation]:
        """The Helmert design at the stations, or None without a frame."""
        return self._helmert

    @property
    def frame_reduction(self) -> Optional[LinearOperator]:
        """In the invariant frame, Z^T, from the station data to the data;
        otherwise None."""
        return self._reduction
