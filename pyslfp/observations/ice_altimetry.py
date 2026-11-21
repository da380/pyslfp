"""
Ice altimetry: the surface height change of the grounded ice at points.

A laser or radar altimeter over an ice sheet (ICESat-2, CryoSat-2) observes
the change in the height of the ice surface in the frame of its orbit. On
grounded ice the surface stands on the solid Earth, so the change is

    h + a + u,

with h the ice-equivalent thickness change, a the change in firn air content
and u the vertical displacement of the solid surface in the response to all
the loads. It is the counterpart over the ice of the geocentric sea level
change over the ocean: gravimetry sees the mass, which is in h alone, and
altimetry a height, to which a field that carries no mass also contributes,
the firn air content here and the ocean density there. A model set-up builds
the observable, `ModelSetup.ice_surface_height`, and
`IceAltimetryObservationModel` samples it.

The displacement is the response's own: in the centre-of-mass frame, which
is the frame of an orbit, with its degree one, and with the deformation that
the change in rotation drives. `pyslfp.core.geocentric_sea_level_operator`
takes the same displacement, so the two altimetries share a frame.

The observable holds on grounded ice alone. Floating ice is in hydrostatic
equilibrium with the ocean: its surface moves with the sea surface, and
with the part of a change in its thickness that stands above the water,
which is other physics. The observation points are therefore confined to
the grounded ice of the state's ice sheets, by the state's own ice and
ocean masks (`on_grounded_ice`), and a model refuses points elsewhere. Corrections applied to the data before they are
compared with the model, for glacial isostatic adjustment or for the
penetration of a radar into the snow, are not part of the operator; their
errors belong in the noise.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from pyshtools import SHGrid
from scipy.interpolate import RegularGridInterpolator

from pygeoinf import LinearOperator

from pyslfp.core import EarthState
from pyslfp.core.response import DISPLACEMENT, GRDResponse, GRDSpace
from pyslfp.core.spaces import check_response_space
from pyslfp.observations.base import Noise, ObservationModel

__all__ = [
    "on_grounded_ice",
    "grounded_ice_altimetry_points",
    "ice_altimetry_point_operator",
    "IceAltimetryObservationModel",
]

Sheets = Optional[Union[str, Sequence[str]]]


# ================================================================ #
#                       The grounded ice                           #
# ================================================================ #


def on_grounded_ice(
    state: EarthState,
    points: Sequence[Tuple[float, float]],
    /,
    *,
    sheets: Sheets = None,
) -> np.ndarray:
    """
    Whether each point lies on the grounded ice of the state's ice sheets.

    The grounded ice is `EarthState.ice_sheet_projection` without the
    floating ice: the nodes of the ice sheets where the ocean function
    vanishes. A point between nodes is on it when the indicator, interpolated
    bilinearly from the four nodes about the point, exceeds one half, the
    rule by which `ocean_altimetry_points` and `ice_altimetry_points` choose
    their points. Longitudes are taken modulo 360 degrees.

    Args:
        state: The Earth state whose ice sheets and ocean function decide.
        points: (latitude, longitude) pairs in degrees.
        sheets: "antarctic", "greenland" or a sequence of them; both if None.

    Returns:
        A boolean array, one entry per point.

    Raises:
        ValueError: If a latitude is outside [-90, 90].
    """
    coordinates = np.asarray(points, dtype=float).reshape(-1, 2)
    if np.any(np.abs(coordinates[:, 0]) > 90.0):
        raise ValueError("latitudes must lie between -90 and 90 degrees")
    mask = state.ice_sheet_projection(
        value=0.0, sheets=sheets, exclude_ice_shelves=True
    )
    latitudes = mask.lats()[::-1]
    longitudes = mask.lons()
    values = mask.data[::-1, :]
    if longitudes[-1] < 360.0:
        longitudes = np.append(longitudes, 360.0)
        values = np.column_stack([values, values[:, 0]])
    interpolator = RegularGridInterpolator((latitudes, longitudes), values)
    clipped = np.clip(coordinates[:, 0], latitudes[0], latitudes[-1])
    wrapped = np.mod(coordinates[:, 1], 360.0)
    return interpolator(np.column_stack([clipped, wrapped])) > 0.5


def grounded_ice_altimetry_points(
    state: EarthState,
    /,
    *,
    spacing: float = 2.0,
    sheets: Sheets = None,
    latitude_min: float = -88.0,
    latitude_max: float = 88.0,
) -> List[Tuple[float, float]]:
    """
    Evenly spaced observation points on the grounded ice of the ice sheets.

    The points lie on parallels `spacing` apart and, along each parallel,
    as near `spacing` apart in arc as a whole number of them allows, so that
    they cover the ice evenly up to the poles, where the meridians of a
    regular grid in longitude would crowd them. They are kept where
    `on_grounded_ice` holds.

    Args:
        state: The Earth state whose ice sheets and ocean function filter
            the points.
        spacing: The spacing in degrees of arc.
        sheets: "antarctic", "greenland" or a sequence of them; both if None.
        latitude_min: The southern limit of the altimetry coverage; the
            default is that of ICESat-2 and CryoSat-2.
        latitude_max: The northern limit.

    Raises:
        ValueError: If the spacing is not positive.
    """
    if not spacing > 0.0:
        raise ValueError("the spacing must be positive")
    candidates = []
    for latitude in np.arange(latitude_min, latitude_max + 1e-9, spacing):
        count = max(1, int(round(360.0 * np.cos(np.radians(latitude)) / spacing)))
        longitudes = 360.0 * np.arange(count) / count
        candidates.append(np.column_stack([np.full(count, latitude), longitudes]))
    candidates = np.concatenate(candidates)
    kept = candidates[on_grounded_ice(state, candidates, sheets=sheets)]
    return [(float(latitude), float(longitude)) for latitude, longitude in kept]


def _grounded_points(
    state: EarthState, points: Sequence[Tuple[float, float]], sheets: Sheets
) -> List[Tuple[float, float]]:
    """The points as a list, refused unless all lie on the grounded ice."""
    points = [(float(latitude), float(longitude)) for latitude, longitude in points]
    if not points:
        raise ValueError("ice altimetry needs at least one observation point")
    off = np.flatnonzero(~on_grounded_ice(state, points, sheets=sheets))
    if off.size:
        examples = ", ".join(str(points[i]) for i in off[:3])
        raise ValueError(
            f"{off.size} of the {len(points)} points are not on the grounded "
            f"ice of the ice sheets, the first being {examples}; ice "
            "altimetry is modelled on grounded ice alone "
            "(see `on_grounded_ice`)"
        )
    return points


# ================================================================ #
#                       Raw Math Operators                         #
# ================================================================ #


def ice_altimetry_point_operator(
    state: EarthState,
    response_space: GRDSpace,
    points: Sequence[Tuple[float, float]],
    /,
    *,
    sheets: Sheets = None,
    matrix_free: bool = False,
) -> LinearOperator[GRDResponse, np.ndarray]:
    """
    Returns an operator mapping the response to its part of the ice surface
    height change at points on the grounded ice: the vertical displacement
    of the solid surface there.

    This is the response's part alone, as `altimetry_point_operator` is over
    the ocean. The thickness changes of the ice and of its firn add to it
    directly; `ModelSetup.ice_surface_height` holds all three.

    Args:
        state: The Earth state whose masks the points are checked against.
        response_space: The response space, with fields of Sobolev order
            above 1.
        points: The observation points, (latitude, longitude) in degrees.
        sheets: The ice sheets the points may lie on; both if None.
        matrix_free: Whether the point evaluation is matrix-free.

    Raises:
        ValueError: If a point is not on the grounded ice.
    """
    check_response_space(response_space)
    points = _grounded_points(state, points, sheets)
    displacement = response_space.field_operator(DISPLACEMENT)
    sampling = displacement.codomain.point_evaluation_operator(
        points, route="local" if matrix_free else "dense"
    )
    return sampling @ displacement


# ================================================================ #
#                     The observation model                        #
# ================================================================ #


class IceAltimetryObservationModel(ObservationModel):
    """
    Ice altimetry: the ice surface height change, a field, sampled at
    observation points on the grounded ice.
    """

    def __init__(
        self,
        observable: LinearOperator[Any, SHGrid],
        state: EarthState,
        points: Sequence[Tuple[float, float]],
        /,
        *,
        sheets: Sheets = None,
        noise: Noise = None,
        matrix_free: bool = False,
    ):
        """
        Args:
            observable: The ice surface height change, from the model space
                into a field space, such as a set-up's
                `ice_surface_height(fingerprint)`.
            state: The Earth state the observable was built on, whose ice
                sheets and ocean function say where the ice is grounded.
            points: The observation points, (latitude, longitude) in degrees,
                all on the grounded ice (`on_grounded_ice`);
                `grounded_ice_altimetry_points` gives such points.
            sheets: The ice sheets the points may lie on; both if None.
            noise: As for `ObservationModel`.
            matrix_free: Whether the point evaluation is matrix-free, slower
                but without the dense matrix of the points' harmonics.

        Raises:
            ValueError: If there are no points, or a point is not on the
                grounded ice.
        """
        self._state = state
        self._points = _grounded_points(state, points, sheets)
        sampling = observable.codomain.point_evaluation_operator(
            self._points, route="local" if matrix_free else "dense"
        )
        super().__init__(observable, sampling, noise=noise)

    @property
    def state(self) -> EarthState:
        """The Earth state whose grounded ice the points lie on."""
        return self._state

    @property
    def points(self) -> List[Tuple[float, float]]:
        """The observation points, (latitude, longitude) in degrees."""
        return list(self._points)
