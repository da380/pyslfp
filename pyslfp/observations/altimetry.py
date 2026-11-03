"""
Satellite altimetry: the geocentric sea level change at points.

Altimetry observes the geocentric sea level change
(`pyslfp.core.geocentric_sea_level_operator`). This module chooses
independent observation points over the ocean or the ice sheets, samples the
geocentric sea level at them, and averages the samples to a global mean;
`AltimetryObservationModel` samples an observable, the geocentric sea level
change from a model space with any direct terms included.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyshtools import SHGrid


from pygeoinf import (
    LinearOperator,
    DirectSum,
    EuclideanSpace,
)

from pyslfp.core import EarthState
from pyslfp.core.observables import geocentric_sea_level_operator
from pyslfp.elastic.operator import FingerPrintOperator
from pyslfp.observations.base import Noise, ObservationModel


# ================================================================ #
#                 Point Generation Utilities                       #
# ================================================================ #


def _filter_independent_points(
    points: np.ndarray, mask: SHGrid
) -> List[Tuple[float, float]]:
    """Interpolates an SHGrid mask to filter a set of coordinates."""
    grid_lats = mask.lats()
    grid_lons = mask.lons()
    lats_asc = grid_lats[::-1]
    data_asc = mask.data[::-1, :]

    interpolator = RegularGridInterpolator(
        (lats_asc, grid_lons), data_asc, bounds_error=False, fill_value=0.0
    )
    mask_values = interpolator(points)
    valid_points = points[mask_values > 0.5]
    return [(float(lat), float(lon)) for lat, lon in valid_points]


def ocean_altimetry_points(
    state: EarthState,
    /,
    *,
    spacing: float = 2.0,
    latitude_min: float = -66.0,
    latitude_max: float = 66.0,
) -> List[Tuple[float, float]]:
    """A regular grid of observation points over the open ocean.

    Args:
        state: The Earth state whose ocean function and ice mask filter
            the points.
        spacing: The grid spacing in degrees.
        latitude_min: The southern limit of the altimetry coverage.
        latitude_max: The northern limit.
    """
    lats = np.arange(latitude_min, latitude_max + 1e-9, spacing)
    lons = np.arange(0.0, 360.0, spacing)
    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
    candidate_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))

    mask = state.altimetry_projection(
        latitude_min=latitude_min, latitude_max=latitude_max, value=0
    )
    return _filter_independent_points(candidate_points, mask)


def ice_altimetry_points(
    state: EarthState,
    /,
    *,
    spacing: float = 2.0,
    sheets: Optional[Union[str, Sequence[str]]] = None,
    exclude_ice_shelves: bool = False,
) -> List[Tuple[float, float]]:
    """A regular grid of observation points over the ice sheets.

    Args:
        state: The Earth state whose ice sheets filter the points
            (`EarthState.ice_sheet_projection`).
        spacing: The grid spacing in degrees.
        sheets: "antarctic", "greenland" or a sequence of them; both if None.
        exclude_ice_shelves: Whether floating shelves are excluded.
    """
    lats = np.arange(-90.0, 90.0 + 1e-9, spacing)
    lons = np.arange(0.0, 360.0, spacing)
    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
    candidate_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))

    mask = state.ice_sheet_projection(
        value=0, sheets=sheets, exclude_ice_shelves=exclude_ice_shelves
    )
    return _filter_independent_points(candidate_points, mask)


# ================================================================ #
#                       Raw Math Operators                         #
# ================================================================ #


def altimetry_point_operator(
    state: EarthState,
    response_space: DirectSum,
    points: List[Tuple[float, float]],
    /,
    *,
    remove_rotational_contribution: bool = True,
    matrix_free: bool = False,
) -> LinearOperator:
    """
    Returns an operator mapping the SLE response space directly to geocentric sea level values
    at discrete altimetry observation points.
    """
    ssh_op = geocentric_sea_level_operator(
        state,
        response_space,
        remove_rotational_contribution=remove_rotational_contribution,
    )
    point_eval_op = ssh_op.codomain.point_evaluation_operator(
        points,
        route="local" if matrix_free else "dense",
    )

    return point_eval_op @ ssh_op


def altimetry_averaging_operator(points: List[Tuple[float, float]]) -> LinearOperator:
    """
    Creates a LinearOperator that maps a vector of altimetry observations
    to a Global Mean Sea Level (GMSL) estimate using a latitude-weighted average.

    Args:
        points: A list of (latitude, longitude) tuples representing the
                observation locations in degrees.

    Returns:
        A LinearOperator mapping from EuclideanSpace(N) to EuclideanSpace(1).
    """
    n_points = len(points)

    if n_points == 0:
        raise ValueError("The list of altimetry points cannot be empty.")

    domain = EuclideanSpace(n_points)
    codomain = EuclideanSpace(1)
    lats = np.array([p[0] for p in points])
    lats_rad = np.radians(lats)
    weights = np.cos(lats_rad)
    weights /= np.sum(weights)
    matrix = weights.reshape(1, n_points)
    return LinearOperator.from_matrix(domain, codomain, matrix, form="components")


class AltimetryObservationModel(ObservationModel):
    """
    Satellite altimetry: the geocentric sea level change, a field, sampled at
    the observation points.
    """

    def __init__(
        self,
        observable: LinearOperator,
        points: List[Tuple[float, float]],
        /,
        *,
        noise: Noise = None,
        matrix_free: bool = False,
    ):
        """
        Args:
            observable: The geocentric sea level change, from the model space
                into a field space, such as a set-up's
                `geocentric_sea_level(fingerprint)`.
            points: The observation points, (latitude, longitude) in degrees.
            noise: As for `ObservationModel`.
            matrix_free: Whether the point evaluation is matrix-free, slower
                but without the dense matrix of the points' harmonics.
        """
        self._points = list(points)
        sampling = observable.codomain.point_evaluation_operator(
            self._points, route="local" if matrix_free else "dense"
        )
        super().__init__(observable, sampling, noise=noise)

    @classmethod
    def from_fingerprint(
        cls,
        fingerprint: FingerPrintOperator,
        points: List[Tuple[float, float]],
        /,
        *,
        remove_rotational_contribution: bool = True,
        noise: Noise = None,
        matrix_free: bool = False,
    ) -> AltimetryObservationModel:
        """Altimetry observing the response of a fingerprint operator to its
        load, with no direct terms; `remove_rotational_contribution` as for
        `geocentric_sea_level_operator`."""
        observable = (
            geocentric_sea_level_operator(
                fingerprint.state,
                fingerprint.codomain,
                remove_rotational_contribution=remove_rotational_contribution,
            )
            @ fingerprint
        )
        return cls(observable, points, noise=noise, matrix_free=matrix_free)

    @property
    def points(self) -> List[Tuple[float, float]]:
        """The observation points, (latitude, longitude) in degrees."""
        return list(self._points)

    def averaging_operator(self) -> LinearOperator:
        """The latitude-weighted mean of the data, an estimate of the global
        mean sea level change, into EuclideanSpace(1)."""
        return altimetry_averaging_operator(self._points)
