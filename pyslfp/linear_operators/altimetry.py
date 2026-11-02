"""
Observation models and operators for satellite altimetry networks.

This module provides the raw mathematical operators for mapping physical
response fields to Sea Surface Height (SSH) anomalies, utilities for generating
independent observation grids, and high-level observation models linking
physical parameter spaces to discrete SSH observations.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyshtools import SHGrid


from pygeoinf import (
    LinearOperator,
    DirectSum,
    RowLinearOperator,
    BlockDiagonalLinearOperator,
    EuclideanSpace,
    ColumnLinearOperator,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.core import EarthState
from pyslfp.core.response import (
    ANGULAR_VELOCITY_CHANGE,
    DISPLACEMENT,
    SEA_LEVEL_CHANGE,
    GRDSpace,
)
from pyslfp.linear_operators.utils import check_response_space
from pyslfp.linear_operators.physics import (
    FingerPrintOperator,
    centrifugal_potential_operator,
)
from pyslfp.linear_operators.loads import (
    ice_thickness_change_to_load_operator,
    joint_ice_ocean_to_load_operator,
)


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


def sea_surface_height_operator(
    state: EarthState,
    response_space: GRDSpace,
    /,
    *,
    remove_rotational_contribution: bool = True,
) -> LinearOperator:
    """
    Returns an operator mapping the response to the sea surface height change,
    a field of the response's field space.

    The sea surface height change is the sea level change plus the vertical
    displacement. With the rotational contribution removed, the centrifugal
    potential over g is added, which needs the response to hold the angular
    velocity change.

    Args:
        state: The EarthState providing the physical parameters.
        response_space: The response space.
        remove_rotational_contribution: If True, adds the centrifugal potential
            correction to the SSH calculation.
    """
    check_response_space(response_space)

    field_space = response_space.field_space
    ssh = response_space.field_operator(
        SEA_LEVEL_CHANGE
    ) + response_space.field_operator(DISPLACEMENT)

    if remove_rotational_contribution and response_space.has_rotation:
        # Match Sobolev parameters if the response space is regularized
        sobolev_params = (
            (field_space.order, field_space.length_scale)
            if field_space.order != 0.0
            else None
        )

        cpc_op = centrifugal_potential_operator(
            state.model, sobolev_parameters=sobolev_params
        )
        g = state.model.parameters.gravitational_acceleration

        # The centrifugal potential over g is its height contribution.
        ssh = ssh + (1.0 / g) * cpc_op @ response_space.projection(
            ANGULAR_VELOCITY_CHANGE
        )
    return ssh


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
    Returns an operator mapping the SLE response space directly to SSH values
    at discrete altimetry observation points.
    """
    ssh_op = sea_surface_height_operator(
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


# ================================================================ #
#                       Observation Models                         #
# ================================================================ #


class JointAltimetryObservationModel:
    """
    An observation model that links satellite altimetry measurements (SSH)
    to a joint parameter space of [Ice Thickness Change, Ocean Dynamic Topography].
    """

    def __init__(
        self,
        fingerprint_operator: FingerPrintOperator,
        points: List[Tuple[float, float]],
        /,
        *,
        ice_space: Optional[Union[Lebesgue, Sobolev]] = None,
        ocean_space: Optional[Union[Lebesgue, Sobolev]] = None,
        remove_rotational_contribution: bool = True,
        matrix_free: bool = False,
    ):
        """
        Args:
            fingerprint_operator: The physics operator that maps loads to Earth responses.
            points: A list of (latitude, longitude) observation coordinates in degrees.
            ice_space: The Hilbert space for the ice prior. Defaults to the SLE load space.
            ocean_space: The Hilbert space for the ocean prior. Defaults to the SLE load space.
            remove_rotational_contribution: If True, adds the centrifugal potential
                correction to the SSH calculation.
            matrix_free: If True, point evaluation is done using a matrix-free implementation.
                This is slower, but avoids building a potentially large matrix internally.
        """
        self._fingerprint_operator = fingerprint_operator
        self._points = points

        state = fingerprint_operator.state
        load_space = fingerprint_operator.domain

        self._ice_space = ice_space if ice_space is not None else load_space
        self._ocean_space = ocean_space if ocean_space is not None else load_space

        self._joint_to_load_operator = joint_ice_ocean_to_load_operator(
            state, self._ice_space, self._ocean_space, load_space
        )
        self._response_to_data_operator = altimetry_point_operator(
            state,
            fingerprint_operator.codomain,
            self._points,
            remove_rotational_contribution=remove_rotational_contribution,
            matrix_free=matrix_free,
        )

        # --- Construct the Corrected Full Composite Forward Operator ---

        # Map [Ice, Ocean] -> [Total Load, Ocean]
        joint_space = DirectSum([self._ice_space, self._ocean_space])
        op1 = ColumnLinearOperator(
            [self._joint_to_load_operator, joint_space.projection(1)]
        )

        # Map [Total Load, Ocean] -> [Static SSH, Ocean]
        static_ssh_op = sea_surface_height_operator(
            state,
            fingerprint_operator.codomain,
            remove_rotational_contribution=remove_rotational_contribution,
        )
        ssh_inclusion = static_ssh_op.codomain.order_inclusion_operator(load_space)

        op2 = BlockDiagonalLinearOperator(
            [
                ssh_inclusion @ static_ssh_op @ self._fingerprint_operator,
                LinearOperator.identity(self._ocean_space),
            ]
        )

        # Map [Static SSH, Ocean] -> Total SSH Field
        ocean_to_ssh = self._ocean_space.order_inclusion_operator(load_space)
        op3 = RowLinearOperator([LinearOperator.identity(load_space), ocean_to_ssh])

        # Evaluate at the discrete altimetry points
        point_eval = load_space.point_evaluation_operator(
            self._points, route="local" if matrix_free else "dense"
        )

        self._forward_operator = point_eval @ op3 @ op2 @ op1

    @property
    def fingerprint_operator(self) -> FingerPrintOperator:
        """The underlying fingerprint operator."""
        return self._fingerprint_operator

    @property
    def points(self) -> List[Tuple[float, float]]:
        """The observation points as (latitude, longitude) pairs."""
        return self._points

    @property
    def forward_operator(self) -> LinearOperator:
        """The composed operator from model space to the observations."""
        return self._forward_operator

    @property
    def joint_to_load_operator(self) -> LinearOperator:
        """Exposes the operator mapping [Ice, Ocean] -> Total Direct Load."""
        return self._joint_to_load_operator


class AltimetryObservationModel:
    """
    An observation model that links satellite altimetry measurements (SSH)
    strictly to an Ice Thickness Change parameter space.
    """

    def __init__(
        self,
        fingerprint_operator: FingerPrintOperator,
        points: List[Tuple[float, float]],
        /,
        *,
        ice_space: Optional[Union[Lebesgue, Sobolev]] = None,
        remove_rotational_contribution: bool = True,
    ):
        """
        Args:
            fingerprint_operator: The physics operator that maps loads to Earth responses.
            points: A list of (latitude, longitude) observation coordinates in degrees.
            ice_space: The Hilbert space for the ice prior. Defaults to the SLE load space.
            remove_rotational_contribution: If True, adds the centrifugal potential
                correction to the SSH calculation.
        """
        self._fingerprint_operator = fingerprint_operator
        self._points = points

        state = fingerprint_operator.state
        load_space = fingerprint_operator.domain
        self._ice_space = ice_space if ice_space is not None else load_space

        # 1. The Pre-Physics Operator: Ice Thickness -> Direct Load
        self._ice_to_load_operator = ice_thickness_change_to_load_operator(
            state, self._ice_space, load_space
        )

        # 2. The Post-Physics Operator: SLE Response -> Discrete SSH Points
        self._response_to_data_operator = altimetry_point_operator(
            state,
            fingerprint_operator.codomain,
            self._points,
            remove_rotational_contribution=remove_rotational_contribution,
        )

        # 3. The Full Composite Forward Operator (No Ocean Dynamic Topography to add here)
        self._forward_operator = (
            self._response_to_data_operator
            @ self._fingerprint_operator
            @ self._ice_to_load_operator
        )

    @property
    def fingerprint_operator(self) -> FingerPrintOperator:
        """The underlying fingerprint operator."""
        return self._fingerprint_operator

    @property
    def points(self) -> List[Tuple[float, float]]:
        """The observation points as (latitude, longitude) pairs."""
        return self._points

    @property
    def response_to_data_operator(self) -> LinearOperator:
        """The operator reading the observable off the response fields."""
        return self._response_to_data_operator

    @property
    def forward_operator(self) -> LinearOperator:
        """The composed operator from model space to the observations."""
        return self._forward_operator
