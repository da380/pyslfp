"""
Observation models and operators for satellite altimetry networks.

This module provides the raw mathematical operators for mapping physical
response fields to Sea Surface Height (SSH) anomalies, utilities for generating
independent observation grids, and high-level observation models linking
physical parameter spaces to discrete SSH observations.
"""

from __future__ import annotations
from typing import List, Tuple, Union, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyshtools import SHGrid

from pygeoinf import (
    LinearOperator,
    HilbertSpaceDirectSum,
    RowLinearOperator,
    GaussianMeasure,
    LinearForwardProblem,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.state import EarthState
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
    spacing_degrees: float = 2.0,
    latitude_min: float = -66.0,
    latitude_max: float = 66.0,
) -> List[Tuple[float, float]]:
    """Generates a regular grid of points returning only those in valid oceans."""
    lats = np.arange(latitude_min, latitude_max + 1e-9, spacing_degrees)
    lons = np.arange(0.0, 360.0, spacing_degrees)
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
    spacing_degrees: float = 2.0,
    exclude_ice_shelves: bool = False,
    exclude_glaciers: bool = True,
) -> List[Tuple[float, float]]:
    """Generates a regular grid of points returning only those over ice sheets."""
    lats = np.arange(-90.0, 90.0 + 1e-9, spacing_degrees)
    lons = np.arange(0.0, 360.0, spacing_degrees)
    lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
    candidate_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))

    mask = state.ice_projection(
        value=0,
        exclude_ice_shelves=exclude_ice_shelves,
        exclude_glaciers=exclude_glaciers,
    )
    return _filter_independent_points(candidate_points, mask)


# ================================================================ #
#                       Raw Math Operators                         #
# ================================================================ #


def sea_surface_height_operator(
    state: EarthState,
    response_space: HilbertSpaceDirectSum,
    /,
    *,
    remove_rotational_contribution: bool = True,
) -> LinearOperator:
    """
    Returns an operator mapping the SLE response space to Sea Surface Height (SSH).

    Mathematically, SSH = Sea Level Change + Vertical Displacement.
    If rotational feedbacks are removed, it also adds (Centrifugal Potential / g).

    Args:
        state: The EarthState providing the physical parameters.
        response_space: The 4-component composite response space.
        remove_rotational_contribution: If True, adds the centrifugal potential
            correction to the SSH calculation.
    """
    check_response_space(response_space)

    field_space = response_space.subspace(0)
    avc_space = response_space.subspace(3)

    id_op = field_space.identity_operator()
    zero_gpc_op = field_space.zero_operator()

    if remove_rotational_contribution:
        # Match Sobolev parameters if the response space is regularized
        sobolev_params = (
            (field_space.order, field_space.scale)
            if isinstance(field_space, Sobolev)
            else None
        )

        cpc_op = centrifugal_potential_operator(
            state.model, sobolev_parameters=sobolev_params
        )
        g = state.model.parameters.gravitational_acceleration

        # Scale the centrifugal potential by 1/g to get the height contribution
        avc_op = cpc_op * (1.0 / g)
    else:
        avc_op = avc_space.zero_operator(codomain=field_space)

    # SSH = [1.0 * SLC] + [1.0 * Disp] + [0.0 * GPC] + [(CPC / g) * AVC]
    return RowLinearOperator([id_op, id_op, zero_gpc_op, avc_op])


def altimetry_point_operator(
    state: EarthState,
    response_space: HilbertSpaceDirectSum,
    points: List[Tuple[float, float]],
    /,
    *,
    remove_rotational_contribution: bool = True,
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
    point_eval_op = ssh_op.codomain.point_evaluation_operator(points)

    return point_eval_op @ ssh_op


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
    ):
        """
        Args:
            fingerprint_operator: The physics operator that maps loads to Earth responses.
            points: A list of (latitude, longitude) observation coordinates in degrees.
            ice_space: The Hilbert space for the ice prior. Defaults to the SLE load space.
            ocean_space: The Hilbert space for the ocean prior. Defaults to the SLE load space.
            remove_rotational_contribution: If True, adds the centrifugal potential
                correction to the SSH calculation.
        """
        self._fingerprint_operator = fingerprint_operator
        self._points = points

        state = fingerprint_operator.state
        load_space = fingerprint_operator.domain

        self._ice_space = ice_space if ice_space is not None else load_space
        self._ocean_space = ocean_space if ocean_space is not None else load_space

        # 1. The Pre-Physics Operator: [Ice, Ocean] -> Direct Load
        self._joint_to_load_operator = joint_ice_ocean_to_load_operator(
            state, self._ice_space, self._ocean_space, load_space
        )

        # 2. The Post-Physics Operator: SLE Response -> Discrete SSH Points
        self._response_to_data_operator = altimetry_point_operator(
            state,
            fingerprint_operator.codomain,
            self._points,
            remove_rotational_contribution=remove_rotational_contribution,
        )

        # 3. The Full Composite Forward Operator
        self._forward_operator = (
            self._response_to_data_operator
            @ self._fingerprint_operator
            @ self._joint_to_load_operator
        )

    @property
    def fingerprint_operator(self) -> FingerPrintOperator:
        return self._fingerprint_operator

    @property
    def points(self) -> List[Tuple[float, float]]:
        return self._points

    @property
    def response_to_data_operator(self) -> LinearOperator:
        return self._response_to_data_operator

    @property
    def forward_operator(self) -> LinearOperator:
        return self._forward_operator

    def create_forward_problem(
        self, noise_std: Union[float, np.ndarray]
    ) -> LinearForwardProblem:
        """
        Wraps the composite forward operator in a Pygeoinf LinearForwardProblem.

        Args:
            noise_std: The non-dimensional standard deviation of the measurement noise.
                Can be a single float (uniform noise) or an array matching the number of points.
        """
        data_space = self.forward_operator.codomain

        if isinstance(noise_std, (float, int)):
            stds = np.full(data_space.dim, float(noise_std))
        else:
            stds = np.asarray(noise_std)
            if len(stds) != data_space.dim:
                raise ValueError(
                    f"Provided noise array length ({len(stds)}) does not match "
                    f"number of altimetry points ({data_space.dim})."
                )

        noise_meas = GaussianMeasure.from_standard_deviations(data_space, stds)
        return LinearForwardProblem(
            self.forward_operator, data_error_measure=noise_meas
        )


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

        # 3. The Full Composite Forward Operator
        self._forward_operator = (
            self._response_to_data_operator
            @ self._fingerprint_operator
            @ self._ice_to_load_operator
        )

    @property
    def fingerprint_operator(self) -> FingerPrintOperator:
        return self._fingerprint_operator

    @property
    def points(self) -> List[Tuple[float, float]]:
        return self._points

    @property
    def response_to_data_operator(self) -> LinearOperator:
        return self._response_to_data_operator

    @property
    def forward_operator(self) -> LinearOperator:
        return self._forward_operator

    def create_forward_problem(
        self, noise_std: Union[float, np.ndarray]
    ) -> LinearForwardProblem:
        """
        Wraps the composite forward operator in a Pygeoinf LinearForwardProblem.
        """
        data_space = self.forward_operator.codomain

        if isinstance(noise_std, (float, int)):
            stds = np.full(data_space.dim, float(noise_std))
        else:
            stds = np.asarray(noise_std)
            if len(stds) != data_space.dim:
                raise ValueError(
                    f"Provided noise array length ({len(stds)}) does not match "
                    f"number of altimetry points ({data_space.dim})."
                )

        noise_meas = GaussianMeasure.from_standard_deviations(data_space, stds)
        return LinearForwardProblem(
            self.forward_operator, data_error_measure=noise_meas
        )
