from __future__ import annotations

import numpy as np

from pyshtools import SHGrid
from pygeoinf import HilbertSpaceDirectSum
from .physics import FingerPrintOperator


def get_sea_surface_height_change(
    self,
    sea_level_change: SHGrid,
    displacement: SHGrid,
    angular_velocity_change: np.ndarray,
    /,
    *,
    remove_rotational_contribution: bool = True,
) -> SHGrid:
    """
    Computes the Sea Surface Height (SSH) change from raw physical fields.

    Args:
        sea_level_change (SHGrid): The relative sea level change field.
        displacement (SHGrid): The solid Earth displacement field.
        angular_velocity_change (np.ndarray): The [omega_x, omega_y] vector.
        remove_rotational_contribution (bool): Whether to remove the
            centrifugal signal from the SSH (standard for altimetry).

    Returns:
        SHGrid: The total Sea Surface Height change.
    """
    ssh_change = sea_level_change + displacement

    if remove_rotational_contribution:
        centrifugal = self.centrifugal_potential_change(angular_velocity_change)
        ssh_change += centrifugal / self._g

    return ssh_change


def sea_surface_height_operator(
    fp_op: FingerPrintOperator,
    response_space: HilbertSpaceDirectSum,
    /,
    *,
    remove_rotational_contribution: bool = True,
):
    """
    Returns as a LinearOperator the mapping from the response space for the fingerprint operator
    to the sea surface height.

    Args:
        fp_op: An instance of FingerPrintOperators.
        remove_rotational_contribution: If True, rotational contribution
                is removed from the sea surface height. Default is True

    Returns:
        A LinearOperator object.

    Note:
        This operator returns only the sea surface height change associated with the
        gravitationally induced sea level change resulting from a given direct load.
        When that direct load has a component linked to ocean dynamic topography,
        the dynamic topography must added to obtain the full sea surface height change.
    """

    # cpc_op = centrifugal_potential_operator(fp_op.model, field_parameters=(codomain))
    # ocean_projection = ocean_projection_operator(fp_op.state, codomain)
