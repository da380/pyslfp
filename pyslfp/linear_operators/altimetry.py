from __future__ import annotations

import numpy as np

from pygeoinf import LinearOperator, HilbertSpaceDirectSum
from .utils import check_response_space, ocean_projection_operator, underlying_space
from .physics import adjoint_angular_momentum_from_potential

from pyslfp.state import EarthState


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
    state: EarthState,
    response_space: HilbertSpaceDirectSum,
    /,
    *,
    remove_rotational_contribution: bool = True,
):
    """
    Returns as a LinearOperator the mapping from the response space for the fingerprint operator
    to the sea surface height.

    Args:
        state: An EarthState instance.
        response_space: The response space, for the fingerprint operator, this being
            a HilbertSpaceDirectSum whose elements take the form [SL, u, phi, omega]
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

    check_response_space(response_space)

    domain = response_space
    codomain = response_space.subspace(0)

    l2_domain = underlying_space(domain)
    l2_codomain = underlying_space(codomain)

    ocean_projection = ocean_projection_operator(state, codomain)

    def mapping(response):
        sea_level_change, displacement, _, angular_velocity_change = response
        sea_surface_height_change = state.get_sea_surface_height_change(
            sea_level_change,
            displacement,
            angular_velocity_change,
            remove_rotational_contribution=remove_rotational_contribution,
        )
        return ocean_projection(sea_surface_height_change)

    def adjoint_mapping(sea_surface_height_change):
        projected_sea_surface_height_change = ocean_projection(
            sea_surface_height_change
        )

        if remove_rotational_contribution:
            angular_momentum_change = (
                -adjoint_angular_momentum_from_potential(
                    projected_sea_surface_height_change, state.model.parameters
                )
                / state.model.parameters.gravitational_acceleration
            )

        else:
            angular_momentum_change = np.zeros(2)

        return [
            projected_sea_surface_height_change,
            projected_sea_surface_height_change,
            codomain.zero,
            angular_momentum_change,
        ]

    l2_operator = LinearOperator(
        l2_domain, l2_codomain, mapping, adjoint_mapping=adjoint_mapping
    )

    return LinearOperator.from_formal_adjoint(domain, codomain, l2_operator)
