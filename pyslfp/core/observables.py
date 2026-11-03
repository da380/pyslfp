"""
Operators that read physical quantities off the response to a load.

`centrifugal_potential_operator` maps an angular velocity change to its
centrifugal potential, and `geocentric_sea_level_operator` maps a response to
the geocentric sea level change, the relative sea level change plus the
vertical displacement, with the centrifugal potential's height on request.
They serve the elastic and viscoelastic physics, the observations and the
models alike.
"""

from __future__ import annotations

import numpy as np
from pyshtools import SHGrid

from pygeoinf import EuclideanSpace, LinearOperator

from .model import EarthModel
from .response import (
    ANGULAR_VELOCITY_CHANGE,
    GRDResponse,
    DISPLACEMENT,
    RELATIVE_SEA_LEVEL_CHANGE,
    GRDSpace,
)
from .spaces import (
    check_response_space,
    lebesgue_load_space,
    sobolev_load_space,
    underlying_space,
)
from .state import EarthState

__all__ = ["centrifugal_potential_operator", "geocentric_sea_level_operator"]


def centrifugal_potential_operator(
    model: EarthModel, /, *, sobolev_parameters=None
) -> LinearOperator[np.ndarray, SHGrid]:
    """
    Returns the LinearOperator the maps an angular velocity perturbation
    [omega_x, omega_y, omega_z] to the corresponding centrifugal potential
    perturbation.

    The transverse components give the degree-2, order-1 potential and the
    axial component the degree-2, order-0 potential together with its
    degree-0 part, a constant on the surface.
    """

    domain = EuclideanSpace(3)

    sobolev_field = sobolev_parameters is not None

    if sobolev_field:
        order, scale = sobolev_parameters
        codomain = sobolev_load_space(model, order, scale)
    else:
        codomain = lebesgue_load_space(model)

    r = model.parameters.rotation_factor
    r_axial = model.parameters.axial_rotation_factor
    r_uniform = model.parameters.uniform_rotation_factor
    b = model.parameters.mean_sea_floor_radius

    def mapping(w: np.ndarray) -> SHGrid:
        cpc_lm = model.zero_coefficients()
        cpc_lm.coeffs[:, 2, 1] = r * w[:2]
        cpc_lm.coeffs[0, 2, 0] = r_axial * w[2]
        cpc_lm.coeffs[0, 0, 0] = r_uniform * w[2]
        return model.expand_coefficient(cpc_lm)

    def adjoint_mapping(cpc: SHGrid) -> np.ndarray:
        cpc_lm = model.expand_field(cpc, lmax_calc=2)
        w = np.empty(3)
        w[:2] = r * b * b * cpc_lm.coeffs[:, 2, 1]
        w[2] = (
            b
            * b
            * (r_axial * cpc_lm.coeffs[0, 2, 0] + r_uniform * cpc_lm.coeffs[0, 0, 0])
        )
        return w

    l2_codomain = underlying_space(codomain)

    l2_operator = LinearOperator.from_callables(
        domain, l2_codomain, mapping, adjoint=adjoint_mapping
    )

    return LinearOperator.from_formal_adjoint(domain, codomain, l2_operator)


def geocentric_sea_level_operator(
    state: EarthState,
    response_space: GRDSpace,
    /,
    *,
    remove_rotational_contribution: bool = True,
) -> LinearOperator[GRDResponse, SHGrid]:
    """
    Returns an operator mapping the response to the geocentric sea level change,
    a field of the response's field space.

    The geocentric sea level change is the relative sea level change plus the vertical
    displacement. With the rotational contribution removed, the centrifugal
    potential over g is added, which needs the response to hold the angular
    velocity change.

    Args:
        state: The EarthState providing the physical parameters.
        response_space: The response space.
        remove_rotational_contribution: If True, adds the centrifugal potential
            correction to the geocentric sea level.
    """
    check_response_space(response_space)

    field_space = response_space.field_space
    geocentric = response_space.field_operator(
        RELATIVE_SEA_LEVEL_CHANGE
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
        geocentric = geocentric + (1.0 / g) * cpc_op @ response_space.projection(
            ANGULAR_VELOCITY_CHANGE
        )
    return geocentric
