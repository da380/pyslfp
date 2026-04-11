"""
Physical operator wrappers for the pyslfp library.

This module bridges the core SeaLevelEquation solver with the pygeoinf
functional analysis library, providing mathematically rigorous LinearOperators
for Bayesian inversions and adjoint calculations.
"""

from typing import List, Union, Optional
import numpy as np
from pyshtools import SHGrid

from pygeoinf import LinearOperator, HilbertSpaceDirectSum, EuclideanSpace
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.physics import SeaLevelEquation
from pyslfp.state import EarthState


# ==================================================================== #
#                       Space Generators                               #
# ==================================================================== #


def lebesgue_load_space(state: EarthState) -> Lebesgue:
    """Defines the L2 mathematical space for square-integrable surface loads."""
    return Lebesgue(
        state.lmax,
        radius=state.model.parameters.mean_sea_floor_radius,
        grid=state.grid_type,
    )


def lebesgue_response_space(state: EarthState) -> HilbertSpaceDirectSum:
    """Defines the composite L2 response space for the physical fields."""
    field_space = lebesgue_load_space(state)
    return HilbertSpaceDirectSum(
        [field_space, field_space, field_space, EuclideanSpace(2)]
    )


def sobolev_load_space(state: EarthState, order: float, scale: float) -> Sobolev:
    """Defines a Sobolev space for surface mass loads with smoothness constraints."""
    return Sobolev(
        state.lmax,
        order,
        scale,
        radius=state.model.parameters.mean_sea_floor_radius,
        grid=state.grid_type,
    )


def sobolev_response_space(
    state: EarthState, order: float, scale: float
) -> HilbertSpaceDirectSum:
    """Defines the response space corresponding to a Sobolev load space (order s+1)."""
    field_space = Sobolev(
        state.lmax,
        order + 1,
        scale,
        radius=state.model.parameters.mean_sea_floor_radius,
        grid=state.grid_type,
    )
    return HilbertSpaceDirectSum(
        [field_space, field_space, field_space, EuclideanSpace(2)]
    )


# ==================================================================== #
#                       Adjoint Math Helpers                           #
# ==================================================================== #


def isolate_gravitational_potential(
    gravity_potential: SHGrid,
    angular_velocity_change: np.ndarray,
    solver: SeaLevelEquation,
    state: EarthState,
) -> SHGrid:
    """Subtracts the centrifugal potential to isolate the gravitational potential."""
    gp_lm = gravity_potential.expand(normalization="ortho", csphase=1)
    gp_lm.coeffs[:, 2, 1] -= solver._rotation_factor * angular_velocity_change
    return gp_lm.expand(grid=state.grid_type, extend=state.extend)


def adjoint_angular_momentum_from_potential(
    gravitational_potential_load: SHGrid, solver: SeaLevelEquation
) -> np.ndarray:
    """Computes the adjoint angular momentum change for a given potential load."""
    gpot_lm = gravitational_potential_load.expand(
        normalization="ortho", csphase=1, lmax_calc=2
    )
    r = solver._rotation_factor
    b = solver.model.parameters.mean_sea_floor_radius
    return -r * b * b * gpot_lm.coeffs[:, 2, 1]


# ==================================================================== #
#                       Linear Operator Factories                      #
# ==================================================================== #


def get_lebesgue_linear_operator(
    solver: SeaLevelEquation,
    state: EarthState,
    *,
    rotational_feedbacks: bool = True,
    rtol: float = 1e-6,
    max_iterations: Optional[int] = None,
    verbose: bool = False,
) -> LinearOperator:
    """
    Constructs the sea-level model as a pygeoinf LinearOperator between Lebesgue spaces.
    """
    domain = lebesgue_load_space(state)
    codomain = lebesgue_response_space(state)
    g = solver.model.parameters.gravitational_acceleration

    def mapping(u: SHGrid) -> List[Union[SHGrid, np.ndarray]]:
        slc, disp, gpc, avc = solver.solve_generalised_equation(
            state,
            direct_load=u,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        if rotational_feedbacks:
            grav_pot_change = isolate_gravitational_potential(gpc, avc, solver, state)
        else:
            grav_pot_change = gpc

        return [slc, disp, grav_pot_change, avc]

    def adjoint_mapping(response: List[Union[SHGrid, np.ndarray]]) -> SHGrid:
        adjoint_direct_load = response[0]
        adjoint_displacement_load = -1 * response[1]
        adjoint_grav_pot_load = -g * response[2]

        if rotational_feedbacks:
            adjoint_avc = -g * (
                response[3]
                + adjoint_angular_momentum_from_potential(response[2], solver)
            )
        else:
            adjoint_avc = None

        adjoint_sea_level, _, _, _ = solver.solve_generalised_equation(
            state,
            direct_load=adjoint_direct_load,
            displacement_load=adjoint_displacement_load,
            gravitational_potential_load=adjoint_grav_pot_load,
            angular_momentum_change=adjoint_avc,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        return adjoint_sea_level

    return LinearOperator(domain, codomain, mapping, adjoint_mapping=adjoint_mapping)


def get_sobolev_linear_operator(
    solver: SeaLevelEquation,
    state: EarthState,
    order: float,
    scale: float,
    *,
    rotational_feedbacks: bool = True,
    rtol: float = 1e-6,
    max_iterations: Optional[int] = None,
    verbose: bool = False,
) -> LinearOperator:
    """
    Constructs the sea-level model as a pygeoinf LinearOperator between Sobolev spaces.
    """
    domain = sobolev_load_space(state, order, scale)
    codomain = sobolev_response_space(state, order, scale)

    lebesgue_operator = get_lebesgue_linear_operator(
        solver,
        state,
        rotational_feedbacks=rotational_feedbacks,
        rtol=rtol,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    return LinearOperator.from_formal_adjoint(domain, codomain, lebesgue_operator)
