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

from pyslfp.core import EarthModelParameters, EarthModel
from pyslfp.physics import SeaLevelEquation
from pyslfp.state import EarthState


# ==================================================================== #
#                       Space Generators                               #
# ==================================================================== #


def lebesgue_load_space(earth_model: EarthModel, /) -> Lebesgue:
    """
    Defines the L2 mathematical space for square-integrable surface loads.

    Args:
        earth_model: The EarthModel relative to which the space is defined

    Returns:
        Lebesgue: The mathematical space for surface loads.
    """
    return Lebesgue(
        earth_model.lmax,
        radius=earth_model.parameters.mean_sea_floor_radius,
        grid=earth_model.grid,
    )


def lebesgue_response_space(earth_model: EarthModel, /) -> HilbertSpaceDirectSum:
    """
    Defines the composite L2 response space for the physical fields.

    Args:
        earth_model: The EarthModel relative to which the space is defined.

    Returns:
        HilbertSpaceDirectSum: A composite space comprising SLC, Displacement,
            Gravitational Potential, and a 2D Euclidean space for Angular Velocity.
    """
    field_space = lebesgue_load_space(earth_model)
    return HilbertSpaceDirectSum(
        [field_space, field_space, field_space, EuclideanSpace(2)]
    )


def sobolev_load_space(
    earth_model: EarthModel, order: float, scale: float, /
) -> Sobolev:
    """
    Defines a Sobolev space for surface mass loads with smoothness constraints.

    Args:
        earth_model: The EarthModel relative to which the space is defined.
        order (float): The Sobolev order (smoothness constraint).
        scale (float): The characteristic length scale.

    Returns:
        Sobolev: The regularized mathematical space for surface loads.
    """
    return Sobolev(
        earth_model.lmax,
        order,
        scale,
        radius=earth_model.parameters.mean_sea_floor_radius,
        grid=earth_model.grid,
    )


def sobolev_response_space(
    earth_model: EarthModel, order: float, scale: float, /
) -> HilbertSpaceDirectSum:
    """
    Defines the response space corresponding to a Sobolev load space.

    Due to elliptic regularity, the response fields are one order smoother
    than the applied load (order + 1).

    Args:
        earth_model: The EarthModel relative to which the space is defined.
        order (float): The Sobolev order of the load space.
        scale (float): The characteristic length scale.

    Returns:
        HilbertSpaceDirectSum: The composite Sobolev response space.
    """
    field_space = Sobolev(
        earth_model.lmax,
        order + 1,
        scale,
        radius=earth_model.parameters.mean_sea_floor_radius,
        grid=earth_model.grid,
    )
    return HilbertSpaceDirectSum(
        [field_space, field_space, field_space, EuclideanSpace(2)]
    )


# ==================================================================== #
#                       Linear Operator Factories                      #
# ==================================================================== #


def get_lebesgue_linear_operator(
    solver: SeaLevelEquation,
    state: EarthState,
    /,
    *,
    rotational_feedbacks: bool = True,
    rtol: float = 1e-6,
    max_iterations: Optional[int] = None,
    verbose: bool = False,
) -> LinearOperator:
    """
    Constructs the sea-level model as a pygeoinf LinearOperator between Lebesgue spaces.

    Args:
        solver (SeaLevelEquation): The configured SLE solver engine.
        state (EarthState): The background Earth state.
        rotational_feedbacks (bool): Whether to calculate polar wander effects.
        rtol (float): Relative tolerance for solver convergence.
        max_iterations (Optional[int]): Hard limit on solver iteration count.
        verbose (bool): If True, prints internal solver metrics.

    Returns:
        LinearOperator: A functional operator mathematically mapping a surface mass
            load to the 4-component physical response fields.
    """
    model = state.model
    parameters = model.parameters
    domain = lebesgue_load_space(model)
    codomain = lebesgue_response_space(model)
    g = parameters.gravitational_acceleration

    def mapping(u: SHGrid) -> List[Union[SHGrid, np.ndarray]]:
        slc, disp, gpc, avc = solver.solve_generalised_equation(
            state,
            direct_load=u,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        return [slc, disp, gpc, avc]

    def adjoint_mapping(response: List[Union[SHGrid, np.ndarray]]) -> SHGrid:
        adjoint_direct_load = response[0]
        adjoint_displacement_load = -1 * response[1]
        adjoint_grav_pot_load = -g * response[2]

        if rotational_feedbacks:
            gpot_lm = model.expand_field(response[2], lmax_calc=2)
            r = parameters.rotation_factor
            b = parameters.mean_sea_floor_radius
            amc = -r * b * b * gpot_lm.coeffs[:, 2, 1]
            adjoint_avc = -g * (response[3] + amc)
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
    /,
    *,
    rotational_feedbacks: bool = True,
    rtol: float = 1e-6,
    max_iterations: Optional[int] = None,
    verbose: bool = False,
) -> LinearOperator:
    """
    Constructs the sea-level model as a pygeoinf LinearOperator between Sobolev spaces.

    This wraps the standard Lebesgue operator but enforces smoothness constraints
    on the input domain, which acts as a powerful regularization technique during
    Bayesian or gradient-based inversions.

    Args:
        solver (SeaLevelEquation): The configured SLE solver engine.
        state (EarthState): The background Earth state.
        order (float): The Sobolev smoothness order (s > 0).
        scale (float): The characteristic Sobolev length scale (λ > 0).
        rotational_feedbacks (bool): Whether to calculate polar wander effects.
        rtol (float): Relative tolerance for solver convergence.
        max_iterations (Optional[int]): Hard limit on solver iteration count.
        verbose (bool): If True, prints internal solver metrics.

    Returns:
        LinearOperator: A functional operator mapping a Sobolev surface mass
            load to the 4-component Sobolev physical response fields.
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
