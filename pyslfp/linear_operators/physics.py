"""
Physical operator wrappers for the pyslfp library.

This module bridges the core SeaLevelEquation solver with the pygeoinf
functional analysis library, providing mathematically rigorous LinearOperators
for Bayesian inversions and adjoint calculations.
"""

from __future__ import annotations

from typing import List, Union, Optional, Tuple
import numpy as np
from pyshtools import SHGrid

from pygeoinf import (
    HilbertSpace,
    LinearOperator,
    HilbertSpaceDirectSum,
    EuclideanSpace,
    GaussianMeasure,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.core import EarthModelParameters, EarthModel
from pyslfp.physics import LinearSeaLevelEquation
from pyslfp.state import EarthState

from pyslfp.linear_operators.utils import underlying_space, check_load_space


class FingerPrintOperator(LinearOperator):
    """
    A pygeoinf LinearOperator associated with the solution of the linearised elastic
    sea level equation.
    """

    def __init__(
        self,
        state: EarthState,
        /,
        *,
        load_parameters: Tuple[float, float] = None,
        response_parameters: Tuple[float, float] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Args:
            state: The backgroud state for the finger print calculations. This state
                also provides access to the associated earth model.
            load_parameters: A tuple of the Sobolev order and scale for the
                load space. If not provided, defaults to a Lebesgue space.
            response_parameters: A tuple of the Sobolev order and scale for the
                response fields. If not provided, defaults to a Lebesgue space.
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.

        Raises:
            ValueError: If load and response parameters are provided, and internal
                checks for mathematical consistency is carried out and an exception
                is raised these fails.

        Notes:
            Due to elliptic regularity, the response fields gain one Sobolev order
                over the loads. This means that the response order can only be less
                than equal to one plus the load order.
        """

        self._state = state

        sobolev_load = load_parameters is not None
        sobolev_response = response_parameters is not None

        if sobolev_load:
            load_order, load_scale = load_parameters
        else:
            load_order = 0
            load_scale = None

        if sobolev_response:
            response_order, response_scale = response_parameters
        else:
            response_order = 0
            response_scale = None

        if response_order > load_order + 1:
            raise ValueError(
                "Response order cannot be greater than one plus the load order"
            )

        domain = (
            sobolev_load_space(state.model, load_order, load_scale)
            if sobolev_load
            else lebesgue_load_space(state.model)
        )

        codomain = (
            sobolev_response_space(state.model, response_order, response_scale)
            if sobolev_response
            else lebesgue_response_space(state.model)
        )

        self._sle = LinearSeaLevelEquation(state)
        self._rotational_feedbacks = rotational_feedbacks
        self._rtol = rtol
        self._max_iterations = max_iterations
        self._verbose = verbose

        self._g = self.parameters.gravitational_acceleration
        self._r = self.parameters.rotation_factor
        self._b = self.parameters.mean_sea_floor_radius

        l2_domain = underlying_space(domain)
        l2_codomain = underlying_space(codomain)

        self._cpc_op = centrifugal_potential_operator(state.model)

        l2_operator = LinearOperator(
            l2_domain,
            l2_codomain,
            self._l2_mapping_impl,
            adjoint_mapping=self._l2_adjoint_mapping_impl,
        )

        operator = LinearOperator.from_formal_adjoint(domain, codomain, l2_operator)

        super().__init__(domain, codomain, operator, adjoint_mapping=operator.adjoint)

    @staticmethod
    def from_defaults(
        *,
        lmax: int = 256,
        load_parameters: Optional[Tuple[float, float]] = None,
        response_parameters: Optional[Tuple[float, float]] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> FingerPrintOperator:
        """
        Returns the FingerPrintOperator using default initialisations for the
        Earth model and state.

        If no parameters are provided, it defaults to a standard L2 (Lebesgue) space.
        To regularize the inversion, provide (order, relative_scale) tuples for the
        Sobolev spaces. The relative scale is automatically multiplied by the Earth's
        mean radius.

        Args:
            lmax: Truncation degree for the calculations.
            load_parameters: Tuple of (Sobolev order, relative length scale).
            response_parameters: Tuple of (Sobolev order, relative length scale).
            rotational_feedbacks: Whether to calculate polar wander effects.
            rtol: The relative tolerance for convergence.
            max_iterations: Hard limit on iteration count.
            verbose: If True, prints iteration metrics.
        """
        state = EarthState.from_defaults(lmax=lmax)
        radius = state.model.parameters.mean_sea_floor_radius

        abs_load = (
            (load_parameters[0], load_parameters[1] * radius)
            if load_parameters
            else None
        )
        abs_resp = (
            (response_parameters[0], response_parameters[1] * radius)
            if response_parameters
            else None
        )

        return FingerPrintOperator(
            state,
            load_parameters=abs_load,
            response_parameters=abs_resp,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    @staticmethod
    def for_testing(
        lmax: int,
        /,
        *,
        load_parameters: Optional[Tuple[float, float]] = None,
        response_parameters: Optional[Tuple[float, float]] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> FingerPrintOperator:
        """
        Returns the FingerPrintOperator using analytical testing initialisations
        for the Earth model and initial state.

        If no parameters are provided, it defaults to a standard L2 (Lebesgue) space.
        To regularize the inversion, provide (order, relative_scale) tuples for the
        Sobolev spaces. The relative scale is automatically multiplied by the Earth's
        mean radius.

        Args:
            lmax: Truncation degree for the calculations.
            load_parameters: Tuple of (Sobolev order, relative length scale).
            response_parameters: Tuple of (Sobolev order, relative length scale).
            rotational_feedbacks: Whether to calculate polar wander effects.
            rtol: The relative tolerance for convergence.
            max_iterations: Hard limit on iteration count.
            verbose: If True, prints iteration metrics.
        """
        state = EarthState.for_testing(lmax)
        radius = state.model.parameters.mean_sea_floor_radius

        abs_load = (
            (load_parameters[0], load_parameters[1] * radius)
            if load_parameters
            else None
        )
        abs_resp = (
            (response_parameters[0], response_parameters[1] * radius)
            if response_parameters
            else None
        )

        return FingerPrintOperator(
            state,
            load_parameters=abs_load,
            response_parameters=abs_resp,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    @property
    def state(self) -> EarthState:
        """
        Returns the EarthState associated with the operator
        """
        return self._state

    @property
    def model(self) -> EarthModel:
        """
        Returns the EarthModel associated with the operator
        """
        return self.state.model

    @property
    def parameters(self) -> EarthModelParameters:
        """
        Returns the EarthModelParameters associated with the operator
        """
        return self.model.parameters

    @property
    def field_space(self) -> HilbertSpace:
        """
        Returns the space for the response fields.
        """
        return self.codomain.subspace(0)

    @property
    def avc_space(self) -> HilbertSpace:
        """
        Retutns the space for the angular velocity changes
        """
        return self.codomain.subspace(3)

    # ================================================================ #
    #                Measures for convenience in testing               #
    # ================================================================ #

    def load_measure_for_testing(self) -> GaussianMeasure:
        """
        Returns a measure on the load space that is suitable for testing purposes.
        """
        return self.domain.heat_kernel_gaussian_measure(
            0.5 * self.parameters.mean_sea_floor_radius
        )

    def response_measure_for_testing(self) -> GaussianMeasure:
        """
        Returns a measure on the response space that is suitable for testing purposes.
        """
        field_measure = self.codomain.subspace(0).heat_kernel_gaussian_measure(
            0.5 * self.parameters.mean_sea_floor_radius
        )

        amc_std = (
            self.parameters.rotation_frequency
            * self.parameters.mean_sea_floor_radius**4
        )

        amc_measure = GaussianMeasure.from_standard_deviation(
            self.codomain.subspace(3), amc_std
        )

        return GaussianMeasure.from_direct_sum(
            [field_measure, field_measure, field_measure, amc_measure]
        )

    # ================================================================= #
    #                          Private methods                          #
    # ================================================================= #

    def _l2_mapping_impl(self, zeta: SHGrid) -> List[Union[SHGrid, np.ndarray]]:
        slc, disp, gpc, avc = self._sle.solve_sea_level_equation(
            zeta,
            rotational_feedbacks=self._rotational_feedbacks,
            rtol=self._rtol,
            max_iterations=self._max_iterations,
            verbose=self._verbose,
        )

        return [slc, disp, gpc, avc]

    def _l2_adjoint_mapping_impl(
        self, response: List[Union[SHGrid, np.ndarray]]
    ) -> SHGrid:
        adjoint_direct_load = response[0]
        adjoint_displacement_load = -1 * response[1]
        adjoint_grav_pot_load = -self._g * response[2]

        if self._rotational_feedbacks:
            adjoint_avc = -self._g * (response[3] - self._cpc_op.adjoint(response[2]))
        else:
            adjoint_avc = None

        adjoint_sea_level, _, _, _ = self._sle.solve_generalised_equation(
            direct_load=adjoint_direct_load,
            displacement_load=adjoint_displacement_load,
            gravitational_potential_load=adjoint_grav_pot_load,
            angular_momentum_change=adjoint_avc,
            rotational_feedbacks=self._rotational_feedbacks,
            rtol=self._rtol,
            max_iterations=self._max_iterations,
            verbose=self._verbose,
        )
        return adjoint_sea_level


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


"""
Core physical operators for the Sea Level Equation.

These operators represent the fundamental mathematical mappings of the Earth's 
elastic, gravitational, and rotational responses. By composing these operators, 
users can construct the full Sea Level Equation as a generalized linear system 
(e.g., for Krylov solvers or adjoint state methods).
"""


# ================================================================ #
#                 Solid Earth Response Operators                   #
# ================================================================ #


def centrifugal_potential_operator(
    model: EarthModel, /, *, sobolev_parameters=None
) -> LinearOperator:
    """
    Returns the LinearOperator the maps an angular velocity perturbation
    to the corresponding centrifugal potential perturbation.
    """

    domain = EuclideanSpace(2)

    sobolev_field = sobolev_parameters is not None

    if sobolev_field:
        order, scale = sobolev_parameters
        codomain = sobolev_load_space(model, order, scale)
    else:
        codomain = lebesgue_load_space(model)

    r = model.parameters.rotation_factor
    b = model.parameters.mean_sea_floor_radius

    def mapping(w: np.ndarray) -> SHGrid:
        cpc_lm = model.zero_coefficients()
        cpc_lm.coeffs[:, 2, 1] = model.parameters.rotation_factor * w
        return model.expand_coefficient(cpc_lm)

    def adjoint_mapping(cpc: SHGrid) -> np.ndarray:
        cpc_lm = model.expand_field(cpc, lmax_calc=2)
        return r * b * b * cpc_lm.coeffs[:, 2, 1]

    l2_codomain = underlying_space(codomain)

    l2_operator = LinearOperator(
        domain, l2_codomain, mapping, adjoint_mapping=adjoint_mapping
    )

    return LinearOperator.from_formal_adjoint(domain, codomain, l2_operator)


def solid_earth_response_operator(
    state: EarthState,
    load_space: Union[Lebesgue, Sobolev],
    /,
    *,
    rotational_feedbacks: bool = True,
) -> LinearOperator:
    """
    Maps a total surface mass load to the solid Earth's elastic and
    gravitational response.

    Args:
        state: The background EarthState.
        load_space: The Hilbert space for the surface mass load.
        rotational_feedbacks: If True, computes the coupled polar wander effect.

    Returns:
        A LinearOperator mapping from `load_space` to a HilbertSpaceDirectSum
        containing [Vertical Displacement, Gravity Potential Change, Angular Velocity].
    """
    check_load_space(load_space)
    l2_load_space = underlying_space(load_space)

    # 4-component space: [Disp, GPC, AVC] (We drop SLC for this pure physical step)
    codomain = HilbertSpaceDirectSum([l2_load_space, l2_load_space, EuclideanSpace(2)])

    # Cache constants
    model = state.model
    h_b = model.love_numbers.h[None, :, None]
    k_b = model.love_numbers.k[None, :, None]
    r = model.parameters.rotation_factor
    i = model.parameters.inertia_factor
    ht = model.love_numbers.ht[2]
    kt = model.love_numbers.kt[2]

    # Closed-form multiplier for the rotational feedback loop
    rot_C = i / (1.0 - i * kt * r) if rotational_feedbacks else 0.0

    def mapping(load: SHGrid) -> list:
        load_lm = model.expand_field(load)

        disp_lm = load_lm.copy()
        gpc_lm = load_lm.copy()

        disp_lm.coeffs *= h_b
        gpc_lm.coeffs *= k_b

        avc = np.zeros(2)

        if rotational_feedbacks:
            # SHCoeffs for l=2, m=1 are at index [:, 2, 1]
            phi_21_load = gpc_lm.coeffs[:, 2, 1]

            # Exact closed-form solution for the angular velocity change
            avc = rot_C * phi_21_load

            centrifugal_coeffs = r * avc
            disp_lm.coeffs[:, 2, 1] += ht * centrifugal_coeffs

            # Total GPC includes the centrifugal potential itself (kt + 1)
            gpc_lm.coeffs[:, 2, 1] += (kt + 1.0) * centrifugal_coeffs

        disp = model.expand_coefficient(disp_lm)
        gpc = model.expand_coefficient(gpc_lm)
        return [disp, gpc, avc]

    def adjoint_mapping(response: list) -> SHGrid:
        disp_star, gpc_star, avc_star = response

        disp_star_lm = model.expand_field(disp_star)
        gpc_star_lm = model.expand_field(gpc_star)

        load_star_lm = disp_star_lm.copy()
        load_star_lm.coeffs = disp_star_lm.coeffs * h_b + gpc_star_lm.coeffs * k_b

        if rotational_feedbacks:
            disp_21_star = disp_star_lm.coeffs[:, 2, 1]
            gpc_21_star = gpc_star_lm.coeffs[:, 2, 1]

            # Adjoint of the rotational coupling
            coupling_star = (
                (ht * r) * disp_21_star + ((kt + 1.0) * r) * gpc_21_star + avc_star
            )

            # k_b[0, 2, 0] is simply the scalar load Love number k_2
            load_star_lm.coeffs[:, 2, 1] += (rot_C * k_b[0, 2, 0]) * coupling_star

        return model.expand_coefficient(load_star_lm)

    l2_operator = LinearOperator(
        l2_load_space, codomain, mapping, adjoint_mapping=adjoint_mapping
    )

    target_codomain = HilbertSpaceDirectSum([load_space, load_space, EuclideanSpace(2)])
    return LinearOperator.from_formal_adjoint(load_space, target_codomain, l2_operator)


# ================================================================ #
#                 Sea Level Assembly Operators                     #
# ================================================================ #


def solid_earth_to_sea_level_operator(
    state: EarthState,
    load_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """
    Maps the solid Earth response [Disp, GPC, AVC] to relative sea level change.

    This computes S_local = (Phi/g) - U, and then subtracts the ocean average
    of S_local to ensure the integral over the oceans is zero.
    """
    check_load_space(load_space)
    l2_load_space = underlying_space(load_space)

    domain = HilbertSpaceDirectSum([l2_load_space, l2_load_space, EuclideanSpace(2)])

    g = state.model.parameters.gravitational_acceleration

    def mapping(response: list) -> SHGrid:
        disp, gpc, avc = response

        # Local relative sea level
        slc_local_data = (gpc.data / g) - disp.data
        slc_local = SHGrid.from_array(slc_local_data, grid=state.grid)

        # Remove the spatial ocean average
        ocean_avg = (
            state.model.integrate(state.ocean_function * slc_local) / state.ocean_area
        )
        slc_local.data -= ocean_avg

        return slc_local

    def adjoint_mapping(slc_star: SHGrid) -> list:
        # Adjoint of removing the ocean average
        avg_star = state.model.integrate(slc_star) / state.ocean_area
        local_star_data = slc_star.data - (avg_star * state.ocean_function.data)
        local_star = SHGrid.from_array(local_star_data, grid=state.grid)

        disp_star = local_star.copy()
        disp_star.data = -local_star.data

        gpc_star = local_star.copy()
        gpc_star.data = local_star.data / g

        avc_star = np.zeros(2)
        return [disp_star, gpc_star, avc_star]

    l2_operator = LinearOperator(
        domain, l2_load_space, mapping, adjoint_mapping=adjoint_mapping
    )

    target_domain = HilbertSpaceDirectSum([load_space, load_space, EuclideanSpace(2)])
    return LinearOperator.from_formal_adjoint(target_domain, load_space, l2_operator)


def eustatic_sea_level_operator(
    state: EarthState,
    load_space: Union[Lebesgue, Sobolev],
) -> LinearOperator:
    """
    Maps a direct mass load to its uniform eustatic sea level shift.

    This enforces strict mass conservation (the volume of water added to the
    oceans must equal the volume of the melted ice).
    """
    check_load_space(load_space)
    l2_load_space = underlying_space(load_space)

    def mapping(direct_load: SHGrid) -> SHGrid:
        eustatic_shift = -state.model.integrate(direct_load) / (
            state.model.parameters.water_density * state.ocean_area
        )
        slc = state.model.zero_grid()
        slc.data += eustatic_shift
        return slc

    def adjoint_mapping(slc_star: SHGrid) -> SHGrid:
        shift_star = state.model.integrate(slc_star)
        load_star_data = -shift_star / (
            state.model.parameters.water_density * state.ocean_area
        )
        load_star = state.model.zero_grid()
        load_star.data += load_star_data
        return load_star

    l2_operator = LinearOperator(
        l2_load_space, l2_load_space, mapping, adjoint_mapping=adjoint_mapping
    )
    return LinearOperator.from_formal_adjoint(load_space, load_space, l2_operator)
