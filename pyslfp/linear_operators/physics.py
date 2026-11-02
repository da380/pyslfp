"""
Physical operator wrappers for the pyslfp library.

This module bridges the core SeaLevelEquation solver with the pygeoinf
functional-analysis library: the solver's forward and adjoint solves
become LinearOperators on the appropriate Hilbert spaces, the form an
inversion consumes.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from pyshtools import SHCoeffs, SHGrid

from pygeoinf import (
    HilbertSpace,
    LinearOperator,
    EuclideanSpace,
    GaussianMeasure,
)
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev

from pyslfp.core import EarthModelParameters, EarthModel
from pyslfp.elastic import LinearSeaLevelEquation
from pyslfp.core import EarthState
from pyslfp.core.response import (
    ANGULAR_VELOCITY_CHANGE,
    GRDResponse,
    GRDSpace,
)

from pyslfp.linear_operators.utils import underlying_space


class FingerPrintOperator(LinearOperator[SHGrid, GRDResponse]):
    """
    A pygeoinf LinearOperator associated with the solution of the linearised elastic
    sea level equation: from a surface mass load to the response, a
    `GRDResponse` of a `GRDSpace`.

    The response holds the sea level change, the displacement -- its
    horizontal part on request -- the gravitational potential change and,
    with rotational feedback, the angular velocity change. The adjoint is one
    adjoint solve whatever the response holds: a dual s on the coefficients of
    the horizontal potential V pairs with the total load through d = l s, the
    tangential Love numbers, and the total load is the direct load plus
    rho_w C xi, so it enters as a further sea level dual rho_w C d, a direct
    term d, and through the centrifugal potential an angular velocity dual.
    """

    def __init__(
        self,
        state: EarthState,
        /,
        *,
        load_parameters: Tuple[float, float] = None,
        response_parameters: Tuple[float, float] = None,
        rotational_feedbacks: bool = True,
        horizontal: bool = False,
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
            rotational_feedbacks (bool | str): True for the full feedback,
                "axial" for the axial component alone, False for none. The
                response holds the angular velocity change unless it is
                False.
            horizontal (bool): Whether the response's displacement holds its
                horizontal part.
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

        rotation = rotational_feedbacks is not False
        codomain = (
            sobolev_response_space(
                state.model,
                response_order,
                response_scale,
                horizontal=horizontal,
                rotation=rotation,
            )
            if sobolev_response
            else lebesgue_response_space(
                state.model, horizontal=horizontal, rotation=rotation
            )
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
        self._l2_field_space = underlying_space(codomain.field_space)
        l2_codomain = GRDSpace(
            self._l2_field_space, horizontal=horizontal, rotation=rotation
        )
        self._l2_codomain = l2_codomain

        self._cpc_op = centrifugal_potential_operator(state.model)

        # The solver's forward and adjoint solves are written on the L2
        # spaces; the Sobolev adjoint follows from the change of metric.
        l2_operator = LinearOperator.from_callables(
            l2_domain,
            l2_codomain,
            self._l2_mapping_impl,
            adjoint=self._l2_adjoint_mapping_impl,
        )
        self._operator = LinearOperator.from_formal_adjoint(
            domain, codomain, l2_operator
        )

        super().__init__(domain, codomain)

    @staticmethod
    def from_defaults(
        *,
        lmax: int = 256,
        load_parameters: Optional[Tuple[float, float]] = None,
        response_parameters: Optional[Tuple[float, float]] = None,
        rotational_feedbacks: bool = True,
        horizontal: bool = False,
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
            rotational_feedbacks: True for the full feedback, "axial" for
                the axial component alone, False for none.
            horizontal: Whether the response holds the horizontal displacement.
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
            horizontal=horizontal,
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
        horizontal: bool = False,
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
            rotational_feedbacks: True for the full feedback, "axial" for
                the axial component alone, False for none.
            horizontal: Whether the response holds the horizontal displacement.
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
            horizontal=horizontal,
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
        Returns the sphere space of the response fields.
        """
        return self.codomain.field_space

    @property
    def avc_space(self) -> HilbertSpace:
        """
        Returns the space of the angular velocity change, EuclideanSpace(3).
        """
        return self.codomain.member_space(ANGULAR_VELOCITY_CHANGE)

    # ================================================================ #
    #                Measures for convenience in testing               #
    # ================================================================ #

    def load_measure_for_testing(self) -> GaussianMeasure:
        """
        Returns a measure on the load space that is suitable for testing purposes.
        """
        return self.domain.heat_measure(0.5 * self.parameters.mean_sea_floor_radius)

    def response_measure_for_testing(self) -> GaussianMeasure:
        """
        Returns a measure on the response space that is suitable for testing purposes.
        """
        amc_std = (
            self.parameters.rotation_frequency
            * self.parameters.mean_sea_floor_radius**4
        )
        return self.codomain.heat_measure(
            0.5 * self.parameters.mean_sea_floor_radius,
            angular_velocity_std=amc_std,
        )

    # ================================================================= #
    #                          Private methods                          #
    # ================================================================= #

    def _value(self, zeta: SHGrid) -> GRDResponse:
        return self._operator(zeta)

    def _adjoint_value(self, response: GRDResponse) -> SHGrid:
        return self._operator.adjoint(response)

    def _l2_mapping_impl(self, zeta: SHGrid) -> GRDResponse:
        return self._sle.solve_response(
            zeta,
            self._l2_codomain,
            rotational_feedbacks=self._rotational_feedbacks,
            rtol=self._rtol,
            max_iterations=self._max_iterations,
            verbose=self._verbose,
        )

    def _l2_adjoint_mapping_impl(self, response: GRDResponse) -> SHGrid:
        field = self._l2_field_space
        sea_level_dual = field.from_components(response.relative_sea_level_change)
        displacement_dual = field.from_components(response.displacement.vertical)
        potential_dual = field.from_components(response.potential_change)
        angular_velocity_dual = (
            response.angular_velocity_change if self._rotational_feedbacks else None
        )

        direct_term = None
        if self._l2_codomain.has_horizontal:
            sea_level_dual, angular_velocity_dual, direct_term = self._horizontal_dual(
                response, sea_level_dual, angular_velocity_dual
            )

        adjoint_direct_load = sea_level_dual
        adjoint_displacement_load = -1 * displacement_dual
        adjoint_grav_pot_load = -self._g * potential_dual

        if self._rotational_feedbacks:
            adjoint_avc = -self._g * (
                angular_velocity_dual - self._cpc_op.adjoint(potential_dual)
            )
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
        if direct_term is not None:
            adjoint_sea_level = adjoint_sea_level + direct_term
        return adjoint_sea_level

    def _horizontal_dual(self, response, sea_level_dual, angular_velocity_dual):
        """The horizontal displacement's dual carried onto the others.

        In the L2 response metric the V components c pair as
        sum l(l + 1) c c', which is sum_lm s_lm V_lm with
        s_lm = b l(l + 1) c' on the coefficients. With d = l s, the tangential
        Love numbers, that is (1 / b^2) [int zeta d + <rho_w C d, xi>], plus,
        with rotational feedback, the pairing of the centrifugal potential
        with e = l^t s at degree 2. Returns the sea level and angular velocity
        duals with these added, and the direct term d / b^2.
        """
        model = self.model
        b = self._b
        b2 = b**2
        field = self._l2_field_space
        ell = field.degrees[field.degrees >= 1]
        orders = field.orders[field.degrees >= 1]
        s_lm = np.zeros((2, model.lmax + 1, model.lmax + 1))
        s_lm[(orders < 0).astype(int), ell, np.abs(orders)] = (
            b * ell * (ell + 1.0) * response.displacement.horizontal
        )
        love = model.love_numbers

        def synthesise(coefficients):
            return model.expand_coefficient(
                SHCoeffs.from_array(
                    coefficients,
                    normalization=model.normalization,
                    csphase=model.csphase,
                )
            )

        d = synthesise(love.l[None, :, None] * s_lm)
        rho_w = self.parameters.water_density
        sea_level_dual = sea_level_dual + (rho_w / b2) * self.state.ocean_function * d
        if self._rotational_feedbacks:
            e_lm = np.zeros_like(s_lm)
            e_lm[:, 2, :2] = love.l_t[2] * s_lm[:, 2, :2]
            angular_velocity_dual = (
                angular_velocity_dual + self._cpc_op.adjoint(synthesise(e_lm)) / b2
            )
        return sea_level_dual, angular_velocity_dual, d * (1.0 / b2)


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
        sampling=earth_model.sampling,
        extend=earth_model.extend,
    )


def lebesgue_response_space(
    earth_model: EarthModel, /, *, horizontal: bool = False, rotation: bool = True
) -> GRDSpace:
    """
    The L2 response space: a `GRDSpace` over the Lebesgue field space.

    Args:
        earth_model: The EarthModel relative to which the space is defined.
        horizontal: Whether the displacement holds its horizontal part.
        rotation: Whether the angular velocity change is held.
    """
    return GRDSpace(
        lebesgue_load_space(earth_model), horizontal=horizontal, rotation=rotation
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
        sampling=earth_model.sampling,
        extend=earth_model.extend,
    )


def sobolev_response_space(
    earth_model: EarthModel,
    order: float,
    scale: float,
    /,
    *,
    horizontal: bool = False,
    rotation: bool = True,
) -> GRDSpace:
    """
    A Sobolev response space: a `GRDSpace` over the Sobolev field space of
    the given order and scale. The sea level equation gains one order, so the
    response's order is at most the load's plus one.

    Args:
        earth_model: The EarthModel relative to which the space is defined.
        order (float): The Sobolev order of the response fields.
        scale (float): The characteristic length scale.
        horizontal: Whether the displacement holds its horizontal part.
        rotation: Whether the angular velocity change is held.
    """
    return GRDSpace(
        sobolev_load_space(earth_model, order, scale),
        horizontal=horizontal,
        rotation=rotation,
    )


# ================================================================ #
#                 Solid Earth Response Operators                   #
# ================================================================ #


def centrifugal_potential_operator(
    model: EarthModel, /, *, sobolev_parameters=None
) -> LinearOperator:
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
