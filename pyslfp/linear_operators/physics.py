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

from pyslfp.linear_operators.utils import (
    underlying_space,
    _check_sobolev,
    spatial_multiplication_operator,
    averaging_operator,
)


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
            load_order, load_scale = _check_sobolev(load_parameters)
        else:
            load_order = 0
            load_scale = None

        if sobolev_response:
            response_order, response_scale = _check_sobolev(response_parameters)
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
    def from_lebesgue_defaults(
        *,
        lmax: int = 256,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> FingerPrintOperator:
        """
        Returns the LinearOperator between Lebesgue spaces using
        default initialisations for the earth model and state.

        Args:
            lmax: Truncation degree for the calculations.
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.
        """

        state = EarthState.from_defaults(lmax=lmax)

        return FingerPrintOperator(
            state,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    @staticmethod
    def from_sobolev_defaults(
        order: float,
        relative_scale: float,
        /,
        *,
        lmax: int = 256,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
        regularity: bool = False,
    ) -> FingerPrintOperator:
        """
        Returns the LinearOperator between Sobolev spaces using
        default initialisations for the earth model and state.

        Args:
            lmax: Truncation degree for the calculations.
            order: Sobolev order.
            relative_scale: Sobolev scale relative to the Earth model's mean radius.
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.
            regularity (bool): If True, the response order is raised by
                one from the load order in accordance with elliptic
                regularity. Otherwise, the response order is the same
                as for the load.
        """

        state = EarthState.from_defaults(lmax=lmax)
        scale = state.model.parameters.mean_sea_floor_radius * relative_scale
        load_parameters = (order, scale)
        response_parameters = (order + 1, scale) if regularity else load_parameters

        return FingerPrintOperator(
            state,
            load_parameters=load_parameters,
            response_parameters=response_parameters,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    @staticmethod
    def for_lebesgue_testing(
        lmax: int,
        /,
        *,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> FingerPrintOperator:
        """
        Returns the LinearOperator between Lebesgue spaces using
        testing initialisations for the Earth model and initial state.

        Args:
            lmax: Truncation degree for the calculations.
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.
        """

        state = EarthState.for_testing(lmax)

        return FingerPrintOperator(
            state,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    @staticmethod
    def for_sobolev_testing(
        lmax: int,
        order: float,
        relative_scale: float,
        /,
        *,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
        regularity: bool = False,
    ) -> FingerPrintOperator:
        """
        Returns the LinearOperator between Sobolev spaces using
        default initialisations for the earth model and state.

        Args:
            lmax: Truncation degree for the calculations.
            order: Sobolev order.
            relative_scale: Sobolev scale relative to the Earth model's mean radius.
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.
            regularity (bool): If True, the response order is raised by
                one from the load order in accordance with ellipstic
                regularity. Otherwise, the response order is the same
                as for the load.
        """

        state = EarthState.for_testing(lmax)
        scale = state.model.parameters.mean_sea_floor_radius * relative_scale
        load_parameters = (order, scale)
        response_parameters = (order + 1, scale) if regularity else load_parameters

        return FingerPrintOperator(
            state,
            load_parameters=load_parameters,
            response_parameters=response_parameters,
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

    # ================================================================ #
    #                    Spatial projection operators                  #
    # ================================================================ #

    def ocean_projection_operator(
        self, /, *, exclude_ice_shelves: bool = False, load_action: bool = True
    ) -> LinearOperator:
        """
        Returns a LinearOperator that multiplies a field by a function that is one
        over the oceans and zero elsewhere.

        Args:
            exclude_ice_shelves: If True, the function is set to zero in ice-shelved regions.
            load_action: If True, the operator acts on the load space. Otherwise, it acts on the
                field space associated with the response space. Default is True.

        Returns:
            A LinearOperator object.
        """
        projection_field = self.state.ocean_projection(
            value=0, exclude_ice_shelves=exclude_ice_shelves
        )
        return spatial_multiplication_operator(
            self.domain if load_action else self.field_space,
            projection_field,
        )

    def ice_projection_operator(
        self, /, *, exclude_ice_shelves: bool = False, load_action: bool = True
    ) -> LinearOperator:
        """
        Returns a LinearOperator that multiplies a field by a function that is one
        over the ice sheets and zero elsewhere.

        Args:
            exclude_ice_shelves: If True, the function is set to zero in ice-shelved regions.
            load_action: If True, the operator acts on the load space. Otherwise, it acts on the
                field space associated with the response space. Default is True.

        Returns:
            A LinearOperator object.
        """
        projection_field = self.state.ice_projection(
            value=0, exclude_ice_shelves=exclude_ice_shelves
        )
        return spatial_multiplication_operator(
            self.domain if load_action else self.field_space,
            projection_field,
        )

    def land_projection_operator(
        self, /, *, exclude_ice: bool = True, load_action: bool = True
    ) -> LinearOperator:
        """
        Returns a LinearOperator that multiplies a field by a function that is one
        over the background land and zero elsewhere.

        Args:
            exclude_ice: If True, the function is set to zero in ice-covered regions.
            load_action: If True, the operator acts on the load space. Otherwise, it acts on the
                field space associated with the response space. Default is True.

        Returns:
            A LinearOperator object.
        """
        projection_field = self.state.land_projection(value=0, exclude_ice=exclude_ice)
        return spatial_multiplication_operator(
            self.domain if load_action else self.field_space,
            projection_field,
        )

    # ================================================================ #
    #                     Spatial averaging operators                  #
    # ================================================================ #

    def ocean_average_operator(
        self, /, *, exclude_ice_shelves: bool = False, load_action: bool = True
    ) -> LinearOperator:
        """
        Returns a LinearOperator that averages a function over the oceans.

        Args:
            exclude_ice_shelves: If True, the function is set to zero in ice-shelved regions.
            load_action: If True, the operator acts on the load space. Otherwise, it acts on the
                field space associated with the response space. Default is True.

        Returns:
            A LinearOperator object.
        """
        projection_field = self.state.ocean_projection(
            value=0, exclude_ice_shelves=exclude_ice_shelves
        )
        return averaging_operator(
            self.state,
            self.domain if load_action else self.field_space,
            [projection_field],
        )

    def ice_average_operator(
        self, /, *, exclude_ice_shelves: bool = False, load_action: bool = True
    ) -> LinearOperator:
        """
        Returns a LinearOperator that averages a function over the ice sheets.

        Args:
            exclude_ice_shelves: If True, the function is set to zero in ice-shelved regions.
            load_action: If True, the operator acts on the load space. Otherwise, it acts on the
                field space associated with the response space. Default is True.

        Returns:
            A LinearOperator object.
        """
        projection_field = self.state.ice_projection(
            value=0, exclude_ice_shelves=exclude_ice_shelves
        )
        return averaging_operator(
            self.state,
            self.domain if load_action else self.field_space,
            [projection_field],
        )

    def land_average_operator(
        self, /, *, exclude_ice: bool = True, load_action: bool = True
    ) -> LinearOperator:
        """
        Returns a LinearOperator that averages a function over the background land.

        Args:
            exclude_ice: If True, the function is set to zero in ice-covered regions.
            load_action: If True, the operator acts on the load space. Otherwise, it acts on the
                field space associated with the response space. Default is True.

        Returns:
            A LinearOperator object.
        """
        projection_field = self.state.land_projection(value=0, exclude_ice=exclude_ice)
        return averaging_operator(
            self.state,
            self.domain if load_action else self.field_space,
            [projection_field],
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
        order, scale = _check_sobolev(sobolev_parameters)
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
