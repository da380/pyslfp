"""
Module for defining some operators related to the sea level problem.
"""

from __future__ import annotations
from typing import Optional, Tuple, List, Union, Callable

import inspect

import numpy as np

from scipy.sparse import diags


from pyshtools import SHGrid, SHCoeffs

from pygeoinf import (
    LinearOperator,
    HilbertSpace,
    EuclideanSpace,
    HilbertSpaceDirectSum,
    RowLinearOperator,
    MassWeightedHilbertSpace,
)

from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev


from . import DATADIR
from .utils import SHVectorConverter
from .physical_parameters import EarthModelParameters
from .love_numbers import LoveNumbers
from .finger_print import FingerPrint


def underlying_space(space: HilbertSpace):
    """
    Returns the underlying space of a HilbertSpace object. The the space
    is not mass weighted, the original space is returned. If the space is
    a direct sum, the method is applied to each subspace recursively.
    """

    if isinstance(space, MassWeightedHilbertSpace):
        return space.underlying_space
    elif isinstance(space, HilbertSpaceDirectSum):
        return HilbertSpaceDirectSum(
            [underlying_space(subspace) for subspace in space.subspaces]
        )
    else:
        return space


def check_load_space(
    load_space: HilbertSpace, /, *, point_values: bool = False
) -> bool:
    """
    Checks that the load space is of a suitable form.
    """

    if not isinstance(load_space, (Lebesgue, Sobolev)):
        raise ValueError("Load space must be a Lebesgue or Sobolev space.")

    if point_values:
        if not isinstance(load_space, Sobolev) and not load_space.order > 1:
            raise ValueError("Load space must be a Sobolev space of order > 1.")

    return True


def check_response_space(
    response_space: HilbertSpace, /, *, point_values: bool = False
) -> None:
    """
    Checks that the response space is of a suitable form.

    Args:
        response_space: The response space.
        point_values: If True, the field spaces must be Sobolev spaces
            for which point-evaluation is defined.
    """

    if not isinstance(response_space, HilbertSpaceDirectSum):
        raise ValueError("Response space must be a HilbertSpaceDirectSum.")

    if not response_space.number_of_subspaces == 4:
        raise ValueError("Response space must have 4 subspaces.")

    field_space = response_space.subspace(0)

    if not isinstance(field_space, (Lebesgue, Sobolev)):
        raise ValueError("Subspace 0 must be a Lebesgue or Sobolev space.")

    if not all(subspace == field_space for subspace in response_space.subspaces[1:3]):
        raise ValueError("Subspaces 1 and 2 must equal subspace 0.")

    angular_velocity_space = response_space.subspace(3)
    if (
        not isinstance(angular_velocity_space, EuclideanSpace)
        or not angular_velocity_space.dim == 2
    ):
        raise ValueError("Subspace 3 must be a 2D Euclidean space.")

    if point_values:
        if not isinstance(field_space, Sobolev) and not field_space.order > 1:
            raise ValueError("Subspace 0 must be a Sobolev space of order > 1.")


def tide_gauge_operator(
    response_space: HilbertSpaceDirectSum, points
) -> LinearOperator:
    """
    Maps the response fields to a vector of sea level change values at
    a discrete set of locations.

    Args:
        response_space: The response space, which is a HilbertSpaceDirectSum
            whose elements are lists of three SHGrid objects: the sea level
            change, displacement, gravitational potential change fields, and
            a numpy array for the angular velocity change.
        points: A list of (latitude, longitude) points in degrees
            where the sea level change is to be evaluated.

    Returns:
        A LinearOperator object.
    """

    check_response_space(response_space, point_values=True)

    field_space = response_space.subspace(0)
    euclidean_space = response_space.subspace(3)
    point_evaluation_operator = field_space.point_evaluation_operator(points)
    codomain = point_evaluation_operator.codomain

    return RowLinearOperator(
        [
            point_evaluation_operator,
            field_space.zero_operator(codomain=codomain),
            field_space.zero_operator(codomain=codomain),
            euclidean_space.zero_operator(codomain=codomain),
        ]
    )


def field_to_sh_coefficient_operator(
    field_space: Union[Lebesgue, Sobolev], lmax: int, lmin: int = 0
) -> LinearOperator:
    """
    Maps a scalar field to a vector of its spherical harmonic coefficients.

    The output coefficients are ordered in the following manner:

    u_{00}, u_{1-1}, u_{10}, u_{11}, u_{2-2}, u_{2-1}, u_{20}, u_{21}, u_{22}, ...

    in this case assuming lmin = 0.

    If lmax is larger than the field's lmax, the output will be padded by zeros.

    This operator can act on both Lebesgue (L2) and Sobolev spaces.

    Args:
        field_space: The domain space for the scalar field.
        lmax: The maximum spherical harmonic degree to include in the output.
        lmin: The minimum spherical harmonic degree to include in the output.
            Defaults to 0.

    Returns:
        A LinearOperator that maps an SHGrid to a NumPy vector of coefficients.
    """
    if not isinstance(field_space, (Lebesgue, Sobolev)):
        raise TypeError("field_space must be a Lebesgue or Sobolev space.")

    is_sobolev = isinstance(field_space, Sobolev)
    l2_space = field_space.underlying_space if is_sobolev else field_space

    converter = SHVectorConverter(lmax=lmax, lmin=lmin)
    codomain = EuclideanSpace(converter.vector_size)

    def mapping(u: SHGrid) -> np.ndarray:
        """L2 forward mapping: Grid -> Coefficients -> Vector"""
        ulm = l2_space.to_coefficient(u)
        return converter.to_vector(ulm.coeffs)

    def adjoint_mapping(data: np.ndarray) -> SHGrid:
        """L2 adjoint mapping: Vector -> Coefficients -> Grid"""
        coeffs = converter.from_vector(data, output_lmax=l2_space.lmax)
        ulm = SHCoeffs.from_array(
            coeffs,
            normalization=l2_space.normalization,
            csphase=l2_space.csphase,
        )
        return l2_space.from_coefficient(ulm) / l2_space.radius**2

    l2_operator = LinearOperator(
        l2_space, codomain, mapping, adjoint_mapping=adjoint_mapping
    )

    if is_sobolev:
        return LinearOperator.from_formal_adjoint(field_space, codomain, l2_operator)
    else:
        return l2_operator


def sh_coefficient_to_field_operator(
    field_space: Union[Lebesgue, Sobolev], lmax: int, lmin: int = 0
) -> LinearOperator:
    """
    Maps a vector spherical harmonic coefficients to a scalar field,
    padding by zero for absent coefficients. This is the right inverse
    of the field_to_sh_coefficient_operator.

    The input coefficients are ordered in the following manner:

    u_{00}, u_{1-1}, u_{10}, u_{11}, u_{2-2}, u_{2-1}, u_{20}, u_{21}, u_{22}, ...

    in this case assuming lmin = 0.

    This operator can map to both Lebesgue (L2) and Sobolev spaces.

    Args:
        field_space: The domain space for the scalar field.
        lmax: The maximum spherical harmonic degree to include in the output.
        lmin: The minimum spherical harmonic degree to include in the output.
            Defaults to 0.

    Returns:
        A LinearOperator that maps an SHGrid to a NumPy vector of coefficients.
    """

    if not isinstance(field_space, (Lebesgue, Sobolev)):
        raise TypeError("field_space must be a Lebesgue or Sobolev space.")

    is_sobolev = isinstance(field_space, Sobolev)
    l2_space = field_space.underlying_space if is_sobolev else field_space

    converter = SHVectorConverter(lmax=lmax, lmin=lmin)
    domain = EuclideanSpace(converter.vector_size)

    def mapping(data: np.ndarray) -> SHGrid:
        coeffs = converter.from_vector(data, output_lmax=l2_space.lmax)
        ulm = SHCoeffs.from_array(
            coeffs,
            normalization=l2_space.normalization,
            csphase=l2_space.csphase,
        )
        return l2_space.from_coefficient(ulm)

    def adjoint_mapping(u: SHGrid) -> np.ndarray:
        ulm = l2_space.to_coefficient(u)
        return converter.to_vector(ulm.coeffs) * l2_space.radius**2

    l2_operator = LinearOperator(
        domain, l2_space, mapping, adjoint_mapping=adjoint_mapping
    )

    if is_sobolev:
        return LinearOperator.from_formal_adjoint(domain, field_space, l2_operator)
    else:
        return l2_operator


def grace_operator(
    response_space: HilbertSpaceDirectSum,
    observation_degree: int,
) -> LinearOperator:
    """
    Maps the response fields to a vector of spherical harmonic coefficients
    of the gravitational potential change, for degrees  2 <= l <= observation_degree.

    The output coefficients are fully normalised and include the Condon-Shortley
    phase factor.

    Args:
        response_space: The response space, which is a HilbertSpaceDirectSum.
        observation_degree: The max degree of the SH coefficient observations.
    Returns:
        A LinearOperator object.
    """

    check_response_space(response_space, point_values=False)

    # Define the non-zero block of the operator by calling the new factory
    grav_potential_space = response_space.subspace(2)
    partial_op = field_to_sh_coefficient_operator(
        grav_potential_space, lmax=observation_degree, lmin=2
    )
    codomain = partial_op.codomain

    # Get the correct field/euclidean spaces for the zero operators
    field_space = response_space.subspace(0)
    euclidean_space = response_space.subspace(3)

    # Assemble the full block operator
    return RowLinearOperator(
        [
            field_space.zero_operator(codomain=codomain),
            field_space.zero_operator(codomain=codomain),
            partial_op,
            euclidean_space.zero_operator(codomain=codomain),
        ]
    )


def averaging_operator(
    load_space: Union[Lebesgue, Sobolev], weighting_functions: List[SHGrid]
) -> LinearOperator:
    """
    Creates an operator that computes a vector of L2 inner products.

    The action of the operator on a function `u` is to return a vector `d`
    where `d_i = <u, w_i>_L2`, with `w_i` being the i-th weighting function.
    The inner product is always the L2 inner product (integration), even if
    the operator's `load_space` is a Sobolev space.

    Args:
        load_space: The Hilbert space for the input function `u`. Must be a
            `Lebesgue` or `Sobolev` space.
        weighting_functions: A list of `SHGrid` objects, `[w_1, w_2, ...]`,
            that will be used to compute the inner products.

    Returns:
        A LinearOperator that maps from the `load_space` to an N-dimensional
        Euclidean space, where N is the number of weighting functions.
    """
    if not isinstance(load_space, (Lebesgue, Sobolev)):
        raise TypeError("load_space must be a Lebesgue or Sobolev space.")

    is_sobolev = isinstance(load_space, Sobolev)
    l2_space = load_space.underlying_space if is_sobolev else load_space

    n_weights = len(weighting_functions)
    codomain = EuclideanSpace(n_weights)

    def mapping(u: SHGrid) -> np.ndarray:
        """Forward map: computes the vector of L2 inner products."""
        results = np.zeros(n_weights)
        for i, w_i in enumerate(weighting_functions):
            results[i] = l2_space.inner_product(u, w_i)
        return results

    def adjoint_mapping(d: np.ndarray) -> SHGrid:
        """Adjoint map: computes a weighted sum of the weighting functions."""
        result_grid = l2_space.zero
        for i, w_i in enumerate(weighting_functions):
            l2_space.axpy(d[i], w_i, result_grid)
        return result_grid

    l2_operator = LinearOperator(
        l2_space, codomain, mapping, adjoint_mapping=adjoint_mapping
    )

    if is_sobolev:
        return LinearOperator.from_formal_adjoint(load_space, codomain, l2_operator)
    else:
        return l2_operator


class WahrMolenaarByranMethod(EarthModelParameters, LoveNumbers):
    """
    A class that groups together functions linked to the method of Wahr, Molenaar, & Bryan (1998)
    for estimating surface loads from GRACE data.
    """

    def __init__(
        self,
        observation_degree: int,
        /,
        *,
        earth_model_parameters: Optional[EarthModelParameters] = None,
        love_number_file: str = DATADIR + "/love_numbers/PREM_4096.dat",
    ):
        """
        Args:
            observation_degree: The maximum degree of the SH coefficient observations.
            earth_model_parameters: Parameters for the Earth model. If None,
                default parameters are used.
            love_number_file: Path to the file containing the Love numbers.
        """

        if earth_model_parameters is None:
            super().__init__()
        else:
            init_kwargs = EarthModelParameters._get_init_kwargs_from_instance(
                earth_model_parameters
            )
            super().__init__(**init_kwargs)

        self._love_number_file = love_number_file

        self._observation_degree = observation_degree

        LoveNumbers.__init__(
            self,
            self._observation_degree,
            self,
            file=self._love_number_file,
        )

    @staticmethod
    def from_finger_print(observation_degree: int, finger_print: FingerPrint):
        """
        Creates a WahrMolenaarByranMethod object from a FingerPrint object.

        Args:
            observation_degree: The maximum degree of the SH coefficient observations.
            finger_print: The FingerPrint object.

        Returns:
            A WahrMolenaarByranMethod object.
        """
        return WahrMolenaarByranMethod(
            observation_degree,
            earth_model_parameters=finger_print,
            love_number_file=finger_print.love_number_file,
        )

    @property
    def observation_degree(self) -> int:
        """The maximum degree of the SH coefficient observations."""
        return self._observation_degree

    def potential_field_to_load_operator(
        self,
        potential_space: Union[Lebesgue, Sobolev],
        load_space: Union[Lebesgue, Sobolev],
    ):
        """
        Returns as a LinearOperator that maps a gravitational potential field to an
        approximation of the causative surface load.

        Args:
            potential_space: The HilbertSpace for the potential field.
            load_space: The HilbertSpace for the load field.

        Returns:
            A LinearOperator object.
        """

        if not isinstance(potential_space, (Lebesgue, Sobolev)):
            raise TypeError("potential_space must be a Lebesgue or Sobolev space.")

        if not isinstance(load_space, (Lebesgue, Sobolev)):
            raise TypeError("load_space must be a Lebesgue or Sobolev space.")

        l2_potential_space = underlying_space(potential_space)

        def scaling_function(k: (int, int)) -> float:
            l, _ = k
            return 1 / self.k[l] if 1 < l <= self.observation_degree else 0

        l2_operator = l2_potential_space.invariant_automorphism_from_index_function(
            scaling_function
        )

        return LinearOperator.from_formal_adjoint(
            potential_space, load_space, l2_operator
        )

    def potential_coefficient_to_load_operator(
        self, load_space: Union[Lebesgue, Sobolev]
    ):
        """
        Returns as a LinearOperator the approximate mapping between a vector of gravitational potential
        coefficients and an approximation to the causative surface load.

        Args:
            load_space: The load_space: The HilbertSpace for the load field.

        Returns:
            A LinearOperator object.

        Notes:

            The input coefficients are ordered in the following manner:

            u_{2-2}, u_{2-1}, u_{20}, u_{21}, u_{22}, u_{3,-3}, u_{3-2},...
        """

        if not isinstance(load_space, (Lebesgue, Sobolev)):
            raise TypeError("load_space must be a Lebesgue or Sobolev space.")

        coeff_to_field_operator = sh_coefficient_to_field_operator(
            load_space, self.observation_degree, lmin=2
        )

        potential_field_to_load_operator = self.potential_field_to_load_operator(
            load_space, load_space
        )

        return potential_field_to_load_operator @ coeff_to_field_operator

    def potential_field_to_load_average_operator(
        self, potential_space, load_space, weighting_functions
    ):
        """
        Returns as a LinearOperator a mapping from a vector of gravitational potential
        field to averages of the causative load.

        Args:
            load_space: The HilbertSpace for the load field.
            weighting_functions: A list of weighting functions.

        Returns:
            A LinearOperator object.

        Notes:

            The input coefficients are ordered in the following manner:

            u_{2-2}, u_{2-1}, u_{20}, u_{21}, u_{22}, u_{3,-3}, u_{3-2},...
        """

        potential_field_to_load_operator = self.potential_field_to_load_operator(
            potential_space, load_space
        )
        load_to_load_averages_operator = averaging_operator(
            load_space, weighting_functions
        )

        return load_to_load_averages_operator @ potential_field_to_load_operator

    def potential_coefficient_to_load_average_operator(
        self, load_space: Union[Lebesgue, Sobolev], weighting_functions: List[SHGrid]
    ):
        """
        Returns as a LinearOperator a mapping from a vector of gravitational potential
        coefficients to averages of the causative load.

        Args:
            load_space: The HilbertSpace for the load field.
            weighting_functions: A list of weighting functions.

        Returns:
            A LinearOperator object.

        Notes:

            The input coefficients are ordered in the following manner:

            u_{2-2}, u_{2-1}, u_{20}, u_{21}, u_{22}, u_{3,-3}, u_{3-2},...
        """

        coefficient_to_load_opeator = self.potential_coefficient_to_load_operator(
            load_space
        )
        load_to_load_averages_operator = averaging_operator(
            load_space, weighting_functions
        )

        return load_to_load_averages_operator @ coefficient_to_load_opeator
