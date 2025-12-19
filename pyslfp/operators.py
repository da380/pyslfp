"""
Module for defining some operators related to the sea level problem.
"""

from __future__ import annotations
from typing import Optional, List, Union


import numpy as np

import pyshtools as pysh
from pyshtools import SHGrid

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
    partial_op = grav_potential_space.to_coefficient_operator(
        observation_degree, lmin=2
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


def sea_surface_height_operator(
    finger_print: FingerPrint,
    response_space: HilbertSpaceDirectSum,
    /,
    *,
    remove_rotational_contribution: bool = True,
):
    """
    Returns as a LinearOperator the mapping from the response space for the fingerprint operator
    to the sea surface height.

    Args:
        finger_print: The FingerPrint object.
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

    ocean_projection = ocean_projection_operator(finger_print, codomain)

    def mapping(response):
        sea_level_change, displacement, _, angular_velocity_change = response
        sea_surface_height_change = finger_print.sea_surface_height_change(
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
                -finger_print.angular_momentum_change_from_potential(
                    projected_sea_surface_height_change
                )
                / finger_print.gravitational_acceleration
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


def spatial_mutliplication_operator(
    projection_field: SHGrid,
    load_space: Union[Lebesgue, Sobolev],
):
    """
    Returns a linear opeator that multiplies a load by a projection field.

    Args:
        projection_field: The projection field.
        load_space: The Hilbert space for the load.

    Returns:
        A LinearOperator object.
    """

    def mapping(load: SHGrid) -> SHGrid:
        return projection_field * load

    l2_load_space = underlying_space(load_space)
    l2_operator = LinearOperator.self_adjoint(l2_load_space, mapping)
    return LinearOperator.from_formally_self_adjoint(load_space, l2_operator)


def ice_projection_operator(
    finger_print: FingerPrint,
    load_space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
):
    """
    Returns a LinearOpeator multiplies a load by a function that is one
    over the background ice sheets and zero elsewhere.

    Args:
        finger_print: The FingerPrint object.
        load_space: The Hilbert space for the load.
        exclude_ice_shelves: If True, the function is set to zero in ice-shelved regions.

    Returns:
        A LinearOperator object.

    """

    projection_field = finger_print.ice_projection(
        value=0, exclude_ice_shelves=exclude_ice_shelves
    )
    return spatial_mutliplication_operator(projection_field, load_space)


def ocean_projection_operator(
    finger_print: FingerPrint,
    load_space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice_shelves: bool = False,
):
    """
    Returns a LinearOpeator multiplies a load by a function that is one
    over the background oceans and zero elsewhere.

    Args:
        finger_print: The FingerPrint object.
        load_space: The Hilbert space for the load.
        exclude_ice_shelves: If True, the function is set to zero in ice-shelved regions.

    Returns:
        A LinearOperator object.

    """

    projection_field = finger_print.ocean_projection(
        value=0, exclude_ice_shelves=exclude_ice_shelves
    )
    return spatial_mutliplication_operator(projection_field, load_space)


def land_projection_operator(
    finger_print: FingerPrint,
    load_space: Union[Lebesgue, Sobolev],
    /,
    *,
    exclude_ice: bool = True,
):
    """
    Returns a LinearOpeator multiplies a load by a function that is one
    over the background land and zero elsewhere.

    Args:
        finger_print: The FingerPrint object.
        load_space: The Hilbert space for the load.
        exclude_ice: If True, the function is set to zero in ice-covered regions.

    Returns:
        A LinearOperator object.

    """

    projection_field = finger_print.land_projection(value=0, exclude_ice=exclude_ice)
    return spatial_mutliplication_operator(projection_field, load_space)


def ice_thickness_change_to_load_operator(
    finger_print: FingerPrint,
    load_space: Union[Lebesgue, Sobolev],
):
    """
    Returns a LinearOperator that maps the ice thickness change to a load.

    Args:
        finger_print: The FingerPrint object.
        load_space: The Hilbert space for the load.

    Returns:
        A LinearOperator object.
    """

    def mapping(ice_thicknes_change: SHGrid) -> SHGrid:
        return finger_print.direct_load_from_ice_thickness_change(ice_thicknes_change)

    l2_load_space = underlying_space(load_space)

    l2_operator = LinearOperator.self_adjoint(l2_load_space, mapping)

    return LinearOperator.from_formally_self_adjoint(load_space, l2_operator)


def sea_level_change_to_load_operator(
    finger_print: FingerPrint,
    load_space: Union[Lebesgue, Sobolev],
):
    """
    Returns a LinearOperator that maps a sea level change to a load.

    Args:
        finger_print: The FingerPrint object.
        load_space: The Hilbert space for the load.

    Returns:
        A LinearOperator object.
    """

    def mapping(ice_thicknes_change: SHGrid) -> SHGrid:
        return finger_print.direct_load_from_sea_level_change(ice_thicknes_change)

    l2_load_space = underlying_space(load_space)

    l2_operator = LinearOperator.self_adjoint(l2_load_space, mapping)

    return LinearOperator.from_formally_self_adjoint(load_space, l2_operator)


def density_change_to_load_operator(
    finger_print: FingerPrint,
    load_space: Union[Lebesgue, Sobolev],
):
    """
    Returns a LinearOperator that maps a density change to a load.

    Args:
        finger_print: The FingerPrint object.
        load_space: The Hilbert space for the load.

    Returns:
        A LinearOperator object.
    """

    def mapping(density_change: SHGrid) -> SHGrid:
        return finger_print.direct_load_from_density_change(density_change)

    l2_load_space = underlying_space(load_space)

    l2_operator = LinearOperator.self_adjoint(l2_load_space, mapping)

    return LinearOperator.from_formally_self_adjoint(load_space, l2_operator)


def remove_ocean_average_operator(
    finger_print: FingerPrint, load_space: Union[Lebesgue, Sobolev]
):
    """
    Returns a LinearOperator that takes a scalar function defined on the Earth's surface, and
    outputs this function adjusted so that its integral over the oceans is zero
    """

    l2_load_space = underlying_space(load_space)

    ocean_function = finger_print.ocean_function
    ocean_area = finger_print.ocean_area

    def mapping(load):
        ocean_average = finger_print.integrate(ocean_function * load) / ocean_area
        new_load = load.copy()
        new_load.data -= ocean_average
        return new_load

    def adjoint_mapping(load):
        average = finger_print.integrate(load)
        return load - average * ocean_function / ocean_area

    l2_operator = LinearOperator(
        l2_load_space, l2_load_space, mapping, adjoint_mapping=adjoint_mapping
    )

    return LinearOperator.from_formal_adjoint(load_space, load_space, l2_operator)


class WMBMethod(EarthModelParameters, LoveNumbers):
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
        return WMBMethod(
            observation_degree,
            earth_model_parameters=finger_print,
            love_number_file=finger_print.love_number_file,
        )

    @property
    def observation_degree(self) -> int:
        """The maximum degree of the SH coefficient observations."""
        return self._observation_degree

    def direct_load_to_load_operator(self, load_space: Union[Lebesgue, Sobolev]):
        """
        Returns a LinearOperator that maps a direct load to an approximation
        of the total load.

        Args:
            load_space: The HilbertSpace for the load field.

        Returns:
            A LinearOperator object.
        """

        if not isinstance(load_space, (Lebesgue, Sobolev)):
            raise TypeError("load_space must be a Lebesgue or Sobolev space.")

        l2_load_space = underlying_space(load_space)

        def scaling_function(k: (int, int)) -> float:
            l, _ = k
            return (
                -(2 * l + 1)
                * self.k[l]
                / (4 * np.pi * self.gravitational_constant * self.mean_sea_floor_radius)
                if 1 < l <= self.observation_degree
                else 0
            )

        l2_operator = l2_load_space.invariant_automorphism_from_index_function(
            scaling_function
        )

        return LinearOperator.from_formally_self_adjoint(load_space, l2_operator)

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
        coeff_to_field_operator = load_space.from_coefficient_operator(
            self.observation_degree, lmin=2
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


def remove_degrees_from_pyshtools_coeffs(coeffs, degrees_to_remove: list[int]):
    """Remove specified spherical harmonic degrees from pyshtools coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
        Pyshtools coefficients array with shape (2, lmax+1, lmax+1).
    degrees_to_remove : list[int]
        List of spherical harmonic degrees to remove.

    Returns
    -------
    np.ndarray
        Modified coefficients with specified degrees set to zero.
    """
    # Create a copy of the coefficients
    modified_coeffs = coeffs.copy()

    # Set specified degrees to zero
    for degree in degrees_to_remove:
        if degree < coeffs.shape[1]:  # Check if degree exists in the array
            modified_coeffs[0, degree, :] = 0.0  # Cosine coefficients
            modified_coeffs[1, degree, :] = 0.0  # Sine coefficients

    return modified_coeffs


def remove_degrees_from_shgrid(grid, degrees_to_remove: list[int]):
    """Remove specified degrees from a pyshtools SHGrid object.

    Parameters
    ----------
    grid : pyshtools.SHGrid
        The input grid from which to remove degrees.
    degrees_to_remove : list[int]
        List of spherical harmonic degrees to remove.

    Returns
    -------
    pyshtools.SHGrid
        Modified grid with specified degrees set to zero.
    """
    # Convert grid to coefficients
    coeffs = grid.expand()

    # Remove specified degrees
    modified_coeffs = remove_degrees_from_pyshtools_coeffs(
        coeffs.coeffs, degrees_to_remove
    )

    # Create new coefficients object
    modified_shcoeffs = pysh.SHCoeffs.from_array(modified_coeffs)

    # Convert back to grid
    modified_grid = modified_shcoeffs.expand(grid=grid.grid, extend=grid.extend)

    return modified_grid
