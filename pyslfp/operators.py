"""
Module for defining some operators related to the sea level problem.
"""

from __future__ import annotations
from typing import Optional, List, Union, Tuple


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
    DiagonalSparseMatrixLinearOperator,
    GaussianMeasure,
    CholeskySolver,
)


from pygeoinf.symmetric_space.symmetric_space import (
    InvariantLinearAutomorphism,
    InvariantGaussianMeasure,
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
    response_space: HilbertSpaceDirectSum, points: List[Tuple[float, float]]
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


def ocean_altimetry_operator(
    finger_print: FingerPrint,
    response_space: HilbertSpaceDirectSum,
    points: List[Tuple[float, float]],
    /,
    *,
    remove_rotational_contribution: bool = True,
):
    """
    Returns as a LinearOperator the mapping from the response space for the fingerprint operator
    to the sea surface height at a given set of points.

    Args:
        finger_print: The FingerPrint object.
        response_space: The response space, for the fingerprint operator, this being
            a HilbertSpaceDirectSum whose elements take the form [SL, u, phi, omega]
        points: A list of (lat,lon) points at which the SHH is observed.
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
    response_to_ssh = sea_surface_height_operator(
        finger_print,
        response_space,
        remove_rotational_contribution=remove_rotational_contribution,
    )
    ssh_to_points = response_to_ssh.codomain.point_evaluation_operator(points)
    return ssh_to_points @ response_to_ssh


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

    def mapping(ice_thickness_change: SHGrid) -> SHGrid:
        return finger_print.direct_load_from_ice_thickness_change(ice_thickness_change)

    l2_load_space = underlying_space(load_space)

    l2_operator = LinearOperator.self_adjoint(l2_load_space, mapping)

    return LinearOperator.from_formally_self_adjoint(load_space, l2_operator)


def sea_level_change_to_load_operator(
    finger_print: FingerPrint,
    sea_level_space: Union[Lebesgue, Sobolev],
    load_space: Union[Lebesgue, Sobolev],
):
    """
    Returns a LinearOperator that maps a sea level change to a load.

    Args:
        finger_print: The FingerPrint object.
        sea_level_space: The Hilbert space for the sea level.
        load_space: The Hilbert space for the load.

    Returns:
        A LinearOperator object.
    """

    def mapping(sea_level_change: SHGrid) -> SHGrid:
        return finger_print.direct_load_from_sea_level_change(sea_level_change)

    l2_sea_level_space = underlying_space(sea_level_space)
    l2_load_space = underlying_space(load_space)

    l2_operator = LinearOperator(
        l2_sea_level_space, l2_load_space, mapping, adjoint_mapping=mapping
    )

    return LinearOperator.from_formal_adjoint(sea_level_space, load_space, l2_operator)


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
    A class implementing the method of Wahr, Molenaar, & Bryan (1998)
    for estimating surface mass loads from satellite gravimetry (e.g., GRACE/FO).

    The WMB method provides a purely spectral approach to isolating surface mass
    changes by dividing observed gravitational potential coefficients by the
    load Love numbers ($k_l$). This class provides highly optimized `LinearOperator`
    mappings between continuous physical spaces (Lebesgue/Sobolev) and the
    truncated Euclidean spaces of satellite observations.
    """

    def __init__(
        self,
        observation_degree: int,
        /,
        *,
        minimum_degree: int = 2,
        earth_model_parameters: Optional[EarthModelParameters] = None,
        love_number_file: str = str(DATADIR / "love_numbers" / "PREM_4096.dat"),
    ):
        """
        Initializes the WMBMethod instance.

        Args:
            observation_degree: The maximum spherical harmonic degree of the
                observed potential coefficients (e.g., 60 or 96 for GRACE).
            minimum_degree: The minimum spherical harmonic degree in the observed
                data. Defaults to 2 (omitting degrees 0 and 1).
            earth_model_parameters: Parameters defining the non-dimensionalization
                and basic Earth structure. Defaults to standard PREM values.
            love_number_file: Path to the data file containing the elastic Love numbers.
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
        self._minimum_degree = minimum_degree

        LoveNumbers.__init__(
            self,
            self._observation_degree,
            self,
            file=self._love_number_file,
        )

    @staticmethod
    def from_finger_print(
        finger_print: FingerPrint,
        observation_degree: int,
        /,
        *,
        minimum_degree: int = 2,
    ) -> "WMBMethod":
        """
        Alternative constructor to initialize the WMB method directly from an
        existing `FingerPrint` instance.

        Args:
            finger_print: An initialized `FingerPrint` model.
            observation_degree: The maximum observed spherical harmonic degree.
            minimum_degree: The minimum observed spherical harmonic degree. Defaults to 2.

        Returns:
            A configured `WMBMethod` instance.
        """
        return WMBMethod(
            observation_degree,
            earth_model_parameters=finger_print,
            love_number_file=finger_print.love_number_file,
            minimum_degree=minimum_degree,
        )

    @property
    def observation_degree(self) -> int:
        """The maximum spherical harmonic degree of the observations."""
        return self._observation_degree

    @property
    def minimum_degree(self) -> int:
        """The minimum spherical harmonic degree of the observations."""
        return self._minimum_degree

    @property
    def observation_dim(self) -> int:
        """The total number of observed spherical harmonic coefficients."""
        return (self.observation_degree + 1) ** 2 - self.minimum_degree**2

    # ---------------------------------------------------------#
    #             Core Euclidean Coefficient Operators         #
    # ---------------------------------------------------------#

    def load_coefficient_to_potential_coefficient_operator(self) -> LinearOperator:
        """
        Constructs a diagonal scaling operator in the truncated observation space.

        This operator maps spherical harmonic coefficients of a mass load to the
        corresponding coefficients of the gravitational potential by multiplying
        each degree $l$ by the elastic Love number $k_l$.

        Returns:
            A `DiagonalSparseMatrixLinearOperator` acting on `EuclideanSpace`.
        """
        domain = EuclideanSpace(self.observation_dim)
        scaling_factors = np.zeros(self.observation_dim)

        for l in range(self.minimum_degree, self.observation_degree + 1):
            idx_start = l**2 - self.minimum_degree**2
            idx_end = (l + 1) ** 2 - self.minimum_degree**2
            scaling_factors[idx_start:idx_end] = self.k[l]

        return DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            domain, domain, scaling_factors
        )

    def potential_coefficient_to_load_coefficient_operator(self) -> LinearOperator:
        """
        Constructs the inverse diagonal scaling operator in the observation space.

        This operator isolates the mass load coefficients from observed potential
        coefficients by applying the inverse Love number scaling ($1 / k_l$).

        Returns:
            A `DiagonalSparseMatrixLinearOperator` acting on `EuclideanSpace`.
        """
        return self.load_coefficient_to_potential_coefficient_operator().inverse

    # ---------------------------------------------------------#
    #               Bridge Spatial/Coefficient Operators       #
    # ---------------------------------------------------------#

    def load_to_potential_coefficient_operator(
        self, load_space: Union[Lebesgue, Sobolev]
    ) -> LinearOperator:
        """
        Maps a continuous surface mass load field to a truncated vector of
        observed gravitational potential coefficients.

        This operator chains the spherical harmonic expansion of the continuous
        field with the forward WMB Love number scaling ($k_l$).

        Args:
            load_space: The function space (`Lebesgue` or `Sobolev`) of the input load.

        Returns:
            A `LinearOperator` mapping from the `load_space` to `EuclideanSpace`.
        """
        if not isinstance(load_space, (Lebesgue, Sobolev)):
            raise TypeError("load_space must be a Lebesgue or Sobolev space.")

        load_to_coeffs = load_space.to_coefficient_operator(
            self.observation_degree, lmin=self.minimum_degree
        )
        scaling_operator = self.load_coefficient_to_potential_coefficient_operator()

        return scaling_operator @ load_to_coeffs

    def potential_coefficient_to_load_operator(
        self, load_space: Union[Lebesgue, Sobolev]
    ) -> LinearOperator:
        """
        Maps a vector of observed gravitational potential coefficients to an
        approximation of the causative continuous surface mass load.

        This operator applies the inverse WMB scaling ($1 / k_l$) in the coefficient
        domain, and then evaluates the resulting truncated spherical harmonic
        series onto the spatial grid defined by the `load_space`.

        Args:
            load_space: The function space (`Lebesgue` or `Sobolev`) for the output load.

        Returns:
            A `LinearOperator` mapping from `EuclideanSpace` to the `load_space`.
        """
        if not isinstance(load_space, (Lebesgue, Sobolev)):
            raise TypeError("load_space must be a Lebesgue or Sobolev space.")

        coeffs_to_load = load_space.from_coefficient_operator(
            self.observation_degree, lmin=self.minimum_degree
        )
        scaling_operator = self.potential_coefficient_to_load_coefficient_operator()

        return coeffs_to_load @ scaling_operator

    # ---------------------------------------------------------#
    #                  Composite Output Operators              #
    # ---------------------------------------------------------#

    def potential_coefficient_to_load_average_operator(
        self, load_space: Union[Lebesgue, Sobolev], weighting_functions: List[SHGrid]
    ) -> LinearOperator:
        """
        Maps a vector of observed gravitational potential coefficients directly
        to a set of regional scalar averages (e.g., basin-scale mass changes).

        This operator composes the inverse WMB scaling expansion with the spatial
        inner products defined by the provided weighting functions.

        Args:
            load_space: The intermediate function space of the load.
            weighting_functions: A list of `SHGrid` objects representing the spatial
                masks/regions to average over.

        Returns:
            A `LinearOperator` mapping from `EuclideanSpace` to a vector of averages.
        """
        coefficient_to_load_operator = self.potential_coefficient_to_load_operator(
            load_space
        )
        load_to_load_averages_operator = averaging_operator(
            load_space, weighting_functions
        )

        return load_to_load_averages_operator @ coefficient_to_load_operator

    def direct_load_to_load_operator(
        self, load_space: Union[Lebesgue, Sobolev]
    ) -> LinearOperator:
        """
        Approximates the total surface load (direct load + induced water load)
        from a given direct load using a purely spectral scaling.

        Args:
            load_space: The function space (`Lebesgue` or `Sobolev`) of the load.

        Returns:
            An `InvariantLinearAutomorphism` acting on the `load_space`.
        """
        if not isinstance(load_space, (Lebesgue, Sobolev)):
            raise TypeError("load_space must be a Lebesgue or Sobolev space.")

        def scaling_function(k: tuple[int, int]) -> float:
            l, _ = k
            return (
                -(2 * l + 1)
                * self.k[l]
                / (4 * np.pi * self.gravitational_constant * self.mean_sea_floor_radius)
                if 1 < l <= self.observation_degree
                else 0
            )

        return InvariantLinearAutomorphism.from_index_function(
            load_space, scaling_function
        )

    # ---------------------------------------------------------#
    #                     Bayesian Helpers                     #
    # ---------------------------------------------------------#

    def load_measure_to_observation_measure(
        self, load_measure: InvariantGaussianMeasure
    ) -> GaussianMeasure:
        """
        Pushes a prior Gaussian measure defined on the continuous load space
        forward into the truncated observation space.

        This propagates the spectral variances of the mass load into the
        expected variances of the observed potential coefficients by scaling
        them with the Love numbers.

        Args:
            load_measure: An `InvariantGaussianMeasure` defining the prior load.

        Returns:
            A `GaussianMeasure` acting on the Euclidean observation space.
        """
        if not isinstance(load_measure, InvariantGaussianMeasure):
            raise TypeError("load_measure must be an InvariantGaussianMeasure.")

        prior_variances = load_measure.spectral_variances
        prior_lmax = load_measure.domain.lmax
        max_mapped_degree = min(prior_lmax, self.observation_degree)

        observed_stds = np.zeros(self.observation_dim)

        for l in range(self.minimum_degree, max_mapped_degree + 1):
            in_start = l**2
            in_end = (l + 1) ** 2

            out_start = l**2 - self.minimum_degree**2
            out_end = (l + 1) ** 2 - self.minimum_degree**2

            prior_stds = np.sqrt(prior_variances[in_start:in_end])
            observed_stds[out_start:out_end] = prior_stds * abs(self.k[l])

        observation_space = EuclideanSpace(self.observation_dim)

        return GaussianMeasure.from_standard_deviations(
            observation_space, observed_stds
        )

    def bayesian_normal_operator_preconditioner(
        self,
        prior_measure: InvariantGaussianMeasure,
        data_error_measure: GaussianMeasure,
        /,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
    ) -> DiagonalSparseMatrixLinearOperator:
        """
        Constructs a computationally efficient diagonal preconditioner for solving
        the Bayesian normal equations.

        In a Bayesian inversion, solving the system $(A Q A^* + R)x = y$ can be
        expensive if the forward operator $A$ involves the full sea-level equation.
        This method approximates $A$ using the simple WMB spectral scaling to form
        an extremely fast diagonal preconditioner, accelerating iterative solvers.

        Args:
            prior_measure: The Gaussian prior on the surface load.
            data_error_measure: The Gaussian noise model for the observations.
            parallel: If True, extracts diagonal from data error covariance
                in parallel. Default is False.
            n_jobs: The number of processors to use for parallel calcualtions.
                Default is -1, which means all available cpus.

        Returns:
            A `DiagonalSparseMatrixLinearOperator` representing the inverse of
            the approximated normal operator.
        """
        mapped_prior = self.load_measure_to_observation_measure(prior_measure)

        aqa_diag = mapped_prior.covariance.extract_diagonal(
            parallel=parallel, n_jobs=n_jobs
        )
        r_diag = data_error_measure.covariance.extract_diagonal()

        normal_diag = aqa_diag + r_diag

        codomain = data_error_measure.domain
        approx_normal_op = DiagonalSparseMatrixLinearOperator.from_diagonal_values(
            codomain, codomain, normal_diag
        )

        return approx_normal_op.inverse


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


def altimetry_averaging_operator(points: List[Tuple[float, float]]) -> LinearOperator:
    """
    Creates a LinearOperator that maps a vector of altimetry observations
    to a Global Mean Sea Level (GMSL) estimate using a latitude-weighted average.

    Args:
        points: A list of (latitude, longitude) tuples representing the
                observation locations in degrees.

    Returns:
        A LinearOperator mapping from EuclideanSpace(N) to EuclideanSpace(1).
    """
    n_points = len(points)

    if n_points == 0:
        raise ValueError("The list of altimetry points cannot be empty.")

    domain = EuclideanSpace(n_points)
    codomain = EuclideanSpace(1)
    lats = np.array([p[0] for p in points])
    lats_rad = np.radians(lats)
    weights = np.cos(lats_rad)
    weights /= np.sum(weights)
    matrix = weights.reshape(1, n_points)
    return LinearOperator.from_matrix(domain, codomain, matrix)


def get_ice_sheet_masks_and_labels(fp, groupings=None):
    """
    Returns combined SHGrid masks and labels for regional groupings.

    Args:
        fp: The FingerPrint object.
        groupings: A list of lists of region names (e.g. [["ANT_A-Ap", "ANT_G-H"], ["GRL_NW"]]).
                   If None, defaults to individual regions for all available basins.
    """

    if isinstance(groupings, str):
        groupings = standard_ice_groupings(fp, scheme=groupings)

    if groupings is None:
        # Default to the full 25-parameter individual breakdown
        ant_names = [f"ANT_{name}" for name in fp.list_imbie_ant_regions()]
        grl_names = [f"GRL_{name}" for name in fp.list_mouginot_grl_regions()]
        groupings = [[name] for name in ant_names + grl_names]

    combined_masks = []
    combined_labels = []

    for group in groupings:
        if not group:
            continue

        accumulated_data = None
        reference_grid = None

        for name in group:
            # Parse the prefix to fetch from the correct Regionmask dataset
            if name.startswith("ANT_"):
                m = fp.imbie_ant_projection(name[4:], value=0.0)
            elif name.startswith("GRL_"):
                m = fp.mouginot_grl_projection(name[4:], value=0.0)
            else:
                raise ValueError(
                    f"Unknown region '{name}'. Must start with 'ANT_' or 'GRL_'."
                )

            # Accumulate the mask data arrays
            if accumulated_data is None:
                accumulated_data = m.data.copy()
                reference_grid = m
            else:
                accumulated_data += m.data

        # Create a new combined SHGrid by injecting the summed data
        combined_grid = reference_grid.copy()
        combined_grid.data = accumulated_data

        combined_masks.append(combined_grid)
        combined_labels.append(" + ".join(group))

    return combined_masks, combined_labels


def ice_sheet_averaging_operator(model_space, fp, groupings=None):
    """
    The 'Averaging' operator B: Maps Global Field -> [N x 1] Averages.
    """
    masks, _ = get_ice_sheet_masks_and_labels(fp, groupings=groupings)

    # Apply the weighting factor (1/Area) to each combined mask
    areas = [fp.integrate(m) for m in masks]
    weighted_masks = [m * (1.0 / a) if a > 0 else m for m, a in zip(masks, areas)]

    return averaging_operator(model_space, weighted_masks)


def ice_sheet_basis_operator(model_space, fp, groupings=None):
    """
    The 'Basis' operator G_tilde: Maps [N x 1] coefficients -> Global Field in H^s.
    Acts as a strict right-inverse to the averaging operator.
    """
    B = ice_sheet_averaging_operator(model_space, fp, groupings=groupings)
    M = B @ B.adjoint
    M_inv = CholeskySolver()(M)
    return B.adjoint @ M_inv


def standard_ice_groupings(fp, scheme="individual"):
    """
    Provides predefined sensible groupings of ice sheet basins.

    Args:
        fp: The FingerPrint object.
        scheme (str): The name of the grouping scheme to use. Options include:
            - 'individual': All 25 basins kept separate (Default).
            - 'ice_sheets': 2 parameters (All Antarctica, All Greenland).
            - 'macro_regions': 4 parameters (WAIS, EAIS, AP, All Greenland).
            - 'grl_focused': Greenland individual, Antarctica lumped.
            - 'ant_focused': Antarctica individual, Greenland lumped.

    Returns:
        A list of lists containing the prefixed basin names.
    """
    ant_names = [f"ANT_{name}" for name in fp.list_imbie_ant_regions()]
    grl_names = [f"GRL_{name}" for name in fp.list_mouginot_grl_regions()]

    if scheme == "individual":
        return [[name] for name in ant_names + grl_names]

    elif scheme == "ice_sheets":
        return [ant_names, grl_names]

    elif scheme == "macro_regions":
        # Standard IMBIE definitions for West Antarctica, East Antarctica, and the Peninsula
        wais = ["ANT_F-G", "ANT_G-H", "ANT_H-Hp"]
        ap = ["ANT_I-Ipp", "ANT_Ipp-J", "ANT_J-Jpp"]
        # EAIS is everything else in Antarctica
        eais = [n for n in ant_names if n not in wais and n not in ap]

        return [wais, eais, ap, grl_names]

    elif scheme == "grl_focused":
        return [ant_names] + [[name] for name in grl_names]

    elif scheme == "ant_focused":
        return [[name] for name in ant_names] + [grl_names]

    else:
        raise ValueError(f"Unknown grouping scheme: '{scheme}'")
