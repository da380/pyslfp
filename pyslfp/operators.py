"""
Module for pygeoinf operators linked to the sea level problem.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union

import numpy as np
from pyshtools import SHCoeffs, SHGrid

import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev
from pygeoinf import LinearForm

from pyslfp.physical_parameters import EarthModelParameters
from pyslfp.finger_print import FingerPrint


class SeaLevelOperator(inf.LinearOperator):
    """
    Maps a direct surface load to the full sea level response.

    This class wraps the FingerPrint solver as a pygeoinf LinearOperator. It
    represents the core forward model, calculating sea level change, vertical
    displacement, gravity change, and rotational perturbations from a given load.
    """

    def __init__(
        self,
        order: float,
        scale: float,
        /,
        *,
        fingerprint: Optional[FingerPrint] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-6,
    ) -> None:
        """
        Args:
            order: The Sobolev order for the domain space of the operator. Must be > 1.
            scale: The Sobolev scale for the domain space. Must be > 0.
            fingerprint: An instance of the FingerPrint class. If None, a default
                instance is created and configured.
            rotational_feedbacks: If True, rotational effects are included.
            rtol: Relative tolerance for the iterative solver in FingerPrint.
        """
        if order <= 1:
            raise ValueError("Sobolev order must be greater than 1.")
        if scale <= 0:
            raise ValueError("Sobolev scale must be greater than 0.")

        self._rotational_feedbacks = rotational_feedbacks
        self._rtol = rtol

        if fingerprint is None:
            self._fingerprint = FingerPrint(
                earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation()
            )
            self._fingerprint.set_state_from_ice_ng()
        else:
            if not fingerprint.background_set:
                raise ValueError(
                    "The provided FingerPrint instance must have its background state set."
                )
            self._fingerprint = fingerprint

        domain = Sobolev(
            self.fingerprint.lmax,
            order,
            scale,
            radius=self.fingerprint.mean_sea_floor_radius,
            grid=self.fingerprint.grid,
        )
        response_space = Sobolev(
            self.fingerprint.lmax,
            order + 1,
            scale,
            radius=self.fingerprint.mean_sea_floor_radius,
            grid=self.fingerprint.grid,
        )
        codomain = inf.HilbertSpaceDirectSum(
            [response_space, response_space, response_space, inf.EuclideanSpace(2)]
        )

        super().__init__(
            domain,
            codomain,
            self._mapping,
            formal_adjoint_mapping=self._formal_adjoint_mapping,
        )

    @property
    def fingerprint(self) -> FingerPrint:
        """Returns the stored FingerPrint instance."""
        return self._fingerprint

    def _mapping(self, direct_load: SHGrid) -> List[Union[SHGrid, np.ndarray]]:
        """The forward mapping from a load to the sea level response fields."""
        (
            sea_level_change,
            vertical_displacement,
            gravity_potential_change,
            angular_velocity_change,
        ) = self.fingerprint(
            direct_load=direct_load,
            rotational_feedbacks=self._rotational_feedbacks,
            rtol=self._rtol,
        )

        if self._rotational_feedbacks:
            gravitational_potential_change = self.fingerprint.gravity_potential_change_to_gravitational_potential_change(
                gravity_potential_change, angular_velocity_change
            )
        else:
            gravitational_potential_change = gravity_potential_change

        return [
            sea_level_change,
            vertical_displacement,
            gravitational_potential_change,
            angular_velocity_change,
        ]

    def _formal_adjoint_mapping(
        self, response_fields: List[Union[SHGrid, np.ndarray]]
    ) -> SHGrid:
        """The formal adjoint mapping from response fields to the adjoint load."""
        g = self.fingerprint.gravitational_acceleration
        zeta_d = response_fields[0]
        zeta_u_d = -1 * response_fields[1]
        zeta_phi_d = -g * response_fields[2]
        angular_momentum_d = response_fields[3]

        if self._rotational_feedbacks:
            kk_d = -g * (
                angular_momentum_d
                + self.fingerprint.adjoint_angular_momentum_change_from_adjoint_gravitational_potential_load(
                    response_fields[2]
                )
            )
        else:
            kk_d = np.zeros(2)

        # Solve the adjoint problem.
        adjoint_sea_level, _, _, _ = self.fingerprint(
            direct_load=zeta_d,
            displacement_load=zeta_u_d,
            gravitational_potential_load=zeta_phi_d,
            angular_momentum_change=kk_d,
            rotational_feedbacks=self._rotational_feedbacks,
            rtol=self._rtol,
        )
        return adjoint_sea_level


class ObservationOperator(ABC, inf.LinearOperator):
    """
    Abstract base class for observation operators.

    These operators map the full physical response (from SeaLevelOperator)
    to a specific space of observations (e.g., GRACE coefficients, tide gauges).
    """

    def __init__(self, sea_level_operator: SeaLevelOperator) -> None:
        """
        Args:
            sea_level_operator: An instance of the SeaLevelOperator class.
        """
        if not isinstance(sea_level_operator, SeaLevelOperator):
            raise TypeError(
                "sea_level_operator must be an instance of SeaLevelOperator."
            )
        self.sea_level_operator = sea_level_operator
        operator = self._operator()
        super().__init__(
            operator.domain,
            operator.codomain,
            operator,
            adjoint_mapping=operator.adjoint,
        )

    @abstractmethod
    def _operator(self) -> inf.LinearOperator:
        """
        Must return a LinearOperator that maps response fields to the data space.
        """
        pass

    @property
    def forward_operator(self) -> inf.LinearOperator:
        """
        Returns the full forward operator (SeaLevelOperator composed with this
        ObservationOperator).
        """
        return self @ self.sea_level_operator


class GraceObservationOperator(ObservationOperator):
    """
    Observation operator for GRACE-like gravity measurements. 🛰️

    Maps the response fields to a vector of spherical harmonic coefficients
    of the gravitational potential change.
    """

    def __init__(
        self, sea_level_operator: SeaLevelOperator, observation_degree: int
    ) -> None:
        """
        Args:
            sea_level_operator: An instance of the SeaLevelOperator.
            observation_degree: The max degree of the spherical harmonic observations.
        """
        self._observation_degree = observation_degree
        self._data_size = (self._observation_degree + 1) ** 2 - 4  # Excludes l=0,1
        self._fingerprint = sea_level_operator.fingerprint
        super().__init__(sea_level_operator)

    def _operator(self) -> inf.LinearOperator:
        """Returns the LinearOperator for this observation type."""
        domain = self.sea_level_operator.codomain
        codomain = inf.EuclideanSpace(self._data_size)
        return inf.LinearOperator(
            domain, codomain, self._mapping, formal_adjoint_mapping=self._formal_adjoint
        )

    def _mapping(self, response_fields: List[Union[SHGrid, np.ndarray]]) -> np.ndarray:
        """Maps response fields to an ordered vector of SH coefficients."""
        gravitational_potential_change = response_fields[2]
        return self._to_ordered_sh_coefficients(gravitational_potential_change)

    def _formal_adjoint(
        self, gravitational_sh_coeffs: np.ndarray
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """Maps an ordered vector of SH coefficients to the adjoint loads."""
        gravitational_potential_change = self._from_ordered_sh_coefficients(
            gravitational_sh_coeffs
        )
        zero_grid = self._fingerprint.zero_grid()
        return (zero_grid, zero_grid, gravitational_potential_change, np.zeros(2))

    def _to_ordered_sh_coefficients(self, grid: SHGrid) -> np.ndarray:
        """Converts a grid to an ordered vector of SH coefficients (l>=2)."""
        coeffs = self._fingerprint._expand_field(grid).coeffs
        vec = np.zeros(self._data_size)
        for l in range(2, self._observation_degree + 1):
            vec[((l) ** 2) - 4 : ((l + 1) ** 2) - 4] = np.concatenate(
                (coeffs[1, l, 1 : l + 1][::-1], coeffs[0, l, 0 : l + 1])
            )
        return vec

    def _from_ordered_sh_coefficients(self, vec: np.ndarray) -> SHGrid:
        """Converts an ordered vector of SH coefficients (l>=2) to a grid."""
        lmax = self._fingerprint.lmax
        coeffs = np.zeros((2, lmax + 1, lmax + 1))
        for l in range(2, self._observation_degree + 1):
            coeffs[1, l, 1 : l + 1] = vec[(l**2) - 4 : (l**2) - 4 + l][::-1]
            coeffs[0, l, 0 : l + 1] = vec[(l**2) - 4 + l : ((l + 1) ** 2) - 4]
        return self._fingerprint._expand_coefficient(
            SHCoeffs.from_array(coeffs, normalization=self._fingerprint.normalization)
        )


class TideGaugeObservationOperator(ObservationOperator):
    """
    Observation operator for tide gauge sea level measurements. 🌊

    Maps the response fields to a vector of sea level change values at
    a discrete set of locations.
    """

    def __init__(
        self,
        sea_level_operator: SeaLevelOperator,
        tide_gauge_locations: List[Tuple[float, float]],
    ) -> None:
        """
        Args:
            sea_level_operator: An instance of the SeaLevelOperator.
            tide_gauge_locations: A list of (latitude, longitude) points in degrees
                where the sea level change is to be evaluated.
        """
        self._fingerprint = sea_level_operator.fingerprint
        self._sl_space = sea_level_operator.codomain.subspaces[0]
        self._point_evaluation_operator = self._sl_space.point_evaluation_operator(
            tide_gauge_locations
        )
        super().__init__(sea_level_operator)

    def _operator(self) -> inf.LinearOperator:
        """Returns a LinearOperator that maps response fields to tide gauge measurements."""
        domain = self.sea_level_operator.codomain
        codomain = self._point_evaluation_operator.codomain
        return inf.LinearOperator(
            domain, codomain, self._mapping, adjoint_mapping=self._adjoint_mapping
        )

    def _mapping(self, response_fields: List[Union[SHGrid, np.ndarray]]) -> np.ndarray:
        """The forward mapping to a vector of sea level change values."""
        sea_level_change = response_fields[0]
        return self._point_evaluation_operator(sea_level_change)

    def _adjoint_mapping(
        self, tide_gauge_measurements: np.ndarray
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """The adjoint mapping from tide gauge measurements back to the response space."""
        zero_grid = self._fingerprint.zero_grid()
        adjoint_sea_level_load = self._point_evaluation_operator.adjoint(
            tide_gauge_measurements
        )
        return (adjoint_sea_level_load, zero_grid, zero_grid, np.zeros(2))


class AveragingOperator(inf.LinearOperator):
    """
    An operator that computes a vector of weighted averages of a field.
    """

    def __init__(
        self,
        space: Sobolev,
        /,
        *,
        weighting_functions: Optional[List[SHGrid]] = None,
        weighting_components: Optional[List[SHCoeffs]] = None,
        fingerprint: Optional[FingerPrint] = None,
    ) -> None:
        """
        Args:
            space: The Sobolev space in which the operator acts.
            weighting_functions: A list of 2D grids to use as weights.
            weighting_components: A list of SH coefficients to use as weights.
            fingerprint: An instance of the FingerPrint class.
        """
        self._space = space

        if weighting_functions is None and weighting_components is None:
            raise ValueError(
                "Either weighting functions or components must be provided."
            )

        if weighting_functions is not None:
            self._weighting_functions = weighting_functions
            self._weighting_components = [
                space.to_components(wf) for wf in weighting_functions
            ]
        else:
            self._weighting_components = weighting_components
            self._weighting_functions = [
                space.from_components(wc) for wc in weighting_components
            ]

        if fingerprint is None:
            self._fingerprint = FingerPrint(
                lmax=space.lmax,
                earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
            )
            self._fingerprint.set_state_from_ice_ng()
        else:
            self._fingerprint = fingerprint

        self._averages_size = len(self._weighting_functions)
        self._averages_space = inf.EuclideanSpace(self._averages_size)

        super().__init__(
            self._space,
            self._averages_space,
            self._mapping,
            dual_mapping=self._dual_mapping,
        )

    @property
    def weighting_functions(self) -> List[SHGrid]:
        """Returns the list of weighting functions (grids)."""
        return self._weighting_functions

    @property
    def weighting_components(self) -> List[SHCoeffs]:
        """Returns the list of weighting functions (SH coefficients)."""
        return self._weighting_components

    def _mapping(self, field: SHGrid) -> np.ndarray:
        """Maps a field to a vector of its weighted averages."""
        averages = np.zeros(self._averages_size)
        for i, w in enumerate(self._weighting_functions):
            averages[i] = self._fingerprint.integrate(field * w)
        return averages

    def _dual_mapping(self, ap: LinearForm) -> LinearForm:
        """The dual mapping."""
        cap = self.codomain.dual.to_components(ap) * self._space.radius**2
        czp = sum([wi * ai for wi, ai in zip(self._weighting_components, cap)])
        return inf.LinearForm(self.domain, components=czp)


class WahrOperator(inf.LinearOperator):
    """
    Applies the Wahr approximation to infer a load average from gravity data.

    This implements a simplified inversion that relates gravitational potential
    coefficients directly to surface mass, scaled by Love numbers. It does not
    account for the full sea level equation.

    NOTE: This operator is incomplete. The adjoint mapping is missing, and the
    forward mapping is implemented inefficiently with nested loops.
    """

    def __init__(
        self,
        observation_degree: int,
        weighting_components: List[np.ndarray],
        love_numbers: np.ndarray,
        radius: float,
    ) -> None:
        """
        Args:
            observation_degree: Max degree of the gravity observation.
            weighting_components: List of weighting functions (as SH coefficient vectors).
            love_numbers: Array of gravitational Love numbers `k`.
            radius: The radius of the sphere.
        """
        self._weighting_components = weighting_components
        self._property_size = len(self._weighting_components)
        self._property_space = inf.EuclideanSpace(self._property_size)

        self._observation_degree = observation_degree
        self._data_size = (self._observation_degree + 1) ** 2 - 4
        self._data_space = inf.EuclideanSpace(self._data_size)

        self._love_numbers = love_numbers
        self._radius = radius

        # The adjoint mapping is missing and should be added here.
        super().__init__(self._data_space, self._property_space, self._mapping)

    def _mapping(self, phi: np.ndarray) -> np.ndarray:
        """
        The forward mapping.

        NOTE: This implementation is very inefficient due to nested Python loops.
        It should be vectorized for any practical application.
        """
        k = self._love_numbers
        b = self._radius
        w = np.zeros(self._property_size)
        for i in range(self._property_size):
            for l in range(2, self._observation_degree + 1):
                for m in range(-1 * l, l + 1):
                    # This indexing assumes the same vectorization as GraceObservationOperator
                    vec_index = (l**2) - 4 + m + l
                    w[i] += (
                        b**2
                        * (1 / k[l])
                        * phi[vec_index]
                        * self._weighting_components[i][vec_index]
                    )

        return w
