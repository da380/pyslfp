"""
Module for defining some operators related to the sea level problem.

These operators map high-dimensional function spaces (like loads or responses)
to finite-dimensional vector spaces of data or properties.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np

from pyshtools import SHCoeffs, SHGrid

from pygeoinf import HilbertSpace, LinearForm, LinearOperator, EuclideanSpace

from .finger_print import FingerPrint
from .utils import SHVectorConverter

Vector = Any

# ==================================================================== #
#                        The Common Base Class                         #
# ==================================================================== #


class FingerPrintOperator(ABC, LinearOperator):
    """
    Abstract base class for operators that map a function space related to the
    sea level problem to a finite-dimensional data or property space.
    """

    def __init_subclass__(cls, **kwargs):
        """
        This hook checks that a new subclass implements AT MOST ONE of the
        three possible adjoint-related methods.
        """
        super().__init_subclass__(**kwargs)
        implementations = {
            "adjoint": cls._adjoint_mapping is not FingerPrintOperator._adjoint_mapping,
            "dual": cls._dual_mapping is not FingerPrintOperator._dual_mapping,
            "formal_adjoint": (
                cls._formal_adjoint_mapping
                is not FingerPrintOperator._formal_adjoint_mapping
            ),
        }
        num_implemented = sum(implementations.values())
        if num_implemented > 1:
            raise TypeError(
                f"Subclass '{cls.__name__}' must implement at most one of "
                "'_adjoint_mapping', '_dual_mapping', or '_formal_adjoint_mapping'. "
                f"Found {num_implemented} implementations."
            )

    def __init__(
        self,
        fingerprint: FingerPrint,
        order: float,
        scale: float,
        domain: HilbertSpace,
    ) -> None:
        """
        Initializes the base operator.

        Args:
            fingerprint: The configured FingerPrint instance.
            order: The Sobolev order of the problem.
            scale: The Sobolev scale of the problem.
            domain: The specific domain (e.g., load or response space).
        """
        if not isinstance(fingerprint, FingerPrint):
            raise TypeError("fingerprint must be an instance of FingerPrint.")
        if not fingerprint.background_set:
            raise ValueError("fingerprint must have a background state set.")

        self._fingerprint = fingerprint
        self._order = order
        self._scale = scale

        mapping_kwargs = {}
        if self.__class__._adjoint_mapping is not FingerPrintOperator._adjoint_mapping:
            mapping_kwargs["adjoint_mapping"] = self._adjoint_mapping
        elif self.__class__._dual_mapping is not FingerPrintOperator._dual_mapping:
            mapping_kwargs["dual_mapping"] = self._dual_mapping
        elif (
            self.__class__._formal_adjoint_mapping
            is not FingerPrintOperator._formal_adjoint_mapping
        ):
            mapping_kwargs["formal_adjoint_mapping"] = self._formal_adjoint_mapping

        super().__init__(
            domain,
            self._data_space(),
            self._mapping,
            **mapping_kwargs,
        )

    @property
    def fingerprint(self) -> FingerPrint:
        """The `FingerPrint` instance associated with the operator."""
        return self._fingerprint

    @property
    def order(self) -> float:
        """The Sobolev order used to define the operator's spaces."""
        return self._order

    @property
    def scale(self) -> float:
        """The Sobolev scale used to define the operator's spaces."""
        return self._scale

    @abstractmethod
    def _data_space(self) -> HilbertSpace:
        """A subclass must implement this to return the data/property space."""
        pass

    @abstractmethod
    def _mapping(self, element: Vector) -> Vector:
        """A subclass must implement the forward mapping."""
        pass

    # --- Adjoint-related methods are abstract by default ---
    def _adjoint_mapping(self, data: Vector) -> Vector:
        raise NotImplementedError

    def _dual_mapping(self, data_dual: LinearForm) -> LinearForm:
        raise NotImplementedError

    def _formal_adjoint_mapping(self, data: Vector) -> Vector:
        raise NotImplementedError


# ==================================================================== #
#                        ObservationOperator Base                      #
# ==================================================================== #


class ObservationOperator(FingerPrintOperator):
    """
    Abstract base class for observation operators.
    Maps the RESPONSE SPACE to a finite-dimensional data space.
    """

    def __init__(self, fingerprint: FingerPrint, order: float, scale: float) -> None:
        """
        Args:
            fingerprint: The configured FingerPrint instance.
            order: The Sobolev order of the problem.
            scale: The Sobolev scale of the problem.
        """
        # Construct the specific domain for this operator type
        response_space = fingerprint.response_space(order, scale)
        # Pass all common arguments up to the base class
        super().__init__(fingerprint, order, scale, response_space)


# ==================================================================== #
#                         PropertyOperator Base                        #
# ==================================================================== #


class PropertyOperator(FingerPrintOperator):
    """
    Abstract base class for property operators.
    Maps the LOAD SPACE to a finite-dimensional property space.
    """

    def __init__(self, fingerprint: FingerPrint, order: float, scale: float) -> None:
        """
        Args:
            fingerprint: The configured FingerPrint instance.
            order: The Sobolev order of the problem.
            scale: The Sobolev scale of the problem.
        """
        # Construct the specific domain for this operator type
        load_space = fingerprint.load_space(order, scale)
        # Pass all common arguments up to the base class
        super().__init__(fingerprint, order, scale, load_space)


# ==================================================================== #
#                     Grace Observation Operator                       #
# ==================================================================== #


class GraceObservationOperator(ObservationOperator):
    """
    Observation operator for GRACE-like gravity measurements.

    Maps the response fields to a vector of spherical harmonic coefficients
    of the gravitational potential change, for degrees l >= 2.
    """

    def __init__(
        self,
        fingerprint: FingerPrint,
        order: float,
        scale: float,
        observation_degree: int,
    ) -> None:
        """
        Args:
            fingerprint: The configured FingerPrint instance.
            order: The Sobolev order of the problem.
            scale: The Sobolev scale of the problem.
            observation_degree: The max degree of the SH coefficient observations.
        """
        self._converter = SHVectorConverter(lmax=observation_degree, lmin=2)
        super().__init__(fingerprint, order, scale)

    def _data_space(self) -> EuclideanSpace:
        """The data space is a vector of the SH coefficients."""
        return EuclideanSpace(self._converter.vector_size)

    def _mapping(self, element: Vector) -> np.ndarray:
        """Maps response fields to an ordered vector of SH coefficients."""
        coeffs = self.fingerprint.expand_field(element[2]).coeffs
        return self._converter.to_vector(coeffs)

    def _formal_adjoint_mapping(self, data: np.ndarray) -> list:
        """Maps a vector of SH coefficients back to the response space."""
        coeffs = self._converter.from_vector(data, output_lmax=self.fingerprint.lmax)
        adjoint_load_lm = SHCoeffs.from_array(
            coeffs,
            lmax=self.fingerprint.lmax,
            normalization=self.fingerprint.normalization,
            csphase=self.fingerprint.csphase,
        )
        adjoint_load = self.fingerprint.expand_coefficient(adjoint_load_lm)
        zero_grid = self.fingerprint.zero_grid()
        return [zero_grid, zero_grid, adjoint_load, np.zeros(2)]


# ==================================================================== #
#                     Grace Observation Operator                       #
# ==================================================================== #


class TideGaugeObservationOperator(ObservationOperator):
    """
    Observation operator for tide gauge sea level measurements.

    Maps the response fields to a vector of sea level change values at
    a discrete set of locations.
    """

    def __init__(
        self,
        fingerprint: FingerPrint,
        order: float,
        scale: float,
        points: List[Tuple[float, float]],
    ) -> None:
        """
        Args:
            fingerprint: The configured FingerPrint instance.
            order: The Sobolev order for the response space. Must be greater than 1.
            scale: The Sobolev scale for the response space. Must be greater than 0.
            points: A list of (latitude, longitude) points in degrees
                where the sea level change is to be evaluated.
        """
        self._points = points
        super().__init__(fingerprint, order, scale)
        field_space = self.domain.subspace(0)
        self._point_evaluation_operator = field_space.point_evaluation_operator(points)

    def _data_space(self):
        """The data space is a vector of sea level change values."""
        return EuclideanSpace(len(self._points))

    def _mapping(self, element: Vector) -> np.ndarray:
        """The forward mapping to a vector of sea level change values."""
        return self._point_evaluation_operator(element[0])

    def _adjoint_mapping(self, data: np.ndarray) -> Vector:
        """The adjoint mapping from sea level change values back to the response space."""
        adjoint_sl_load = self._point_evaluation_operator.adjoint(data)
        zero_grid = self.fingerprint.zero_grid()
        return [adjoint_sl_load, zero_grid, zero_grid, np.zeros(2)]


# ====================================================== #
#                 Load averaging operator                #
# ====================================================== #


class LoadAveragingOperator(PropertyOperator):
    """
    An operator that computes a vector of weighted averages of a load
    (or similar scalar field). A list of weighting functions defined on
    the appropriate SHGrid must be provided.
    """

    def __init__(
        self,
        fingerprint: FingerPrint,
        order: float,
        scale: float,
        weighting_functions: List[SHGrid],
    ):
        """
        Args:
            fingerprint: The configured FingerPrint instance.
            order: The Sobolev order for the load space.
            scale: The Sobolev scale for the load space.
            weighting_functions: A list of 2D grids to use as weights.
        """
        for w in weighting_functions:
            if not isinstance(w, SHGrid):
                raise TypeError("weighting_functions must be a list of SHGrids.")
            if not fingerprint.check_field(w):
                raise ValueError("weighting_functions not defined on compatible grids")

        self._weighting_functions = weighting_functions
        super().__init__(fingerprint, order, scale)

    def _data_space(self):
        """The data space is a vector of the weighted averages."""
        return EuclideanSpace(len(self._weighting_functions))

    def _mapping(self, element: Vector) -> np.ndarray:
        """Maps a load to a vector of its weighted averages."""
        averages = np.zeros(len(self._weighting_functions))
        for i, w in enumerate(self._weighting_functions):
            averages[i] = self.fingerprint.integrate(element * w)
        return averages

    def _formal_adjoint_mapping(self, data: np.ndarray) -> Vector:
        """Maps a vector of weighted averages back to the load space."""
        adjoint_load = self.fingerprint.zero_grid()
        radius = self.fingerprint.mean_sea_floor_radius
        for i, w in enumerate(self._weighting_functions):
            adjoint_load += radius**2 * data[i] * w
        return adjoint_load
