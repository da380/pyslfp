"""
Module for pygeoinf operators linked to the sea level problem that map the response space
(i.e., the space of solutions to the sea level equation) to an appropriate data space.
"""

from abc import ABC, abstractmethod

from typing import Any

from pygeoinf import LinearOperator, HilbertSpace, HilbertSpaceDirectSum, LinearForm
from pyslfp.finger_print import FingerPrint

Vector = Any


class ObservationOperator(ABC, LinearOperator):
    """
    Abstract base class for observation operators.
    A subclass must implement _mapping and data_space. It may optionally
    implement at most one of _adjoint_mapping, _dual_mapping, or
    _formal_adjoint_mapping for efficiency.
    """

    def __init_subclass__(cls, **kwargs):
        """
        This hook checks that a new subclass implements AT MOST ONE of the
        three possible adjoint-related methods.
        """
        super().__init_subclass__(**kwargs)

        implementations = {
            "adjoint": cls._adjoint_mapping is not ObservationOperator._adjoint_mapping,
            "dual": cls._dual_mapping is not ObservationOperator._dual_mapping,
            "formal_adjoint": (
                cls._formal_adjoint_mapping
                is not ObservationOperator._formal_adjoint_mapping
            ),
        }
        num_implemented = sum(implementations.values())

        if num_implemented > 1:
            raise TypeError(
                f"Subclass '{cls.__name__}' must implement at most one of "
                "'_adjoint_mapping', '_dual_mapping', or '_formal_adjoint_mapping'. "
                f"Found {num_implemented} implementations."
            )

    def __init__(self, fingerprint: FingerPrint, order: float, scale: float) -> None:
        if not isinstance(fingerprint, FingerPrint):
            raise TypeError("fingerprint must be an instance of FingerPrint.")

        if not fingerprint.background_set:
            raise ValueError("fingerprint must have a background state set.")

        self._fingerprint = fingerprint
        self._order = order
        self._scale = scale

        self._response_space = self.fingerprint.response_space(order, scale)

        mapping_kwargs = {}
        if self._adjoint_mapping is not ObservationOperator._adjoint_mapping:
            mapping_kwargs["adjoint_mapping"] = self._adjoint_mapping
        elif self._dual_mapping is not ObservationOperator._dual_mapping:
            mapping_kwargs["dual_mapping"] = self._dual_mapping
        elif (
            self._formal_adjoint_mapping
            is not ObservationOperator._formal_adjoint_mapping
        ):
            mapping_kwargs["formal_adjoint_mapping"] = self._formal_adjoint_mapping

        super().__init__(
            self.response_space,
            self.data_space(),
            self._mapping,
            **mapping_kwargs,
        )

    @property
    def fingerprint(self) -> FingerPrint:
        """Returns the fingerprint."""
        return self._fingerprint

    @property
    def order(self) -> float:
        """Returns the Sobolev order."""
        return self._order

    @property
    def scale(self) -> float:
        """Returns the Sobolev scale."""
        return self._scale

    @property
    def response_space(self) -> HilbertSpaceDirectSum:
        """Returns the response space."""
        return self._response_space

    @abstractmethod
    def data_space(self) -> HilbertSpace:
        """Returns the data space."""
        pass

    @abstractmethod
    def _mapping(self, response):
        """
        Implementation of the operators mapping.
        """
        pass

    @abstractmethod
    def _adjoint_mapping(self, data) -> Vector:
        """Implementation of the operators adjoint mapping."""
        raise NotImplementedError

    @abstractmethod
    def _dual_mapping(self, data_dual) -> LinearForm:
        """Implementation of the operators dual mapping."""
        raise NotImplementedError

    @abstractmethod
    def _formal_adjoint_mapping(self, data) -> Vector:
        """Implementation of the operators formal adjoint mapping."""
        raise NotImplementedError
