"""
Module for pygeoinf operators linked to the sea level problem. 
"""

import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev

from pyslfp.physical_parameters import EarthModelParameters
from pyslfp.finger_print import FingerPrint


class SeaLevelOperator(inf.LinearOperator):
    """
    The mapping from a direct load to the sea level change
    as a pygeoinf LinearOpeartor.
    """

    def __init__(self, order, scale, /, *, fingerprint=None):
        """
        Args:
            fingerprint (FingerPrint): An instance of the FingerPrint class that
            must have its background state set. Default is None, in which case
            an instance is set internally using the default options.
        """

        if order <= 1:
            raise ValueError("Sobolev order must be greater than 1.")

        if scale <= 0:
            raise ValueError("Sobolev scale must be greater than 0")

        if fingerprint is None:
            self._fingerprint = FingerPrint(
                earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation()
            )
            self._fingerprint.set_state_from_ice_ng()
        else:
            if not fingerprint.background_set:
                raise ValueError("fingerprint must have its background state set.")
            self._fingerprint = fingerprint

        domain = Sobolev(
            self._fingerprint.lmax,
            order,
            scale,
            radius=self._fingerprint.mean_sea_floor_radius,
        )

        codomain = Sobolev(
            self._fingerprint.lmax,
            order + 1,
            scale,
            radius=self._fingerprint.mean_sea_floor_radius,
        )

        operator = inf.LinearOperator.from_formal_adjoint(
            domain, codomain, self._mapping, self._mapping
        )

        super().__init__(domain, codomain, operator, adjoint_mapping=operator.adjoint)

    def _mapping(self, direct_load):
        return self._fingerprint(direct_load=direct_load)[0]
