"""
Module for pygeoinf operators linked to the sea level problem. 
"""

import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev

from pyslfp.physical_parameters import EarthModelParameters
from pyslfp.finger_print import FingerPrint

import numpy as np
from pyshtools import SHCoeffs, SHGrid

from abc import ABC, abstractmethod

class SeaLevelOperator0(inf.LinearOperator):
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
    
class SeaLevelOperator(inf.LinearOperator):
    """
    The mapping from a direct load to sea level change, vertical deisplacement, gravity change, and rotational perturbation
    as a pygeoinf LinearOpeartor.
    """

    def __init__(self, order, scale, /, *, fingerprint=None, rotational_feedbacks=False, rtol=1e-6):
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
        
        self._rotational_feedbacks = rotational_feedbacks
        self._rtol = rtol

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

        response_space = Sobolev(
            self._fingerprint.lmax,
            order + 1,
            scale,
            radius=self._fingerprint.mean_sea_floor_radius,
        )

        codomain = inf.HilbertSpaceDirectSum(
            [response_space,
             response_space, 
             response_space, 
             inf.EuclideanSpace(2)]
        )

        operator = inf.LinearOperator.from_formal_adjoint(
            domain, codomain, self._mapping, self._formal_adjoint
        )

        super().__init__(domain, codomain, operator, adjoint_mapping=operator.adjoint)

    def _mapping(self, direct_load):
         
        sea_level_change, vertical_displacement, gravity_potential_change, angular_velocity_change = self._fingerprint(direct_load=direct_load, rotational_feedbacks=self._rotational_feedbacks, rtol=self._rtol)
        
        if self._rotational_feedbacks:
            gravitational_potential_change = (
                self._fingerprint.gravity_potential_change_to_gravitational_potential_change(
                    gravity_potential_change, angular_velocity_change
                )
            )
        else:
            gravitational_potential_change = gravity_potential_change

        return (
            sea_level_change,
            vertical_displacement,
            gravitational_potential_change,
            angular_velocity_change
        )
    
    def _formal_adjoint(self, response_fields):

            g = self._fingerprint.gravitational_acceleration

            zeta_d = response_fields[0]
            zeta_u_d = -1 * response_fields[1]
            zeta_phi_d = -g * response_fields[2]
            if self._rotational_feedbacks:
                kk_d = -g * (response_fields[3] - self._fingerprint.adjoint_angular_momentum_change_from_adjoint_gravitational_potential_load(response_fields[2]))
            else:
                kk_d = np.zeros(2)
            
            # Solve the adjoint problem.
            return self._fingerprint(
                direct_load = zeta_d, 
                displacement_load = zeta_u_d,
                gravitational_potential_load = zeta_phi_d,
                angular_momentum_change = kk_d,
                rotational_feedbacks=self._rotational_feedbacks, 
                rtol=self._rtol
            )[0]
    
class ObservationOperator(ABC, inf.LinearOperator):
    """
    Abstract base class for observation operators.
    """

    def __init__(self, sea_level_operator):
        """
        Args:
            sea_level_operator (SeaLevelOperator): An instance of the SeaLevelOperator class.
        """
        if not isinstance(sea_level_operator, SeaLevelOperator):
            raise TypeError("sea_level_operator must be an instance of SeaLevelOperator.")
        
        self.sea_level_operator = sea_level_operator
        
        domain = sea_level_operator.codomain
        codomain = self._data_space()

        operator = inf.LinearOperator.from_formal_adjoint(
            domain, codomain, self._mapping, self._formal_adjoint
        )

        super().__init__(domain, codomain, operator, adjoint_mapping=operator.adjoint)
        
    @abstractmethod
    def _mapping(self, response_fields):
        """
        Maps the response fields to the data space.
        """
        pass
    
    @abstractmethod
    def _formal_adjoint(self, data):
        """
        Maps from the data space to the response fields.
        """
        pass

    @abstractmethod
    def _data_space(self):
        """
        Returns the data space of the observation operator.
        """
        pass

    @property
    def forward_operator(self):
        """
        Returns the full forward operator; the sea level operator composed with the observation operator.
        """
        return self @ self.sea_level_operator
    
class GraceObservationOperator(ObservationOperator):
    """
    The mapping from a set of four response fields to a vector of spherical harmonic coefficients of gravitational potential change
    as a pygeoinf LinearOperator.
    """

    def __init__(self, sea_level_operator, observation_degree):
        """
        Args:
            sea_level_operator (SeaLevelOperator): An instance of the SeaLevelOperator class.
            observation_degree (int): The degree of the spherical harmonics used for the observations.
            rotational_feedbacks (bool): Whether to include rotational feedbacks in the operator.
        """
        self._observation_degree = observation_degree
        self._data_size = (self._observation_degree+1)**2 - 4
        self._fingerprint = sea_level_operator._fingerprint
        self._rotational_feedbacks = sea_level_operator._rotational_feedbacks
        super().__init__(sea_level_operator)


    def _mapping(self, response_fields):

        # Retrieve the relevant fields from the response.
        gravity_potential_change = response_fields[2]
        angular_velocity_change = response_fields[3]

        # Get the gravitational potential change. todo!!
        """if self._rotational_feedbacks:
            gravitational_potential_change = (
                self._fingerprint.gravity_potential_change_to_gravitational_potential_change(
                    gravity_potential_change, angular_velocity_change
                )
            )
        else:
            gravitational_potential_change = gravity_potential_change"""
        
        gravitational_potential_change = gravity_potential_change

        # Convert the gravitational potential change to spherical harmonic coefficients.
        gravitational_sh_coeffs = self._to_ordered_sh_coefficients(gravitational_potential_change)

        # Return the spherical harmonic coefficients.
        return gravitational_sh_coeffs
    
    def _formal_adjoint(self, gravitational_sh_coeffs):

        # Convert the spherical harmonic coefficients to a grid.
        gravitational_potential_change = self._from_ordered_sh_coefficients(gravitational_sh_coeffs)

        # Convert the gravitational potential change to gravity potential change.
        """if self._rotational_feedbacks:
            gravity_potential_change = self._fingerprint.gravitational_potential_change_to_gravity_potential_change(gravitational_potential_change, np.zeros(2))
        else:
            gravity_potential_change = gravitational_potential_change"""
        
        gravity_potential_change = gravitational_potential_change

        # Create a zero grid for the dual loads.
        zero_grid = self._fingerprint.zero_grid()

        return (
            zero_grid,
            zero_grid,
            gravity_potential_change,
            np.zeros(2)
        )
    
    def _data_space(self):
        """Returns the data space of the observation operator."""
        return inf.EuclideanSpace(self._data_size)
    
    def _to_ordered_sh_coefficients(self, grid):
        """Converts a grid to ordered spherical harmonic coefficients."""
        coeffs = self._fingerprint._expand_field(grid).coeffs
        vec = np.zeros(self._data_size)       
        for l in range(2,self._observation_degree+1):
            vec[((l)**2)-4:((l+1)**2)-4] = np.concatenate((coeffs[1,l,1:l+1][::-1],coeffs[0,l,0:l+1]))
        return vec    
    
    def _from_ordered_sh_coefficients(self, vec):
        """Converts ordered spherical harmonic coefficients to a grid."""
        lmax = self._fingerprint.lmax
        coeffs = np.zeros((2, lmax + 1, lmax + 1))
        for l in range(2, self._observation_degree + 1):
            coeffs[1, l, 1:l+1] = vec[(l**2)-4:(l**2)-4+l][::-1]
            coeffs[0, l, 0:l+1] = vec[(l**2)-4+l:((l+1)**2)-4]
        return self._fingerprint._expand_coefficient(SHCoeffs.from_array(coeffs, normalization=self._fingerprint.normalization))  


class GraceObservationOperator2(inf.LinearOperator):
    """
    The mapping from a set of four response fields to a vector of spherical harmonic coefficients of gravitational potantial change
    as a pygeoinf LinearOperator.
    """

    def __init__(self, order, scale, observation_degree, /, *, fingerprint=None, rotational_feedbacks=False):
        

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

        response_space = Sobolev(
            self._fingerprint.lmax,
            order + 1,
            scale,
            radius=self._fingerprint.mean_sea_floor_radius,
        )

        domain = inf.HilbertSpaceDirectSum(
            [response_space,
             response_space, 
             response_space, 
             inf.EuclideanSpace(2)]
        )

        self._observation_degree = observation_degree
        self._data_size = (self._observation_degree+1)**2 - 4 

        codomain = inf.EuclideanSpace(self._data_size)

        operator = inf.LinearOperator.from_formal_adjoint(
            domain, codomain, self._mapping, self._formal_adjoint
        )

        super().__init__(domain, codomain, operator, adjoint_mapping=operator.adjoint)

    def _mapping(self, response_fields):

        # Retrieve the relevant fields from the response.
        gravity_potential_change = response_fields[2]
        angular_velocity_change = response_fields[3]

        # Get the gravitational potential change. todo!!
        if self._rotational_feedbacks:
            gravitational_potential_change = (
                self._fingerprint.gravity_potential_change_to_gravitational_potential_change(
                    gravity_potential_change, angular_velocity_change
                )
            )
        else:
            gravitational_potential_change = gravity_potential_change

        # Convert the gravitational potential change to spherical harmonic coefficients.
        gravitational_sh_coeffs = self._to_ordered_sh_coefficients(gravitational_potential_change)

        # Return the spherical harmonic coefficients.
        return gravitational_sh_coeffs
    
    def _formal_adjoint(self, gravitational_sh_coeffs):

        # Convert the spherical harmonic coefficients to a grid.
        gravitational_potential_change = self._from_ordered_sh_coefficients(gravitational_sh_coeffs)

        # Convert the gravitational potential change to gravity potential change.
        if self._rotational_feedbacks:
            gravity_potential_change = self._fingerprint.gravitational_potential_change_to_gravity_potential_change(gravitational_potential_change, np.zeros(2))
        else:
            gravity_potential_change = gravitational_potential_change

        # Create a zero grid for the dual loads.
        zero_grid = self._fingerprint.zero_grid()

        return (
            zero_grid,
            zero_grid,
            gravity_potential_change,
            np.zeros(2)
        )
    
    def _to_ordered_sh_coefficients(self, grid):
        """Converts a grid to ordered spherical harmonic coefficients."""
        coeffs = self._fingerprint._expand_field(grid).coeffs
        vec = np.zeros(self._data_size)       
        for l in range(2,self._observation_degree+1):
            vec[((l)**2)-4:((l+1)**2)-4] = np.concatenate((coeffs[1,l,1:l+1][::-1],coeffs[0,l,0:l+1]))
        return vec    
    
    def _from_ordered_sh_coefficients(self, vec):
        """Converts ordered spherical harmonic coefficients to a grid."""
        coeffs = np.zeros((2, self._truncation_degree+1, self._truncation_degree+1))
        for l in range(2,self._observation_degree+1):
            coeffs[1,l,1:l+1] = vec[(l**2)-4:(l**2)-4+l][::-1]
            coeffs[0,l,0:l+1] = vec[(l**2)-4+l:((l+1)**2)-4]
        return self._fingerprint._expand_coefficient(SHCoeffs.from_array(coeffs, normalization=self._fingerprint.normalization))  