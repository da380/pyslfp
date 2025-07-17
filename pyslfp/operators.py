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

        super().__init__(domain, codomain, self._mapping, formal_adjoint_mapping=self._formal_adjoint)

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
                kk_d = -g * (response_fields[3] + self._fingerprint.adjoint_angular_momentum_change_from_adjoint_gravitational_potential_load(response_fields[2]))
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
        
        operator = self._operator()
        if not isinstance(operator, inf.LinearOperator):
            raise TypeError("The operator must be a pygeoinf LinearOperator.")

        super().__init__(operator.domain, operator.codomain, operator, adjoint_mapping=operator.adjoint)
        
    @abstractmethod
    def _operator(self):
        """
        LinearOperator instance that implements the mapping from the response fields to the data space.
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
        """
        self._observation_degree = observation_degree
        self._data_size = (self._observation_degree+1)**2 - 4
        self._fingerprint = sea_level_operator._fingerprint
        self._rotational_feedbacks = sea_level_operator._rotational_feedbacks
        super().__init__(sea_level_operator)

    def _operator(self):
        """Returns a LinearOperator instance which maps the response fields to spherical harmonic coefficients of gravitational potential change."""
        domain = self.sea_level_operator.codomain
        codomain = inf.EuclideanSpace(self._data_size)
        return inf.LinearOperator(
            domain, codomain, self._mapping, formal_adjoint_mapping=self._formal_adjoint
        )

    def _mapping(self, response_fields):
        """Maps the response fields to spherical harmonic coefficients of gravitational potential change."""
        gravitational_potential_change = response_fields[2]
        gravitational_sh_coeffs = self._to_ordered_sh_coefficients(gravitational_potential_change)
        return gravitational_sh_coeffs
    
    def _formal_adjoint(self, gravitational_sh_coeffs):
        """Maps from spherical harmonic coefficients of gravitational potential change to the adjoint loads"""
        gravitational_potential_change = self._from_ordered_sh_coefficients(gravitational_sh_coeffs)
        zero_grid = self._fingerprint.zero_grid()
        return (
            zero_grid,
            zero_grid,
            gravitational_potential_change,
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
        lmax = self._fingerprint.lmax
        coeffs = np.zeros((2, lmax + 1, lmax + 1))
        for l in range(2, self._observation_degree + 1):
            coeffs[1, l, 1:l+1] = vec[(l**2)-4:(l**2)-4+l][::-1]
            coeffs[0, l, 0:l+1] = vec[(l**2)-4+l:((l+1)**2)-4]
        return self._fingerprint._expand_coefficient(SHCoeffs.from_array(coeffs, normalization=self._fingerprint.normalization))  

class TideGaugeObservationOperator(ObservationOperator):
    """
    The mapping from a set of four response fields to a vector of sea level change at tide gauge locations
    as a pygeoinf LinearOperator.
    """
    
    def __init__(self, sea_level_operator, tide_gauge_locations):
        """
        Args:
            sea_level_operator (SeaLevelOperator): An instance of the SeaLevelOperator class.
            tide_gauge_locations (list): A list of points ([lat, lon]) where the sea level change is to be evaluated.    
        """
        self._fingerprint = sea_level_operator._fingerprint
        self._sl_space = sea_level_operator.codomain.subspaces[0]
        self._point_evaluation_operator = self._sl_space.point_evaluation_operator(tide_gauge_locations)
        super().__init__(sea_level_operator)

    def _operator(self):
        """
        Returns a LinearOperator instance that maps the response fields to a vector of tide gauge measurements.
        """
        domain = self.sea_level_operator.codomain
        codomain = self._point_evaluation_operator.codomain
        return inf.LinearOperator(
            domain, codomain, self._mapping, adjoint_mapping=self._adjoint_mapping
        )

    def _mapping(self, response_fields):
        return self._point_evaluation_operator(response_fields[0])
    
    def _adjoint_mapping(self, tide_gauge_measurements):
        zero_grid = self._fingerprint.zero_grid()
        return (
            self._point_evaluation_operator.adjoint(tide_gauge_measurements),
            zero_grid,
            zero_grid,
            np.zeros(2)
        )

class AveragingOperator(inf.LinearOperator):
    """
    Class for an operator which computes a vector of weighted averages of a field.
    Weighting functions can be given as 2D fields or as components.
    """
    def __init__(self, space, /, *, weighting_functions=None, weighting_components=None, fingerprint=None):
        """
        Args:
            space (Sobolev): The Sobolev space in which the operator acts.
            weighting_functions (list of SHGrid): A list of 2D fields to use as weights.
            weighting_components (list of SHCoeffs): A list of spherical harmonic coefficients to use as weights.
            fingerprint (FingerPrint): An instance of the FingerPrint class that must have its background state set.
        """
        self._space = space

        assert weighting_components is not None or weighting_functions is not None, "Either weighting functions or components must be given."
        if weighting_functions is not None:
            self._weighting_functions = weighting_functions
            self._weighting_components = [space.to_components(wf) for wf in weighting_functions]
        elif weighting_components is not None:
            self._weighting_components = weighting_components
            self._weighting_functions = [space.from_components(wc) for wc in weighting_components]

        if fingerprint is None:
            self._fingerprint = FingerPrint(
                earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation()
            )
            self._fingerprint.set_state_from_ice_ng()
        else:
            if not fingerprint.background_set:
                raise ValueError("fingerprint must have its background state set.")
            self._fingerprint = fingerprint

        self._averages_size = len(self._weighting_functions)
        self._averages_space = inf.EuclideanSpace(self._averages_size)

        super().__init__(self._space, self._averages_space, self._mapping, dual_mapping = self._dual_mapping)

    @property
    def weighting_functions(self):
        """Returns the weighting functions."""
        return self._weighting_functions
    
    @property
    def weighting_components(self):
        """Returns the weighting components."""
        return self._weighting_components
    
    def _mapping(self, field):
        """Maps a field to a vector of weighted averages."""
        averages = np.zeros(self._averages_size)
        for i, w in enumerate(self._weighting_functions):
            averages[i] = self._fingerprint.integrate(field * w)
        return averages
    
    def _dual_mapping(self, ap):
        """The dual mapping"""
        cap = self.codomain.dual.to_components(ap) * self._space.radius**2
        czp = sum([wi * ai for wi, ai in zip(self._weighting_components, cap)])
        return inf.LinearForm(self.domain, components=czp)

class WahrOperator(inf.LinearOperator):
    """
    Class for an operator which acts the wahr method on a vector of gravitational potential coefficients to produce a load average.
    """
    def __init__(self, observation_degree, weighting_components, love_numbers, radius):

        self._weighting_components = weighting_components
        self._property_size = len(self._weighting_components)
        self._property_space = inf.EuclideanSpace(self._property_size)

        self._observation_degree = observation_degree
        self._data_size = (self._observation_degree+1)**2 - 4
        self._data_space = inf.EuclideanSpace(self._data_size)

        self._love_numbers = love_numbers
        self._radius = radius

        super().__init__(self._data_space, self._property_space, self._mapping)    
    
    def _mapping(self, phi):
        """The forward mapping."""
        ## Loops over l and m, and computes sigma = k^-1*phi_lm*w_i
        k = self._love_numbers
        b = self._radius
        w = np.zeros(self._property_size)
        for i in range(self._property_size):
            for l in range(2,self._observation_degree+1):
                for m in range(-1*l,l+1):
                    vec_index = (l**2)-4+m+l
                    w[i] += b**2 * (1/k[l]) * phi[vec_index] * self._weighting_components[i][vec_index]

        return w
    
class AveragingOperator2(inf.LinearOperator):
    """
    Class for an operator which computes a vector of weighted averages of a field.
    Weighting functions can be given as 2D fields or as components.
    """
    def __init__(self, space, /, *, weighting_functions=None, weighting_components=None, fingerprint=None):
        """
        Args:
            space (Sobolev): The Sobolev space in which the operator acts.
            weighting_functions (list of SHGrid): A list of 2D fields to use as weights.
            weighting_components (list of SHCoeffs): A list of spherical harmonic coefficients to use as weights.
            fingerprint (FingerPrint): An instance of the FingerPrint class that must have its background state set.
        """
        self._space = space

        assert weighting_components is not None or weighting_functions is not None, "Either weighting functions or components must be given."
        if weighting_functions is not None:
            self._weighting_functions = weighting_functions
            self._weighting_components = [space.to_components(wf) for wf in weighting_functions]
        elif weighting_components is not None:
            self._weighting_components = weighting_components
            self._weighting_functions = [space.from_components(wc) for wc in weighting_components]

        if fingerprint is None:
            self._fingerprint = FingerPrint(
                earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation()
            )
            self._fingerprint.set_state_from_ice_ng()
        else:
            if not fingerprint.background_set:
                raise ValueError("fingerprint must have its background state set.")
            self._fingerprint = fingerprint

        self._averages_size = len(self._weighting_functions)
        self._averages_space = inf.EuclideanSpace(self._averages_size)

        operator = inf.LinearOperator.formally_self_adjoint(
            self._space,
            self._mapping,
        )

        super().__init__(operator.domain, operator.codomain, operator, adjoint_mapping=operator.adjoint)

    @property
    def weighting_functions(self):
        """Returns the weighting functions."""
        return self._weighting_functions
    
    @property
    def weighting_components(self):
        """Returns the weighting components."""
        return self._weighting_components
    
    def _mapping(self, field):
        """Maps a field to a vector of weighted averages."""
        averages = np.zeros(self._averages_size)
        for i, w in enumerate(self._weighting_functions):
            averages[i] = self._fingerprint.integrate(field * w)
        return averages

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