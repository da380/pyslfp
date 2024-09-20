
from . import DATADIR
from pyslfp.ice_ng import IceNG
from pyslfp.fields import ResponseFields,ResponseCoefficients
from pyshtools import SHGrid, SHCoeffs
import numpy as np


if __name__ == "__main__":
    pass


class Solver:


    def __init__(self, lmax=256, /,*, length_scale=1, mass_scale=1, time_scale=1,
                 grid = "DH", love_number_file=DATADIR + "/love_numbers/PREM_4096.dat"):
        
        # Set the base units. 
        self._length_scale = length_scale        
        self._mass_scale = mass_scale
        self._time_scale = time_scale                

        # Set the derived units. Parameter
        self._frequency_scale = 1 / self.time_scale
        self._density_scale = self.mass_scale / self.length_scale**3
        self._load_scale = self.mass_scale / self.length_scale**2
        self._velocity_scale = self.length_scale / self.time_scale
        self._acceleration_scale = self.velocity_scale / self.time_scale
        self._gravitational_potential_scale = self.acceleration_scale * self.length_scale
        self._moment_of_inertia_scale = self.mass_scale * self.length_scale**2

        # Set the physical constants. 
        self._equatorial_radius = 6378137 / self.length_scale
        self._polar_radius = 6356752  / self.length_scale
        self._mean_radius = 6371000 / self.length_scale
        self._mean_sea_floor_radius = 6368000 / self.length_scale
        self._mass = 5.974e24 / self.mass_scale
        self._gravitational_acceleration = 9.825652323 / self.acceleration_scale
        self._gravitational_constant = 6.6723e-11 * self.mass_scale * self.time_scale**2 / self.length_scale**3        
        self._equatorial_moment_of_inertia = 8.0096e37 / self.moment_of_inertia_scale
        self._polar_moment_of_inertia =  8.0359e37 / self.moment_of_inertia_scale
        self._rotation_frequency = 7.27220521664304e-05 / self.frequency_scale
        self._water_density = 1000 / self.density_scale
        self._ice_density = 917 / self.density_scale

        # Set some options. 
        self._lmax = lmax        
        if grid == "DH2":
            self._grid = "DH"
            self._sampling = 2
        else:
            self._grid = grid
            self._sampling = 1
        self._extend = True
        self._normalization = "ortho"
        self._csphase = 1        
        
        # Read in the Love numbers.
        self._read_love_numbers(love_number_file)

        # Pre-compute some constants.
        self._rotation_factor =  np.sqrt((4*np.pi)/15.) * self.rotation_frequency *  self.mean_sea_floor_radius**2
        self._integration_factor =  np.sqrt((4*np.pi)) * self._mean_sea_floor_radius**2
        self._inertia_factor = (np.sqrt(5/(12*np.pi))  * self.rotation_frequency * self.mean_sea_floor_radius**3 / 
                               (self.gravitational_constant   * (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)))        

        # Background model not set.        
        self._sea_level = None
        self._ice_thickness = None
        self._ocean_function = None
        self._ocean_area = None
        

    #------------------------------------------------#
    #          Properties related to units           #
    #------------------------------------------------#

    @property
    def length_scale(self):
        """Return length for non-dimensionalisation."""
        return self._length_scale

    @property
    def mass_scale(self):
        """Return mass for non-dimensionalisation."""
        return self._mass_scale

    @property
    def time_scale(self):
        """Return time for non-dimensionalisation."""
        return self._time_scale

    @property
    def frequency_scale(self):
        """Return frequency for non-dimensionalisation."""
        return self._frequency_scale

    @property
    def density_scale(self):
        """Return density for non-dimensionalisation."""
        return self._density_scale

    @property
    def load_scale(self):
        """Return load for non-dimensionalisation."""
        return self._load_scale

    @property
    def velocity_scale(self):
        """Return velocity for non-dimensionalisation."""
        return self._velocity_scale

    @property
    def acceleration_scale(self):
        """Return acceleration for non-dimensionalisation."""
        return self._acceleration_scale

    @property
    def gravitational_potential_scale(self):
        """Return gravitational potential for non-dimensionalisation."""
        return self._gravitational_potential_scale

    @property
    def moment_of_inertia_scale(self):
        """Return moment of intertia for non-dimensionalisation."""
        return self._moment_of_inertia_scale

    #-----------------------------------------------------#
    #      Properties related to physical constants       #
    #-----------------------------------------------------#

    @property
    def equatorial_radius(self):
        """Return Earth's equatorial radius."""
        return self._equatorial_radius

    @property
    def polar_radius(self):
        """Return Earth's polar radius."""
        return self._polar_radius

    @property
    def mean_radius(self):
        """Return Earth's mean radius."""
        return self._mean_radius

    @property
    def mean_sea_floor_radius(self):
        """Return Earth's mean sea floor radius."""
        return self._mean_sea_floor_radius

    @property
    def mass(self):
        """Return Earth's mass."""
        return self._mass

    @property
    def gravitational_acceleration(self):
        """Return Earth's surface gravitational acceleration."""
        return self._gravitational_acceleration

    @property 
    def gravitational_constant(self):
        """Return Gravitational constant."""
        return self._gravitational_constant    

    @property
    def equatorial_moment_of_inertia(self):
        """Return Earth's equatorial moment of inertia."""
        return self._equatorial_moment_of_inertia

    @property
    def polar_moment_of_inertia(self):
        """Return Earth's polar moment of inertia."""
        return self._polar_moment_of_inertia

    @property
    def rotation_frequency(self):
        """Return Earth's rotational frequency."""
        return self._rotation_frequency

    @property
    def water_density(self):
        """Return density of water."""
        return self._water_density

    @property
    def ice_density(self):
        """Return density of ice."""
        return self._ice_density

    #-----------------------------------------------#
    #        Properties related to options          #
    #-----------------------------------------------#

    @property
    def lmax(self):
        """Return truncation degree for expansions."""
        return self._lmax

    @property 
    def normalization(self):
        """Return spherical harmonic normalisation convention."""
        return self._normalization

    @property
    def csphase(self):
        """Return Condon-Shortley phase option."""
        return self._csphase

    @property
    def grid(self):
        """Return spatial grid option."""
        return self._grid

    @property
    def extend(self):
        """True if grid extended to include 360 degree longitude."""
        return self._extend


    #----------------------------------------------------#
    #     Properties related to the background state     #
    #----------------------------------------------------#

    @property
    def sea_level(self):
        """Returns the backgroud sea level."""
        if self._sea_level is None:
            raise NotImplementedError("Sea level not set.")
        else:
            return self._sea_level                

    @sea_level.setter
    def sea_level(self, value):
        self._check_field(value)
        self._sea_level = value

    @property
    def ice_thickness(self):
        """Returns the backgroud ice thickness."""
        if self._ice_thickness is None:
            raise NotImplementedError("Ice thickness not set.")
        else:
            return self._ice_thickness

    @ice_thickness.setter
    def ice_thickness(self, value):
        self._check_field(value)
        self._ice_thickness = value

    @property 
    def ocean_function(self):
        """Returns the ocean function."""
        if self._ocean_function is None:            
            self._compute_ocean_function()
        return self._ocean_function

    @property
    def one_minus_ocean_function(self):
        """Returns 1 - C, with C the ocean function."""
        tmp = self.ocean_function.copy()
        tmp.data = 1 - tmp.data
        return tmp

    @property
    def ocean_area(self):
        """Returns the ocean area."""
        if self._ocean_area is None:
            self._compute_ocean_area()
        return self._ocean_area
    
    #---------------------------------------------------------#
    #                     Private methods                     #
    #---------------------------------------------------------#

    
    def _read_love_numbers(self, file):
        # Read in the Love numbers from a given file and non-dimensionalise. 

        data = np.loadtxt(file) 
        data_degree = len(data[:,0]) -1
    
        if self.lmax > data_degree:
            raise ValueError("maximum degree is larger than present in data file")
        
        self._h_u = data[:self.lmax+1,1] * self.load_scale / self.length_scale
        self._k_u = data[:self.lmax+1,2] * self.load_scale / self.gravitational_potential_scale
        self._h_phi = data[:self.lmax+1,3] * self.load_scale / self.length_scale
        self._k_phi = data[:self.lmax+1,4] * self.load_scale / self.gravitational_potential_scale

        self._h = self._h_u + self._h_phi
        self._k = self._k_u + self._k_phi

        self._ht = data[:self.lmax+1,5] * self.gravitational_potential_scale / self.length_scale
        self._kt = data[:self.lmax+1,6]

    def _check_field(self, f):
        # Check SHGrid object is compatible with options.         
        return (f.lmax == self.lmax and f.grid == self.grid and f.extend == self.extend)

    def _check_coefficient(self,f):
        # Check SHCoeff object is compatible with options. 
        return (f.lmax == self.lmax and f.normalization == self.normalization 
                and f.csphase == self.csphase)            
    
    def _expand_field(self, f, /, *, lmax_calc = None):
        # Expand a SHGrid object using stored parameters. 
        assert self._check_field(f)
        if lmax_calc is None:
            return f.expand(normalization = self.normalization, csphase = self.csphase)
        else:   
            return f.expand(lmax_calc = lmax_calc, normalization = self.normalization, csphase = self.csphase)

    def _expand_coefficient(self,f):
        # Expand a SHCoeff object using stored parameters. 
        assert self._check_coefficient(f)
        if self._sampling == 2:
            grid = "DH2"
        else:
            grid = self.grid
        return f.expand(grid=grid, extend=self.extend)

    def _compute_ocean_function(self):
        # Copmutes and stores the ocean function. 
        if self._sea_level is None or self._ice_thickness is None:
            raise NotImplementedError("Sea level and/or ice thickness not set")
        self._ocean_function = SHGrid.from_array(np.where(self.water_density * self.sea_level.data - 
                                                          self.ice_density * self.ice_thickness.data > 0, 1, 0),
                                                          grid=self.grid)

    def _compute_ocean_area(self):
        # Computes and stores the ocean area. 
        if self._ocean_function is None:
            self._compute_ocean_function()
        self._ocean_area = self.integrate(self._ocean_function)                

    def _iterate_solver(self, sigma, sl_uniform, /, *, rotational_feedbacks=True):
        # Given a load, returns the response.

        assert self._check_field(sigma)            
        sigma_lm = self._expand_field(sigma)
        u_lm = sigma_lm.copy()
        phi_lm = sigma_lm.copy()
        for l in range(self.lmax+1):
            u_lm.coeffs[:,l,:] *= self._h[l]
            phi_lm.coeffs[:,l,:] *= self._k[l]

        if rotational_feedbacks:
            r = self._rotation_factor
            i = self._inertia_factor
            kt = self._kt[2]
            ht = self._ht[2]
            f = r * i /(1 - r * i * kt)
            u_lm.coeffs[:,2,1] += ht * f * phi_lm.coeffs[:,2,1]
            phi_lm.coeffs[:,2,1] += kt * f * phi_lm.coeffs[:,2,1]
            omega = i * phi_lm.coeffs[:,2,1]
            phi_lm.coeffs[:,2,1] += r * omega
        else:
            omega = np.zeros(2)        

        g = self.gravitational_acceleration        
        u = self._expand_coefficient(u_lm)
        phi = self._expand_coefficient(phi_lm)
        sl = (-1/g) * (g * u + phi)
        sl_average = self.ocean_average(sl)            
        sl.data[:,:] += sl_uniform - sl_average        

        return ResponseFields(u, phi, omega, sl)
        
    #--------------------------------------------------------#
    #                       Public methods                   #
    #--------------------------------------------------------#


    def integrate(self, f):        
        """ Integrate function over the surface."""
        return self._integration_factor * self._expand_field(f).coeffs[0,0,0]

    def ocean_average(self,f):        
        """Return average of a function over the oceans."""
        return self.integrate(self.ocean_function * f) / self.ocean_area

    

    def set_background_state_from_ice_ng(self,  /, *, version = 7, date=0):
        """
        Sets background state from ice_7g, ice_6g, or ice_5g.

        Args:
            version (int): Selects the model to use. 
            data (float): Selects the date from which values are taken. 

        Notes:
            To detemrine the fields, linear interpolation between 
            model values is applied. If the date is out of range, 
            constant extrapolation of the boundary values is used. 
        """
        # Read and interpolate the data.
        ice_ng = IceNG(version = version)
        ice_thickness, topography = ice_ng.get_time_slice(date, self.lmax, grid = self.grid,                                                          
                                                          sampling=self._sampling, extend = self.extend)

        # Non-dimensionalise the values. 
        ice_thickness /= self.length_scale
        topography /= self.length_scale

        # Compute the sea level using isostatic balance within ice shelves. 
        ice_shelf_thickness = SHGrid.from_array(np.where(np.logical_and(topography.data < 0, ice_thickness.data > 0), 
                                                ice_thickness.data,0),grid = self.grid)
        sea_level = SHGrid.from_array(np.where(topography.data < 0, -topography.data, -topography.data + ice_thickness.data),
                                               grid = self.grid)
        sea_level += self.water_density * ice_shelf_thickness / self.ice_density

        # Set the values. 
        self.sea_level = sea_level
        self.ice_thickness = ice_thickness
    

    
    def __call__(self, zeta, /, *, rotational_feedbacks=True, rtol = 1.e-6, verbose = False):
        """
        Returns the solution to the fingerprint problem for a given direct load.

        Args:
            zeta (SHGrid): The direct load. 
            rotational_feedbacks (bool): If true, rotational feedbacks included.
            rtol (float): Relative tolerance used in assessing convergence of iterations. 
            verbose (bool): If true, information on iterations printed. 

        Returns:
            ResponseField: Instance of the class containing the displacement, 
                gravitaty potential perturbation, rotational perturbation, and 
                sea level change. 

        Notes:
            If rotational feedbacks are included, the potential perturbation 
            is that of gravity, this being a sum of the gravitational and 
            centrifugal perturbations. 
        """
        assert self._check_field(zeta)
        sl_uniform = -self.integrate(zeta) / (self.water_density * self.ocean_area)            
        sigma = zeta + self.water_density * self.ocean_function * sl_uniform

        err = 1
        count = 0
        while err > rtol:
            response = self._iterate_solver(sigma, sl_uniform, rotational_feedbacks=rotational_feedbacks)
            sigma_new = zeta + self.water_density * self.ocean_function * response.sl
            err = np.max(np.abs((sigma_new - sigma).data)) / np.max(np.abs(sigma.data))
            sigma = sigma_new
            if verbose:
                count += 1
                print(f'Iteration = {count}, relative error = {err:6.4e}')

        return response

    def gravity_potential_to_gravitational_potential(self, response):
        """Converts the gravity potential within a ResponseField to the gravitational potential."""    
        phi_lm = self._expand_field(response.phi)
        phi_lm[:,2,1] -= self._rotation_factor * response.omega
        response.phi = self._expand_coefficient(phi_lm)
        return response

    def gravitational_potential_to_gravity_potential(self, response):
        """Converts the gravitational potential within a ResponseField to the gravity potential."""    
        phi_lm = self._expand_field(response.phi)
        phi_lm[:,2,1] += self._rotation_factor * response.omega
        response.phi = self._expand_coefficient(phi_lm)
        return response

    def ocean_mask(self, value = np.nan):
        """Return a mask over the oceans."""
        return SHGrid.from_array(np.where(self.ocean_function.data > 0, 1, value), grid = self.grid)        

    def land_mask(self, value = np.nan):
        """Return mask over the land."""
        return SHGrid.from_array(np.where(self._ocean_function.data == 0, 1, value), grid = self.grid)        

    def northern_hemisphere_mask(self, value = np.nan):
        """Return mask over the northern hemisphere."""
        lats, _ = np.meshgrid(self.ice_thickness.lats(), self.ice_thickness.lons(), indexing="ij")
        return SHGrid.from_array(np.where(lats > 0, 1, value), grid = self.grid)

    def southern_hemisphere_mask(self, value = np.nan):
        """Return mask over the southern hemisphere."""
        lats, _ = np.meshgrid(self.ice_thickness.lats(), self.ice_thickness.lons(), indexing="ij")
        return SHGrid.from_array(np.where(lats < 0, 1, value), grid = self.grid)        


    def disk_load(self, delta, latitutude, longitude, amplitude):
        """Return a disk load."""
        return amplitude * SHGrid.from_cap(delta, latitutude, longitude, lmax = self.lmax, grid=self.grid,
                               extend = self._extend, sampling = self._sampling)





    
