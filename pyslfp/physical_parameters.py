"""
Module for a class that stores basic Earth model data along with a non-dimensionalisation scheme. 
"""


class EarthModelParameters:
    """
    Class for storing Earth model parmaters and a non-dimensionalisation scheme.
    """

    def __init__(
        self,
        /,
        *,
        length_scale=1,
        density_scale=1,
        time_scale=1,
        equatorial_radius=6378137,
        polar_radius=6356752,
        mean_radius=6371000,
        mean_sea_floor_radius=6368000,
        mass=5.974e24,
        gravitational_acceleration=9.825652323,
        equatorial_moment_of_inertia=8.0096e37,
        polar_moment_of_inertia=8.0359e37,
        rotation_frequency=7.27220521664304e-05,
        water_density=1000,
        ice_density=917,
    ):

        # Set the base units.
        self._length_scale = length_scale
        self._density_scale = density_scale
        self._time_scale = time_scale

        # Set the derived units.
        self._mass_scale = self._density_scale * self._length_scale**3
        self._frequency_scale = 1 / self.time_scale
        self._load_scale = self.mass_scale / self.length_scale**2
        self._velocity_scale = self.length_scale / self.time_scale
        self._acceleration_scale = self.velocity_scale / self.time_scale
        self._gravitational_potential_scale = (
            self.acceleration_scale * self.length_scale
        )
        self._moment_of_inertia_scale = self.mass_scale * self.length_scale**2

        # Set the physical constants.
        self._equatorial_radius = equatorial_radius / self.length_scale
        self._polar_radius = polar_radius / self.length_scale
        self._mean_radius = mean_radius / self.length_scale
        self._mean_sea_floor_radius = mean_sea_floor_radius / self.length_scale
        self._mass = mass / self.mass_scale
        self._gravitational_acceleration = (
            gravitational_acceleration / self.acceleration_scale
        )
        self._gravitational_constant = (
            6.6723e-11 * self.mass_scale * self.time_scale**2 / self.length_scale**3
        )
        self._equatorial_moment_of_inertia = (
            equatorial_moment_of_inertia / self.moment_of_inertia_scale
        )
        self._polar_moment_of_inertia = (
            polar_moment_of_inertia / self.moment_of_inertia_scale
        )
        self._rotation_frequency = rotation_frequency / self.frequency_scale
        self._water_density = water_density / self.density_scale
        self._ice_density = ice_density / self.density_scale

        # Set the derived units.
        self._frequency_scale = 1 / self.time_scale
        self._density_scale = self.mass_scale / self.length_scale**3
        self._load_scale = self.mass_scale / self.length_scale**2
        self._velocity_scale = self.length_scale / self.time_scale
        self._acceleration_scale = self.velocity_scale / self.time_scale
        self._gravitational_potential_scale = (
            self.acceleration_scale * self.length_scale
        )
        self._moment_of_inertia_scale = self.mass_scale * self.length_scale**2

        # Set the physical constants.
        self._equatorial_radius = 6378137 / self.length_scale
        self._polar_radius = 6356752 / self.length_scale
        self._mean_radius = 6371000 / self.length_scale
        self._mean_sea_floor_radius = 6368000 / self.length_scale
        self._mass = 5.974e24 / self.mass_scale
        self._gravitational_acceleration = 9.825652323 / self.acceleration_scale
        self._gravitational_constant = (
            6.6723e-11 * self.mass_scale * self.time_scale**2 / self.length_scale**3
        )
        self._equatorial_moment_of_inertia = 8.0096e37 / self.moment_of_inertia_scale
        self._polar_moment_of_inertia = 8.0359e37 / self.moment_of_inertia_scale
        self._rotation_frequency = 7.27220521664304e-05 / self.frequency_scale
        self._water_density = 1000 / self.density_scale
        self._ice_density = 917 / self.density_scale

    @staticmethod
    def from_standard_non_dimensionalisation():
        """
        Returns parameters using a non-dimensionalisation scheme based
        on the mean radius of the Earth, the density of water, and the length
        of a year.
        """
        return EarthModelParameters(
            length_scale=6371000, density_scale=1000, time_scale=365 * 24 * 3600
        )

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

    # -----------------------------------------------------#
    #      Properties related to physical constants       #
    # -----------------------------------------------------#

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
