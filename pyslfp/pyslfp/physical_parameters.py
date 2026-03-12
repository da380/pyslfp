"""
Module for a class that stores basic Earth model data along with a
non-dimensionalisation scheme.
"""

from __future__ import annotations

import inspect

# =====================================================================
# Default Earth Model Physical Parameters
# Based on the Preliminary Reference Earth Model (PREM) and other
# standard values. All units are SI.
# =====================================================================

# Radii (m)
EQUATORIAL_RADIUS: float = 6378137.0
POLAR_RADIUS: float = 6356752.0
MEAN_RADIUS: float = 6371000.0
MEAN_SEA_FLOOR_RADIUS: float = 6368000.0

# Mass and Gravity
MASS: float = 5.974e24  # Earth mass (kg)
GRAVITATIONAL_ACCELERATION: float = 9.825652323  # Surface gravity (m/s^2)

# Moments of Inertia (kg*m^2)
EQUATORIAL_MOMENT_OF_INERTIA: float = 8.0096e37
POLAR_MOMENT_OF_INERTIA: float = 8.0359e37

# Rotational Frequency (rad/s)
ROTATION_FREQUENCY: float = 7.27220521664304e-05

# Densities (kg/m^3)
WATER_DENSITY: float = 1000.0
ICE_DENSITY: float = 917.0
# =====================================================================


class EarthModelParameters:
    """
    A data class for storing Earth model parameters and a non-dimensionalisation scheme.

    All physical parameters are stored internally in their non-dimensional form.
    The properties provide access to these values.
    """

    def __init__(
        self,
        /,
        *,
        # Scales have simple defaults
        length_scale: float = 1.0,
        density_scale: float = 1.0,
        time_scale: float = 1.0,
        # Physical parameters now use the named constants as defaults
        equatorial_radius: float = EQUATORIAL_RADIUS,
        polar_radius: float = POLAR_RADIUS,
        mean_radius: float = MEAN_RADIUS,
        mean_sea_floor_radius: float = MEAN_SEA_FLOOR_RADIUS,
        mass: float = MASS,
        gravitational_acceleration: float = GRAVITATIONAL_ACCELERATION,
        equatorial_moment_of_inertia: float = EQUATORIAL_MOMENT_OF_INERTIA,
        polar_moment_of_inertia: float = POLAR_MOMENT_OF_INERTIA,
        rotation_frequency: float = ROTATION_FREQUENCY,
        water_density: float = WATER_DENSITY,
        ice_density: float = ICE_DENSITY,
    ) -> None:
        """
        Args:
            length_scale: The characteristic length for non-dimensionalisation (m).
            density_scale: The characteristic density (kg/m^3).
            time_scale: The characteristic time (s).
            equatorial_radius: Earth's equatorial radius (m).
            polar_radius: Earth's polar radius (m).
            mean_radius: Earth's mean radius (m).
            mean_sea_floor_radius: Earth's mean sea floor radius (m).
            mass: Earth's mass (kg).
            gravitational_acceleration: Surface gravitational acceleration (m/s^2).
            equatorial_moment_of_inertia: Earth's equatorial moment of inertia (kg*m^2).
            polar_moment_of_inertia: Earth's polar moment of inertia (kg*m^2).
            rotation_frequency: Earth's angular rotation frequency (rad/s).
            water_density: Density of water (kg/m^3).
            ice_density: Density of ice (kg/m^3).
        """

        # Set the base units.
        self._length_scale: float = length_scale
        self._density_scale: float = density_scale
        self._time_scale: float = time_scale

        # Set the derived units.
        self._mass_scale: float = self._density_scale * self._length_scale**3
        self._frequency_scale: float = 1.0 / self.time_scale
        self._load_scale: float = self.mass_scale / self.length_scale**2
        self._velocity_scale: float = self.length_scale / self.time_scale
        self._acceleration_scale: float = self.velocity_scale / self.time_scale
        self._gravitational_potential_scale: float = (
            self.acceleration_scale * self.length_scale
        )
        self._moment_of_inertia_scale: float = self.mass_scale * self.length_scale**2

        # Set the non-dimensional physical constants.
        self._equatorial_radius: float = equatorial_radius / self.length_scale
        self._polar_radius: float = polar_radius / self.length_scale
        self._mean_radius: float = mean_radius / self.length_scale
        self._mean_sea_floor_radius: float = mean_sea_floor_radius / self.length_scale
        self._mass: float = mass / self.mass_scale
        self._gravitational_acceleration: float = (
            gravitational_acceleration / self.acceleration_scale
        )
        self._gravitational_constant: float = (
            6.6723e-11 * self.mass_scale * self.time_scale**2 / self.length_scale**3
        )
        self._equatorial_moment_of_inertia: float = (
            equatorial_moment_of_inertia / self.moment_of_inertia_scale
        )
        self._polar_moment_of_inertia: float = (
            polar_moment_of_inertia / self.moment_of_inertia_scale
        )
        self._rotation_frequency: float = rotation_frequency / self.frequency_scale
        self._water_density: float = water_density / self.density_scale
        self._ice_density: float = ice_density / self.density_scale

    @staticmethod
    def from_standard_non_dimensionalisation() -> "EarthModelParameters":
        """
        Returns parameters using a standard non-dimensionalisation scheme based
        on the mean radius of the Earth, the density of water, and the length
        of an hour.
        """
        return EarthModelParameters(
            length_scale=6371000.0, density_scale=1000.0, time_scale=3600
        )

    @staticmethod
    def _get_init_kwargs_from_instance(
        emp_instance: "EarthModelParameters",
    ) -> dict[str, float]:
        """
        Extracts __init__ keyword arguments from an existing EarthModelParameters instance.

        This is a helper method to facilitate re-initializing a subclass with
        the parameters of an existing parent class instance.

        Args:
            emp_instance: An instance of EarthModelParameters or its subclass.

        Returns:
            A dictionary of keyword arguments suitable for initializing a new
            EarthModelParameters instance.
        """
        signature = inspect.signature(EarthModelParameters.__init__)
        param_names = [
            p for p in signature.parameters if p not in ("self", "args", "kwargs")
        ]
        scale_names = {"length_scale", "density_scale", "time_scale"}

        kwargs = {}
        for name in param_names:
            if name in scale_names:
                kwargs[name] = getattr(emp_instance, name)
            else:
                # Assumes a corresponding '_si' property for other parameters
                kwargs[name] = getattr(emp_instance, f"{name}_si")
        return kwargs

    @property
    def length_scale(self) -> float:
        """The characteristic length scale (m)."""
        return self._length_scale

    @property
    def mass_scale(self) -> float:
        """The characteristic mass scale (kg)."""
        return self._mass_scale

    @property
    def time_scale(self) -> float:
        """The characteristic time scale (s)."""
        return self._time_scale

    @property
    def frequency_scale(self) -> float:
        """The characteristic frequency scale (1/s)."""
        return self._frequency_scale

    @property
    def density_scale(self) -> float:
        """The characteristic density scale (kg/m^3)."""
        return self._density_scale

    @property
    def load_scale(self) -> float:
        """The characteristic load scale (kg/m^2)."""
        return self._load_scale

    @property
    def velocity_scale(self) -> float:
        """The characteristic velocity scale (m/s)."""
        return self._velocity_scale

    @property
    def acceleration_scale(self) -> float:
        """The characteristic acceleration scale (m/s^2)."""
        return self._acceleration_scale

    @property
    def gravitational_potential_scale(self) -> float:
        """The characteristic gravitational potential scale (m^2/s^2)."""
        return self._gravitational_potential_scale

    @property
    def moment_of_inertia_scale(self) -> float:
        """The characteristic moment of inertia scale (kg*m^2)."""
        return self._moment_of_inertia_scale

    # -----------------------------------------------------#
    #      Properties related to physical constants       #
    # -----------------------------------------------------#

    @property
    def equatorial_radius(self) -> float:
        """Earth's equatorial radius (non-dimensional)."""
        return self._equatorial_radius

    @property
    def polar_radius(self) -> float:
        """Earth's polar radius (non-dimensional)."""
        return self._polar_radius

    @property
    def mean_radius(self) -> float:
        """Earth's mean radius (non-dimensional)."""
        return self._mean_radius

    @property
    def mean_sea_floor_radius(self) -> float:
        """Earth's mean sea floor radius (non-dimensional)."""
        return self._mean_sea_floor_radius

    @property
    def mass(self) -> float:
        """Earth's mass (non-dimensional)."""
        return self._mass

    @property
    def gravitational_acceleration(self) -> float:
        """Earth's surface gravitational acceleration (non-dimensional)."""
        return self._gravitational_acceleration

    @property
    def gravitational_constant(self) -> float:
        """The gravitational constant (non-dimensional)."""
        return self._gravitational_constant

    @property
    def equatorial_moment_of_inertia(self) -> float:
        """Earth's equatorial moment of inertia (non-dimensional)."""
        return self._equatorial_moment_of_inertia

    @property
    def polar_moment_of_inertia(self) -> float:
        """Earth's polar moment of inertia (non-dimensional)."""
        return self._polar_moment_of_inertia

    @property
    def rotation_frequency(self) -> float:
        """Earth's rotational frequency (non-dimensional)."""
        return self._rotation_frequency

    @property
    def water_density(self) -> float:
        """The density of water (non-dimensional)."""
        return self._water_density

    @property
    def ice_density(self) -> float:
        """The density of ice (non-dimensional)."""
        return self._ice_density

    # -----------------------------------------------------#
    #    Properties for Dimensional (SI) Constants         #
    # -----------------------------------------------------#

    @property
    def equatorial_radius_si(self) -> float:
        """Earth's equatorial radius in SI units (m)."""
        return self._equatorial_radius * self.length_scale

    @property
    def polar_radius_si(self) -> float:
        """Earth's polar radius in SI units (m)."""
        return self._polar_radius * self.length_scale

    @property
    def mean_radius_si(self) -> float:
        """Earth's mean radius in SI units (m)."""
        return self._mean_radius * self.length_scale

    @property
    def mean_sea_floor_radius_si(self) -> float:
        """Earth's mean sea floor radius in SI units (m)."""
        return self._mean_sea_floor_radius * self.length_scale

    @property
    def mass_si(self) -> float:
        """Earth's mass in SI units (kg)."""
        return self._mass * self.mass_scale

    @property
    def gravitational_acceleration_si(self) -> float:
        """Earth's surface gravitational acceleration in SI units (m/s^2)."""
        return self._gravitational_acceleration * self.acceleration_scale

    @property
    def equatorial_moment_of_inertia_si(self) -> float:
        """Earth's equatorial moment of inertia in SI units (kg*m^2)."""
        return self._equatorial_moment_of_inertia * self.moment_of_inertia_scale

    @property
    def polar_moment_of_inertia_si(self) -> float:
        """Earth's polar moment of inertia in SI units (kg*m^2)."""
        return self._polar_moment_of_inertia * self.moment_of_inertia_scale

    @property
    def rotation_frequency_si(self) -> float:
        """Earth's rotational frequency in SI units (rad/s)."""
        return self._rotation_frequency * self.frequency_scale

    @property
    def water_density_si(self) -> float:
        """The density of water in SI units (kg/m^3)."""
        return self._water_density * self.density_scale

    @property
    def ice_density_si(self) -> float:
        """The density of ice in SI units (kg/m^3)."""
        return self._ice_density * self.density_scale
