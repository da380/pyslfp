"""
Core physics solvers for the pyslfp library.

This module contains the SeaLevelEquation class, which acts as the primary
engine for calculating gravitationally consistent sea-level fingerprints,
handling rotational feedbacks, and computing non-linear shoreline migration.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from pyshtools import SHGrid, SHCoeffs

from .core import EarthModel
from .state import EarthState


class SeaLevelEquation:
    """
    The core solver for the gravitationally consistent Sea Level Equation (SLE).

    This class encapsulates the algorithms required to solve both the linear
    and non-linear forms of the SLE, mapping surface mass redistributions to
    global sea level, surface displacement, and gravity anomalies.
    """

    def __init__(self, model: EarthModel, /) -> None:
        """
        Initializes the Sea Level Equation solver.

        Args:
            model (EarthModel): The physical configuration of the Earth,
                including non-dimensional scales and Love numbers. Must be
                passed positionally.
        """
        self._model = model

        # Cache frequently used constants locally for performance
        self._g = self._model.parameters.gravitational_acceleration
        self._water_density = self._model.parameters.water_density
        self._radius = self._model.parameters.mean_sea_floor_radius

        self._rotation_factor = (
            np.sqrt((4 * np.pi) / 15.0)
            * self._model.parameters.rotation_frequency
            * self._radius**2
        )

        self._inertia_factor = (
            np.sqrt(5 / (12 * np.pi))
            * self._model.parameters.rotation_frequency
            * self._radius**3
            / (
                self._model.parameters.gravitational_constant
                * (
                    self._model.parameters.polar_moment_of_inertia
                    - self._model.parameters.equatorial_moment_of_inertia
                )
            )
        )
        self._solver_counter: int = 0

    @property
    def model(self) -> EarthModel:
        """The EarthModel configuration driving the solver."""
        return self._model

    @property
    def solver_counter(self) -> int:
        """The number of times the solver has been executed."""
        return self._solver_counter

    @property
    def rotation_factor(self) -> float:
        """The precomputed centrifugal potential rotation factor."""
        return self._rotation_factor

    @property
    def inertia_factor(self) -> float:
        """The precomputed polar wander inertia factor."""
        return self._inertia_factor

    # ---------------------------------------------------------#
    #                 Internal Helpers                         #
    # ---------------------------------------------------------#

    def _expand(self, grid: SHGrid) -> SHCoeffs:
        """Expands an SHGrid to SHCoeffs using normalized conventions."""
        return grid.expand(normalization="ortho", csphase=1)

    def _synthesize(self, coeffs: SHCoeffs, target_grid: str, extend: bool) -> SHGrid:
        """Synthesizes SHCoeffs explicitly respecting the grid sampling type."""
        return coeffs.expand(grid=target_grid, extend=extend)

    def _mean_sea_level_change(self, state: EarthState, direct_load: SHGrid) -> float:
        """Computes the mean eustatic sea level change for a given load."""
        return -state.integrate(direct_load) / (self._water_density * state.ocean_area)

    def _ocean_average(self, state: EarthState, f: SHGrid) -> float:
        """Computes the spatial average of a field over the oceans."""
        return state.integrate(state.ocean_function * f) / state.ocean_area

    # ---------------------------------------------------------#
    #                 Observable Generators                    #
    # ---------------------------------------------------------#

    def centrifugal_potential_change(
        self, state: EarthState, angular_velocity_change: np.ndarray, /
    ) -> SHGrid:
        """
        Computes the change in centrifugal potential due to polar wander.

        Args:
            state (EarthState): The background state defining the grid.
            angular_velocity_change (np.ndarray): The 2-element [omega_x, omega_y] vector.

        Returns:
            SHGrid: The centrifugal potential change field.
        """
        coeffs = SHCoeffs.from_zeros(
            lmax=self._model.lmax, normalization="ortho", csphase=1
        )
        coeffs.coeffs[:, 2, 1] = self._rotation_factor * angular_velocity_change
        return self._synthesize(coeffs, state.grid_type, state.extend)

    def get_sea_surface_height_change(
        self,
        state: EarthState,
        sea_level_change: SHGrid,
        displacement: SHGrid,
        angular_velocity_change: np.ndarray,
        /,
        *,
        remove_rotational_contribution: bool = True,
    ) -> SHGrid:
        """
        Computes the Sea Surface Height (SSH) change from raw physical fields.

        Args:
            state (EarthState): The background state defining the grid.
            sea_level_change (SHGrid): The relative sea level change field.
            displacement (SHGrid): The solid Earth displacement field.
            angular_velocity_change (np.ndarray): The [omega_x, omega_y] vector.
            remove_rotational_contribution (bool): Whether to remove the
                centrifugal signal from the SSH (standard for altimetry).

        Returns:
            SHGrid: The total Sea Surface Height change.
        """
        ssh_change = sea_level_change + displacement

        if remove_rotational_contribution:
            centrifugal = self.centrifugal_potential_change(
                state, angular_velocity_change
            )
            ssh_change += centrifugal / self._g

        return ssh_change

    # ---------------------------------------------------------#
    #                 Primary Solvers                          #
    # ---------------------------------------------------------#

    def solve_sea_level_equation(
        self,
        state: EarthState,
        direct_load: SHGrid,
        /,
        *,
        rotational_feedbacks: bool = True,
        rtol: float = 1.0e-6,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the standard linear Sea Level Equation for a surface mass load.

        Args:
            state (EarthState): The unperturbed background Earth state.
            direct_load (SHGrid): The mass redistribution forcing the system.
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.

        Returns:
            Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]: A 4-tuple containing:
                - Relative Sea Level Change
                - Vertical Displacement
                - Gravity Potential Change
                - Angular Velocity Change [omega_x, omega_y]
        """
        return self.solve_generalised_equation(
            state,
            direct_load=direct_load,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def solve_generalised_equation(
        self,
        state: EarthState,
        /,
        *,
        direct_load: Optional[SHGrid] = None,
        displacement_load: Optional[SHGrid] = None,
        gravitational_potential_load: Optional[SHGrid] = None,
        angular_momentum_change: Optional[np.ndarray] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1.0e-6,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the generalized linear SLE for an arbitrary combination of forcings.

        Useful for adjoint calculations or complex, multi-physical inversions.

        Args:
            state (EarthState): The unperturbed background Earth state.
            direct_load (Optional[SHGrid]): Standard surface mass forcing.
            displacement_load (Optional[SHGrid]): External vertical surface displacement forcing.
            gravitational_potential_load (Optional[SHGrid]): External gravitational potential forcing.
            angular_momentum_change (Optional[np.ndarray]): External angular momentum perturbation.
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.

        Returns:
            Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]: The physical response fields:
                (Sea Level Change, Displacement, Gravity Potential, Angular Velocity)
        """
        if direct_load is not None and (
            direct_load.lmax != state.lmax or direct_load.grid != state.grid
        ):
            raise ValueError("Direct load grid must match the EarthState grid.")

        h_b = self._model.love_numbers.h[None, :, None]
        k_b = self._model.love_numbers.k[None, :, None]
        h_u_b = self._model.love_numbers.h_u[None, :, None]
        k_u_b = self._model.love_numbers.k_u[None, :, None]
        h_phi_b = self._model.love_numbers.h_phi[None, :, None]
        k_phi_b = self._model.love_numbers.k_phi[None, :, None]

        loads_present = False
        non_zero_rhs = False

        if direct_load is not None:
            loads_present = True
            mean_slc = self._mean_sea_level_change(state, direct_load)
            non_zero_rhs = non_zero_rhs or np.max(np.abs(direct_load.data)) > 0
        else:
            direct_load = SHGrid.from_zeros(
                state.lmax,
                grid=state.grid,
                sampling=state.sampling,
                extend=state.extend,
            )
            mean_slc = 0.0

        static_disp_coeffs = 0.0
        static_grav_coeffs = 0.0
        has_static_loads = False

        if displacement_load is not None:
            loads_present = True
            disp_lm = self._expand(displacement_load)
            non_zero_rhs = non_zero_rhs or np.max(np.abs(displacement_load.data)) > 0
            static_disp_coeffs += h_u_b * disp_lm.coeffs
            static_grav_coeffs += k_u_b * disp_lm.coeffs
            has_static_loads = True

        if gravitational_potential_load is not None:
            loads_present = True
            grav_lm = self._expand(gravitational_potential_load)
            non_zero_rhs = (
                non_zero_rhs or np.max(np.abs(gravitational_potential_load.data)) > 0
            )
            static_disp_coeffs += h_phi_b * grav_lm.coeffs
            static_grav_coeffs += k_phi_b * grav_lm.coeffs
            has_static_loads = True

        if angular_momentum_change is not None:
            loads_present = True
            non_zero_rhs = non_zero_rhs or np.max(np.abs(angular_momentum_change)) > 0

        if not loads_present or not non_zero_rhs:
            zero = SHGrid.from_zeros(
                state.lmax,
                grid=state.grid,
                sampling=state.sampling,
                extend=state.extend,
            )
            return zero, zero, zero, np.zeros(2)

        self._solver_counter += 1

        load = direct_load + self._water_density * state.ocean_function * mean_slc
        angular_velocity_change = np.zeros(2)

        r = self._rotation_factor
        i = self._inertia_factor
        m = 1 / (
            self._model.parameters.polar_moment_of_inertia
            - self._model.parameters.equatorial_moment_of_inertia
        )
        ht = self._model.love_numbers.ht[2]
        kt = self._model.love_numbers.kt[2]

        sea_level_change = SHGrid.from_zeros(
            state.lmax, grid=state.grid, sampling=state.sampling, extend=state.extend
        )
        slc_data = sea_level_change.data
        load_data = load.data.copy()
        direct_load_data = direct_load.data
        ocean_func_data = state.ocean_function.data

        err = 1.0
        count = 0
        count_print = 0
        iter_limit = max_iterations if max_iterations is not None else 10000

        while err > rtol and count < iter_limit:

            displacement_lm = self._expand(load)
            gravity_potential_change_lm = displacement_lm.copy()

            displacement_lm.coeffs *= h_b
            gravity_potential_change_lm.coeffs *= k_b

            if has_static_loads:
                displacement_lm.coeffs += static_disp_coeffs
                gravity_potential_change_lm.coeffs += static_grav_coeffs

            if rotational_feedbacks:
                centrifugal_coeffs = r * angular_velocity_change

                displacement_lm.coeffs[:, 2, 1] += ht * centrifugal_coeffs
                gravity_potential_change_lm.coeffs[:, 2, 1] += kt * centrifugal_coeffs

                angular_velocity_change = (
                    i * gravity_potential_change_lm.coeffs[:, 2, 1]
                )

                if angular_momentum_change is not None:
                    angular_velocity_change -= m * angular_momentum_change

                gravity_potential_change_lm.coeffs[:, 2, 1] += (
                    r * angular_velocity_change
                )

            displacement = self._synthesize(
                displacement_lm, state.grid_type, state.extend
            )
            gravity_potential_change = self._synthesize(
                gravity_potential_change_lm, state.grid_type, state.extend
            )

            slc_data[:] = (-1.0 / self._g) * (
                self._g * displacement.data + gravity_potential_change.data
            )

            slc_data += mean_slc - self._ocean_average(state, sea_level_change)

            load_new_data = direct_load_data + (
                self._water_density * ocean_func_data * slc_data
            )

            if count > 1 or mean_slc != 0:
                max_load = np.max(np.abs(load_data))
                err = (
                    np.max(np.abs(load_new_data - load_data)) / max_load
                    if max_load > 0
                    else 0
                )
                if verbose:
                    count_print += 1
                    print(f"Iteration = {count_print}, relative error = {err:6.4e}")

            load_data[:] = load_new_data
            load.data[:] = load_new_data
            count += 1

        return (
            sea_level_change,
            displacement,
            gravity_potential_change,
            angular_velocity_change,
        )

    def solve_nonlinear_equation(
        self,
        initial_state: EarthState,
        /,
        *,
        ice_thickness_change: SHGrid,
        sediment_thickness_change: Optional[SHGrid] = None,
        dynamic_sea_level_change: Optional[SHGrid] = None,
        sediment_density: float = 2300.0,
        rotational_feedbacks: bool = True,
        rtol: float = 1.0e-6,
        max_iterations: int = 50,
        verbose: bool = False,
    ) -> Tuple[EarthState, SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the full non-linear Sea Level Equation with shifting shorelines.

        Incorporates dynamic ocean function updates, explicit mass conservation,
        and allows for combined ice, sediment, and dynamic sea level forcings.

        Args:
            initial_state (EarthState): The unperturbed background Earth state.
            ice_thickness_change (SHGrid): Change in ice thickness.
            sediment_thickness_change (Optional[SHGrid]): Change in sediment.
            dynamic_sea_level_change (Optional[SHGrid]): Ocean dynamic sea level forcing.
            sediment_density (float): Density of sediment layer in kg/m^3.
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence.
            max_iterations (int): Hard limit on non-linear iteration count.
            verbose (bool): If True, prints iteration metrics.

        Returns:
            Tuple[EarthState, SHGrid, SHGrid, SHGrid, np.ndarray]:
                - The new equilibrium EarthState
                - Relative Sea Level Change
                - Vertical Displacement
                - Gravity Potential Change
                - Angular Velocity Change [omega_x, omega_y]
        """
        if (
            ice_thickness_change.lmax != initial_state.lmax
            or ice_thickness_change.grid != initial_state.grid
        ):
            raise ValueError(
                "Ice thickness change grid must match the initial state grid."
            )

        h_b = self._model.love_numbers.h[None, :, None]
        k_b = self._model.love_numbers.k[None, :, None]
        r = self._rotation_factor
        i = self._inertia_factor
        ht = self._model.love_numbers.ht[2]
        kt = self._model.love_numbers.kt[2]

        initial_bathy = initial_state.sea_level.data
        initial_ice = initial_state.ice_thickness.data
        initial_ocean_func = initial_state.ocean_function.data

        new_ice = initial_ice + ice_thickness_change.data

        static_load_data = (
            self._model.parameters.ice_density * ice_thickness_change.data
        )
        if sediment_thickness_change is not None:
            static_load_data += sediment_density * sediment_thickness_change.data

        meltwater_mass = -initial_state.integrate(
            self._model.parameters.ice_density * ice_thickness_change
        )
        initial_water_mass = initial_state.integrate(
            self._water_density * initial_state.ocean_function * initial_state.sea_level
        )
        target_water_mass = initial_water_mass + meltwater_mass

        current_ocean_func = initial_ocean_func.copy()
        current_bathy = initial_bathy.copy()
        slc_data = np.zeros_like(initial_bathy)
        angular_velocity_change = np.zeros(2)

        grid_template = SHGrid.from_zeros(
            initial_state.lmax,
            grid=initial_state.grid,
            sampling=initial_state.sampling,
            extend=initial_state.extend,
        )

        exclude_caspian = initial_state.exclude_caspian
        caspian_mask = initial_state.caspian_sea_projection(value=0).data

        err = 1.0
        count = 0

        self._solver_counter += 1

        while err > rtol and count < max_iterations:

            ocean_mass_change = self._water_density * (
                current_ocean_func * current_bathy - initial_ocean_func * initial_bathy
            )
            total_load_data = static_load_data + ocean_mass_change

            grid_template.data[:] = total_load_data
            load_lm = self._expand(grid_template)

            displacement_lm = load_lm.copy()
            gpc_lm = load_lm.copy()

            displacement_lm.coeffs *= h_b
            gpc_lm.coeffs *= k_b

            if rotational_feedbacks:
                centrifugal_coeffs = r * angular_velocity_change
                displacement_lm.coeffs[:, 2, 1] += ht * centrifugal_coeffs
                gpc_lm.coeffs[:, 2, 1] += kt * centrifugal_coeffs

                angular_velocity_change = i * gpc_lm.coeffs[:, 2, 1]
                gpc_lm.coeffs[:, 2, 1] += r * angular_velocity_change

            displacement = self._synthesize(
                displacement_lm, initial_state.grid_type, initial_state.extend
            )
            gpc = self._synthesize(
                gpc_lm, initial_state.grid_type, initial_state.extend
            )

            slc_local = (-1.0 / self._g) * (self._g * displacement.data + gpc.data)

            raw_bathy = initial_bathy + slc_local
            if sediment_thickness_change is not None:
                raw_bathy -= sediment_thickness_change.data
            if dynamic_sea_level_change is not None:
                raw_bathy += dynamic_sea_level_change.data

            grid_template.data[:] = self._water_density * current_ocean_func * raw_bathy
            current_raw_water_mass = initial_state.integrate(grid_template)

            grid_template.data[:] = self._water_density * current_ocean_func
            current_ocean_density_area = initial_state.integrate(grid_template)

            eustatic_shift = (
                target_water_mass - current_raw_water_mass
            ) / current_ocean_density_area

            new_slc_data = slc_local + eustatic_shift
            new_bathy = raw_bathy + eustatic_shift

            potential_ocean = np.where(
                self._water_density * new_bathy
                - self._model.parameters.ice_density * new_ice
                > 0,
                1,
                0,
            )

            if exclude_caspian:
                new_ocean_func = np.where(caspian_mask == 1, 0, potential_ocean)
            else:
                new_ocean_func = potential_ocean

            max_slc = np.max(np.abs(slc_data))
            err = (
                np.max(np.abs(new_slc_data - slc_data)) / max_slc
                if max_slc > 0
                else 1.0
            )

            if verbose:
                print(
                    f"Non-Linear Iteration = {count + 1}, relative error = {err:6.4e}"
                )

            slc_data[:] = new_slc_data
            current_bathy[:] = new_bathy
            current_ocean_func[:] = new_ocean_func
            count += 1

        final_sea_level = SHGrid.from_array(current_bathy, grid=initial_state.grid)
        final_ice = SHGrid.from_array(new_ice, grid=initial_state.grid)

        # Inherit the Caspian masking policy properly from the initial state
        final_state = EarthState(
            final_ice, final_sea_level, self._model, exclude_caspian=exclude_caspian
        )

        final_slc = SHGrid.from_array(slc_data, grid=initial_state.grid)

        return final_state, final_slc, displacement, gpc, angular_velocity_change
