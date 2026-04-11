"""
Core physics solvers for the pyslfp library.

This module contains the SeaLevelEquation class, which acts as the primary
engine for calculating gravitationally consistent sea-level fingerprints and
handling rotational feedbacks.
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
    """

    def __init__(self, model: EarthModel):
        self.model = model

        self._g = self.model.parameters.gravitational_acceleration
        self._water_density = self.model.parameters.water_density
        self._radius = self.model.parameters.mean_sea_floor_radius

        self._rotation_factor = (
            np.sqrt((4 * np.pi) / 15.0)
            * self.model.parameters.rotation_frequency
            * self._radius**2
        )
        self._inertia_factor = (
            np.sqrt(5 / (12 * np.pi))
            * self.model.parameters.rotation_frequency
            * self._radius**3
            / (
                self.model.parameters.gravitational_constant
                * (
                    self.model.parameters.polar_moment_of_inertia
                    - self.model.parameters.equatorial_moment_of_inertia
                )
            )
        )
        self.solver_counter: int = 0

    def _expand(self, grid: SHGrid) -> SHCoeffs:
        return grid.expand(normalization="ortho", csphase=1)

    def _synthesize(self, coeffs: SHCoeffs, target_grid: str, extend: bool) -> SHGrid:
        """Synthesizes SHCoeffs explicitly respecting the grid sampling type."""
        return coeffs.expand(grid=target_grid, extend=extend)

    def _mean_sea_level_change(self, state: EarthState, direct_load: SHGrid) -> float:
        return -state.integrate(direct_load) / (self._water_density * state.ocean_area)

    def _ocean_average(self, state: EarthState, f: SHGrid) -> float:
        return state.integrate(state.ocean_function * f) / state.ocean_area

    def centrifugal_potential_change(
        self, state: EarthState, angular_velocity_change: np.ndarray
    ) -> SHGrid:
        coeffs = SHCoeffs.from_zeros(
            lmax=self.model.lmax, normalization="ortho", csphase=1
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
        ssh_change = sea_level_change + displacement

        if remove_rotational_contribution:
            centrifugal = self.centrifugal_potential_change(
                state, angular_velocity_change
            )
            ssh_change += centrifugal / self._g

        return ssh_change

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
        Solves the standard Sea Level Equation for a given surface mass load.
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
        Solves the generalized linear SLE for any combination of forcings.
        """
        if direct_load is not None and (
            direct_load.lmax != state.lmax or direct_load.grid != state.grid
        ):
            raise ValueError("Direct load grid must match the EarthState grid.")

        h_b = self.model.love_numbers.h[None, :, None]
        k_b = self.model.love_numbers.k[None, :, None]
        h_u_b = self.model.love_numbers.h_u[None, :, None]
        k_u_b = self.model.love_numbers.k_u[None, :, None]
        h_phi_b = self.model.love_numbers.h_phi[None, :, None]
        k_phi_b = self.model.love_numbers.k_phi[None, :, None]

        loads_present = False
        non_zero_rhs = False

        if direct_load is not None:
            loads_present = True
            mean_slc = self._mean_sea_level_change(state, direct_load)
            non_zero_rhs = non_zero_rhs or np.max(np.abs(direct_load.data)) > 0
        else:
            # Force exactly aligned empty grids
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

        self.solver_counter += 1

        load = direct_load + self._water_density * state.ocean_function * mean_slc
        angular_velocity_change = np.zeros(2)

        r = self._rotation_factor
        i = self._inertia_factor
        m = 1 / (
            self.model.parameters.polar_moment_of_inertia
            - self.model.parameters.equatorial_moment_of_inertia
        )
        ht = self.model.love_numbers.ht[2]
        kt = self.model.love_numbers.kt[2]

        # Force exact alignment for the iteration state
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
        and allows for ice, sediment, and dynamic sea level forcings.
        """
        # Validate grid alignments
        if (
            ice_thickness_change.lmax != initial_state.lmax
            or ice_thickness_change.grid != initial_state.grid
        ):
            raise ValueError(
                "Ice thickness change grid must match the initial state grid."
            )

        # 1. Setup Love Numbers and Constants
        h_b = self.model.love_numbers.h[None, :, None]
        k_b = self.model.love_numbers.k[None, :, None]
        r = self._rotation_factor
        i = self._inertia_factor
        ht = self.model.love_numbers.ht[2]
        kt = self.model.love_numbers.kt[2]

        # 2. Extract Initial State Arrays
        initial_bathy = initial_state.sea_level.data
        initial_ice = initial_state.ice_thickness.data
        initial_ocean_func = initial_state.ocean_function.data

        new_ice = initial_ice + ice_thickness_change.data

        # 3. Precompute Static Mass Loads
        static_load_data = self.model.parameters.ice_density * ice_thickness_change.data
        if sediment_thickness_change is not None:
            static_load_data += sediment_density * sediment_thickness_change.data

        # 4. Global Water Mass Conservation Target
        # The total water mass in the final ocean must equal initial water mass + meltwater
        meltwater_mass = -initial_state.integrate(
            self.model.parameters.ice_density * ice_thickness_change
        )
        initial_water_mass = initial_state.integrate(
            self._water_density * initial_state.ocean_function * initial_state.sea_level
        )
        target_water_mass = initial_water_mass + meltwater_mass

        # 5. Iteration State Variables
        current_ocean_func = initial_ocean_func.copy()
        current_bathy = initial_bathy.copy()
        slc_data = np.zeros_like(initial_bathy)
        angular_velocity_change = np.zeros(2)

        # We need an empty grid template to use the integrator safely
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

        self.solver_counter += 1

        # 6. The Non-Linear Loop
        while err > rtol and count < max_iterations:

            # --- A. Determine the new Load ---
            # Total Load = Static (Ice + Sed) + Change in Ocean Water Mass
            ocean_mass_change = self._water_density * (
                current_ocean_func * current_bathy - initial_ocean_func * initial_bathy
            )
            total_load_data = static_load_data + ocean_mass_change

            # Expand load to Spherical Harmonics
            grid_template.data[:] = total_load_data
            load_lm = self._expand(grid_template)

            # --- B. Solve Elasticity & Gravity ---
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

            # Local Sea Level Change (spatially varying, no eustatic shift yet)
            slc_local = (-1.0 / self._g) * (self._g * displacement.data + gpc.data)

            # --- C. Enforce Mass Conservation (Solve for Eustatic Shift) ---
            # Raw bathymetry = Initial + SLC_local - Sediment + DSL
            raw_bathy = initial_bathy + slc_local
            if sediment_thickness_change is not None:
                raw_bathy -= sediment_thickness_change.data
            if dynamic_sea_level_change is not None:
                raw_bathy += dynamic_sea_level_change.data

            # Calculate the water mass currently captured by the raw bathymetry
            grid_template.data[:] = self._water_density * current_ocean_func * raw_bathy
            current_raw_water_mass = initial_state.integrate(grid_template)

            # Calculate current ocean area (weighted by density for the shift equation)
            grid_template.data[:] = self._water_density * current_ocean_func
            current_ocean_density_area = initial_state.integrate(grid_template)

            # The eustatic shift needed to balance the global water budget
            eustatic_shift = (
                target_water_mass - current_raw_water_mass
            ) / current_ocean_density_area

            # --- D. Update Topography and Flotation ---
            new_slc_data = slc_local + eustatic_shift
            new_bathy = raw_bathy + eustatic_shift

            # 2. Update Ocean Function using the state's policy
            potential_ocean = np.where(
                self._water_density * new_bathy
                - self.model.parameters.ice_density * new_ice
                > 0,
                1,
                0,
            )

            if exclude_caspian:
                # Ensure the Caspian basin remains land-locked land
                new_ocean_func = np.where(caspian_mask == 1, 0, potential_ocean)
            else:
                new_ocean_func = potential_ocean

            # --- E. Check Convergence ---
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

        # 7. Construct Final State and Return Outputs
        final_sea_level = SHGrid.from_array(current_bathy, grid=initial_state.grid)
        final_ice = SHGrid.from_array(new_ice, grid=initial_state.grid)

        # Create the new equilibrium background state
        final_state = EarthState(final_ice, final_sea_level, self.model)

        # Package the outputs identically to the linear solver
        final_slc = SHGrid.from_array(slc_data, grid=initial_state.grid)

        return final_state, final_slc, displacement, gpc, angular_velocity_change
