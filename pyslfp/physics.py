"""
Core physics solvers for the pyslfp library.

This module contains the SeaLevelEquation class, which acts as the primary
engine for calculating gravitationally consistent sea-level fingerprints,
handling rotational feedbacks, and computing non-linear shoreline migration.

Numerical scheme
----------------
Both solvers iterate on the surface load. Within each iteration the sea
level change is synthesised directly from the load coefficients using the
combined factor -(h_l + k_l / g), so a single inverse transform is needed
per iteration; displacement and gravitational potential are synthesised
once from the converged load. Surface integrals use the quadrature weights
held by the EarthModel rather than a truncated transform.

The rotational feedback lives at degree 2: the transverse angular velocity
components at order 1 and the axial component at order 0, the latter
with a degree-0 part through the axial Love numbers of the table. Each is
a scalar fixed point solved in closed form within every iteration, so the
returned displacement, potential and angular velocity are all consistent
with the same load; see ``SeaLevelEquation._rotation``.

The linear solver accelerates the fixed-point iteration with Anderson
mixing on the load coefficients. Set ``SeaLevelEquation.anderson_memory``
to zero to recover a plain Picard iteration.

Spherical harmonic transforms are performed by pyshtools. With the ``ducc0``
package installed (a declared dependency) pyshtools uses its multi-threaded
backend, taking the thread count from the ``OMP_NUM_THREADS`` environment
variable (all cores if unset). When parallelising over many solves with
worker processes, set ``OMP_NUM_THREADS=1`` before Python starts.
"""

from __future__ import annotations
from typing import Optional, Tuple
import warnings

import numpy as np
from pyshtools import SHCoeffs, SHGrid

from .core import EarthModel
from .state import EarthState


class _AndersonMixer:
    """
    Anderson acceleration for a fixed-point iteration x = G(x).

    The last ``memory`` differences of iterates and residuals are kept in
    ring buffers and combined in the least-squares sense (Walker and Ni,
    2011). The small Gram matrix is updated incrementally, so each update
    costs a handful of passes over the vector. Dot products are formed with
    ``numpy.einsum`` rather than BLAS: BLAS worker threads spin-wait after a
    call and slow down single-threaded transforms that follow.
    """

    def __init__(self, memory: int) -> None:
        self._memory = memory
        self._x_prev: Optional[np.ndarray] = None
        self._f_prev: Optional[np.ndarray] = None
        self._d_x: Optional[np.ndarray] = None  # (memory, n) ring buffer
        self._d_f: Optional[np.ndarray] = None  # (memory, n) ring buffer
        self._gram: Optional[np.ndarray] = None  # (memory, memory)
        self._count = 0  # differences stored so far (capped at memory)
        self._slot = 0  # next ring-buffer slot to overwrite

    def update(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """Returns the next iterate given the current one and G applied to it."""
        if self._memory <= 0:
            return gx

        x_flat = x.ravel()
        f_flat = gx.ravel() - x_flat

        if self._x_prev is None:
            n = x_flat.size
            self._d_x = np.empty((self._memory, n))
            self._d_f = np.empty((self._memory, n))
            self._gram = np.zeros((self._memory, self._memory))
            self._x_prev = x_flat.copy()
            self._f_prev = f_flat
            return gx

        slot = self._slot
        np.subtract(x_flat, self._x_prev, out=self._d_x[slot])
        np.subtract(f_flat, self._f_prev, out=self._d_f[slot])
        self._x_prev = x_flat.copy()
        self._f_prev = f_flat

        self._count = min(self._count + 1, self._memory)
        self._slot = (slot + 1) % self._memory
        m = self._count

        # Refresh the row and column of the Gram matrix for the new entry
        gram_row = np.einsum("ij,j->i", self._d_f[:m], self._d_f[slot])
        self._gram[slot, :m] = gram_row
        self._gram[:m, slot] = gram_row

        rhs = np.einsum("ij,j->i", self._d_f[:m], f_flat)
        gamma = np.linalg.lstsq(self._gram[:m, :m], rhs, rcond=None)[0]

        x_new = gx.ravel() - np.einsum("i,ij->j", gamma, self._d_x[:m])
        x_new -= np.einsum("i,ij->j", gamma, self._d_f[:m])
        return x_new.reshape(x.shape)


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
        parameters = model.parameters

        # Cache frequently used constants locally for performance
        self._g = parameters.gravitational_acceleration
        self._water_density = parameters.water_density
        self._ice_density = parameters.ice_density
        self._radius = parameters.mean_sea_floor_radius
        self._rotation_factor = parameters.rotation_factor
        self._inertia_factor = parameters.inertia_factor
        self._axial_rotation_factor = parameters.axial_rotation_factor
        self._axial_inertia_factor = parameters.axial_inertia_factor
        self._uniform_rotation_factor = parameters.uniform_rotation_factor
        self._inverse_inertia_difference = 1.0 / (
            parameters.polar_moment_of_inertia - parameters.equatorial_moment_of_inertia
        )
        self._inverse_polar_inertia = 1.0 / parameters.polar_moment_of_inertia

        # Love numbers shaped for broadcasting over coefficient arrays
        love_numbers = model.love_numbers
        self._h = love_numbers.h[None, :, None]
        self._k = love_numbers.k[None, :, None]
        self._h_u = love_numbers.h_u[None, :, None]
        self._k_u = love_numbers.k_u[None, :, None]
        self._h_phi = love_numbers.h_phi[None, :, None]
        self._k_phi = love_numbers.k_phi[None, :, None]

        # Combined factor mapping load coefficients to sea level change
        self._slc_factor = -(self._h + self._k / self._g)

        # Degree-2 quantities for the rotational feedback
        self._h2 = love_numbers.h[2]
        self._k2 = love_numbers.k[2]
        self._ht2 = love_numbers.h_t[2]
        self._kt2 = love_numbers.k_t[2]
        # Degree-0 quantities for the axial feedback: the response to the
        # uniform part of the centrifugal potential and the inertia moments
        # of the degree-0 responses, zero when the table does not hold them
        if love_numbers.has_axial:
            self._hc = love_numbers.h_c
            self._kc = love_numbers.k_c
            self._mu = love_numbers.m_u
            self._mphi = love_numbers.m_phi
            self._mc = love_numbers.m_c
        else:
            self._hc = self._kc = self._mu = self._mphi = self._mc = 0.0
        omega = parameters.rotation_frequency
        polar_inertia = parameters.polar_moment_of_inertia
        # (4/3) Omega / C on the moments, and the inertia of a uniform shell
        # (2/3) Omega a^4 sqrt(4 pi) / C on degree-0 mass-like sources
        self._trace_factor = (4.0 / 3.0) * omega / polar_inertia
        self._shell_factor = (
            (2.0 / 3.0) * omega * self._radius**4 * np.sqrt(4.0 * np.pi) / polar_inertia
        )

        # Fixed-point denominators for the transverse (order 1) and axial
        # (order 0) components; the axial sign follows from the polar
        # entry -C of the inertia matrix in the Euler equation.
        self._rotation_denominator = (
            1.0 - self._inertia_factor * self._kt2 * self._rotation_factor
        )
        self._axial_rotation_denominator = (
            1.0
            + self._axial_inertia_factor * self._kt2 * self._axial_rotation_factor
            + self._trace_factor * self._mc * self._uniform_rotation_factor
        )

        # Number of previous iterates retained by the Anderson acceleration
        # in the linear solver. Zero gives a plain Picard iteration.
        self.anderson_memory: int = 5

        self._solver_counter: int = 0

    @property
    def solver_counter(self) -> int:
        """The number of times the solver has been executed."""
        return self._solver_counter

    # ---------------------------------------------------------#
    #                 Internal Helpers                         #
    # ---------------------------------------------------------#

    def _mean_sea_level_change(self, state: EarthState, direct_load: SHGrid) -> float:
        """Computes the mean eustatic sea level change for a given load."""
        return -self._model.integrate(direct_load) / (
            self._water_density * state.ocean_area
        )

    def _ocean_average(self, state: EarthState, f: SHGrid) -> float:
        """Computes the spatial average of a field over the oceans."""
        return (
            self._model._integrate_data(state.ocean_function.data * f.data)
            / state.ocean_area
        )

    def _expand(self, data: np.ndarray) -> np.ndarray:
        """Forward transform of grid values, returning the coefficient array."""
        grid = SHGrid.from_array(data, grid=self._model.grid, copy=False)
        return self._model.expand_field(grid).coeffs

    def _synthesise(self, coeffs: np.ndarray) -> np.ndarray:
        """Inverse transform of a coefficient array, returning grid values."""
        clm = SHCoeffs.from_array(
            coeffs,
            normalization=self._model.normalization,
            csphase=self._model.csphase,
            copy=False,
        )
        return self._model.expand_coefficient(clm).data

    def _rotation(
        self,
        load_2: np.ndarray,
        /,
        *,
        load_0: float = 0.0,
        static_disp_2: Optional[np.ndarray] = None,
        static_grav_2: Optional[np.ndarray] = None,
        displacement_load_0: float = 0.0,
        potential_load_0: float = 0.0,
        angular_momentum_change: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solves the rotational feedback exactly.

        The components of the angular velocity change orthogonal to the
        rotation axis are set by the degree-2, order-1 coefficients of the
        potential and the axial component by the order-0 coefficient
        (MacCullagh's formula), while each feeds back through the tidal
        Love numbers at the same order. The axial component also depends on
        the trace of the inertia perturbation, which the degree-2 potential
        does not see: the inertia of degree-0 mass-like sources (the load
        and the potential load, as a uniform shell) and the inertia moments
        of the degree-0 deformation they and the uniform part of the
        centrifugal potential drive, through the axial Love numbers. A
        table without those numbers gives the deformation terms as zero.

        Args:
            load_2: The load coefficients of degree 2 at orders 0 and 1,
                shaped (2, 2) with rows (cos, sin) and columns (m=0, m=1).
            load_0: The degree-0 load coefficient, zero for a load that
                conserves mass.
            static_disp_2: The same block of displacement from static loads.
            static_grav_2: The same block of potential from static loads.
            displacement_load_0: The degree-0 displacement load coefficient.
            potential_load_0: The degree-0 potential load coefficient.
            angular_momentum_change: External angular momentum perturbation,
                [x, y, z].

        Returns:
            Tuple of angular velocity change [omega_x, omega_y, omega_z],
            the (2, 2) blocks of displacement coefficients, gravitational
            potential coefficients (without the centrifugal term) and
            centrifugal potential coefficients, and the degree-0 centrifugal
            potential coefficient.
        """
        grav_forcing = self._k2 * load_2
        disp_forcing = self._h2 * load_2
        if static_grav_2 is not None:
            grav_forcing = grav_forcing + static_grav_2
        if static_disp_2 is not None:
            disp_forcing = disp_forcing + static_disp_2

        transverse = self._inertia_factor * grav_forcing[:, 1]
        axial = (
            -self._axial_inertia_factor * grav_forcing[0, 0]
            - self._trace_factor
            * (
                (self._mu + self._mphi) * load_0
                + self._mu * displacement_load_0
                + self._mphi * potential_load_0
            )
            - self._shell_factor * (load_0 + potential_load_0)
        )
        if angular_momentum_change is not None:
            transverse = (
                transverse
                - self._inverse_inertia_difference * angular_momentum_change[:2]
            )
            axial = axial + self._inverse_polar_inertia * angular_momentum_change[2]

        omega = np.empty(3)
        omega[:2] = transverse / self._rotation_denominator
        omega[2] = axial / self._axial_rotation_denominator

        centrifugal = np.zeros((2, 2))
        centrifugal[:, 1] = self._rotation_factor * omega[:2]
        centrifugal[0, 0] = self._axial_rotation_factor * omega[2]
        centrifugal_0 = self._uniform_rotation_factor * omega[2]
        disp_2 = disp_forcing + self._ht2 * centrifugal
        grav_2 = grav_forcing + self._kt2 * centrifugal
        return omega, disp_2, grav_2, centrifugal, centrifugal_0

    def _warn_not_converged(self, name: str, err: float, count: int) -> None:
        warnings.warn(
            f"{name} did not converge in {count} iterations "
            f"(final relative error {err:.3e}).",
            RuntimeWarning,
            stacklevel=3,
        )

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
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the standard linear Sea Level Equation for a surface mass load.

        Args:
            state (EarthState): The unperturbed background Earth state.
            direct_load (SHGrid): The mass redistribution forcing the system.
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence, measured as the
                change in sea level over the oceans between iterations relative
                to its magnitude.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.

        Returns:
            Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]: A 4-tuple containing:
                - Relative Sea Level Change
                - Vertical Displacement
                - Gravity Potential Change
                - Angular Velocity Change [omega_x, omega_y, omega_z]
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
        rtol: float = 1e-9,
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
            angular_momentum_change (Optional[np.ndarray]): External angular momentum perturbation, [x, y, z].
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence, measured as the
                change in sea level over the oceans between iterations relative
                to its magnitude.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.

        Returns:
            Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]: The physical response fields:
                (Sea Level Change, Displacement, Gravitational Potential, Angular Velocity)
        """
        model = self._model
        g = self._g
        water_density = self._water_density

        loads_present = False
        non_zero_rhs = False

        if direct_load is not None:
            model.check_field(direct_load)
            loads_present = True
            non_zero_rhs = non_zero_rhs or np.max(np.abs(direct_load.data)) > 0

        static_disp = None
        static_grav = None

        if displacement_load is not None:
            model.check_field(displacement_load)
            loads_present = True
            non_zero_rhs = non_zero_rhs or np.max(np.abs(displacement_load.data)) > 0
            disp_lm = self._expand(displacement_load.data)
            static_disp = self._h_u * disp_lm
            static_grav = self._k_u * disp_lm

        if gravitational_potential_load is not None:
            model.check_field(gravitational_potential_load)
            loads_present = True
            non_zero_rhs = (
                non_zero_rhs or np.max(np.abs(gravitational_potential_load.data)) > 0
            )
            grav_lm = self._expand(gravitational_potential_load.data)
            if static_disp is None:
                static_disp = self._h_phi * grav_lm
                static_grav = self._k_phi * grav_lm
            else:
                static_disp += self._h_phi * grav_lm
                static_grav += self._k_phi * grav_lm

        if angular_momentum_change is not None:
            loads_present = True
            angular_momentum_change = np.asarray(angular_momentum_change, dtype=float)
            non_zero_rhs = non_zero_rhs or np.max(np.abs(angular_momentum_change)) > 0

        if not loads_present or not non_zero_rhs:
            return (
                model.zero_grid(),
                model.zero_grid(),
                model.zero_grid(),
                np.zeros(3),
            )

        self._solver_counter += 1

        ocean_function = state.ocean_function.data
        ocean_area = state.ocean_area

        if direct_load is not None:
            direct_data = direct_load.data
            mean_slc = -model._integrate_data(direct_data) / (
                water_density * ocean_area
            )
        else:
            direct_data = np.zeros_like(ocean_function)
            mean_slc = 0.0

        if static_disp is not None:
            static_slc = -(static_disp + static_grav / g)
            static_disp_2 = static_disp[:, 2, :2].copy()
            static_grav_2 = static_grav[:, 2, :2].copy()
        else:
            static_slc = None
            static_disp_2 = None
            static_grav_2 = None
        displacement_load_0 = 0.0 if displacement_load is None else disp_lm[0, 0, 0]
        potential_load_0 = (
            0.0 if gravitational_potential_load is None else grav_lm[0, 0, 0]
        )

        def sea_level_from_load(load_lm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Synthesises the mass-conserving sea level change for a load."""
            slc_lm = self._slc_factor * load_lm
            if static_slc is not None:
                slc_lm += static_slc
            omega = np.zeros(3)
            if rotational_feedbacks:
                omega, disp_2, grav_2, centrifugal, centrifugal_0 = self._rotation(
                    load_lm[:, 2, :2],
                    load_0=load_lm[0, 0, 0],
                    static_disp_2=static_disp_2,
                    static_grav_2=static_grav_2,
                    displacement_load_0=displacement_load_0,
                    potential_load_0=potential_load_0,
                    angular_momentum_change=angular_momentum_change,
                )
                slc_lm[:, 2, :2] = -(disp_2 + (grav_2 + centrifugal) / g)
                # uniform, and so absorbed by mass conservation below
                slc_lm[0, 0, 0] -= (self._hc + (self._kc + 1.0) / g) * centrifugal_0
            slc = self._synthesise(slc_lm)
            slc += mean_slc - model._integrate_data(ocean_function * slc) / ocean_area
            return slc, omega

        load_lm = self._expand(direct_data + water_density * ocean_function * mean_slc)
        mixer = _AndersonMixer(self.anderson_memory)
        iter_limit = max_iterations if max_iterations is not None else 1000

        ocean_slc_prev = None
        err = np.inf
        converged = False
        count = 0

        while count < iter_limit:
            sea_level_change, angular_velocity_change = sea_level_from_load(load_lm)
            ocean_slc = ocean_function * sea_level_change

            if ocean_slc_prev is not None:
                scale = np.max(np.abs(ocean_slc))
                err = (
                    np.max(np.abs(ocean_slc - ocean_slc_prev)) / scale
                    if scale > 0
                    else 0.0
                )
                if verbose:
                    print(f"Iteration = {count}, relative error = {err:6.4e}")
                if err <= rtol:
                    converged = True
                    break

            ocean_slc_prev = ocean_slc
            load_new_lm = self._expand(direct_data + water_density * ocean_slc)
            load_lm = mixer.update(load_lm, load_new_lm)
            count += 1

        if not converged:
            self._warn_not_converged("Sea level equation solver", err, count)

        # Synthesise the remaining fields from the same load as the sea level
        displacement_lm = self._h * load_lm
        potential_lm = self._k * load_lm
        if static_disp is not None:
            displacement_lm += static_disp
            potential_lm += static_grav
        if rotational_feedbacks:
            _, disp_2, grav_2, _, centrifugal_0 = self._rotation(
                load_lm[:, 2, :2],
                load_0=load_lm[0, 0, 0],
                static_disp_2=static_disp_2,
                static_grav_2=static_grav_2,
                displacement_load_0=displacement_load_0,
                potential_load_0=potential_load_0,
                angular_momentum_change=angular_momentum_change,
            )
            displacement_lm[:, 2, :2] = disp_2
            potential_lm[:, 2, :2] = grav_2
            displacement_lm[0, 0, 0] += self._hc * centrifugal_0
            potential_lm[0, 0, 0] += self._kc * centrifugal_0

        grid = model.grid
        return (
            SHGrid.from_array(sea_level_change, grid=grid, copy=False),
            SHGrid.from_array(self._synthesise(displacement_lm), grid=grid, copy=False),
            SHGrid.from_array(self._synthesise(potential_lm), grid=grid, copy=False),
            angular_velocity_change,
        )

    def solve_nonlinear_equation(
        self,
        initial_state: EarthState,
        /,
        *,
        ice_thickness_change: Optional[SHGrid] = None,
        sediment_thickness_change: Optional[SHGrid] = None,
        dynamic_sea_level_change: Optional[SHGrid] = None,
        sediment_density: Optional[float] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: int = 50,
        verbose: bool = False,
    ) -> Tuple[EarthState, SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the full non-linear Sea Level Equation with shifting shorelines.

        Incorporates dynamic ocean function updates, explicit mass conservation,
        and allows for combined ice, sediment, and dynamic sea level forcings.

        Args:
            initial_state (EarthState): The unperturbed background Earth state.
            ice_thickness_change (Optional[SHGrid]): Change in ice thickness.
            sediment_thickness_change (Optional[SHGrid]): Change in sediment.
            dynamic_sea_level_change (Optional[SHGrid]): Ocean dynamic sea level forcing.
            sediment_density (Optional[float]): Non-dimensional density of the
                sediment layer. Defaults to 2300 kg/m^3 non-dimensionalised.
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
                - Angular Velocity Change [omega_x, omega_y, omega_z]
        """
        model = self._model
        g = self._g
        water_density = self._water_density
        ice_density = self._ice_density

        if sediment_density is None:
            sediment_density = 2300.0 / model.parameters.density_scale

        for field in (
            ice_thickness_change,
            sediment_thickness_change,
            dynamic_sea_level_change,
        ):
            if field is not None:
                model.check_field(field)

        initial_bathy = initial_state.sea_level.data
        initial_ice = initial_state.ice_thickness.data
        initial_ocean_func = initial_state.ocean_function.data

        if ice_thickness_change is not None:
            new_ice = initial_ice + ice_thickness_change.data
        else:
            new_ice = initial_ice.copy()

        if sediment_thickness_change is not None:
            sediment_load = sediment_density * sediment_thickness_change.data
            sediment_mass_change = model._integrate_data(sediment_load)
        else:
            sediment_load = None
            sediment_mass_change = 0.0

        initial_water_mass = water_density * model._integrate_data(
            initial_ocean_func * initial_bathy
        )

        current_ocean_func = initial_ocean_func.copy()
        current_bathy = initial_bathy.copy()
        slc_data = np.zeros_like(initial_bathy)
        angular_velocity_change = np.zeros(3)

        exclude_caspian = initial_state.exclude_caspian
        caspian_mask = (
            initial_state.caspian_sea_projection(value=0).data
            if exclude_caspian
            else None
        )

        self._solver_counter += 1

        err = np.inf
        converged = False
        count = 0
        disp_2 = grav_2 = None
        centrifugal_0 = 0.0

        while count < max_iterations:
            grounded_ice_change = ice_density * (
                (1.0 - current_ocean_func) * new_ice
                - (1.0 - initial_ocean_func) * initial_ice
            )

            ocean_mass_change = water_density * (
                current_ocean_func * current_bathy - initial_ocean_func * initial_bathy
            )

            total_load = grounded_ice_change + ocean_mass_change
            if sediment_load is not None:
                total_load += sediment_load

            load_lm = self._expand(total_load)

            slc_lm = self._slc_factor * load_lm
            if rotational_feedbacks:
                angular_velocity_change, disp_2, grav_2, centrifugal, centrifugal_0 = (
                    self._rotation(load_lm[:, 2, :2], load_0=load_lm[0, 0, 0])
                )
                slc_lm[:, 2, :2] = -(disp_2 + (grav_2 + centrifugal) / g)
                slc_lm[0, 0, 0] -= (self._hc + (self._kc + 1.0) / g) * centrifugal_0

            slc_local = self._synthesise(slc_lm)

            raw_bathy = initial_bathy + slc_local
            if sediment_thickness_change is not None:
                raw_bathy -= sediment_thickness_change.data
            if dynamic_sea_level_change is not None:
                raw_bathy += dynamic_sea_level_change.data

            total_ice_mass_change = model._integrate_data(grounded_ice_change)
            target_water_mass = (
                initial_water_mass - total_ice_mass_change - sediment_mass_change
            )
            current_raw_water_mass = water_density * model._integrate_data(
                current_ocean_func * raw_bathy
            )
            current_ocean_density_area = water_density * model._integrate_data(
                current_ocean_func
            )

            eustatic_shift = (
                target_water_mass - current_raw_water_mass
            ) / current_ocean_density_area

            new_slc_data = slc_local + eustatic_shift
            new_bathy = raw_bathy + eustatic_shift

            new_ocean_func = np.where(
                water_density * new_bathy - ice_density * new_ice > 0, 1.0, 0.0
            )
            if exclude_caspian:
                new_ocean_func = np.where(caspian_mask == 1, 0.0, new_ocean_func)

            max_slc = np.max(np.abs(slc_data))
            if max_slc > 0:
                err = np.max(np.abs(new_slc_data - slc_data)) / max_slc
            elif np.max(np.abs(new_slc_data)) == 0:
                err = 0.0  # zero forcing: nothing to iterate
            else:
                err = 1.0

            if verbose:
                print(
                    f"Non-Linear Iteration = {count + 1}, relative error = {err:6.4e}"
                )

            slc_data[:] = new_slc_data
            current_bathy[:] = new_bathy
            current_ocean_func[:] = new_ocean_func
            count += 1

            if err <= rtol:
                converged = True
                break

        if not converged:
            self._warn_not_converged("Non-linear sea level equation solver", err, count)

        # Displacement and potential for the load of the final iteration
        displacement_lm = self._h * load_lm
        potential_lm = self._k * load_lm
        if rotational_feedbacks:
            displacement_lm[:, 2, :2] = disp_2
            potential_lm[:, 2, :2] = grav_2
            displacement_lm[0, 0, 0] += self._hc * centrifugal_0
            potential_lm[0, 0, 0] += self._kc * centrifugal_0

        grid = model.grid
        final_sea_level = SHGrid.from_array(current_bathy, grid=grid)
        final_ice = SHGrid.from_array(new_ice, grid=grid)

        # Inherit the Caspian masking policy properly from the initial state
        final_state = EarthState(
            final_ice, final_sea_level, model, exclude_caspian=exclude_caspian
        )

        return (
            final_state,
            SHGrid.from_array(slc_data, grid=grid),
            SHGrid.from_array(self._synthesise(displacement_lm), grid=grid, copy=False),
            SHGrid.from_array(self._synthesise(potential_lm), grid=grid, copy=False),
            angular_velocity_change,
        )


class LinearSeaLevelEquation:
    """
    Specialisation of the SeaLevelEquation class that is limited to
    the solution of linear problems, and for which the initial state
    is taken in during construction.
    """

    def __init__(self, state: EarthState, /) -> None:
        """
        Initializes the linear Sea Level Equation solver.

        Args:
            state (Earthstate): The initial state for the earth model.
        """

        self._state = state
        self._sle = SeaLevelEquation(state.model)

    @staticmethod
    def from_defaults(*, lmax: int = 256) -> LinearSeaLevelEquation:
        """
        Sets up the linear solver using the default parameters.
        The Earth model is PREM with standard non-dimensionalisations,
        while the initial state is present-day Ice-7g.

        Args:
            lmax (int): Truncation degree for the discretisation. Defaults to 256.
        """
        state = EarthState.from_defaults(lmax=lmax)
        return LinearSeaLevelEquation(state)

    @staticmethod
    def for_testing(lmax: int) -> LinearSeaLevelEquation:
        """
        Sets up the linear solver using the testing state.
        The Earth model is PREM weith standard non-dimensionalisartion,
        while the initial state takes a simple analytical form.
        """
        state = EarthState.for_testing(lmax)
        return LinearSeaLevelEquation(state)

    @property
    def state(self) -> EarthState:
        """
        Returns the background state.
        """
        return self._state

    def solve_sea_level_equation(
        self,
        direct_load: SHGrid,
        /,
        *,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the standard linear Sea Level Equation for a surface mass load.

        Args:
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
                - Angular Velocity Change [omega_x, omega_y, omega_z]
        """
        return self._sle.solve_sea_level_equation(
            self.state,
            direct_load,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def solve_generalised_equation(
        self,
        /,
        *,
        direct_load: Optional[SHGrid] = None,
        displacement_load: Optional[SHGrid] = None,
        gravitational_potential_load: Optional[SHGrid] = None,
        angular_momentum_change: Optional[np.ndarray] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the generalized linear SLE for an arbitrary combination of forcings.

        Useful for adjoint calculations or complex, multi-physical inversions.

        Args:
            direct_load (Optional[SHGrid]): Standard surface mass forcing.
            displacement_load (Optional[SHGrid]): External vertical surface displacement forcing.
            gravitational_potential_load (Optional[SHGrid]): External gravitational potential forcing.
            angular_momentum_change (Optional[np.ndarray]): External angular momentum perturbation, [x, y, z].
            rotational_feedbacks (bool): Whether to calculate polar wander effects.
            rtol (float): The relative tolerance for convergence.
            max_iterations (Optional[int]): Hard limit on iteration count.
            verbose (bool): If True, prints iteration metrics.

        Returns:
            Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]: The physical response fields:
                (Sea Level Change, Displacement, Gravitational Potential, Angular Velocity)
        """
        return self._sle.solve_generalised_equation(
            self._state,
            direct_load=direct_load,
            displacement_load=displacement_load,
            gravitational_potential_load=gravitational_potential_load,
            angular_momentum_change=angular_momentum_change,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )
