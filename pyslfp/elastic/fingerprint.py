"""The elastic fingerprint driver.

An elastic earth responds instantaneously, so a fingerprint is one
equilibrium solve of the sea level equation: this module wraps the
engine of `pyslfp.core` with a fixed background state, which is all
the elastic problem adds to it.  The operator layer for inference,
built on these solves, is `pyslfp.linear_operators`.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from pyshtools import SHCoeffs, SHGrid

from ..core import EarthState, SeaLevelEquation

__all__ = ["LinearSeaLevelEquation"]


class LinearSeaLevelEquation:
    """
    Specialisation of the SeaLevelEquation class that is limited to
    the solution of linear problems, and for which the initial state
    is taken in during construction.
    """

    def __init__(self, state: EarthState, /) -> None:
        """
        Initialises the linear sea level equation solver.

        Args:
            state (EarthState): The background state the fingerprints
                perturb.
        """

        self._state = state
        self._sle = SeaLevelEquation(state.model)

    @staticmethod
    def from_defaults(*, lmax: int = 256) -> LinearSeaLevelEquation:
        """
        Sets up the linear solver with the default parameters: PREM
        with the standard non-dimensionalisation, and present-day
        ICE-7G as the background state.

        Args:
            lmax (int): Truncation degree for the discretisation. Defaults to 256.
        """
        state = EarthState.from_defaults(lmax=lmax)
        return LinearSeaLevelEquation(state)

    @staticmethod
    def for_testing(lmax: int) -> LinearSeaLevelEquation:
        """
        Sets up the linear solver on the analytical testing state:
        PREM with the standard non-dimensionalisation, and the simple
        analytical geometry of `EarthState.for_testing`.
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
        Solves the standard linear sea level equation for a surface mass load.

        Args:
            direct_load (SHGrid): The mass redistribution forcing the system.
            rotational_feedbacks (bool | str): True for the full feedback,
                "axial" for the axial component alone, False for none.
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
        displacement_response: Optional[SHCoeffs] = None,
        gravitational_potential_response: Optional[SHCoeffs] = None,
        inertia_moment_change: float = 0.0,
        angular_momentum_change: Optional[np.ndarray] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1e-9,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the generalised linear sea level equation for an
        arbitrary combination of forcings, on the construction's
        background state; see
        `SeaLevelEquation.solve_generalised_equation` for the engine.

        Args:
            direct_load (Optional[SHGrid]): Standard surface mass forcing.
            displacement_load (Optional[SHGrid]): External vertical surface displacement forcing.
            gravitational_potential_load (Optional[SHGrid]): External gravitational potential forcing.
            displacement_response (Optional[SHCoeffs]): A ready displacement
                response in coefficient form, joined to the static terms
                with no Love number applied.
            gravitational_potential_response (Optional[SHCoeffs]): The
                matching ready potential response.
            inertia_moment_change (float): The inertia-moment change
                accompanying the responses, for the axial feedback.
            angular_momentum_change (Optional[np.ndarray]): External angular momentum perturbation, [x, y, z].
            rotational_feedbacks (bool | str): True for the full feedback,
                "axial" for the axial component alone, False for none.
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
            displacement_response=displacement_response,
            gravitational_potential_response=gravitational_potential_response,
            inertia_moment_change=inertia_moment_change,
            angular_momentum_change=angular_momentum_change,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            max_iterations=max_iterations,
            verbose=verbose,
        )
