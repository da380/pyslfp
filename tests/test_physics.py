"""
Test suite for the core Sea Level Equation solvers.
"""

import pytest
import numpy as np
from pyshtools import SHGrid

from pyslfp.state import EarthState
from pyslfp.physics import SeaLevelEquation, LinearSeaLevelEquation


# ==================================================================== #
#                          Fixtures & Helpers                          #
# ==================================================================== #


@pytest.fixture(scope="module")
def analytical_state():
    """Provides a low-resolution, mathematically smooth EarthState."""
    return EarthState.for_testing(lmax=16)


@pytest.fixture(scope="module")
def linear_solver(analytical_state):
    """Provides a LinearSeaLevelEquation solver wrapped around the analytical state."""
    return LinearSeaLevelEquation(analytical_state)


def random_load(state: EarthState) -> SHGrid:
    """Generates a physically plausible, randomized surface mass load."""
    f = np.random.uniform(0.1, 0.9)
    load1 = state.northern_hemisphere_load(fraction=1.0)
    load2 = state.southern_hemisphere_load(fraction=1.0)
    combined_load = f * load1 + (1 - f) * load2
    return combined_load


# ==================================================================== #
#                  1. Basic Solvers & Mass Conservation                #
# ==================================================================== #


def test_zero_load_input(linear_solver):
    """Sanity check: A zero load should produce a zero response across all fields."""
    zero_load = linear_solver.state.model.zero_grid()
    slc, disp, potc, avc = linear_solver.solve_sea_level_equation(zero_load)

    assert np.allclose(slc.data, 0.0)
    assert np.allclose(disp.data, 0.0)
    assert np.allclose(potc.data, 0.0)
    assert np.allclose(avc, 0.0)


def test_mass_conservation(analytical_state):
    """
    Check for mass conservation: the mass of water added to the ocean
    must equal the mass removed by the direct load (e.g., melted ice).
    """
    sle = SeaLevelEquation(analytical_state.model)
    direct_load = random_load(analytical_state)

    mass_removed = -analytical_state.model.integrate(direct_load)

    # Internal method calculates eustatic sea level change
    mean_slc = sle._mean_sea_level_change(analytical_state, direct_load)

    mass_added = mean_slc * analytical_state.ocean_area * sle._water_density

    assert np.isclose(mass_removed, mass_added, rtol=1e-5)


# ==================================================================== #
#                  2. Reciprocity (Adjoint) Tests                      #
# ==================================================================== #


@pytest.mark.parametrize("rotational_feedbacks", [True, False])
def test_sea_level_reciprocity(linear_solver, rotational_feedbacks):
    """
    Check the fundamental sea level reciprocity relation using random loads.
    Integral of (Load_1 * SLC_2) == Integral of (Load_2 * SLC_1)
    """
    load_1 = random_load(linear_solver.state)
    load_2 = random_load(linear_solver.state)
    rtol = 1e-9

    slc_1, _, _, _ = linear_solver.solve_sea_level_equation(
        load_1, rotational_feedbacks=rotational_feedbacks, rtol=rtol
    )
    slc_2, _, _, _ = linear_solver.solve_sea_level_equation(
        load_2, rotational_feedbacks=rotational_feedbacks, rtol=rtol
    )

    lhs = linear_solver.state.model.integrate(load_2 * slc_1)
    rhs = linear_solver.state.model.integrate(load_1 * slc_2)

    assert np.isclose(lhs, rhs, rtol=1e-6)


# ==================================================================== #
#                  3. Non-Linear Solver Tests                          #
# ==================================================================== #


def test_nonlinear_solver_smoke_test(analytical_state):
    """
    Ensures the non-linear solver with migrating shorelines executes successfully
    and correctly returns an updated EarthState object.
    """
    sle = SeaLevelEquation(analytical_state.model)

    # Create a non-dimensional ice melt scenario (melt 100 meters globally)
    ice_melt_nd = -100.0 / analytical_state.model.parameters.length_scale
    ice_thickness_change = analytical_state.model.zero_grid()

    # Only melt where there is currently ice
    ice_thickness_change.data = np.where(
        analytical_state.ice_thickness.data > 0, ice_melt_nd, 0.0
    )

    initial_counter = sle.solver_counter

    new_state, slc, disp, potc, avc = sle.solve_nonlinear_equation(
        analytical_state,
        ice_thickness_change=ice_thickness_change,
        max_iterations=5,  # Keep it low for the smoke test
        rotational_feedbacks=True,
    )

    # Verify the solver returned the correct physical containers
    assert isinstance(new_state, EarthState)
    assert isinstance(slc, SHGrid)

    # Verify that solver iteration tracking works
    assert sle.solver_counter > initial_counter

    # Verify the Caspian Sea policy inherited correctly
    assert new_state.exclude_caspian == analytical_state.exclude_caspian


def test_nonlinear_mass_conservation_with_complex_loads(analytical_state):
    """
    Tests mass conservation for the non-linear solver with migrating shorelines
    (changing ocean function) while applying ice, sediment, and ocean density loads.
    """
    sle = SeaLevelEquation(analytical_state.model)

    # 1. Ice Melt (Mass leaving the continents)
    ice_melt_nd = -50.0 / analytical_state.model.parameters.length_scale
    ice_thickness_change = analytical_state.model.zero_grid()
    ice_thickness_change.data = np.where(
        analytical_state.ice_thickness.data > 0, ice_melt_nd, 0.0
    )

    # 2. Sediment Load (Mass entering the ocean floor)
    sediment_thickness_change = analytical_state.model.zero_grid()
    lats, _ = np.meshgrid(
        analytical_state.lats(), analytical_state.lons(), indexing="ij"
    )
    sediment_thickness_change.data = np.where(
        np.abs(lats) < 10, 5.0 / analytical_state.model.parameters.length_scale, 0.0
    )

    # 3. Ocean Density Change
    dynamic_sea_level_change = analytical_state.model.zero_grid()
    dynamic_sea_level_change.data = np.where(
        analytical_state.ocean_function.data > 0, 0.001, 0.0
    )

    sediment_density_nd = 2300.0 / analytical_state.model.parameters.density_scale

    # Run the non-linear solver
    new_state, slc, disp, potc, avc = sle.solve_nonlinear_equation(
        analytical_state,
        ice_thickness_change=ice_thickness_change,
        sediment_thickness_change=sediment_thickness_change,
        dynamic_sea_level_change=dynamic_sea_level_change,
        sediment_density=sediment_density_nd,
        max_iterations=5,
        rotational_feedbacks=False,
    )

    # ==================================================================== #
    #                     Mass Conservation Check                          #
    # ==================================================================== #

    # A. Solid Load Mass Changes (Ice + Sediment)
    ice_mass = (
        analytical_state.model.parameters.ice_density
        * analytical_state.model.integrate(ice_thickness_change)
    )
    sediment_mass = sediment_density_nd * analytical_state.model.integrate(
        sediment_thickness_change
    )

    # B. Ocean Density Mass Change (Volume * Density_Change)
    ocean_density_mass = analytical_state.model.integrate(
        dynamic_sea_level_change
        * analytical_state.ocean_function
        * analytical_state.sea_level
    )

    # C. Ocean Water Mass Change (Accounting for Shoreline Migration)
    rho_w = analytical_state.model.parameters.water_density
    old_water_mass = rho_w * analytical_state.model.integrate(
        analytical_state.ocean_function * analytical_state.sea_level
    )
    new_water_mass = rho_w * new_state.model.integrate(
        new_state.ocean_function * new_state.sea_level
    )

    water_mass_change = new_water_mass - old_water_mass

    # The sum of all mass injected/removed from the system must be balanced
    # perfectly by the redistributed ocean water volume.
    total_mass_change = (
        ice_mass + sediment_mass + ocean_density_mass + water_mass_change
    )

    assert np.isclose(
        total_mass_change, 0.0, atol=1e-5
    ), f"Mass non-conservation detected: {total_mass_change}"


def test_linear_nonlinear_consistency(analytical_state):
    """
    Tests that the non-linear solver converges perfectly to the linear solver
    in the limit of small perturbations.

    Because shoreline migration is mathematically a second-order effect
    O(load^2), the difference between the two solvers must vanish relative
    to the total signal for infinitesimally small melts.
    """
    lin_sle = LinearSeaLevelEquation(analytical_state)
    nonlin_sle = SeaLevelEquation(analytical_state.model)

    # We use a very small melt (e.g., 10 cm) to stay strictly in the linear regime.
    tiny_melt_nd = -0.1 / analytical_state.model.parameters.length_scale
    tiny_ice_change = analytical_state.model.zero_grid()
    tiny_ice_change.data = np.where(
        analytical_state.ice_thickness.data > 0, tiny_melt_nd, 0.0
    )

    # CONVERT thickness change (meters) to a direct mass load (kg/m^2) for the linear solver
    tiny_direct_load = analytical_state.direct_load_from_ice_thickness_change(
        tiny_ice_change
    )

    # 1. Solve via the strictly Linear operator
    slc_lin, _, _, _ = lin_sle.solve_sea_level_equation(
        tiny_direct_load, rotational_feedbacks=False
    )

    # 2. Solve via the Non-Linear shoreline migration loop
    _, slc_nonlin, _, _, _ = nonlin_sle.solve_nonlinear_equation(
        analytical_state,
        ice_thickness_change=tiny_ice_change,
        max_iterations=4,
        rotational_feedbacks=False,
    )

    # 3. Consistency Check
    max_signal = np.max(np.abs(slc_lin.data))
    max_difference = np.max(np.abs(slc_lin.data - slc_nonlin.data))

    # The difference caused by the non-linear terms should be practically zero
    assert (
        max_difference < 0.001 * max_signal
    ), f"Solvers diverge! Max Signal: {max_signal}, Difference: {max_difference}"

    # Furthermore, verify that a massive load (e.g., 1000m) *does* trigger
    # divergence, proving the non-linear solver is actually capturing shoreline migration.
    massive_melt_nd = -1000.0 / analytical_state.model.parameters.length_scale
    massive_ice_change = analytical_state.model.zero_grid()
    massive_ice_change.data = np.where(
        analytical_state.ice_thickness.data > 0, massive_melt_nd, 0.0
    )

    massive_direct_load = analytical_state.direct_load_from_ice_thickness_change(
        massive_ice_change
    )

    slc_lin_massive, _, _, _ = lin_sle.solve_sea_level_equation(
        massive_direct_load, rotational_feedbacks=False
    )
    _, slc_nonlin_massive, _, _, _ = nonlin_sle.solve_nonlinear_equation(
        analytical_state,
        ice_thickness_change=massive_ice_change,
        max_iterations=4,
        rotational_feedbacks=False,
    )

    massive_diff = np.max(np.abs(slc_lin_massive.data - slc_nonlin_massive.data))

    # For a 1000m melt, shoreline migration must be significant (> 0.5% divergence)
    assert massive_diff > 0.005 * np.max(np.abs(slc_lin_massive.data))
