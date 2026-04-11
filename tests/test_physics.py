"""
Test suite for the core SeaLevelEquation solver.

Validates the mathematical correctness, mass conservation, and
reciprocity of the decoupled physical engine.
"""

import pytest
import numpy as np
from pyshtools import SHGrid

from pyslfp.core import EarthModel
from pyslfp.state import EarthState
from pyslfp.physics import SeaLevelEquation
from pyslfp.ice_ng import IceNG, IceModel

# ==================================================================== #
#                            Fixtures                                  #
# ==================================================================== #


@pytest.fixture(scope="module")
def physics_setup():
    """
    Provides a pre-configured (Model, State, Solver) triad for testing.
    Uses lmax=64 to keep the iterative reciprocity tests fast.
    """
    model = EarthModel(lmax=64)

    # Generate realistic background state using IceNG
    ice_ng = IceNG(version=IceModel.ICE7G)
    ice_thickness, sea_level = ice_ng.get_ice_thickness_and_sea_level(0.0, 64)

    # Non-dimensionalize the state grids
    ice_thickness = ice_thickness / model.parameters.length_scale
    sea_level = sea_level / model.parameters.length_scale

    state = EarthState(ice_thickness, sea_level, model)
    solver = SeaLevelEquation(model)

    return model, state, solver


# ==================================================================== #
#                            Helpers                                   #
# ==================================================================== #


def random_load(state: EarthState) -> SHGrid:
    """Generates a random surface mass load based on ice thickness changes."""
    f = np.random.uniform(0, 1)

    # Create a mock ice thickness change in both hemispheres
    ice_change = (
        -f * state.ice_thickness * state.northern_hemisphere_projection(value=0)
    ) + (-(1 - f) * state.ice_thickness * state.southern_hemisphere_projection(value=0))

    # Convert thickness change to a direct surface load (density * thickness)
    return state.direct_load_from_ice_thickness_change(ice_change)


def random_angular_momentum(model: EarthModel, state: EarthState) -> np.ndarray:
    """Generates a random angular momentum jump based on a random load."""
    b = model.parameters.mean_sea_floor_radius
    omega = model.parameters.rotation_frequency

    load = random_load(state)
    load_lm = load.expand(normalization="ortho", csphase=1, lmax_calc=2)

    return omega * b**4 * load_lm.coeffs[:, 2, 1]


# ==================================================================== #
#                     Unit and Sanity Check Tests                      #
# ==================================================================== #


def test_zero_load_input(physics_setup):
    """Sanity check: A zero load should produce a zero response."""
    _, state, solver = physics_setup
    zero_load = SHGrid.from_zeros(
        state.lmax, grid=state.grid, sampling=state.sampling, extend=state.extend
    )

    # Testing the standard wrapper
    slc, disp, gpc, avc = solver.solve_sea_level_equation(state, zero_load)

    assert np.all(slc.data == 0)
    assert np.all(disp.data == 0)
    assert np.all(gpc.data == 0)
    assert np.all(avc == 0)


def test_mass_conservation(physics_setup):
    """
    Check for mass conservation: the mass of water added to the ocean
    must equal the mass removed by the direct load (e.g., melted ice).
    """
    model, state, solver = physics_setup
    direct_load = random_load(state)

    # Testing the standard wrapper
    slc, _, _, _ = solver.solve_sea_level_equation(state, direct_load)

    mass_removed = -state.integrate(direct_load)

    # Mass added = Integral of (sea level change * water density) over the ocean
    ocean_slc = slc * state.ocean_function
    mass_added = state.integrate(ocean_slc) * model.parameters.water_density

    assert np.isclose(mass_removed, mass_added, rtol=1e-5)


def test_get_sea_surface_height_change(physics_setup):
    """Tests the SSH observable calculation logic."""
    _, state, solver = physics_setup

    # Create dummy outputs ensuring we pass ALL parameters to get an exactly aligned grid
    slc = SHGrid.from_zeros(
        state.lmax, grid=state.grid, sampling=state.sampling, extend=state.extend
    )
    slc.data += 1.0
    disp = SHGrid.from_zeros(
        state.lmax, grid=state.grid, sampling=state.sampling, extend=state.extend
    )
    disp.data += 0.5

    # Macroscopically large angular velocity change to clear numpy's allclose tolerance
    avc = np.array([1.0, 1.0])

    # Test without rotational feedbacks
    ssh_no_rot = solver.get_sea_surface_height_change(
        state, slc, disp, avc, remove_rotational_contribution=False
    )
    assert np.allclose(ssh_no_rot.data, 1.5)

    # Test with rotational feedbacks (should differ from 1.5)
    ssh_rot = solver.get_sea_surface_height_change(
        state, slc, disp, avc, remove_rotational_contribution=True
    )
    assert not np.allclose(ssh_rot.data, 1.5)


# ==================================================================== #
#           High-Level Physics-Based Reciprocity Tests                 #
# ==================================================================== #


@pytest.mark.parametrize("rotational_feedbacks", [True, False])
def test_sea_level_reciprocity(physics_setup, rotational_feedbacks: bool):
    """
    Check the sea level reciprocity relation using random loads.
    This tests that the standard wrapper accurately passes configuration down.
    """
    _, state, solver = physics_setup

    direct_load_1 = random_load(state)
    direct_load_2 = random_load(state)
    rtol = 1e-9

    slc_1, _, _, _ = solver.solve_sea_level_equation(
        state,
        direct_load_1,
        rotational_feedbacks=rotational_feedbacks,
        rtol=rtol,
    )

    slc_2, _, _, _ = solver.solve_sea_level_equation(
        state,
        direct_load_2,
        rotational_feedbacks=rotational_feedbacks,
        rtol=rtol,
    )

    lhs = state.integrate(direct_load_2 * slc_1)
    rhs = state.integrate(direct_load_1 * slc_2)

    assert np.isclose(lhs, rhs, rtol=1000 * rtol)


@pytest.mark.parametrize("rotational_feedbacks", [True, False])
def test_generalised_sea_level_reciprocity(physics_setup, rotational_feedbacks: bool):
    """Test the generalised reciprocity relation using all available forcing terms."""
    model, state, solver = physics_setup

    direct_load_1 = random_load(state)
    direct_load_2 = random_load(state)
    displacement_load_1 = random_load(state)
    displacement_load_2 = random_load(state)
    gravitational_potential_load_1 = random_load(state)
    gravitational_potential_load_2 = random_load(state)

    if rotational_feedbacks:
        angular_momentum_change_1 = random_angular_momentum(model, state)
        angular_momentum_change_2 = random_angular_momentum(model, state)
    else:
        angular_momentum_change_1 = np.zeros(2)
        angular_momentum_change_2 = np.zeros(2)

    rtol = 1e-9

    # Testing the generalized engine with strictly keyword arguments
    slc_1, disp_1, gpc_1, avc_1 = solver.solve_generalised_equation(
        state,
        direct_load=direct_load_1,
        displacement_load=displacement_load_1,
        gravitational_potential_load=gravitational_potential_load_1,
        angular_momentum_change=angular_momentum_change_1,
        rotational_feedbacks=rotational_feedbacks,
        rtol=rtol,
    )

    slc_2, disp_2, gpc_2, avc_2 = solver.solve_generalised_equation(
        state,
        direct_load=direct_load_2,
        displacement_load=displacement_load_2,
        gravitational_potential_load=gravitational_potential_load_2,
        angular_momentum_change=angular_momentum_change_2,
        rotational_feedbacks=rotational_feedbacks,
        rtol=rtol,
    )

    g = model.parameters.gravitational_acceleration

    lhs_integrand = direct_load_2 * slc_1 - (1.0 / g) * (
        g * displacement_load_2 * disp_1 + gravitational_potential_load_2 * gpc_1
    )
    lhs = state.integrate(lhs_integrand) - np.dot(angular_momentum_change_2, avc_1) / g

    rhs_integrand = direct_load_1 * slc_2 - (1.0 / g) * (
        g * displacement_load_1 * disp_2 + gravitational_potential_load_1 * gpc_2
    )
    rhs = state.integrate(rhs_integrand) - np.dot(angular_momentum_change_1, avc_2) / g

    assert np.isclose(lhs, rhs, rtol=1000 * rtol)


# ==================================================================== #
#               Non-Linear Shifting Shoreline Tests                    #
# ==================================================================== #


def test_nonlinear_zero_load(physics_setup):
    """Sanity check: A zero ice change should produce zero response and identical state."""
    _, state, solver = physics_setup
    zero_ice = SHGrid.from_zeros(
        state.lmax, grid=state.grid, sampling=state.sampling, extend=state.extend
    )

    final_state, slc, disp, gpc, avc = solver.solve_nonlinear_equation(
        state, ice_thickness_change=zero_ice, max_iterations=3
    )

    # Physics outputs should be effectively zero (accounting for float noise in the transforms)
    assert np.allclose(slc.data, 0.0, atol=1e-5)
    assert np.allclose(disp.data, 0.0, atol=1e-5)

    # The new state masks should be physically identical
    assert np.all(final_state.ocean_function.data == state.ocean_function.data)
    assert np.allclose(final_state.sea_level.data, state.sea_level.data, atol=1e-5)


def test_nonlinear_mass_conservation(physics_setup):
    """
    Critical test: Validates that the iterative eustatic shift perfectly balances
    the global water budget even as the ocean area changes (shorelines shift).
    """
    model, state, solver = physics_setup

    # Melt 50% of the Northern Hemisphere ice to force a massive shoreline shift
    ice_melt_fraction = 0.5
    ice_change = (
        -ice_melt_fraction
        * state.ice_thickness
        * state.northern_hemisphere_projection(value=0)
    )

    # Run the non-linear solver
    final_state, slc, disp, gpc, avc = solver.solve_nonlinear_equation(
        state,
        ice_thickness_change=ice_change,
        rtol=1e-5,  # Slightly looser tolerance to keep the test fast
        verbose=False,
    )

    # 1. Calculate Initial Ocean Water Mass
    initial_water_mass = state.integrate(
        model.parameters.water_density * state.ocean_function * state.sea_level
    )

    # 2. Calculate Melted Ice Mass
    melt_mass = -state.integrate(model.parameters.ice_density * ice_change)

    # 3. Calculate Final Ocean Water Mass (using the dynamically updated shoreline and bathymetry!)
    final_water_mass = final_state.integrate(
        model.parameters.water_density
        * final_state.ocean_function
        * final_state.sea_level
    )

    # The final ocean must exactly contain the initial ocean plus the melted ice
    assert np.isclose(initial_water_mass + melt_mass, final_water_mass, rtol=1e-5)


def test_nonlinear_multi_forcing_execution(physics_setup):
    """
    Smoke test to ensure the solver correctly processes sediment and DSL grids.
    """
    _, state, solver = physics_setup
    zero_ice = SHGrid.from_zeros(
        state.lmax, grid=state.grid, sampling=state.sampling, extend=state.extend
    )

    # Create dummy sediment and DSL grids
    sediment = SHGrid.from_zeros(
        state.lmax, grid=state.grid, sampling=state.sampling, extend=state.extend
    )
    sediment.data += 10.0  # 10 meters of sediment

    dsl = SHGrid.from_zeros(
        state.lmax, grid=state.grid, sampling=state.sampling, extend=state.extend
    )
    dsl.data += 1.0  # 1 meter of dynamic sea level

    # Just run it for a couple of iterations to ensure no broadcasting errors occur
    final_state, slc, _, _, _ = solver.solve_nonlinear_equation(
        state,
        ice_thickness_change=zero_ice,
        sediment_thickness_change=sediment,
        dynamic_sea_level_change=dsl,
        max_iterations=2,
    )

    assert isinstance(final_state, EarthState)
    assert slc.data.shape == state.sea_level.data.shape
