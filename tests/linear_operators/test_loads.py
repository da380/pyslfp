"""
Test suite for mass load conversions and composite domain operators.
"""

import pytest
import pygeoinf as inf

import numpy as np

from pyslfp.core import EarthState
from pyslfp.linear_operators.physics import lebesgue_load_space, sobolev_load_space
from pyslfp.linear_operators.loads import (
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    ocean_density_change_to_load_operator,
    steric_sea_level_operator,
    ocean_mass_conservation_operator,
    joint_ice_ocean_to_load_operator,
)

# ==================================================================== #
#                          Fixtures                                    #
# ==================================================================== #


@pytest.fixture(scope="module")
def operator_lmax():
    return 16


@pytest.fixture(scope="module")
def testing_state(operator_lmax):
    return EarthState.for_testing(operator_lmax)


# ==================================================================== #
#                  1. Individual Conversion Operators                  #
# ==================================================================== #


@pytest.mark.parametrize("sobolev", [False, True])
def test_mass_conversion_operators(testing_state, sobolev):
    """
    Tests the adjoint identities for physical density multipliers that
    explicitly map from a parameter space to a load space.
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        param_space = sobolev_load_space(model, 1.0, 0.1 * b)
        load_space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        param_space = lebesgue_load_space(model)
        load_space = lebesgue_load_space(model)

    ops = [
        ice_thickness_change_to_load_operator(testing_state, param_space, load_space),
        sea_level_change_to_load_operator(testing_state, param_space, load_space),
        ocean_density_change_to_load_operator(testing_state, param_space, load_space),
    ]

    domain_measure = param_space.heat_kernel_gaussian_measure(0.5 * b)
    codomain_measure = load_space.heat_kernel_gaussian_measure(0.5 * b)

    for A in ops:
        A.check(
            n_checks=2,
            check_rtol=1e-4,
            check_atol=1e-4,
            domain_measure=domain_measure,
            codomain_measure=codomain_measure,
        )


# ==================================================================== #
#             2. Steric Response and Mass Conservation                 #
# ==================================================================== #


@pytest.mark.parametrize("sobolev", [False, True])
def test_steric_and_conservation_operators(testing_state, sobolev):
    """The adjoint identities, and the physics that defines the pair:
    the steric response of a density change loads nothing (its water
    load cancels the density load pointwise), so a (steric eta, drho)
    pair lies exactly in the mass-conservation operator's kernel."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        param_space = sobolev_load_space(model, 1.0, 0.1 * b)
        load_space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        param_space = lebesgue_load_space(model)
        load_space = lebesgue_load_space(model)

    steric = steric_sea_level_operator(testing_state, param_space, load_space)
    conservation = ocean_mass_conservation_operator(
        testing_state, load_space, param_space
    )

    domain_measure = param_space.heat_kernel_gaussian_measure(0.5 * b)
    codomain_measure = load_space.heat_kernel_gaussian_measure(0.5 * b)
    steric.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )
    conservation.check(n_checks=2, check_rtol=1e-4, check_atol=1e-4)

    drho = domain_measure.sample()
    eta_s = steric(drho)
    # the steric response loads nothing
    water_load = sea_level_change_to_load_operator(
        testing_state, load_space, load_space
    )(eta_s)
    density_load = ocean_density_change_to_load_operator(
        testing_state, param_space, load_space
    )(drho)
    scale = np.max(np.abs(density_load.data))
    assert np.allclose(water_load.data + density_load.data, 0.0, atol=1e-12 * scale)
    # and so the pair conserves ocean mass exactly
    residual = conservation([eta_s, drho])
    assert abs(residual[0]) < 1e-12 * scale
    # while a bare sea surface height change does not
    violating = conservation([eta_s, param_space.zero])
    assert abs(violating[0]) > 1e-6 * scale


def test_barystatic_weighting_and_mean_depth(testing_state):
    """The state's mass-conservation helpers: the barystatic weighting
    integrates a uniform grounded melt to the bathtub answer, and the
    mean ocean depth is the ocean average of the sea level."""
    state = testing_state
    model = state.model
    parameters = model.parameters
    grounded_area = model.integrate(state.one_minus_ocean_function)
    uniform = model.constant_grid(1.0)
    gmsl = model.integrate(state.barystatic_gmsl_weighting * uniform)
    expected = (
        -parameters.ice_density
        * grounded_area
        / (parameters.water_density * state.ocean_area)
    )
    assert np.isclose(gmsl, expected)

    depth = model.integrate(state.sea_level * state.ocean_projection(value=0.0))
    assert np.isclose(state.mean_ocean_depth, depth / state.ocean_area)
    assert state.mean_ocean_depth > 0.0


# ==================================================================== #
#                  3. Composite Joint Load Operators                   #
# ==================================================================== #


@pytest.mark.parametrize("sobolev", [False, True])
def test_joint_ice_ocean_to_load_operator(testing_state, sobolev):
    """
    Tests the adjoint identity for the RowLinearOperator combining ice
    and ocean loads. This ensures the adjoint correctly splits the gradient
    back into the individual direct sum subspaces.
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        ice_space = sobolev_load_space(model, 1.0, 0.1 * b)
        ocean_space = sobolev_load_space(model, 1.0, 0.1 * b)
        load_space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        ice_space = lebesgue_load_space(model)
        ocean_space = lebesgue_load_space(model)
        load_space = lebesgue_load_space(model)

    A = joint_ice_ocean_to_load_operator(
        testing_state, ice_space, ocean_space, load_space
    )

    # The domain is a HilbertSpaceDirectSum, so we must build a direct sum measure
    ice_measure = ice_space.heat_kernel_gaussian_measure(0.5 * b)
    ocean_measure = ocean_space.heat_kernel_gaussian_measure(0.5 * b)
    domain_measure = inf.GaussianMeasure.from_direct_sum([ice_measure, ocean_measure])

    codomain_measure = load_space.heat_kernel_gaussian_measure(0.5 * b)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )
