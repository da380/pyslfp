"""
Test suite for mass load conversions and composite domain operators.
"""

import pytest
import pygeoinf as inf

from pyslfp.state import EarthState
from pyslfp.linear_operators.physics import lebesgue_load_space, sobolev_load_space
from pyslfp.linear_operators.loads import (
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    ocean_density_change_to_load_operator,
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
#                  2. Composite Joint Load Operators                   #
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
