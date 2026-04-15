"""
Test suite for the spatial projection, averaging, and conversion operators.
"""

import pytest
import numpy as np

import pygeoinf as inf
from pyslfp.state import EarthState
from pyslfp.linear_operators.physics import lebesgue_load_space, sobolev_load_space
from pyslfp.linear_operators.spatial import (
    ocean_projection_operator,
    ice_projection_operator,
    land_projection_operator,
    ocean_average_operator,
    ice_average_operator,
    land_average_operator,
    remove_ocean_average_operator,
    ice_sheet_averaging_operator,
    ice_sheet_basis_operator,
    region_average_operator,
    region_projection_operator,
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
#                  1. Spatial Projection Operators                     #
# ==================================================================== #


@pytest.mark.parametrize("sobolev", [False, True])
def test_spatial_projection_operators(testing_state, sobolev):
    """Tests the adjoint identities for all basic masking operators."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        space = lebesgue_load_space(model)

    ops = [
        ocean_projection_operator(testing_state, space),
        ice_projection_operator(testing_state, space),
        land_projection_operator(testing_state, space),
    ]

    domain_measure = space.heat_kernel_gaussian_measure(0.5 * b)

    for A in ops:
        A.check(
            n_checks=2,
            check_rtol=1e-4,
            check_atol=1e-4,
            domain_measure=domain_measure,
            codomain_measure=domain_measure,
        )


# ==================================================================== #
#                  2. Spatial Averaging Operators                      #
# ==================================================================== #


@pytest.mark.parametrize("sobolev", [False, True])
def test_spatial_averaging_operators(testing_state, sobolev):
    """Tests the adjoint identities for all regional averaging operators."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        space = lebesgue_load_space(model)

    ops = [
        ocean_average_operator(testing_state, space),
        ice_average_operator(testing_state, space),
        land_average_operator(testing_state, space),
    ]

    domain_measure = space.heat_kernel_gaussian_measure(0.5 * b)
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(ops[0].codomain, 1.0)

    for A in ops:
        A.check(
            n_checks=2,
            check_rtol=1e-4,
            check_atol=1e-4,
            domain_measure=domain_measure,
            codomain_measure=codomain_measure,
        )


# ==================================================================== #
#                  4. Specialized Physics/Basin Operators              #
# ==================================================================== #


def test_remove_ocean_average_operator_logic(testing_state):
    """
    Verifies that applying the remove_ocean_average operator results
    in a field whose integral strictly over the ocean is zero.
    """
    space = lebesgue_load_space(testing_state.model)
    A = remove_ocean_average_operator(testing_state, space)

    # Start with a constant field of 10.0 everywhere
    constant_field = testing_state.model.constant_grid(10.0)

    # Apply operator
    adjusted_field = A(constant_field)

    # The integral of the adjusted field *over the ocean* must be zero
    ocean_integral = testing_state.model.integrate(
        testing_state.ocean_function * adjusted_field
    )
    assert np.isclose(ocean_integral, 0.0, atol=1e-7)


@pytest.mark.parametrize("sobolev", [False, True])
def test_remove_ocean_average_adjoint(testing_state, sobolev):
    """Tests the adjoint identity for the remove_ocean_average operator."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        space = lebesgue_load_space(model)

    A = remove_ocean_average_operator(testing_state, space)

    domain_measure = space.heat_kernel_gaussian_measure(0.5 * b)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=domain_measure,
    )


@pytest.mark.slow
@pytest.mark.parametrize("sobolev", [False, True])
def test_ice_sheet_basin_operators(testing_state, sobolev):
    """
    Tests the basin averaging and basis operators.
    Specifically checks that B is a strict right-inverse to A (i.e., A @ B = I).
    """
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        space = lebesgue_load_space(model)

    # FIX: Use a valid grouping scheme from regions.py
    grouping = "ice_sheets"

    A = ice_sheet_averaging_operator(testing_state, space, groupings=grouping)
    B = ice_sheet_basis_operator(testing_state, space, groupings=grouping)

    # 1. Adjoint checks for the averager
    domain_measure = space.heat_kernel_gaussian_measure(0.5 * b)
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )

    # 2. Right-Inverse check
    # Generate random vector in Euclidean space
    num_basins = A.codomain.dim
    random_coeffs = np.random.randn(num_basins)

    # Project coeffs -> Global Grid -> Project back to coeffs
    reconstructed_coeffs = A(B(random_coeffs))

    assert np.allclose(random_coeffs, reconstructed_coeffs)


# ==================================================================== #
#                  5. Universal Region Routing Operators               #
# ==================================================================== #


def test_region_average_operator_routing(testing_state):
    """Test that the average operator correctly interprets list nesting."""
    space = lebesgue_load_space(testing_state.model)

    # 1. Single string -> EuclideanSpace(1)
    op1 = region_average_operator(testing_state, space, "Greenland/Iceland")
    assert isinstance(op1.codomain, inf.EuclideanSpace)
    assert op1.codomain.dim == 1

    # 2. Flat list (Composite Region) -> EuclideanSpace(1)
    op2 = region_average_operator(
        testing_state, space, ["Greenland/Iceland", "W.Antarctica"]
    )
    assert op2.codomain.dim == 1

    # 3. List of Lists -> EuclideanSpace(2)
    op3 = region_average_operator(
        testing_state, space, [["Greenland/Iceland", "W.Antarctica"], ["E.Antarctica"]]
    )
    assert op3.codomain.dim == 2


def test_region_projection_operator_routing(testing_state):
    """Test that the projection operator returns a simple Space -> Space mapping."""
    space = lebesgue_load_space(testing_state.model)

    # 1. Single string -> Space to Space
    op1 = region_projection_operator(testing_state, space, "Greenland/Iceland")
    assert isinstance(op1.codomain, type(space))

    # 2. Flat list (Composite Region) -> Space to Space
    op2 = region_projection_operator(
        testing_state, space, ["Greenland/Iceland", "W.Antarctica"]
    )
    assert isinstance(op2.codomain, type(space))


@pytest.mark.slow
@pytest.mark.parametrize("sobolev", [False, True])
def test_region_routing_adjoints(testing_state, sobolev):
    """Tests the adjoint identities for the universal routing operators."""
    model = testing_state.model
    b = model.parameters.mean_sea_floor_radius

    if sobolev:
        space = sobolev_load_space(model, 1.0, 0.1 * b)
    else:
        space = lebesgue_load_space(model)

    # Averager takes a nested list for multiple distinct outputs
    avg_regions = [["Greenland/Iceland"], ["W.Antarctica"]]
    avg_op = region_average_operator(testing_state, space, avg_regions)

    # Projector takes a flat list for a single composite mask
    proj_regions = ["Greenland/Iceland", "W.Antarctica"]
    proj_op = region_projection_operator(testing_state, space, proj_regions)

    domain_measure = space.heat_kernel_gaussian_measure(0.5 * b)

    # 1. Adjoint check for average operator (Space -> Euclidean)
    avg_codomain_measure = inf.GaussianMeasure.from_standard_deviation(
        avg_op.codomain, 1.0
    )
    avg_op.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=avg_codomain_measure,
    )

    # 2. Adjoint check for projection operator (Space -> Space)
    # Codomain measure is just the domain measure again!
    proj_op.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=domain_measure,
    )
