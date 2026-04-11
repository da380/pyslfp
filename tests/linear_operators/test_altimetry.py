"""
Test suite for the Altimetry observation operators.

Rigorously verifies the adjoint identities of the Sea Surface Height mapping,
ocean-masking projections, and Global Mean Sea Level averaging.
"""

import pytest
import pygeoinf as inf

from pyslfp.core import EarthModel
from pyslfp.state import EarthState
from pyslfp.physics import SeaLevelEquation
from pyslfp.ice_ng import IceNG, IceModel

from pyslfp.linear_operators.physics import sobolev_response_space
from pyslfp.linear_operators.altimetry import (
    ocean_projection_operator,
    sea_surface_height_operator,
    ocean_altimetry_operator,
    global_mean_sea_level_operator,
)


@pytest.fixture(scope="module")
def altimetry_setup():
    """Provides the triad of (State, Solver, Sobolev Space) for testing."""
    # Using lmax=64 for high-speed randomized checks
    model = EarthModel(64)

    # Initialize state with IceNG natively non-dimensionalized
    ice_ng = IceNG(version=IceModel.ICE7G, length_scale=model.parameters.length_scale)
    ice_thickness, sea_level = ice_ng.get_ice_thickness_and_sea_level(0.0, 64)

    state = EarthState(ice_thickness, sea_level, model)
    solver = SeaLevelEquation(model)

    # SSH and Altimetry tracks require Sobolev order s > 1 for point evaluation
    order = 1.0
    scale = 1000.0 * model.parameters.mean_sea_floor_radius
    response_space = sobolev_response_space(state, order, scale)

    return state, solver, response_space


def test_ocean_projection_operator_adjoint_identity(altimetry_setup):
    """Verifies that the ocean mask is a mathematically self-adjoint operator."""
    state, _, response_space = altimetry_setup
    field_space = response_space.subspace(0)

    # Construct the projection operator
    A = ocean_projection_operator(state, field_space)

    # Verification measures
    measure = field_space.heat_kernel_gaussian_measure(0.1)

    # pygeoinf .check() handles internal assertions
    A.check(
        n_checks=3,
        domain_measure=measure,
        codomain_measure=measure,
    )


@pytest.mark.parametrize("remove_rotational", [True, False])
def test_ssh_operator_adjoint_identity(altimetry_setup, remove_rotational: bool):
    """Rigorously tests the adjoint identity of the response-to-SSH mapping."""
    state, solver, response_space = altimetry_setup

    A = sea_surface_height_operator(
        solver, state, response_space, remove_rotational_contribution=remove_rotational
    )

    # Domain Measure: Composite space [SLC, Disp, GPC, AVC]
    field_space = response_space.subspace(0)
    field_measure = field_space.heat_kernel_gaussian_measure(0.1)

    euclidean_space = response_space.subspace(3)
    euclidean_measure = inf.GaussianMeasure.from_standard_deviation(
        euclidean_space, 1.0
    )

    domain_measure = inf.GaussianMeasure.from_direct_sum(
        [field_measure, field_measure, field_measure, euclidean_measure]
    )

    # Codomain Measure: Masked SSH Field
    codomain_measure = A.codomain.heat_kernel_gaussian_measure(0.1)

    A.check(
        n_checks=3,
        check_rtol=1e-5,
        check_atol=1e-5,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


@pytest.mark.parametrize("remove_rotational", [True, False])
def test_ocean_altimetry_operator_adjoint_identity(
    altimetry_setup, remove_rotational: bool
):
    """Tests the adjoint identity of SSH evaluated at discrete random tracks."""
    state, solver, response_space = altimetry_setup

    field_space = response_space.subspace(0)
    points = field_space.random_points(20)

    A = ocean_altimetry_operator(
        solver,
        state,
        response_space,
        points,
        remove_rotational_contribution=remove_rotational,
    )

    # Reuse measures from composite response space
    field_measure = field_space.heat_kernel_gaussian_measure(0.1)
    euclidean_measure = inf.GaussianMeasure.from_standard_deviation(
        response_space.subspace(3), 1.0
    )

    domain_measure = inf.GaussianMeasure.from_direct_sum(
        [field_measure, field_measure, field_measure, euclidean_measure]
    )

    # Codomain Measure: Euclidean observation space
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=3,
        check_rtol=1e-5,
        check_atol=1e-5,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


def test_global_mean_sea_level_operator_adjoint_identity(altimetry_setup):
    """Tests the adjoint identity of the weighted averaging GMSL matrix."""
    _, _, response_space = altimetry_setup

    field_space = response_space.subspace(0)
    points = field_space.random_points(50)

    A = global_mean_sea_level_operator(points)

    domain_measure = inf.GaussianMeasure.from_standard_deviation(A.domain, 1.0)
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=3,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )
