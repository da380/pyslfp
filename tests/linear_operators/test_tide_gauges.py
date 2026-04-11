"""
Test suite for the Tide Gauge observation operator.

Validates the instantiation of point-evaluation operators and rigorously
tests their adjoint identities using Gaussian measures and random coordinates.
"""

import pytest

import pygeoinf as inf
from pygeoinf import HilbertSpaceDirectSum, EuclideanSpace

from pyslfp.core import EarthModel
from pyslfp.state import EarthState
from pyslfp.ice_ng import IceNG, IceModel
from pyslfp.linear_operators.physics import sobolev_response_space
from pyslfp.linear_operators.tide_gauges import tide_gauge_operator


@pytest.fixture(scope="module")
def tide_gauge_setup():
    """Provides a basic EarthState and Sobolev response space for testing."""
    model = EarthModel(32)
    ice_ng = IceNG(version=IceModel.ICE7G, length_scale=model.parameters.length_scale)
    ice_thickness, sea_level = ice_ng.get_ice_thickness_and_sea_level(0.0, 32)

    state = EarthState(ice_thickness, sea_level, model)

    # Tide gauges require point evaluation, so the load space must be order s > 0
    # The response space is order s+1, so an input load of s=1 gives a response of s=2.
    order = 1.0
    scale = 1000.0 * model.parameters.mean_sea_floor_radius
    response_space = sobolev_response_space(state, order, scale)

    return state, response_space


def test_tide_gauge_operator_instantiation(tide_gauge_setup):
    """Smoke test for the instantiation and shape mapping of the operator."""
    _, response_space = tide_gauge_setup

    # Generate 15 random (lat, lon) observation points
    field_space = response_space.subspace(0)
    points = field_space.random_points(15)

    operator = tide_gauge_operator(response_space, points)

    assert isinstance(operator.domain, HilbertSpaceDirectSum)
    assert isinstance(operator.codomain, EuclideanSpace)
    assert operator.codomain.dim == 15


def test_tide_gauge_operator_adjoint_identity(tide_gauge_setup):
    """
    Rigorously tests the mathematical adjoint identity <Au, v> == <u, A*v>
    for the tide gauge operator using Gaussian measures.
    """
    _, response_space = tide_gauge_setup
    field_space = response_space.subspace(0)

    num_points = 20
    points = field_space.random_points(num_points)

    # 1. Construct the target operator
    A = tide_gauge_operator(response_space, points)

    # 2. Domain measure (Composite physical response space)
    # We apply standard normal noise to the spherical harmonic fields
    smoothness_scale = 0.1
    field_measure = field_space.heat_kernel_gaussian_measure(smoothness_scale)

    euclidean_space = response_space.subspace(3)
    euclidean_measure = inf.GaussianMeasure.from_standard_deviation(
        euclidean_space, 1.0
    )

    domain_measure = inf.GaussianMeasure.from_direct_sum(
        [field_measure, field_measure, field_measure, euclidean_measure]
    )

    # 3. Codomain measure (Euclidean observation space)
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    # 4. Run the comprehensive self-checks
    A.check(
        n_checks=3,
        check_rtol=1e-5,
        check_atol=1e-5,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )
