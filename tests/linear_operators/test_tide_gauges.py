"""
Test suite for tide gauge observation models and operators.
"""

import pytest

import pygeoinf as inf
from pyslfp.linear_operators.physics import FingerPrintOperator
from pyslfp.linear_operators.tide_gauges import (
    read_gloss_tide_gauge_data,
    tide_gauge_operator,
    TideGaugeObservationModel,
)


# ==================================================================== #
#                          Fixtures                                    #
# ==================================================================== #


@pytest.fixture(scope="module")
def operator_lmax():
    # Keep lmax low for fast tests
    return 16


@pytest.fixture(scope="module")
def fingerprint_operator(operator_lmax):
    """
    Provides a base FingerPrintOperator with a Sobolev load space.
    Note: Point evaluation REQUIRES a Sobolev order > 1.0.
    A load order of 1.0 means the response fields are order 2.0,
    which is sufficient for continuous point evaluation on a 2D sphere.
    """
    return FingerPrintOperator.for_testing(
        operator_lmax,
        load_parameters=(1.0, 0.1),
        response_parameters=(2.0, 0.1),
        rotational_feedbacks=False,
    )


@pytest.fixture
def sample_points():
    """Returns a small list of geographically distinct points."""
    return [
        (45.0, -60.0),  # Halifax, Canada
        (-33.8, 151.2),  # Sydney, Australia
        (51.5, 0.0),  # London, UK
    ]


# ==================================================================== #
#                  1. Data Loader Tests                                #
# ==================================================================== #


@pytest.mark.slow
def test_read_gloss_tide_gauge_data():
    """
    Tests that the GLOSS network downloader retrieves and parses the data.
    Marked as slow because it triggers a network request.
    """
    names, points = read_gloss_tide_gauge_data()

    assert isinstance(names, list)
    assert isinstance(points, list)
    assert len(names) > 0
    assert len(names) == len(points)
    assert isinstance(names[0], str)
    assert isinstance(points[0], tuple)


@pytest.mark.slow
def test_read_gloss_tide_gauge_data_with_filter():
    """Tests the filtering logic during parsing."""

    def filter_uk(name, lat, lon):
        # Very crude filter just to prove the callable works
        return "NEWLYN" in name.upper()

    names, points = read_gloss_tide_gauge_data(filter_func=filter_uk)

    assert len(names) > 0
    assert "NEWLYN" in names[0].upper()


# ==================================================================== #
#                  2. Raw Math Operators                               #
# ==================================================================== #


def test_tide_gauge_operator_validation(operator_lmax):
    """
    Tests that the point evaluation operator strictly enforces the
    Sobolev > 1.0 requirement to guarantee mathematically continuous fields.
    """
    # Create an operator with Lebesgue (L2) spaces
    fp_l2 = FingerPrintOperator.for_testing(operator_lmax)

    with pytest.raises(ValueError, match="must be a Sobolev space of order > 1"):
        tide_gauge_operator(fp_l2.codomain, [(0.0, 0.0)])


def test_tide_gauge_operator_adjoint(fingerprint_operator, sample_points):
    """
    Tests the adjoint identity of the isolated point-evaluation operator.
    """
    A = tide_gauge_operator(fingerprint_operator.codomain, sample_points)

    domain_measure = fingerprint_operator.response_measure_for_testing()
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )


# ==================================================================== #
#                  3. Composite Observation Models                     #
# ==================================================================== #


def test_tide_gauge_observation_model_structure(fingerprint_operator, sample_points):
    """
    Verifies that the TideGaugeObservationModel correctly assembles the
    composite forward operator (Mapping: Load Space -> Euclidean Data Space).
    """
    names = ["A", "B", "C"]
    model = TideGaugeObservationModel(fingerprint_operator, sample_points, names=names)

    assert model.names == names
    assert model.points == sample_points

    # The composite forward operator should map from the Sobolev load space
    # directly to a 3-dimensional Euclidean space.
    assert model.forward_operator.domain == fingerprint_operator.domain
    assert isinstance(model.forward_operator.codomain, inf.EuclideanSpace)
    assert model.forward_operator.codomain.dim == 3


def test_tide_gauge_forward_operator_adjoint(fingerprint_operator, sample_points):
    """
    Rigorously tests the adjoint of the fully assembled forward operator
    (PointEval @ Fingerprint).
    """
    model = TideGaugeObservationModel(fingerprint_operator, sample_points)
    A = model.forward_operator

    domain_measure = fingerprint_operator.load_measure_for_testing()
    codomain_measure = inf.GaussianMeasure.from_standard_deviation(A.codomain, 1.0)

    A.check(
        n_checks=2,
        check_rtol=1e-4,
        check_atol=1e-4,
        domain_measure=domain_measure,
        codomain_measure=codomain_measure,
    )
