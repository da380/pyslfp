import pytest
import numpy as np
import pygeoinf as inf

from typing import List, Tuple

from pyslfp.finger_print import FingerPrint, EarthModelParameters, IceModel
from pyslfp.operators import (
    FingerPrintOperator,
    GraceObservationOperator,
    ObservationOperator,
    TideGaugeObservationOperator,
)

# ==================================================================== #
#                              Fixtures                                #
# ==================================================================== #


@pytest.fixture(scope="module")
def configured_fingerprint():
    """Provides a standard, configured FingerPrint instance for tests."""
    fp = FingerPrint(
        lmax=32,
        grid="DH",
        earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng(version=IceModel.ICE7G)
    return fp


# ==================================================================== #
#                  Tests for Abstract Base Classes                     #
# ==================================================================== #


def test_abc_multiple_adjoints_error():
    """
    Tests that the __init_subclass__ hook in FingerPrintOperator correctly
    raises a TypeError if a subclass tries to implement more than one
    type of adjoint mapping.
    """
    with pytest.raises(TypeError, match="must implement at most one"):
        # This class definition itself should raise the error
        class BadOperator(FingerPrintOperator):
            def __init__(self):
                pass

            def _data_space(self):
                pass

            def _mapping(self, element):
                pass

            def _adjoint_mapping(self, data):
                pass

            def _formal_adjoint_mapping(self, data):
                pass


def test_abc_instantiation_error():
    """
    Tests that you cannot instantiate an abstract operator without
    implementing all of its abstract methods.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        # This class is incomplete because it's missing _data_space and _mapping
        class IncompleteOperator(ObservationOperator):
            pass

        # Attempting to instantiate it should fail
        _ = IncompleteOperator()


# ==================================================================== #
#                Tests for GraceObservationOperator                    #
# ==================================================================== #


class TestGraceObservationOperator:
    """A test suite for the GraceObservationOperator."""

    def test_grace_initialization(self, configured_fingerprint):
        """Tests that the operator initializes correctly."""
        obs_deg = 16
        op = GraceObservationOperator(
            fingerprint=configured_fingerprint,
            order=0.0,
            scale=1.0,
            observation_degree=obs_deg,
        )

        assert op.fingerprint is configured_fingerprint
        assert isinstance(op.domain, inf.HilbertSpaceDirectSum)
        assert isinstance(op.codomain, inf.EuclideanSpace)
        # Check that the data space has the correct size
        expected_size = (obs_deg + 1) ** 2 - 4  # l>=2
        assert op.codomain.dim == expected_size

    def test_grace_initialization_no_background_error(self):
        """
        Tests that GraceObservationOperator raises a ValueError if the
        provided fingerprint instance does not have a background state set.
        """
        # Create a FingerPrint instance but DO NOT set the state
        unconfigured_fp = FingerPrint(lmax=32)
        with pytest.raises(ValueError, match="must have a background state set"):
            _ = GraceObservationOperator(
                fingerprint=unconfigured_fp,
                order=0.0,
                scale=1.0,
                observation_degree=16,
            )

    @pytest.mark.parametrize(
        "fingerprint_lmax, observation_degree",
        [
            (32, 16),  # Standard case: observation degree is lower
            (16, 16),  # Edge case: degrees are equal
            (16, 32),  # Edge case: observation degree is higher
        ],
        ids=["obs_deg < fp_lmax", "obs_deg == fp_lmax", "obs_deg > fp_lmax"],
    )
    def test_grace_adjoint_identity_various_lmax(
        self, fingerprint_lmax, observation_degree
    ):
        """
        Performs the dot-product test for various lmax configurations to
        ensure the mappings and their adjoints are always correct.
        """
        # 1. Create a fingerprint instance with the desired lmax for this test
        fp = FingerPrint(
            lmax=fingerprint_lmax,
            earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
        )
        fp.set_state_from_ice_ng()

        # 2. Create the operator
        op = GraceObservationOperator(
            fingerprint=fp,
            order=1.0,
            scale=0.5,
            observation_degree=observation_degree,
        )

        # 3. Perform the dot-product test
        u = op.domain.random()
        v = op.codomain.random()
        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))

        assert np.isclose(lhs, rhs, rtol=1e-6)


# ==================================================================== #
#              Tests for TideGaugeObservationOperator                  #
# ==================================================================== #


class TestTideGaugeObservationOperator:
    """A test suite for the TideGaugeObservationOperator."""

    def generate_random_points(self, num_points: int) -> List[Tuple[float, float]]:
        """Helper to generate random (lat, lon) points for testing."""
        lats = np.random.uniform(-90, 90, num_points)
        lons = np.random.uniform(-180, 180, num_points)
        return list(zip(lats, lons))

    def test_tide_gauge_initialization(self, configured_fingerprint):
        """Tests that the operator initializes correctly."""
        points = self.generate_random_points(10)
        op = TideGaugeObservationOperator(
            fingerprint=configured_fingerprint,
            order=2.0,  # Point evaluation requires Sobolev order > 1
            scale=1.0,
            points=points,
        )

        assert op.fingerprint is configured_fingerprint
        assert isinstance(op.domain, inf.HilbertSpaceDirectSum)
        assert isinstance(op.codomain, inf.EuclideanSpace)
        assert op.codomain.dim == len(points)

    @pytest.mark.parametrize("num_points", [1, 5, 20])
    def test_tide_gauge_adjoint_identity(self, configured_fingerprint, num_points):
        """
        Performs the dot-product test to verify the adjoint identity for
        the tide gauge operator.
        """
        # 1. Create a set of random locations for the tide gauges
        points = self.generate_random_points(num_points)

        # 2. Create the operator
        op = TideGaugeObservationOperator(
            fingerprint=configured_fingerprint,
            order=2.0,  # Point evaluation requires Sobolev order > 1
            scale=1.0,
            points=points,
        )

        # 3. Perform the dot-product test
        u = op.domain.random()
        v = op.codomain.random()
        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))

        assert np.isclose(lhs, rhs, rtol=1e-6)
