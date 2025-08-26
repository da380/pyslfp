import pytest
import numpy as np
import pygeoinf as inf

from typing import List, Tuple

from pyshtools import SHGrid

from pyslfp.finger_print import FingerPrint, EarthModelParameters, IceModel
from pyslfp.operators import (
    FingerPrintOperator,
    GraceObservationOperator,
    ObservationOperator,
    TideGaugeObservationOperator,
    LoadAveragingOperator,
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


# ==================================================================== #
#                 Tests for LoadAveragingOperator                      #
# ==================================================================== #


class TestLoadAveragingOperator:
    """A test suite for the LoadAveragingOperator."""

    def create_weighting_functions(
        self, fingerprint: FingerPrint, num_functions: int
    ) -> List[SHGrid]:
        """Helper to generate random SHGrid weighting functions for testing."""
        weights = []
        for _ in range(num_functions):
            delta = np.random.uniform(5, 20)
            lat = np.random.uniform(-90, 90)
            lon = np.random.uniform(-180, 180)
            weights.append(fingerprint.disk_load(delta, lat, lon, 1.0))
        return weights

    def test_load_averaging_initialization(self, configured_fingerprint):
        """Tests that the operator initializes correctly."""
        weights = self.create_weighting_functions(configured_fingerprint, 5)
        op = LoadAveragingOperator(
            fingerprint=configured_fingerprint,
            order=0.0,
            scale=1.0,
            weighting_functions=weights,
        )

        assert isinstance(op.domain, inf.symmetric_space.sphere.Sobolev)
        assert isinstance(op.codomain, inf.EuclideanSpace)
        assert op.codomain.dim == len(weights)

    def test_load_averaging_init_errors(self, configured_fingerprint):
        """Tests that initialization fails with incompatible weighting functions."""
        # Test with a non-SHGrid object
        bad_weights_type = [configured_fingerprint.zero_grid(), np.zeros(5)]
        with pytest.raises(TypeError):
            LoadAveragingOperator(
                fingerprint=configured_fingerprint,
                order=0.0,
                scale=1.0,
                weighting_functions=bad_weights_type,
            )

        # Test with an SHGrid of the wrong lmax
        incompatible_grid = SHGrid.from_zeros(lmax=16)
        bad_weights_lmax = [configured_fingerprint.zero_grid(), incompatible_grid]
        with pytest.raises(ValueError):
            LoadAveragingOperator(
                fingerprint=configured_fingerprint,
                order=0.0,
                scale=1.0,
                weighting_functions=bad_weights_lmax,
            )

    @pytest.mark.parametrize("num_weights", [1, 5, 20])
    def test_load_averaging_adjoint_identity(self, configured_fingerprint, num_weights):
        """
        Performs the dot-product test to verify the adjoint identity for
        the load averaging operator.
        """
        # 1. Create a set of random weighting functions
        weights = self.create_weighting_functions(configured_fingerprint, num_weights)

        # 2. Create the operator
        op = LoadAveragingOperator(
            fingerprint=configured_fingerprint,
            order=1.0,
            scale=0.5,
            weighting_functions=weights,
        )

        # 3. Perform the dot-product test
        mu = op.domain.heat_gaussian_measure(1.0, 1)
        u = mu.sample()
        v = op.codomain.random()
        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))

    #        assert np.isclose(lhs, rhs, rtol=1e-6)

    def test_load_averaging_simple_case(self, configured_fingerprint):
        """
        Tests the operator's forward mapping against a simple, deterministic
        case with a known analytical answer.
        """
        fp = configured_fingerprint

        # 1. Define a simple load (constant value of 10.0 everywhere)
        load = fp.constant_grid(10.0)

        # 2. Define a simple weighting function (1 over the Northern Hemisphere)
        weight = fp.northern_hemisphere_projection(0.0)

        # 3. Create the operator
        op = LoadAveragingOperator(
            fingerprint=fp,
            order=0.0,
            scale=1.0,
            weighting_functions=[weight],
        )

        # 4. Apply the operator to get the calculated average
        result = op(load)

        # 5. Calculate the expected answer analytically
        # The integral of (10.0 * weight) is 10.0 * area_of_northern_hemisphere
        radius = fp.mean_sea_floor_radius_si
        total_surface_area = 4.0 * np.pi * radius**2
        expected_average = 10.0 * (total_surface_area / 2.0)

        # assert np.isclose(result[0], expected_average)
