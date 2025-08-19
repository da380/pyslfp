import pytest
import numpy as np
from pyslfp.finger_print import FingerPrint
from pyslfp.operators import (
    SeaLevelOperator,
    GraceObservationOperator,
    TideGaugeObservationOperator,
    AveragingOperator,
)
import pygeoinf as inf
from pygeoinf.symmetric_space.sphere import Sobolev


@pytest.mark.parametrize("version", [6, 7])
# @pytest.mark.parametrize("rotational_feedbacks", [True, False])
@pytest.mark.parametrize("rtol", [1e-6])
@pytest.mark.parametrize("lmax", [60, 128])
@pytest.mark.parametrize("order", [2])
@pytest.mark.parametrize("scale", [0.1])
# @pytest.mark.parametrize("grace_observation_degree", [10])
# @pytest.mark.parametrize("number_of_tide_gauges", [50])
# @pytest.mark.parametrize("number_of_weighting_functions", [10])
class TestOperators:

    def random_ocean_locations(self, fingerprint, n):
        """
        Returns a set of n points within the oceans.
        """
        points = []
        while len(points) < n:
            lat = np.random.uniform(-90, 90)
            lon = np.random.uniform(-180, 180)
            sl = fingerprint.point_evaulation(fingerprint.sea_level, lat, lon)
            if sl > 0:
                points.append([lat, lon])
        return points

    def random_load(self, fingerprint):
        """
        Return a random disk load.
        """
        delta = np.random.uniform(10, 30)
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        amp = np.random.randn()
        return fingerprint.disk_load(delta, lat, lon, amp)

    def random_weighting_functions(self, fingerprint, n):
        """
        Return a set of n random weighting functions.
        """
        gaussian_params = []
        for _ in range(n):
            width = np.random.uniform(0.0001, 0.0005)
            lat = np.random.uniform(-90, 90)
            lon = np.random.uniform(-180, 180)
            gaussian_params.append((width, lat, lon))
        return [
            fingerprint.gaussian_averaging_function(width, lat, lon)
            for width, lat, lon in gaussian_params
        ]

    def set_up_fingerprint(self, version, lmax):
        """
        Set up a FingerPrint instance of the given trunction
        degree and set the equilibrium model using the
        present-day ice-7g values.
        """
        fingerprint = FingerPrint(lmax=lmax)
        fingerprint.set_state_from_ice_ng(version=version)
        return fingerprint

    def set_up_sea_level_operator(
        self, version, rotational_feedbacks, rtol, lmax, order, scale
    ):
        """
        Set up a sea level operator with the given parameters.
        """
        fingerprint = self.set_up_fingerprint(version, lmax)
        return SeaLevelOperator(
            order,
            scale,
            fingerprint=fingerprint,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )

    def test_sea_level_operator_self_adjoint(
        self, version, rotational_feedbacks, rtol, lmax, order, scale
    ):
        """
        Test the self-adjointness of the sea level operator.
        """
        sea_level_operator = self.set_up_sea_level_operator(
            version,
            rotational_feedbacks,
            rtol,
            lmax,
            order,
            scale,
        )
        load1 = self.random_load(sea_level_operator.fingerprint)
        load2 = self.random_load(sea_level_operator.fingerprint)
        response2 = sea_level_operator(load2)

        model_space = sea_level_operator.domain
        response_space = sea_level_operator.codomain

        lhs = response_space.inner_product(sea_level_operator(load1), response2)
        rhs = model_space.inner_product(load1, sea_level_operator.adjoint(response2))

        assert np.isclose(lhs, rhs, rtol=1000 * rtol)

    def test_grace_observation_operator_self_adjoint(
        self,
        version,
        rotational_feedbacks,
        rtol,
        lmax,
        order,
        scale,
        grace_observation_degree,
    ):
        """
        Test the self-adjointness of the Grace observation operator.
        """
        sea_level_operator = self.set_up_sea_level_operator(
            version,
            rotational_feedbacks,
            rtol,
            lmax,
            order,
            scale,
        )
        grace_observation_operator = GraceObservationOperator(
            sea_level_operator, grace_observation_degree
        )
        forward_operator = grace_observation_operator.forward_operator

        load1 = self.random_load(sea_level_operator.fingerprint)
        load2 = self.random_load(sea_level_operator.fingerprint)
        response1 = grace_observation_operator(load1)
        data2 = forward_operator(load2)

        model_space = sea_level_operator.domain
        response_space = sea_level_operator.codomain
        data_space = grace_observation_operator.codomain

        # Test the self-adjointness of the observation operator
        lhs = data_space.inner_product(grace_observation_operator(response1), data2)
        rhs = response_space.inner_product(
            response1, grace_observation_operator.adjoint(data2)
        )
        assert np.isclose(lhs, rhs, rtol=1000 * rtol)

        # Test the self-adjointness of the forward operator
        lhs = data_space.inner_product(forward_operator(load1), data2)
        rhs = model_space.inner_product(load1, forward_operator.adjoint(data2))
        assert np.isclose(lhs, rhs, rtol=1000 * rtol)

    def test_tide_gauge_observation_operator_self_adjoint(
        self,
        version,
        rotational_feedbacks,
        rtol,
        lmax,
        order,
        scale,
        number_of_tide_gauges,
    ):
        """
        Test the self-adjointness of the Tide Gauge observation operator.
        """
        sea_level_operator = self.set_up_sea_level_operator(
            version,
            rotational_feedbacks,
            rtol,
            lmax,
            order,
            scale,
        )
        points = self.random_ocean_locations(
            sea_level_operator.fingerprint, number_of_tide_gauges
        )
        tide_gauge_observation_operator = TideGaugeObservationOperator(
            sea_level_operator, points
        )
        forward_operator = tide_gauge_observation_operator.forward_operator

        load1 = self.random_load(sea_level_operator.fingerprint)
        load2 = self.random_load(sea_level_operator.fingerprint)
        response1 = tide_gauge_observation_operator(load1)
        data2 = forward_operator(load2)

        model_space = sea_level_operator.domain
        response_space = sea_level_operator.codomain
        data_space = tide_gauge_observation_operator.codomain

        # Test the self-adjointness of the observation operator
        lhs = data_space.inner_product(
            tide_gauge_observation_operator(response1), data2
        )
        rhs = response_space.inner_product(
            response1, tide_gauge_observation_operator.adjoint(data2)
        )
        assert np.isclose(lhs, rhs, rtol=1000 * rtol)

        # Test the self-adjointness of the forward operator
        lhs = data_space.inner_product(forward_operator(load1), data2)
        rhs = model_space.inner_product(load1, forward_operator.adjoint(data2))
        assert np.isclose(lhs, rhs, rtol=1000 * rtol)

    def test_averaging_operator_self_adjoint(
        self, version, lmax, order, scale, number_of_weighting_functions, rtol
    ):
        """
        Test the self-adjointness of the Averaging operator.
        """
        fingerprint = self.set_up_fingerprint(version, lmax)
        weighting_functions = self.random_weighting_functions(
            fingerprint, number_of_weighting_functions
        )
        model_space = Sobolev(
            lmax, order, scale, radius=fingerprint.mean_sea_floor_radius
        )
        averaging_operator = AveragingOperator(
            model_space,
            weighting_functions=weighting_functions,
            fingerprint=fingerprint,
        )

        load1 = self.random_load(fingerprint)
        load2 = self.random_load(fingerprint)
        averages2 = averaging_operator(load2)

        averages_space = averaging_operator.codomain

        lhs = averages_space.inner_product(averaging_operator(load1), averages2)
        rhs = model_space.inner_product(load1, averaging_operator.adjoint(averages2))

        assert np.isclose(lhs, rhs, rtol=1000 * rtol)
