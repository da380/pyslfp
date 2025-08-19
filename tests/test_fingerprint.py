import pytest
import numpy as np
from pyslfp.finger_print import FingerPrint, EarthModelParameters, IceModel


@pytest.mark.parametrize("version", [IceModel.ICE6G, IceModel.ICE7G])
@pytest.mark.parametrize("rotational_feedbacks", [True, False])
@pytest.mark.parametrize("rtol", [1e-6])
@pytest.mark.parametrize("lmax", [128])
class TestFingerPrint:

    def set_up_fingerprint(self, version, lmax):
        """
        Set up a FingerPrint instance of the given truncation
        degree and set the equilibrium model using the
        present-day ice-ng values.
        """
        fingerprint = FingerPrint(
            lmax=lmax,
            earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
        )
        fingerprint.set_state_from_ice_ng(version=version)
        return fingerprint

    def random_load(self, fingerprint):
        """
        Return a random disk load.
        """
        delta = np.random.uniform(10, 30)
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)
        amp = np.random.randn()
        return fingerprint.disk_load(delta, lat, lon, amp)

    def random_angular_momentum(self, fingerprint):
        """
        Return a random angular momentum jump
        with magnitude comparable to the disk loads.
        """
        b = fingerprint.mean_sea_floor_radius
        omega = fingerprint.rotation_frequency
        load = self.random_load(fingerprint)
        load_lm = load.expand(lmax_calc=2, normalization="ortho")
        return omega * b**4 * load_lm.coeffs[:, 2, 1]

    def test_sea_level_reciprocity(self, version, rotational_feedbacks, rtol, lmax):
        """
        Check the sea level reciprocity relation using random loads.
        """
        fingerprint = self.set_up_fingerprint(version, lmax)
        direct_load_1 = self.random_load(fingerprint)
        direct_load_2 = self.random_load(fingerprint)
        sea_level_change_1, _, _, _ = fingerprint(
            direct_load=direct_load_1,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )
        sea_level_change_2, _, _, _ = fingerprint(
            direct_load=direct_load_2,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
        )
        lhs = fingerprint.integrate(direct_load_2 * sea_level_change_1)
        rhs = fingerprint.integrate(direct_load_1 * sea_level_change_2)
        assert np.isclose(lhs, rhs, rtol=1000 * rtol)

    def test_generalised_sea_level_reciprocity(
        self, version, rotational_feedbacks, rtol, lmax
    ):
        """
        Test the generalised reciprocity relation using random forces.
        """
        fingerprint = self.set_up_fingerprint(version, lmax)
        direct_load_1 = self.random_load(fingerprint)
        direct_load_2 = self.random_load(fingerprint)
        displacement_load_1 = self.random_load(fingerprint)
        displacement_load_2 = self.random_load(fingerprint)
        gravitational_potential_load_1 = self.random_load(fingerprint)
        gravitational_potential_load_2 = self.random_load(fingerprint)
        angular_momentum_change_1 = (
            self.random_angular_momentum(fingerprint)
            if rotational_feedbacks
            else np.zeros(2)
        )
        angular_momentum_change_2 = (
            self.random_angular_momentum(fingerprint)
            if rotational_feedbacks
            else np.zeros(2)
        )

        (
            sea_level_change_1,
            displacement_1,
            gravity_potential_change_1,
            angular_velocity_change_1,
        ) = fingerprint(
            direct_load=direct_load_1,
            displacement_load=displacement_load_1,
            gravitational_potential_load=gravitational_potential_load_1,
            angular_momentum_change=angular_momentum_change_1,
        )

        (
            sea_level_change_2,
            displacement_2,
            gravity_potential_change_2,
            angular_velocity_change_2,
        ) = fingerprint(
            direct_load=direct_load_2,
            displacement_load=displacement_load_2,
            gravitational_potential_load=gravitational_potential_load_2,
            angular_momentum_change=angular_momentum_change_2,
        )

        g = fingerprint.gravitational_acceleration

        lhs_integrand = direct_load_2 * sea_level_change_1 - (1 / g) * (
            g * displacement_load_2 * displacement_1
            + gravitational_potential_load_2 * gravity_potential_change_1
        )

        lhs = (
            fingerprint.integrate(lhs_integrand)
            - np.dot(angular_momentum_change_2, angular_velocity_change_1) / g
        )

        rhs_integrand = direct_load_1 * sea_level_change_2 - (1 / g) * (
            g * displacement_load_1 * displacement_2
            + gravitational_potential_load_1 * gravity_potential_change_2
        )

        rhs = (
            fingerprint.integrate(rhs_integrand)
            - np.dot(angular_momentum_change_1, angular_velocity_change_2) / g
        )

        assert np.isclose(lhs, rhs, rtol=1000 * rtol)

    def test_alternative_generalised_sea_level_reciprocity(
        self, version, rotational_feedbacks, rtol, lmax
    ):
        """
        Test the generalised reciprocity relation using random forces.
        """
        fingerprint = self.set_up_fingerprint(version, lmax)
        direct_load_1 = self.random_load(fingerprint)
        direct_load_2 = self.random_load(fingerprint)
        displacement_load_1 = self.random_load(fingerprint)
        displacement_load_2 = self.random_load(fingerprint)
        gravitational_potential_load_1 = self.random_load(fingerprint)
        gravitational_potential_load_2 = self.random_load(fingerprint)
        angular_momentum_change_1 = (
            self.random_angular_momentum(fingerprint)
            if rotational_feedbacks
            else np.zeros(2)
        )
        angular_momentum_change_2 = (
            self.random_angular_momentum(fingerprint)
            if rotational_feedbacks
            else np.zeros(2)
        )

        (
            sea_level_change_1,
            displacement_1,
            gravity_potential_change_1,
            angular_velocity_change_1,
        ) = fingerprint(
            direct_load=direct_load_1,
            displacement_load=displacement_load_1,
            gravitational_potential_load=gravitational_potential_load_1,
            angular_momentum_change=angular_momentum_change_1,
        )

        (
            sea_level_change_2,
            displacement_2,
            gravity_potential_change_2,
            angular_velocity_change_2,
        ) = fingerprint(
            direct_load=direct_load_2,
            displacement_load=displacement_load_2,
            gravitational_potential_load=gravitational_potential_load_2,
            angular_momentum_change=angular_momentum_change_2,
        )

        gravitational_potential_change_1 = (
            fingerprint.gravity_potential_change_to_gravitational_potential_change(
                gravity_potential_change_1, angular_velocity_change_1
            )
        )
        gravitational_potential_change_2 = (
            fingerprint.gravity_potential_change_to_gravitational_potential_change(
                gravity_potential_change_2, angular_velocity_change_2
            )
        )

        g = fingerprint.gravitational_acceleration

        lhs_integrand = direct_load_2 * sea_level_change_1 - (1 / g) * (
            g * displacement_load_2 * displacement_1
            + gravitational_potential_load_2 * gravitational_potential_change_1
        )

        lhs = (
            fingerprint.integrate(lhs_integrand)
            - np.dot(
                angular_momentum_change_2
                - fingerprint.adjoint_angular_momentum_change_from_adjoint_gravitational_potential_load(
                    gravitational_potential_load_2
                ),
                angular_velocity_change_1,
            )
            / g
        )

        rhs_integrand = direct_load_1 * sea_level_change_2 - (1 / g) * (
            g * displacement_load_1 * displacement_2
            + gravitational_potential_load_1 * gravitational_potential_change_2
        )

        rhs = (
            fingerprint.integrate(rhs_integrand)
            - np.dot(
                angular_momentum_change_1
                - fingerprint.adjoint_angular_momentum_change_from_adjoint_gravitational_potential_load(
                    gravitational_potential_load_1
                ),
                angular_velocity_change_2,
            )
            / g
        )

        assert np.isclose(lhs, rhs, rtol=1000 * rtol)
