"""
UPDATED test suite for operators in the pyslfp library.

This version is simplified and expanded to use the pygeoinf
LinearOperator.check() method for comprehensive axiom validation.

(This version corrects all fixture issues by deriving spaces
directly from the FingerPrint instance.)
"""

import pytest
import numpy as np
from pyshtools import SHCoeffs
from pygeoinf import EuclideanSpace

from pyslfp import FingerPrint
from pyslfp.finger_print import IceModel, EarthModelParameters
from pyslfp.operators import (
    field_to_sh_coefficient_operator,
    sh_coefficient_to_field_operator,
    tide_gauge_operator,
    grace_operator,
    averaging_operator,
    WMBMethod,
    ice_thickness_change_to_load_operator,
    sea_surface_height_operator,
    ice_projection_operator,
    ocean_projection_operator,
    land_projection_operator,
    sea_level_change_to_load_operator,
    remove_ocean_average_operator,
)

# Use standard non-dimensionalisation for sensible numbers
standard_nondim = EarthModelParameters.from_standard_non_dimensionalisation()


# ================== Fixtures ==================


@pytest.fixture
def fp_instance(lmax):
    """
    Provides a configured FingerPrint instance for the tests,
    using the parameterized lmax and standard non-dimensionalisation.
    """
    fp = FingerPrint(lmax=lmax, earth_model_parameters=standard_nondim)
    fp.set_state_from_ice_ng(version=IceModel.ICE7G)
    return fp


@pytest.fixture
def wmb_method(lmax):
    """Provides a configured WMBMethod instance for testing."""
    fp = FingerPrint(lmax=lmax, earth_model_parameters=standard_nondim)
    return WMBMethod.from_finger_print(observation_degree=lmax, finger_print=fp)


# ================== Helper Functions for Spaces ==================


def get_load_space(fp_instance, space_type):
    """Helper to get the correct load space from the fp instance."""
    if space_type == "lebesgue":
        return fp_instance.lebesgue_load_space()
    else:  # sobolev
        # Use a consistent order/scale for Sobolev tests
        return fp_instance.sobolev_load_space(
            2, 0.5 * fp_instance.mean_sea_floor_radius
        )


def get_response_space(fp_instance, space_type):
    """Helper to get the correct response space from the fp instance."""
    if space_type == "lebesgue":
        return fp_instance.lebesgue_response_space()
    else:  # sobolev
        # Use a consistent order/scale for Sobolev tests
        return fp_instance.sobolev_response_space(
            2, 0.5 * fp_instance.mean_sea_floor_radius
        )


# ================== Tests for field_to_sh_coefficient_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestFieldToShCoefficientOperator:
    """A test suite for the field_to_sh_coefficient_operator factory."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_forward_mapping_known_input(self, lmax, space_type, fp_instance):
        """Tests the forward mapping with a known spherical harmonic field."""
        field_space = get_load_space(fp_instance, space_type)

        l, m = 3, 2
        known_field_lm = SHCoeffs.from_zeros(
            lmax=lmax,
            normalization=field_space.normalization,
            csphase=field_space.csphase,
        )
        known_field_lm.set_coeffs(values=[1.0], ls=[l], ms=[m])
        known_field = field_space.from_coefficients(known_field_lm)

        op = field_to_sh_coefficient_operator(field_space, lmax=5, lmin=2)

        result_vec = op(known_field)

        expected_vec = op.codomain.zero
        idx = (l**2 - 2**2) + (m + l)
        expected_vec[idx] = 1.0

        assert np.allclose(result_vec, expected_vec)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    @pytest.mark.parametrize("lmax_obs", [8, 16])
    @pytest.mark.parametrize("lmin_obs", [0, 2])
    def test_axiom_checks(self, lmax, space_type, fp_instance, lmax_obs, lmin_obs):
        """Tests the operator axioms using the pygeoinf check() method."""
        if lmax_obs > lmax:
            pytest.skip("Observation degree exceeds grid resolution.")

        field_space = get_load_space(fp_instance, space_type)

        op = field_to_sh_coefficient_operator(field_space, lmax=lmax_obs, lmin=lmin_obs)

        op.check(n_checks=3)


# ================== Tests for sh_coefficient_to_field_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestShCoefficientToFieldOperator:
    """A test suite for the sh_coefficient_to_field_operator factory."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_right_inverse_property(self, lmax, space_type, fp_instance):
        """Tests that applying synthesis then analysis returns the original vector."""
        field_space = get_load_space(fp_instance, space_type)
        lmax_obs, lmin_obs = 8, 2
        if lmax_obs > lmax:
            pytest.skip("Observation degree exceeds grid resolution.")

        op_synthesis = sh_coefficient_to_field_operator(
            field_space, lmax=lmax_obs, lmin=lmin_obs
        )
        op_analysis = field_to_sh_coefficient_operator(
            field_space, lmax=lmax_obs, lmin=lmin_obs
        )

        v_initial = op_synthesis.domain.random()
        v_final = op_analysis(op_synthesis(v_initial))

        assert np.isclose(
            op_synthesis.domain.norm(v_initial - v_final)
            / (op_synthesis.domain.norm(v_initial) + 1e-12),
            0.0,
            atol=1e-9,
        )
        assert np.allclose(v_initial, v_final)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_axiom_checks(self, lmax, space_type, fp_instance):
        """Tests the operator axioms using the pygeoinf check() method."""
        field_space = get_load_space(fp_instance, space_type)
        lmax_obs, lmin_obs = 8, 2
        if lmax_obs > lmax:
            pytest.skip("Observation degree exceeds grid resolution.")

        op = sh_coefficient_to_field_operator(field_space, lmax=lmax_obs, lmin=lmin_obs)

        op.check(n_checks=3)


# ================== Tests for block operators ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestBlockOperators:
    """A test suite for operators that act on the response space."""

    @pytest.mark.parametrize("space_type", ["sobolev"])
    def test_tide_gauge_forward_mapping(self, lmax, space_type, fp_instance):
        """Tests the tide gauge forward mapping with a constant field."""
        response_space = get_response_space(fp_instance, space_type)
        points = [(-5, 8), (45, -30)]
        op = tide_gauge_operator(response_space, points)

        field_space = response_space.subspace(0)

        # Ensure grid is compatible with fp_instance
        const_grid = fp_instance.zero_grid()
        const_grid.data[:, :] = 5.0

        response_vector = [
            const_grid,
            field_space.zero,
            field_space.zero,
            EuclideanSpace(2).zero,
        ]
        result = op(response_vector)

        expected = np.full(len(points), 5.0)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("space_type", ["sobolev"])
    def test_tide_gauge_axiom_checks(self, lmax, space_type, fp_instance):
        """Tests the adjoint identity for the tide gauge operator."""
        response_space = get_response_space(fp_instance, space_type)
        points1 = [(10, 20), (-30, 50)]
        op = tide_gauge_operator(response_space, points1)

        op.check(n_checks=3)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_grace_operator_axiom_checks(self, lmax, space_type, fp_instance):
        """Tests the adjoint identity for the grace operator."""
        response_space = get_response_space(fp_instance, space_type)
        obs_degree = lmax // 2
        op = grace_operator(response_space, obs_degree)

        op.check(n_checks=3)


# ================== Tests for averaging_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestAveragingOperators:
    """A test suite for the averaging_operator."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_averaging_axiom_checks(self, lmax, space_type, fp_instance):
        """Tests the adjoint identity for the basic averaging operator."""
        load_space = get_load_space(fp_instance, space_type)
        l2_space = (
            load_space.underlying_space if space_type == "sobolev" else load_space
        )
        weighting_functions1 = [l2_space.random() for _ in range(3)]

        op = averaging_operator(load_space, weighting_functions1)

        op.check(n_checks=3)


# ================== Tests for WMBMethod Operators ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestWMBMethod:
    """A test suite for the WMBMethod class."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_potential_to_load_axiom_checks(
        self, lmax, space_type, wmb_method, fp_instance
    ):
        """Tests the potential field -> load field operator."""
        space = get_load_space(fp_instance, space_type)
        op = wmb_method.potential_field_to_load_operator(space, space)

        op.check(n_checks=3)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_coeff_to_load_axiom_checks(
        self, lmax, space_type, wmb_method, fp_instance
    ):
        """Tests the potential coefficients -> load field operator."""
        load_space = get_load_space(fp_instance, space_type)
        op = wmb_method.potential_coefficient_to_load_operator(load_space)

        op.check(n_checks=3)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_coeff_to_avg_axiom_checks(self, lmax, space_type, wmb_method, fp_instance):
        """Tests the potential coefficients -> load average operator."""
        load_space = get_load_space(fp_instance, space_type)
        l2_space = (
            load_space.underlying_space if space_type == "sobolev" else load_space
        )
        weighting_functions1 = [l2_space.random() for _ in range(3)]

        op = wmb_method.potential_coefficient_to_load_average_operator(
            load_space, weighting_functions1
        )

        op.check(n_checks=3)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_direct_load_to_load_axiom_checks(
        self, lmax, space_type, wmb_method, fp_instance
    ):
        """NEW TEST: Tests the WMB direct_load_to_load_operator."""
        load_space = get_load_space(fp_instance, space_type)
        op = wmb_method.direct_load_to_load_operator(load_space)

        op.check(n_checks=3)


# ================== Tests for ice_thickness_change_to_load_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestIceThicknessToLoadOperator:
    """A test suite for the ice_thickness_change_to_load_operator."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_forward_mapping(self, lmax, space_type, fp_instance):
        """
        Tests the forward mapping of the operator.
        """
        space = get_load_space(fp_instance, space_type)
        op = ice_thickness_change_to_load_operator(fp_instance, space)
        ice_thickness_change = space.random()
        load_actual = op(ice_thickness_change)
        load_expected = (
            fp_instance.ice_density
            * fp_instance.one_minus_ocean_function
            * ice_thickness_change
        )
        assert np.allclose(load_actual.data, load_expected.data, rtol=1e-10)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_axiom_checks(self, lmax, space_type, fp_instance):
        """
        Tests the operator axioms using the pygeoinf check() method.
        """
        space = get_load_space(fp_instance, space_type)
        op = ice_thickness_change_to_load_operator(fp_instance, space)

        op.check(n_checks=3)


# ================== NEW: Tests for sea_level_change_to_load_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestSeaLevelChangeToLoadOperator:
    """A test suite for the sea_level_change_to_load_operator."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_axiom_checks(self, lmax, space_type, fp_instance):
        """
        Tests the operator axioms using the pygeoinf check() method.
        """
        space = get_load_space(fp_instance, space_type)
        op = sea_level_change_to_load_operator(fp_instance, space)

        op.check(n_checks=3)


# ================== NEW: Tests for spatial_mutliplication_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestSpatialMultiplicationOperators:
    """A test suite for the spatial multiplication projection operators."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_ice_projection_axiom_checks(self, lmax, space_type, fp_instance):
        space = get_load_space(fp_instance, space_type)
        op = ice_projection_operator(fp_instance, space, exclude_ice_shelves=False)
        op.check(n_checks=3)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_ocean_projection_axiom_checks(self, lmax, space_type, fp_instance):
        space = get_load_space(fp_instance, space_type)
        op = ocean_projection_operator(fp_instance, space, exclude_ice_shelves=False)
        op.check(n_checks=3)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_land_projection_axiom_checks(self, lmax, space_type, fp_instance):
        space = get_load_space(fp_instance, space_type)
        op = land_projection_operator(fp_instance, space, exclude_ice=True)
        op.check(n_checks=3)


# ================== NEW: Tests for sea_surface_height_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestSeaSurfaceHeightOperator:
    """A test suite for the sea_surface_height_operator."""

    # Must be sobolev for point evaluation in child operators
    @pytest.mark.parametrize("space_type", ["sobolev"])
    @pytest.mark.parametrize("remove_rotational_contribution", [True, False])
    def test_axiom_checks(
        self,
        lmax,
        space_type,
        fp_instance,
        remove_rotational_contribution,
    ):
        """Tests the operator axioms using the pygeoinf check() method."""
        response_space = get_response_space(fp_instance, space_type)
        op = sea_surface_height_operator(
            fp_instance,
            response_space,
            remove_rotational_contribution=remove_rotational_contribution,
        )

        op.check(n_checks=3)


# ================== NEW: Tests for ocean averate operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestRemoveOceanAverageOperator:
    """A test suite for the remove_ocean_average_operator."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_axiom_checks(
        self,
        lmax,
        space_type,
        fp_instance,
    ):
        """Tests the operator axioms using the pygeoinf check() method."""
        load_space = get_load_space(fp_instance, space_type)
        op = remove_ocean_average_operator(
            fp_instance,
            load_space,
        )

        op.check(n_checks=3)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_result(
        self,
        lmax,
        space_type,
        fp_instance,
    ):
        """Tests the operator axioms using the pygeoinf check() method."""
        load_space = get_load_space(fp_instance, space_type)
        op = remove_ocean_average_operator(
            fp_instance,
            load_space,
        )

        u = load_space.project_function(lambda point: 1)
        v = op(u)
        int = fp_instance.ocean_average(v)

        np.testing.assert_allclose(int, 0, rtol=1e-8)
