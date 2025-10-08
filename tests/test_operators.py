import pytest
import numpy as np
from pyshtools import SHGrid, SHCoeffs
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev
from pygeoinf import HilbertSpaceDirectSum, EuclideanSpace

from pyslfp import FingerPrint
from pyslfp.operators import (
    field_to_sh_coefficient_operator,
    sh_coefficient_to_field_operator,
    tide_gauge_operator,
    grace_operator,
    averaging_operator,
    WMBMethod,
    ice_thickness_change_to_load_operator,
)

# Define a standard radius for creating test spaces
RADIUS = 6371.0


# ================== Tests for field_to_sh_coefficient_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestFieldToShCoefficientOperator:
    """A test suite for the field_to_sh_coefficient_operator factory."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_forward_mapping_known_input(self, lmax, space_type):
        """Tests the forward mapping with a known spherical harmonic field."""
        if space_type == "lebesgue":
            field_space = Lebesgue(lmax, radius=RADIUS)
        else:  # sobolev
            field_space = Sobolev(lmax, 2, 0.5, radius=RADIUS)

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
    def test_adjoint_identity(self, lmax, space_type, lmax_obs, lmin_obs):
        """Tests the adjoint identity for a variety of observation degrees."""
        if lmax_obs > lmax:
            pytest.skip("Observation degree exceeds grid resolution.")

        if space_type == "lebesgue":
            field_space = Lebesgue(lmax, radius=RADIUS)
        else:
            field_space = Sobolev(lmax, 2, 0.5, radius=RADIUS)

        op = field_to_sh_coefficient_operator(field_space, lmax=lmax_obs, lmin=lmin_obs)

        u = field_space.random()
        v = op.codomain.random()

        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))

        assert np.isclose(lhs, rhs, rtol=1e-10)


# ================== Tests for sh_coefficient_to_field_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestShCoefficientToFieldOperator:
    """A test suite for the sh_coefficient_to_field_operator factory."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_right_inverse_property(self, lmax, space_type):
        """Tests that applying synthesis then analysis returns the original vector."""
        if space_type == "lebesgue":
            field_space = Lebesgue(lmax, radius=RADIUS)
        else:
            field_space = Sobolev(lmax, 2, 0.5, radius=RADIUS)

        lmax_obs, lmin_obs = 8, 2

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
            / op_synthesis.domain.norm(v_initial),
            0.0,
        )
        assert np.allclose(v_initial, v_final)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_adjoint_identity(self, lmax, space_type):
        """Tests the adjoint identity for the synthesis operator."""
        if space_type == "lebesgue":
            field_space = Lebesgue(lmax, radius=RADIUS)
        else:
            field_space = Sobolev(lmax, 2, 0.5, radius=RADIUS)

        op = sh_coefficient_to_field_operator(field_space, lmax=8, lmin=2)

        u = op.domain.random()
        v = op.codomain.random()

        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))

        assert np.isclose(lhs, rhs, rtol=1e-10)


# ================== Helper to build response spaces ==================


def _create_response_space(lmax, space_type):
    """Helper function to build response spaces for tests."""
    if space_type == "lebesgue":
        field_space = Lebesgue(lmax, radius=RADIUS)
    else:  # sobolev
        field_space = Sobolev(lmax, 2, 0.5, radius=RADIUS)

    return HilbertSpaceDirectSum(
        [field_space, field_space, field_space, EuclideanSpace(2)]
    )


# ================== Tests for block operators ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestBlockOperators:
    """A test suite for operators that act on the response space."""

    def test_tide_gauge_forward_mapping(self, lmax):
        """Tests the tide gauge forward mapping with a constant field."""
        response_space = _create_response_space(lmax, "sobolev")
        points = [(-5, 8), (45, -30)]
        op = tide_gauge_operator(response_space, points)

        field_space = response_space.subspace(0)
        const_grid = SHGrid.from_zeros(lmax=lmax)
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

    def test_tide_gauge_adjoint_identity(self, lmax):
        """Tests the adjoint identity for the tide gauge operator."""
        response_space = _create_response_space(lmax, "sobolev")
        points = [(10, 20), (-30, 50)]
        op = tide_gauge_operator(response_space, points)

        u = op.domain.random()
        v = op.codomain.random()

        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))
        assert np.isclose(lhs, rhs, rtol=1e-10)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_grace_operator_adjoint_identity(self, lmax, space_type):
        """Tests the adjoint identity for the grace operator."""
        response_space = _create_response_space(lmax, space_type)
        obs_degree = lmax // 2
        op = grace_operator(response_space, obs_degree)

        u = op.domain.random()
        v = op.codomain.random()

        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))
        assert np.isclose(lhs, rhs, rtol=1e-10)


# ================== Tests for averaging_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestAveragingOperators:
    """A test suite for the averaging_operator."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_averaging_adjoint_identity(self, lmax, space_type):
        """Tests the adjoint identity for the basic averaging operator."""
        if space_type == "lebesgue":
            load_space = Lebesgue(lmax, radius=RADIUS)
        else:
            load_space = Sobolev(lmax, 2, 0.5, radius=RADIUS)

        l2_space = (
            load_space.underlying_space if space_type == "sobolev" else load_space
        )
        weighting_functions = [l2_space.random() for _ in range(3)]
        op = averaging_operator(load_space, weighting_functions)

        u = op.domain.random()
        v = op.codomain.random()

        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))
        assert np.isclose(lhs, rhs, rtol=1e-10)


# ================== Tests for WMBMethod Operators ==================


@pytest.fixture(scope="module")
def wmb_method(lmax=32):
    """Provides a configured WMBMethod instance for testing."""
    # The factory requires an object with Earth Model parameters.
    # A FingerPrint instance is a convenient way to provide one.
    fp = FingerPrint(lmax=lmax)
    return WMBMethod.from_finger_print(observation_degree=lmax, finger_print=fp)


@pytest.mark.parametrize("lmax", [16, 32])
class TestWMBMethod:
    """A test suite for the WMBMethod class."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_potential_to_load_adjoint_identity(self, lmax, space_type, wmb_method):
        """Tests the potential field -> load field operator."""
        if space_type == "lebesgue":
            space = Lebesgue(lmax, radius=RADIUS)
        else:
            space = Sobolev(lmax, 2, 0.5, radius=RADIUS)

        op = wmb_method.potential_field_to_load_operator(space, space)

        u = op.domain.random()
        v = op.codomain.random()

        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))
        assert np.isclose(lhs, rhs, rtol=1e-10)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_coeff_to_load_adjoint_identity(self, lmax, space_type, wmb_method):
        """Tests the potential coefficients -> load field operator."""
        if space_type == "lebesgue":
            load_space = Lebesgue(lmax, radius=RADIUS)
        else:
            load_space = Sobolev(lmax, 2, 0.5, radius=RADIUS)

        op = wmb_method.potential_coefficient_to_load_operator(load_space)

        u = op.domain.random()
        v = op.codomain.random()

        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))
        assert np.isclose(lhs, rhs, rtol=1e-10)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_coeff_to_avg_adjoint_identity(self, lmax, space_type, wmb_method):
        """Tests the potential coefficients -> load average operator."""
        if space_type == "lebesgue":
            load_space = Lebesgue(lmax, radius=RADIUS)
        else:
            load_space = Sobolev(lmax, 2, 0.5, radius=RADIUS)

        # Create some random averaging functions
        l2_space = (
            load_space.underlying_space if space_type == "sobolev" else load_space
        )
        weighting_functions = [l2_space.random() for _ in range(3)]

        op = wmb_method.potential_coefficient_to_load_average_operator(
            load_space, weighting_functions
        )

        u = op.domain.random()
        v = op.codomain.random()

        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))
        assert np.isclose(lhs, rhs, rtol=1e-10)


# ================== Tests for ice_thickness_change_to_load_operator ==================


@pytest.mark.parametrize("lmax", [16, 32])
class TestIceThicknessToLoadOperator:
    """A test suite for the ice_thickness_change_to_load_operator."""

    @pytest.fixture
    def fp_instance(self, lmax):
        """Provides a configured FingerPrint instance for the tests."""
        fp = FingerPrint(lmax=lmax)
        fp.set_state_from_ice_ng()
        return fp

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_forward_mapping(self, lmax, space_type, fp_instance):
        """
        Tests the forward mapping of the operator to ensure it correctly
        converts an ice thickness change grid to a surface mass load grid.
        """
        if space_type == "lebesgue":
            space = Lebesgue(lmax, radius=fp_instance.mean_sea_floor_radius)
        else:
            space = Sobolev(lmax, 2, 0.5, radius=fp_instance.mean_sea_floor_radius)

        # Create the operator to be tested
        op = ice_thickness_change_to_load_operator(fp_instance, space)

        # 1. Create a random ice thickness change field
        ice_thickness_change = space.random()

        # 2. Apply the operator to get the actual load
        load_actual = op(ice_thickness_change)

        # 3. Manually calculate the expected load based on the physical definition
        load_expected = (
            fp_instance.ice_density
            * fp_instance.one_minus_ocean_function
            * ice_thickness_change
        )

        # 4. Assert that the operator's output matches the expected calculation
        assert np.allclose(load_actual.data, load_expected.data, rtol=1e-10)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_adjoint_identity(self, lmax, space_type, fp_instance):
        """
        Tests the adjoint identity for the operator. This confirms that the
        operator is correctly defined as formally self-adjoint.
        """
        if space_type == "lebesgue":
            space = Lebesgue(lmax, radius=fp_instance.mean_sea_floor_radius)
        else:
            space = Sobolev(lmax, 2, 0.5, radius=fp_instance.mean_sea_floor_radius)

        op = ice_thickness_change_to_load_operator(fp_instance, space)

        # Create two random fields in the space
        u = space.random()
        v = space.random()

        # Test the adjoint identity: <A(u), v> = <u, A*(v)>
        lhs = space.inner_product(op(u), v)
        rhs = space.inner_product(u, op.adjoint(v))

        assert np.isclose(lhs, rhs, rtol=1e-10)
