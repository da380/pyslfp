import pytest
import numpy as np
import pygeoinf as inf
from pyslfp import FingerPrint
from pyslfp.operators import (
    sh_coefficient_operator,
    tide_gauge_operator,
    grace_operator,
    averaging_operator,
)

# ================== Fixtures ==================


@pytest.fixture(scope="module", params=[16, 32], ids=["lmax16", "lmax32"])
def fingerprint(request):
    """Provides a FingerPrint instance for creating test spaces."""
    return FingerPrint(lmax=request.param)


# ================== Tests for sh_coefficient_operator ==================


class TestShCoefficientOperator:
    """A test suite for the sh_coefficient_operator factory."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    @pytest.mark.parametrize("lmax_obs", [8, 16])
    @pytest.mark.parametrize("lmin_obs", [0, 2])
    def test_adjoint_identity(self, fingerprint, space_type, lmax_obs, lmin_obs):
        """
        Tests the adjoint identity for a variety of observation degrees
        and for both Lebesgue and Sobolev spaces.
        """
        if lmax_obs > fingerprint.lmax:
            pytest.skip("Observation degree exceeds grid resolution.")

        if space_type == "lebesgue":
            field_space = fingerprint.lebesgue_load_space()
        else:  # sobolev
            field_space = fingerprint.sobolev_load_space(2, 0.5)

        op = sh_coefficient_operator(field_space, lmax=lmax_obs, lmin=lmin_obs)

        # Create random elements in the domain and codomain to test the identity
        u = field_space.random()
        v = op.codomain.random()

        # Manually perform the adjoint identity test (dot-product test)
        # ⟨A(u), v⟩ = ⟨u, A*(v)⟩
        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))

        assert np.isclose(lhs, rhs, rtol=1e-12)


# ================== Tests for tide_gauge_operator ==================


class TestTideGaugeOperator:
    """A test suite for the tide_gauge_operator factory."""

    @pytest.mark.parametrize(
        "points",
        [[(10, 20)], [(-5, 8), (45, -30), (22, 120)]],
        ids=["single_point", "multi_point"],
    )
    def test_forward_mapping(self, fingerprint, points):
        """
        Tests the forward mapping with a constant field to ensure it
        returns the correct values at the specified points.
        """
        # Point evaluation requires a Sobolev space of order > 1
        response_space = fingerprint.sobolev_response_space(order=2, scale=0.5)
        op = tide_gauge_operator(response_space, points)

        # Create a response where the sea level is a constant field of value 5.0
        field_space = response_space.subspace(0)
        const_slc = fingerprint.constant_grid(5.0)
        zero_field = field_space.zero
        zero_vec = response_space.subspace(3).zero

        response_vector = [const_slc, zero_field, zero_field, zero_vec]

        # Apply the operator
        result = op(response_vector)

        # The result should be a vector where every entry is 5.0
        expected = np.full(len(points), 5.0)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("sobolev_order", [2, 3])
    def test_adjoint_identity(self, fingerprint, sobolev_order):
        """Tests the adjoint identity for the tide gauge operator."""
        response_space = fingerprint.sobolev_response_space(
            order=sobolev_order, scale=0.5
        )
        points = [(10, 20), (-30, 50)]
        op = tide_gauge_operator(response_space, points)

        u = op.domain.random()
        v = op.codomain.random()

        # Manually perform the adjoint identity test (dot-product test)
        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))

        assert np.isclose(lhs, rhs, rtol=1e-12)


# ================== Tests for grace_operator ==================


class TestGraceOperator:
    """A test suite for the grace_operator factory."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    @pytest.mark.parametrize("obs_degree", [8, 16])
    def test_adjoint_identity(self, fingerprint, space_type, obs_degree):
        """
        Tests the adjoint identity for the full block operator for both
        Lebesgue and Sobolev response spaces.
        """
        if obs_degree > fingerprint.lmax:
            pytest.skip("Observation degree exceeds grid resolution.")

        if space_type == "lebesgue":
            response_space = fingerprint.lebesgue_response_space()
        else:  # sobolev
            response_space = fingerprint.sobolev_response_space(order=2, scale=0.5)

        op = grace_operator(response_space, obs_degree)

        u = op.domain.random()
        v = op.codomain.random()

        # Manually perform the adjoint identity test (dot-product test)
        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))

        assert np.isclose(lhs, rhs, rtol=1e-12)


# ================== Tests for averaging_operator ==================


class TestAveragingOperator:
    """A test suite for the averaging_operator factory."""

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    def test_forward_mapping_known_input(self, fingerprint, space_type):
        """
        Tests the forward mapping with a known input. The inner product of a
        function with itself should be its squared L2 norm.
        """
        if space_type == "lebesgue":
            load_space = fingerprint.lebesgue_load_space()
        else:  # sobolev
            load_space = fingerprint.sobolev_load_space(2, 0.5)

        # Use a single, non-trivial weighting function
        l2_space = (
            load_space.underlying_space if space_type == "sobolev" else load_space
        )
        weighting_function = l2_space.random()

        op = averaging_operator(load_space, [weighting_function])

        # The input u is the same as the weighting function w
        u = weighting_function

        # The result should be a single-element vector containing <w, w>_L2
        result = op(u)

        # Calculate the expected L2 norm squared directly
        expected_norm_sq = l2_space.inner_product(
            weighting_function, weighting_function
        )

        assert np.isclose(result[0], expected_norm_sq)

    @pytest.mark.parametrize("space_type", ["lebesgue", "sobolev"])
    @pytest.mark.parametrize("num_weights", [1, 3])
    def test_adjoint_identity(self, fingerprint, space_type, num_weights):
        """
        Tests the adjoint identity for the averaging operator.
        """
        if space_type == "lebesgue":
            load_space = fingerprint.lebesgue_load_space()
        else:  # sobolev
            load_space = fingerprint.sobolev_load_space(order=2, scale=0.5)

        # Create a list of random weighting functions
        l2_space = (
            load_space.underlying_space if space_type == "sobolev" else load_space
        )
        weighting_functions = [l2_space.random() for _ in range(num_weights)]

        op = averaging_operator(load_space, weighting_functions)

        u = op.domain.random()
        v = op.codomain.random()

        # Manually perform the adjoint identity test (dot-product test)
        lhs = op.codomain.inner_product(op(u), v)
        rhs = op.domain.inner_product(u, op.adjoint(v))

        assert np.isclose(lhs, rhs, rtol=1e-12)
