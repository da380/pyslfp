import pytest
import numpy as np
from pyshtools import SHCoeffs
from pyslfp.utils import SHVectorConverter

# ==================================================================== #
#                       Tests for SHVectorConverter                      #
# ==================================================================== #


class TestSHVectorConverter:
    """A test suite for the SHVectorConverter class."""

    def test_initialization(self):
        """Tests that the converter initializes with correct parameters."""
        lmax, lmin = 10, 2
        converter = SHVectorConverter(lmax=lmax, lmin=lmin)
        assert converter.lmax == lmax
        assert converter.lmin == lmin
        expected_size = (lmax + 1) ** 2 - lmin**2
        assert converter.vector_size == expected_size

    def test_initialization_errors(self):
        """Tests that __init__ raises errors for invalid arguments."""
        with pytest.raises(ValueError, match="lmin cannot be greater than lmax"):
            SHVectorConverter(lmax=5, lmin=10)

        with pytest.raises(TypeError, match="must be integers"):
            SHVectorConverter(lmax=10.5, lmin=2)

    @pytest.mark.parametrize("lmax", [8, 32])
    @pytest.mark.parametrize("lmin", [0, 1, 2])
    def test_round_trip_conversion(self, lmax, lmin):
        """
        Tests the default case where a vector is converted back to a
        coefficient array of the same lmax.
        """
        power = np.ones(lmax + 1)
        coeffs_in_obj = SHCoeffs.from_random(power, lmax=lmax, kind="real")
        coeffs_in = coeffs_in_obj.to_array()

        converter = SHVectorConverter(lmax=lmax, lmin=lmin)

        vector = converter.to_vector(coeffs_in)
        coeffs_out = converter.from_vector(vector)

        mask = np.zeros_like(coeffs_in)
        mask[:, lmin : lmax + 1, :] = 1.0

        assert np.allclose(coeffs_in * mask, coeffs_out)

    def test_to_vector_with_smaller_input_lmax(self):
        """
        Tests that to_vector correctly zero-pads the output vector when the
        input coefficient array has a smaller lmax.
        """
        converter_lmax = 20
        input_lmax = 10
        lmin = 2

        converter = SHVectorConverter(lmax=converter_lmax, lmin=lmin)

        power_small = np.ones(input_lmax + 1)
        coeffs_in_small = SHCoeffs.from_random(
            power_small, lmax=input_lmax, kind="real"
        ).to_array()

        vector_full = converter.to_vector(coeffs_in_small)

        converter_small = SHVectorConverter(lmax=input_lmax, lmin=lmin)
        vector_small = converter_small.to_vector(coeffs_in_small)

        assert np.allclose(vector_full[: converter_small.vector_size], vector_small)
        assert np.all(vector_full[converter_small.vector_size :] == 0)

    def test_from_vector_truncation(self):
        """
        Tests that from_vector correctly truncates the coefficients when
        output_lmax is smaller than the converter's lmax.
        """
        converter_lmax = 20
        output_lmax = 10
        lmin = 2

        converter_full = SHVectorConverter(lmax=converter_lmax, lmin=lmin)
        vector_full = np.random.rand(converter_full.vector_size)

        # Truncate the full vector to a smaller coefficient array
        coeffs_truncated = converter_full.from_vector(
            vector_full, output_lmax=output_lmax
        )

        # Check that the shape is correct
        assert coeffs_truncated.shape == (2, output_lmax + 1, output_lmax + 1)

        # Check that the content is correct by comparing to a direct conversion
        converter_small = SHVectorConverter(lmax=output_lmax, lmin=lmin)
        vector_small_part = vector_full[: converter_small.vector_size]
        expected_coeffs = converter_small.from_vector(vector_small_part)

        assert np.allclose(coeffs_truncated, expected_coeffs)

    def test_from_vector_zero_padding(self):
        """
        Tests that from_vector correctly zero-pads the coefficients when
        output_lmax is larger than the converter's lmax.
        """
        converter_lmax = 10
        output_lmax = 20
        lmin = 2

        converter_small = SHVectorConverter(lmax=converter_lmax, lmin=lmin)
        vector_small = np.random.rand(converter_small.vector_size)

        # Zero-pad the small vector to a larger coefficient array
        coeffs_padded = converter_small.from_vector(
            vector_small, output_lmax=output_lmax
        )

        # Check that the shape is correct
        assert coeffs_padded.shape == (2, output_lmax + 1, output_lmax + 1)

        # Check that the lower-degree content is correct
        coeffs_small_expected = converter_small.from_vector(vector_small)
        assert np.allclose(
            coeffs_padded[:, : converter_lmax + 1, : converter_lmax + 1],
            coeffs_small_expected,
        )

        # Check that the higher-degree coefficients are all zero
        assert np.all(coeffs_padded[:, converter_lmax + 1 :, :] == 0)

    def test_from_vector_wrong_size_error(self):
        """
        Tests that from_vector raises a ValueError if the input vector
        has an incorrect size.
        """
        converter = SHVectorConverter(lmax=10, lmin=2)
        bad_vector = np.zeros(converter.vector_size - 1)

        with pytest.raises(ValueError, match="Input vector has incorrect size"):
            converter.from_vector(bad_vector)
