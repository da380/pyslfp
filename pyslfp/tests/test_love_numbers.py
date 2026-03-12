import pytest
import numpy as np
from pyslfp.love_numbers import LoveNumbers
from pyslfp.physical_parameters import EarthModelParameters


class TestLoveNumbers:
    """A test suite for the LoveNumbers class."""

    def test_initialization_and_loading(self):
        """
        Tests that the LoveNumbers class initializes correctly, loads the
        default data file, and creates numpy arrays of the correct size.
        """
        lmax = 64
        params = EarthModelParameters.from_standard_non_dimensionalisation()

        # Create an instance of the class
        ln = LoveNumbers(lmax, params)

        # Assert that the Love number arrays have been loaded and are the correct type and size
        assert isinstance(ln.h, np.ndarray)
        assert len(ln.h) == lmax + 1

        assert isinstance(ln.k, np.ndarray)
        assert len(ln.k) == lmax + 1

        assert isinstance(ln.ht, np.ndarray)
        assert len(ln.ht) == lmax + 1

    def test_lmax_too_large_raises_error(self):
        """
        Tests that initializing with an lmax greater than what's available
        in the data file correctly raises a ValueError.
        """
        params = EarthModelParameters.from_standard_non_dimensionalisation()

        # The default Love number file goes up to degree 4096
        lmax_too_large = 5000

        with pytest.raises(ValueError, match="is larger than the maximum degree"):
            LoveNumbers(lmax_too_large, params)
