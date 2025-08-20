"""
Module collecting some small helper functions or classes. 
"""

import numpy as np
from typing import Optional


class SHVectorConverter:
    """
    Handles conversion between pyshtools 3D coefficient arrays and 1D vectors.

    This class provides a bridge between the pyshtools data structure for
    spherical harmonic coefficients, a 3D array of shape (2, lmax+1, lmax+1),
    and the 1D vector format often used in linear algebra and inverse problems.

    The vector is ordered by degree l, and within each degree, by order m,
    from -l to +l.

    Args:
        lmax (int): The maximum spherical harmonic degree to include.
        lmin (int): The minimum spherical harmonic degree to include. Defaults to 2.
    """

    def __init__(self, lmax: int, lmin: int = 2):
        if not isinstance(lmax, int) or not isinstance(lmin, int):
            raise TypeError("lmax and lmin must be integers.")
        if lmin > lmax:
            raise ValueError("lmin cannot be greater than lmax.")

        self.lmax = lmax
        self.lmin = lmin
        self.vector_size = (self.lmax + 1) ** 2 - self.lmin**2

    def to_vector(self, coeffs: np.ndarray) -> np.ndarray:
        """Converts a pyshtools 3D coefficient array to a 1D vector.

        If the input coefficients have a smaller lmax than the converter,
        the missing high-degree coefficients in the output vector will be zero.

        Args:
            coeffs (np.ndarray): A pyshtools-compatible coefficient array
                of shape (2, l_in+1, l_in+1).

        Returns:
            np.ndarray: A 1D vector of the coefficients from lmin to lmax.
        """
        lmax_in = coeffs.shape[1] - 1
        vec = np.zeros(self.vector_size)
        loop_lmax = min(self.lmax, lmax_in)

        for l in range(self.lmin, loop_lmax + 1):
            start_idx = l**2 - self.lmin**2
            sin_part = coeffs[1, l, 1 : l + 1][::-1]
            cos_part = coeffs[0, l, 0 : l + 1]
            vec[start_idx : start_idx + l] = sin_part
            vec[start_idx + l : start_idx + 2 * l + 1] = cos_part

        return vec

    def from_vector(
        self, vec: np.ndarray, output_lmax: Optional[int] = None
    ) -> np.ndarray:
        """Converts a 1D vector back to a pyshtools 3D coefficient array.

        This method can create an array that is larger (zero-padding) or
        smaller (truncating) than the lmax of the converter.

        Args:
            vec (np.ndarray): A 1D vector of coefficients.
            output_lmax (Optional[int]): The desired lmax for the output array.
                If None, defaults to the converter's lmax.

        Returns:
            np.ndarray: A pyshtools-compatible coefficient array.
        """
        if vec.size != self.vector_size:
            raise ValueError("Input vector has incorrect size.")

        # If output_lmax is not specified, default to the converter's lmax
        lmax_out = output_lmax if output_lmax is not None else self.lmax

        # Create the output array of the desired size, initialized to zeros
        coeffs = np.zeros((2, lmax_out + 1, lmax_out + 1))

        # Determine the loop range: iterate up to the minimum of the two lmax values
        loop_lmax = min(self.lmax, lmax_out)

        for l in range(self.lmin, loop_lmax + 1):
            start_idx = l**2 - self.lmin**2
            coeffs[1, l, 1 : l + 1] = vec[start_idx : start_idx + l][::-1]
            coeffs[0, l, 0 : l + 1] = vec[start_idx + l : start_idx + 2 * l + 1]

        return coeffs
