import pytest
import numpy as np
import matplotlib
from pyshtools import SHCoeffs

# Use a non-interactive backend for testing to prevent plot windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet

from pyslfp.plotting import plot


@pytest.fixture(scope="module")
def sample_grid():
    """
    Provides a simple, non-trivial SHGrid object for plotting tests.
    This creates a simple dipole (degree 1, order 1) field.
    """
    lmax = 16
    coeffs = SHCoeffs.from_zeros(lmax)
    coeffs.set_coeffs(values=1, ls=[1], ms=[1])
    return coeffs.expand(grid="DH")


# ==================================================================== #
#                       Tests for plotting.py                          #
# ==================================================================== #


def test_plot_smoke_test(sample_grid):
    """
    A simple smoke test to ensure the plot function runs without errors
    using default settings. The figure is closed to free memory.
    """
    try:
        fig, ax, im = plot(sample_grid)
        assert fig is not None
        assert ax is not None
        assert im is not None
    finally:
        plt.close("all")


def test_plot_return_types(sample_grid):
    """
    Tests that the plot function returns the correct object types for both
    pcolormesh (default) and contour plots.
    """
    try:
        # Test default pcolormesh plot
        fig_pcm, ax_pcm, im_pcm = plot(sample_grid, contour=False)
        assert isinstance(fig_pcm, Figure)
        assert isinstance(ax_pcm, GeoAxes)
        assert isinstance(im_pcm, QuadMesh)

        # Test contour plot
        fig_cf, ax_cf, im_cf = plot(sample_grid, contour=True)
        assert isinstance(fig_cf, Figure)
        assert isinstance(ax_cf, GeoAxes)
        assert isinstance(im_cf, QuadContourSet)
    finally:
        plt.close("all")


def test_plot_symmetric_option(sample_grid):
    """

    Tests that the `symmetric=True` option correctly sets the color limits
    to be symmetric around zero.
    """
    try:
        fig, ax, im = plot(sample_grid, symmetric=True, contour=False)
        vmin, vmax = im.get_clim()
        assert np.isclose(vmin, -vmax)
    finally:
        plt.close("all")


def test_plot_raises_error_for_wrong_input_type():
    """
    Tests that the plot function raises a ValueError when the input
    is not an SHGrid object, as expected.
    """
    not_a_grid = np.zeros((10, 10))
    with pytest.raises(ValueError, match="must be of SHGrid type"):
        plot(not_a_grid)
