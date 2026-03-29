import pytest
import numpy as np
import matplotlib
from pyshtools import SHCoeffs

# Use a non-interactive backend for testing to prevent plot windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet
from unittest.mock import patch

from pyslfp.plotting import plot, plot_corner_distributions


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


@pytest.fixture
def mock_measure():
    """
    Provides a duck-typed mock of a GaussianMeasure for testing
    the corner plot's secondary axis logic without needing pygeoinf's full setup.
    """

    class MockCovariance:
        def matrix(self, **kwargs):
            return np.array([[1.0, 0.5], [0.5, 1.0]])

    class MockMeasure:
        def __init__(self):
            self.expectation = np.array([0.0, 1.0])
            self.covariance = MockCovariance()

    return MockMeasure()


# ==================================================================== #
#                       Tests for plotting.py                          #
# ==================================================================== #


def test_plot_smoke_test(sample_grid):
    """
    A simple smoke test to ensure the plot function runs without errors
    using default settings, unpacking the updated (ax, im) tuple.
    """
    try:
        ax, im = plot(sample_grid)
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
        ax_pcm, im_pcm = plot(sample_grid, contour=False)
        assert isinstance(ax_pcm, GeoAxes)
        assert isinstance(im_pcm, QuadMesh)

        # Test contour plot
        ax_cf, im_cf = plot(sample_grid, contour=True)
        assert isinstance(ax_cf, GeoAxes)
        assert isinstance(im_cf, QuadContourSet)
    finally:
        plt.close("all")


def test_plot_with_existing_ax(sample_grid):
    """
    Tests that passing an existing GeoAxes object works and does not
    create a redundant axis.
    """
    try:
        fig = plt.figure()
        ax_in = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

        ax_out, im = plot(sample_grid, ax=ax_in)

        # Ensure the function used the axis we provided
        assert ax_out is ax_in
    finally:
        plt.close("all")


def test_plot_symmetric_option(sample_grid):
    """
    Tests that the `symmetric=True` option correctly sets the color limits
    to be symmetric around zero.
    """
    try:
        ax, im = plot(sample_grid, symmetric=True, contour=False)
        vmin, vmax = im.get_clim()
        assert np.isclose(vmin, -vmax)
    finally:
        plt.close("all")


def test_plot_raises_error_for_wrong_input_type():
    """
    Tests that the plot function raises a ValueError when the input
    is not an SHGrid object.
    """
    not_a_grid = np.zeros((10, 10))
    with pytest.raises(ValueError, match="f must be of SHGrid type"):
        plot(not_a_grid)


@patch("pyslfp.plotting.pygeoinf_corner_plot")
def test_corner_plot_injects_prior_axis(mock_base_plot, mock_measure):
    """
    Tests that plot_corner_distributions correctly injects the custom
    secondary x-axis when a prior_measure is provided.
    """
    try:
        # Mock the pygeoinf return to be a standard 2x2 axes array
        fig, axes = plt.subplots(2, 2)
        mock_base_plot.return_value = axes

        out_axes = plot_corner_distributions(mock_measure, prior_measure=mock_measure)

        # Ensure the mock was called
        mock_base_plot.assert_called_once()

        # Check that the diagonal axes now have a secondary x-axis injected
        # In Matplotlib, secondary axes are added as children to the primary axis
        assert len(out_axes[0, 0].child_axes) > 0
        assert len(out_axes[1, 1].child_axes) > 0

        # Verify the custom label was applied to the secondary axis
        sec_ax = out_axes[0, 0].child_axes[0]
        assert "Distance from Prior Mean" in sec_ax.get_xlabel()

    finally:
        plt.close("all")
