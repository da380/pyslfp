import pytest
import numpy as np
import matplotlib
from pyshtools import SHCoeffs

# Use a non-interactive backend for testing to prevent plot windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.crs as ccrs

from pyslfp.plotting import plot, create_map_figure


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


def test_create_map_figure_defaults_to_robinson():
    """
    Tests that pyslfp's figure creator defaults to a Robinson projection.
    """
    fig, ax = create_map_figure()

    assert isinstance(ax, GeoAxes)
    assert isinstance(ax.projection, ccrs.Robinson)
    plt.close(fig)


def test_plot_smoke_test_with_pyslfp_defaults(sample_grid):
    """
    A simple smoke test to ensure the plot function runs without errors
    using default settings, and verifies pyslfp's colorbar injection.
    """
    try:
        ax, im = plot(sample_grid)
        assert ax is not None
        assert im is not None

        # Verify pyslfp default of `colorbar=True` worked by checking for a second axis
        assert len(ax.figure.axes) > 1
    finally:
        plt.close("all")


def test_plot_with_existing_ax(sample_grid):
    """
    Tests that passing an existing GeoAxes object works and does not
    create a redundant axis.
    """
    try:
        # Pass a specific projection to verify it doesn't get overridden
        proj = ccrs.PlateCarree()
        fig, ax_in = create_map_figure(projection=proj)

        ax_out, im = plot(sample_grid, ax=ax_in, colorbar=False)

        assert ax_out is ax_in
        assert ax_out.projection == proj
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
