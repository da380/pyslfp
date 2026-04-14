"""
Test suite for the abstract BaseIceModel.
"""

import pytest
from pyshtools import SHGrid
from pyslfp.ice.ice_models import BaseIceModel


class DummyIceModel(BaseIceModel):
    """A minimal concrete implementation of BaseIceModel for pure isolation testing."""

    def get_ice_thickness_and_sea_level(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ):
        return SHGrid.from_zeros(lmax), SHGrid.from_zeros(lmax)


def test_base_ice_model_initialization():
    """Tests basic initialization and length scale inheritance."""
    model = DummyIceModel(length_scale=2000.0)
    assert model.length_scale == 2000.0


def test_base_ice_model_animate_validation():
    """Ensures the BaseIceModel animation wrapper validates field inputs."""
    model = DummyIceModel()

    with pytest.raises(ValueError, match="Field must be one of"):
        # We don't actually want to render a video in CI, so testing the
        # validation catch before ffmpeg spins up is perfect.
        model.animate("dummy.mp4", field="invalid_field")
