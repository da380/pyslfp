"""The ice-model base classes: the thickness-only contract, the richer
sea-level type, and the animation field validation."""

import pytest
from pyshtools import SHGrid
from pyslfp.ice.ice_models import BaseIceModel, BaseIceSeaLevelModel


class DummyIceModel(BaseIceModel):
    """A minimal thickness-only model."""

    def get_ice_thickness(
        self,
        date: float,
        lmax: int,
        /,
        *,
        grid: str = "DH",
        sampling: int = 1,
        extend: bool = True,
    ):
        return SHGrid.from_zeros(lmax)


class DummyIceSeaLevelModel(BaseIceSeaLevelModel):
    """A minimal model that also carries a sea level."""

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


def test_thickness_only_contract():
    """A thickness-only model needs no sea level, and the richer type
    derives its single-field accessors from the pair."""
    assert DummyIceModel().get_ice_thickness(0.0, 4).data.shape[0] > 0
    rich = DummyIceSeaLevelModel()
    assert rich.get_ice_thickness(0.0, 4).data is not None
    assert rich.get_sea_level(0.0, 4).data is not None


def test_animate_validation_follows_the_type():
    """The animation wrapper validates fields against what the type
    carries: a thickness-only model cannot animate a sea level."""
    with pytest.raises(ValueError, match="Field must be one of"):
        DummyIceModel().animate("dummy.mp4", field="invalid_field")
    with pytest.raises(ValueError, match="Field must be one of"):
        DummyIceModel().animate("dummy.mp4", field="sea_level")
    with pytest.raises(ValueError, match="Field must be one of"):
        DummyIceSeaLevelModel().animate("dummy.mp4", field="invalid_field")
