import pytest
import numpy as np
from dataclasses import FrozenInstanceError

from pyslfp.core import (
    EarthModelParameters,
    LoveNumbers,
    EarthModel,
    MEAN_RADIUS,
    MASS,
    WATER_DENSITY,
    ICE_DENSITY,
)


class TestEarthModelParameters:
    def test_default_initialization(self):
        """Tests default SI values and unit scales."""
        params = EarthModelParameters()

        assert params.length_scale == 1.0
        assert params.density_scale == 1.0

        # Updated to check the 'raw_' properties for SI units
        assert params.raw_mean_radius == MEAN_RADIUS
        assert params.raw_mass == MASS
        assert params.raw_water_density == WATER_DENSITY

    def test_custom_initialization(self):
        """Tests keyword-only custom initialization."""
        custom_radius = 7.0e6
        custom_mass = 6.0e24

        # kw_only=True enforces explicit keyword arguments
        params = EarthModelParameters(
            raw_mean_radius=custom_radius, raw_mass=custom_mass
        )

        assert params.raw_mean_radius == custom_radius
        assert params.raw_mass == custom_mass

    def test_immutability(self):
        """Proves that the dataclass is frozen and prevents accidental mutation."""
        params = EarthModelParameters()

        with pytest.raises(FrozenInstanceError):
            params.length_scale = 2.0

        with pytest.raises(FrozenInstanceError):
            params.raw_mean_radius = 100.0

    def test_standard_non_dimensionalisation_factory(self):
        params = EarthModelParameters.from_standard_non_dimensionalisation()

        assert params.length_scale == 6371000.0
        assert params.density_scale == 1000.0
        assert params.time_scale == 3600.0

        assert params.mean_radius == 1.0
        assert params.water_density == 1.0

        expected_ice_density = ICE_DENSITY / WATER_DENSITY
        assert np.isclose(params.ice_density, expected_ice_density)


class TestLoveNumbers:
    def test_initialization_and_loading(self):
        lmax = 64
        params = EarthModelParameters.from_standard_non_dimensionalisation()
        # lmax and params are strictly positional-only arguments
        ln = LoveNumbers(lmax, params)

        assert isinstance(ln.h, np.ndarray)
        assert len(ln.h) == lmax + 1
        assert isinstance(ln.k, np.ndarray)
        assert len(ln.k) == lmax + 1
        assert isinstance(ln.ht, np.ndarray)
        assert len(ln.ht) == lmax + 1

    def test_lmax_too_large_raises_error(self):
        params = EarthModelParameters.from_standard_non_dimensionalisation()
        lmax_too_large = 5000

        with pytest.raises(ValueError, match="exceeds Love number file max degree"):
            LoveNumbers(lmax_too_large, params)


class TestEarthModel:
    def test_earth_model_composition(self):
        """Tests that the EarthModel correctly wires up parameters and love numbers."""
        lmax = 32

        # lmax is strictly positional; optional arguments must be keywords
        model = EarthModel(lmax)

        # Did it instantiate default parameters?
        assert isinstance(model.parameters, EarthModelParameters)
        assert model.parameters.length_scale == 6371000.0  # Standard ND factory default

        # Did it successfully initialize the LoveNumbers with those parameters?
        assert isinstance(model.love_numbers, LoveNumbers)
        assert model.love_numbers.lmax == lmax
        assert len(model.love_numbers.h) == lmax + 1

    def test_earth_model_default_factory(self):
        """Tests the generation of an EarthModel using the classmethod factory."""
        lmax = 64
        model = EarthModel.default(lmax)

        assert isinstance(model, EarthModel)
        assert model.lmax == lmax
        assert model.parameters.length_scale == 6371000.0
