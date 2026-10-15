"""What the elastic and viscoelastic sides share: the Earth model and
its parameters, the background state, and the sea level engine.

`SeaLevelEquation` solves the equilibrium sea level equation -- linear,
generalised or nonlinear -- for whatever Love numbers its Earth model
carries: the elastic table of a fingerprint, or the per-step effective
table of a viscoelastic evolution.  Nothing in the engine is elastic
but the table it is handed.
"""

from .model import (
    EQUATORIAL_MOMENT_OF_INERTIA,
    EQUATORIAL_RADIUS,
    GRAVITATIONAL_ACCELERATION,
    GRAVITATIONAL_CONSTANT,
    ICE_DENSITY,
    MASS,
    MEAN_RADIUS,
    MEAN_SEA_FLOOR_RADIUS,
    POLAR_MOMENT_OF_INERTIA,
    POLAR_RADIUS,
    ROTATION_FREQUENCY,
    WATER_DENSITY,
    EarthModel,
    EarthModelParameters,
)
from .anderson import AndersonMixer
from .sea_level import SeaLevelEquation
from .state import EarthState

__all__ = [
    "EarthModel",
    "EarthModelParameters",
    "EarthState",
    "AndersonMixer",
    "SeaLevelEquation",
    "EQUATORIAL_RADIUS",
    "POLAR_RADIUS",
    "MEAN_RADIUS",
    "MEAN_SEA_FLOOR_RADIUS",
    "MASS",
    "GRAVITATIONAL_ACCELERATION",
    "GRAVITATIONAL_CONSTANT",
    "EQUATORIAL_MOMENT_OF_INERTIA",
    "POLAR_MOMENT_OF_INERTIA",
    "ROTATION_FREQUENCY",
    "WATER_DENSITY",
    "ICE_DENSITY",
]
