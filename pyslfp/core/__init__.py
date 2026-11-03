"""What the elastic and viscoelastic sides share: the Earth model and
its parameters, the background state, and the sea level engine.

`SeaLevelEquation` solves the equilibrium sea level equation -- linear,
generalised or nonlinear -- for whatever Love numbers its Earth model
carries: the elastic table of a fingerprint, or the per-step effective
table of a viscoelastic evolution.  Nothing in the engine is elastic
but the table it is handed.

`response` holds the response to a load, the displacement included, as one
spectral object (`GRDResponse`, `SurfaceDisplacement`) with its Hilbert space,
and `water_bodies` decides which water below a level is ocean and which is
landlocked, the regions a state holds as land.
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
from .response import (
    GRDResponse,
    GRDSpace,
    SurfaceDisplacement,
    SurfaceDisplacementSpace,
)
from .history import FieldHistory, FieldHistorySpace
from .spaces import (
    lebesgue_load_space,
    lebesgue_response_space,
    sobolev_load_space,
    sobolev_response_space,
)

__all__ = [
    "EarthModel",
    "EarthModelParameters",
    "EarthState",
    "AndersonMixer",
    "SeaLevelEquation",
    "GRDResponse",
    "GRDSpace",
    "SurfaceDisplacement",
    "SurfaceDisplacementSpace",
    "FieldHistory",
    "FieldHistorySpace",
    "lebesgue_load_space",
    "lebesgue_response_space",
    "sobolev_load_space",
    "sobolev_response_space",
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
