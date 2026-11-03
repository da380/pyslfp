"""What the elastic and viscoelastic sides share: the Earth model and
its parameters, the background state, and the sea level engine.

`SeaLevelEquation` solves the equilibrium sea level equation -- linear,
generalised or nonlinear -- for whatever Love numbers its Earth model
carries: the elastic table of a fingerprint, or the per-step effective
table of a viscoelastic evolution.  Nothing in the engine is elastic
but the table it is handed.

`response` holds the response to a load, the displacement included, as one
spectral object (`GRDResponse`, `SurfaceDisplacement`) with its Hilbert space,
and `history` a field through time (`FieldHistory`) with its space. `spaces`
builds the load and response spaces over a model's sphere, `observables`
reads physical quantities off a response (the centrifugal potential, the
geocentric sea level), and `spatial` projects and averages fields over the
ocean, the ice and named regions. `water_bodies` decides which water below a
level is ocean and which is landlocked, the regions a state holds as land.
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
    check_load_space,
    check_response_space,
    lebesgue_load_space,
    lebesgue_response_space,
    sobolev_load_space,
    sobolev_response_space,
    underlying_space,
)
from .observables import centrifugal_potential_operator, geocentric_sea_level_operator
from .spatial import (
    averaging_operator,
    ice_average_operator,
    ice_projection_operator,
    ice_sheet_averaging_operator,
    ice_sheet_basis_operator,
    ice_sheet_projection_operator,
    land_average_operator,
    land_projection_operator,
    ocean_average_operator,
    ocean_projection_operator,
    region_average_operator,
    region_projection_operator,
    remove_ocean_average_operator,
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
    "underlying_space",
    "check_load_space",
    "check_response_space",
    "centrifugal_potential_operator",
    "geocentric_sea_level_operator",
    "averaging_operator",
    "ice_average_operator",
    "ice_projection_operator",
    "ice_sheet_averaging_operator",
    "ice_sheet_basis_operator",
    "ice_sheet_projection_operator",
    "land_average_operator",
    "land_projection_operator",
    "ocean_average_operator",
    "ocean_projection_operator",
    "region_average_operator",
    "region_projection_operator",
    "remove_ocean_average_operator",
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
