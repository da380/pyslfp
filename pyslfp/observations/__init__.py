"""
Observations of the response to a load: tide gauges, satellite altimetry and
GRACE.

Each module reads its observable off a response, samples or averages it as
the measurement does, and handles the network's data: the GLOSS tide gauges,
independent altimetry points over the ocean and the ice sheets, and GRACE's
Stokes coefficients. The observation models compose these with the elastic
fingerprint operator.
"""

from .altimetry import (
    altimetry_averaging_operator,
    altimetry_point_operator,
    ice_altimetry_points,
    ocean_altimetry_points,
)
from .grace import (
    GRACE_GRAVITATIONAL_PARAMETER,
    GRACE_REFERENCE_RADIUS,
    GraceObservationModel,
    WMBMethod,
    grace_observation_operator,
    stokes_coefficient_operator,
)
from .tide_gauges import (
    TideGaugeObservationModel,
    read_gloss_tide_gauge_data,
    tide_gauge_operator,
)

__all__ = [
    "ocean_altimetry_points",
    "ice_altimetry_points",
    "altimetry_point_operator",
    "altimetry_averaging_operator",
    "grace_observation_operator",
    "stokes_coefficient_operator",
    "GraceObservationModel",
    "WMBMethod",
    "GRACE_GRAVITATIONAL_PARAMETER",
    "GRACE_REFERENCE_RADIUS",
    "read_gloss_tide_gauge_data",
    "tide_gauge_operator",
    "TideGaugeObservationModel",
]
