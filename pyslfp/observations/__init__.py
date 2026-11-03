"""
Observations of the response to a load: tide gauges, satellite altimetry and
GRACE.

An observation model (`ObservationModel`) takes its observable as an operator
from the model space, a set-up's or a viscoelastic operator's, and adds what
the measurement does: the sampling, the network's data and the noise. The
physics, direct terms included, stays with whatever built the observable. Each
module also samples a response directly, and handles its network's data: the
GLOSS tide gauges, independent altimetry points over the ocean and the ice
sheets, and GRACE's Stokes coefficients.
"""

from .altimetry import (
    AltimetryObservationModel,
    altimetry_averaging_operator,
    altimetry_point_operator,
    ice_altimetry_points,
    ocean_altimetry_points,
)
from .base import ObservationModel
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
    "ObservationModel",
    "AltimetryObservationModel",
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
