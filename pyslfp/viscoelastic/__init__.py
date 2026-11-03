"""Glacial isostatic adjustment: viscoelastic loading of a spherically
symmetric earth coupled to the sea level equation, its sensitivity
kernels by discrete and continuous adjoints, and the functionals
built on them.

The formulation follows Al-Attar & Tromp (2014), internal variables
in the degree basis; the time-stepping algebra is shared with the
AdGIA library so the two can benchmark one another.  The kernel
drivers are a computational layer: they return kernels as plain grids
and nodal arrays.  `IceHistoryOperator` is the inversion-facing form
of the frozen-shoreline map, a pygeoinf operator on a space of ice
histories, whose adjoint gives the kernels as Riesz representers.
"""

from .earth_models import maxwell_mantle, prem_maxwell, viscoelastic_earth
from .functionals import (
    DecayTimeFit,
    decay_time_measurement,
    fit_decay_time,
)
from .evolution import (
    FinalSeaLevelMisfit,
    ForwardTrajectory,
    IceKernels,
    InitialSeaLevelCalibration,
    Measurement,
    SeaLevelEvolution,
    SeaLevelSnapshot,
    calibrate_initial_sea_level,
)
from .evolution_operator import SeaLevelEvolutionOperator
from .ice_history import IceHistoryOperator
from .operator import (
    SCHEMES,
    ViscoelasticOperator,
    ViscoelasticState,
    ViscoelasticStep,
)
from .rheology import Rheology
from .viscosity import LogViscositySpace
from .weights import backward_euler_weight, exponential_trapezoid_weights

__all__ = [
    "DecayTimeFit",
    "decay_time_measurement",
    "fit_decay_time",
    "FinalSeaLevelMisfit",
    "ForwardTrajectory",
    "SCHEMES",
    "Rheology",
    "LogViscositySpace",
    "IceHistoryOperator",
    "SeaLevelEvolutionOperator",
    "IceKernels",
    "InitialSeaLevelCalibration",
    "Measurement",
    "SeaLevelEvolution",
    "calibrate_initial_sea_level",
    "SeaLevelSnapshot",
    "ViscoelasticOperator",
    "ViscoelasticState",
    "ViscoelasticStep",
    "backward_euler_weight",
    "exponential_trapezoid_weights",
    "maxwell_mantle",
    "prem_maxwell",
    "viscoelastic_earth",
]
