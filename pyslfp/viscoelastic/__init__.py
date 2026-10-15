"""Forward GIA modelling: viscoelastic loading of a spherically
symmetric earth, coupled to the sea level equation.

The theory basis is Al-Attar & Tromp (2014), internal variables in the
degree basis; the time-stepping algebra is shared with the AdGIA
library for reciprocal benchmarking.  See docs/planning/gia_plan.md
and gia_notes.md.
"""

from .evolution import (
    IceKernels,
    Measurement,
    SeaLevelEvolution,
    SeaLevelSnapshot,
    calibrate_initial_sea_level,
)
from .operator import (
    SCHEMES,
    ViscoelasticOperator,
    ViscoelasticState,
    ViscoelasticStep,
)
from .rheology import Rheology
from .weights import backward_euler_weight, exponential_trapezoid_weights

__all__ = [
    "SCHEMES",
    "Rheology",
    "IceKernels",
    "Measurement",
    "SeaLevelEvolution",
    "calibrate_initial_sea_level",
    "SeaLevelSnapshot",
    "ViscoelasticOperator",
    "ViscoelasticState",
    "ViscoelasticStep",
    "backward_euler_weight",
    "exponential_trapezoid_weights",
]
