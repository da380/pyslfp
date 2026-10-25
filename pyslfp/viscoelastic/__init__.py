"""Glacial isostatic adjustment: viscoelastic loading of a spherically
symmetric earth coupled to the sea level equation, its sensitivity
kernels by discrete and continuous adjoints, and the functionals
built on them.

The formulation follows Al-Attar & Tromp (2014), internal variables
in the degree basis; the time-stepping algebra is shared with the
AdGIA library so the two can benchmark one another.  The kernel
drivers here are a computational layer, in the role the sea level
engine plays beneath `pyslfp.linear_operators` on the elastic side:
they return kernels as plain grids and nodal arrays, without a
Hilbert-space structure of their own.
"""

from .functionals import (
    DecayTimeFit,
    decay_time_measurement,
    fit_decay_time,
)
from .evolution import (
    ForwardTrajectory,
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
    "DecayTimeFit",
    "decay_time_measurement",
    "fit_decay_time",
    "ForwardTrajectory",
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
