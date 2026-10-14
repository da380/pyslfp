"""What the elastic and viscoelastic sides share: the Earth model and
its parameters, the background state, and the sea level engine.

`SeaLevelEquation` solves the equilibrium sea level equation -- linear,
generalised or nonlinear -- for whatever Love numbers its Earth model
carries: the elastic table of a fingerprint, or the per-step effective
table of a viscoelastic evolution.  Nothing in the engine is elastic
but the table it is handed.
"""

from .model import EarthModel, EarthModelParameters
from .sea_level import SeaLevelEquation
from .state import EarthState

__all__ = [
    "EarthModel",
    "EarthModelParameters",
    "EarthState",
    "SeaLevelEquation",
]
