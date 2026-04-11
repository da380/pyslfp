import matplotlib.pyplot as plt
import pyslfp as sl

from pyslfp.state import EarthState
from pyslfp.physics import SeaLevelEquation


lmax = 256


state = EarthState.default(lmax)

sle = SeaLevelEquation(state.model)


ice_thickness_change = (
    -1 * state.east_antarctic_projection(value=0) * state.ice_thickness
)
load = state.direct_load_from_ice_thickness_change(ice_thickness_change)


# sea_level_change, _, _, _ = sle.solve_sea_level_equation(state, load, verbose=True)


new_state, sea_level_change, _, _, _ = sle.solve_nonlinear_equation(
    state, ice_thickness_change=ice_thickness_change, verbose=True
)

sl.plot(sea_level_change)


plt.show()
