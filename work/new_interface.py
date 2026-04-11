import matplotlib.pyplot as plt
import pyslfp as sl

from pyslfp.core import EarthModel
from pyslfp.state import EarthState
from pyslfp.ice_ng import IceNG
from pyslfp.physics import SeaLevelEquation


lmax = 1028
model = EarthModel(lmax)
ice_model = IceNG()

ice_thickness, sea_level = ice_model.get_ice_thickness_and_sea_level(0, lmax)

ice_thickness /= model.parameters.length_scale
sea_level /= model.parameters.length_scale

state = EarthState(ice_thickness, sea_level, model, exclude_caspian=True)

sle = SeaLevelEquation(model)


ice_thickness_change = (
    -1 * state.east_antarctic_projection(value=0) * state.ice_thickness
)
load = state.direct_load_from_ice_thickness_change(ice_thickness_change)


# sea_level_change, _, _, _ = sle.solve_sea_level_equation(state, load, verbose=True)


new_state, sea_level_change, _, _, _ = sle.solve_nonlinear_equation(
    state, ice_thickness_change=ice_thickness_change, verbose=True
)

sl.plot(state.ocean_function)
sl.plot(new_state.ocean_function)

plt.show()
