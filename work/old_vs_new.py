import numpy as np
import matplotlib.pyplot as plt

import pyslfp as sl
from pyslfp.core import EarthModel, EarthModelParameters
from pyslfp.state import EarthState
from pyslfp.ice_ng import IceNG
from pyslfp.physics import LinearSeaLevelEquation

LMAX = 128

# ---------------------------------------------------------
# Old Solver Setup
# ---------------------------------------------------------
fp = sl.FingerPrint(lmax=LMAX)
fp.set_state_from_ice_ng()

direct_load = fp.greenland_load()

sea_level_change1, displ1, pot1, omega1 = fp(
    direct_load=direct_load, rotational_feedbacks=True, rtol=1e-9, verbose=True
)

# Strip centrifugal potential from the old solver to match the new solver's definition
pot1_gravitational = fp.gravity_potential_change_to_gravitational_potential_change(
    pot1, omega1
)


# ---------------------------------------------------------
# New Solver Setup
# ---------------------------------------------------------
parameters = EarthModelParameters()
earth_model = EarthModel(LMAX, parameters=parameters)
ice_model = IceNG(length_scale=parameters.length_scale)
ice_thickness, sea_level = ice_model.get_ice_thickness_and_sea_level(0, LMAX)
state = EarthState(ice_thickness, sea_level, earth_model)

sle = LinearSeaLevelEquation(state)

sea_level_change2, displ2, pot2, omega2 = sle.solve_sea_level_equation(
    direct_load, rotational_feedbacks=True, rtol=1e-9, verbose=True
)

# ---------------------------------------------------------
# Visual Comparison
# ---------------------------------------------------------

# 1. Sea Level Change
ax, _ = sl.plot(sea_level_change1, colorbar=True)
ax.set_title("Sea Level Change (Old)")

ax, _ = sl.plot(sea_level_change2, colorbar=True)
ax.set_title("Sea Level Change (New)")

ax, _ = sl.plot(sea_level_change2 - sea_level_change1, colorbar=True)
ax.set_title("Sea Level Change (Difference)")

# 2. Vertical Displacement
ax, _ = sl.plot(displ1, colorbar=True)
ax.set_title("Vertical Displacement (Old)")

ax, _ = sl.plot(displ2, colorbar=True)
ax.set_title("Vertical Displacement (New)")

ax, _ = sl.plot(displ2 - displ1, colorbar=True)
ax.set_title("Vertical Displacement (Difference)")

# 3. Gravitational Potential Change
ax, _ = sl.plot(pot1_gravitational, colorbar=True)
ax.set_title("Gravitational Potential (Old)")

ax, _ = sl.plot(pot2, colorbar=True)
ax.set_title("Gravitational Potential (New)")

ax, _ = sl.plot(pot2 - pot1_gravitational, colorbar=True)
ax.set_title("Gravitational Potential (Difference)")

# 4. Angular Velocity Change
print("\n--- Angular Velocity Change Comparison ---")
print(f"Old Omega: {omega1}")
print(f"New Omega: {omega2}")
print(f"Difference : {omega2 - omega1}")
print(f"Max Abs Error: {np.max(np.abs(omega2 - omega1)):.4e}")

plt.show()
