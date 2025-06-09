# Import necessary modules for this notebook.
import numpy as np
import matplotlib.pyplot as plt
from pyslfp import FingerPrint

# Set up the fingerprint instance.
fingerprint = FingerPrint(lmax=128)
fingerprint.set_state_from_ice_ng()


# Set the direct load
direct_load = fingerprint.southern_hemisphere_load()

# Solve the forward problem
_, _, gravity_potential_change, angular_velocity_change = fingerprint(
    direct_load=direct_load, rtol=1e-9
)

# Get the gravitational potential change.
gravitational_potential_change = (
    fingerprint.gravity_potential_change_to_gravitational_potential_change(
        gravity_potential_change, angular_velocity_change
    )
)

# set the observation degree and order
l = 2
m = 2


# Print the coefficient.
observed_coefficient = fingerprint.coefficient_evaluation(
    gravitational_potential_change, l, m
)
print(f"The degree {l} and order {m} coefficient = {observed_coefficient}")

# Solve the adjoint problem
_, _, adjoint_gravitational_potential_load, adjoint_angular_momentum_change = (
    fingerprint.adjoint_loads_for_gravitational_potential_coefficient(l, m)
)
adjoint_sea_level_change, _, _, _ = fingerprint(
    gravitational_potential_load=adjoint_gravitational_potential_load,
    angular_momentum_change=adjoint_angular_momentum_change,
)

predicted_coefficient = fingerprint.integrate(adjoint_sea_level_change * direct_load)
print(f"Integral of direct load against adjoint sea level {predicted_coefficient}")

print(
    f"Relative difference = {np.abs(observed_coefficient-predicted_coefficient)/ np.abs(observed_coefficient)}"
)
