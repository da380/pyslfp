from pyslfp import FingerPrint, plot_SHGrid


# Set up the fingerprint solver.
finger_print = FingerPrint()

# Set the background sea level and ice thickness.
finger_print.set_background_state_from_ice_ng()


# Set the load.
zeta1 = finger_print.northern_hemisphere_load()
zeta2 = finger_print.southern_hemisphere_load()


# Compute the response.
response1 = finger_print.solver(zeta1)
response2 = finger_print.solver(zeta1, rotational_feedbacks=False)

# print(finger_print.integrate(response1.sl * zeta2))
# print(finger_print.integrate(response2.sl * zeta1))

# Plot the sea level change over the oceans.
plot_SHGrid((response1.sl-response2.sl) * finger_print.ocean_mask(),
            contour=True, levels=60, colorbar=True)
