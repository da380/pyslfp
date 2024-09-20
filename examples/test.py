from pyslfp import Solver, plot_SHGrid



# Set up the fingerprint solver. 
solver = Solver()

# Set the background sea level and ice thickness. 
solver.set_background_state_from_ice_ng()

# Set the load. 
zeta = -solver.ice_density * solver.one_minus_ocean_function \
       * solver.ice_thickness * solver.northern_hemisphere_mask(0)

# Compute the response. 
response = solver(zeta, verbose=True)

# Plot the sea level change over the oceans. 
plot_SHGrid(response.sl * solver.ocean_mask(), contour=True, levels=60, colorbar=True)













