from pyslfp import Solver, plot_SHGrid



solver = Solver(256)

solver.set_background_state_from_ice_ng()

solver.rotational_feedbacks = False

zeta = solver.ice_density * solver.ice_thickness * solver.northern_hemisphere_mask(value=0)

response1 = solver(zeta, verbose=True)


plot_SHGrid(response1.sl * solver.ocean_mask())













