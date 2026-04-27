import matplotlib.pyplot as plt
import pyslfp as sl


state = sl.EarthState.from_defaults(lmax=128)

regions = state.list_iho_seas()

print(regions)

_, ax = sl.create_map_figure(figsize=(16, 10))
ax.set_global()

state.plot_coastline(ax)

state.plot_boundaries(ax, ["Tasman Sea", "Gulf of Mexico", "Greenland Sea"])

plt.show()
