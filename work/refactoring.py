import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyslfp as sl

state = sl.EarthState.from_defaults(lmax=128)

# Assuming 'state' is your initialized EarthState
fig, ax = sl.create_map_figure(figsize=(15, 10), projection=ccrs.PlateCarree())
ax.stock_img()

# List all Asian basins (starts with '4')
asian_basins = [b for b in state.list_hydrobasins() if b.startswith("4")]

# Plot their boundaries
state.plot_hydrobasin_boundaries(
    ax, region_ids=asian_basins, edgecolor="blue", linewidth=1.5
)

for basin_id in asian_basins:
    # 1. Map the string ID to its internal regionmask index
    idx = state.hydrobasins_regions.map_keys(basin_id)

    # 2. Extract the centroid (longitude, latitude)
    lon, lat = state.hydrobasins_regions.centroids[idx]

    # 3. Draw the text label
    ax.text(
        lon,
        lat,
        basin_id,
        transform=ccrs.PlateCarree(),
        ha="center",
        va="center",  # Center the text on the coordinate
        fontsize=8,
        fontweight="bold",
        color="black",
        bbox=dict(
            facecolor="white", alpha=0.7, edgecolor="none", pad=1.5
        ),  # Adds a readable backdrop
    )

plt.show()
