import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import pyslfp as sl
import pygeoinf as inf

# ==========================================
# 1. Setup the space and generate points
# ==========================================
print("Initializing model space and generating random points...")
fp = sl.FingerPrint(
    lmax=64,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)

model_space = inf.symmetric_space.sphere.Sobolev(
    fp.lmax, 2.0, 0.1, radius=fp.mean_sea_floor_radius
)

n_points = 100000
grid_size_degrees = 10

print(f"Generating {n_points} random points...")
points = model_space.random_points(n_points)

# ==========================================
# 2. Partition the points using pyslfp
# ==========================================
# Call the function directly from your library
blocks = sl.partition_points_by_grid(points, grid_size_degrees)
print(f"Partitioned into {len(blocks)} blocks.")

# ==========================================
# 3. Plot the results (Checkerboard Style)
# ==========================================
print("Plotting...")
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})

ax.set_global()
ax.coastlines(linewidth=0.5, zorder=1)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5, zorder=1)
ax.gridlines(linestyle="--", draw_labels=True, alpha=0.5, zorder=0)

# Two shades of a single color (dark blue and light blue)
shades = ["#08519c", "#9ecae1"]

for block_indices in blocks:
    if not block_indices:
        continue

    # Reverse-engineer the grid cell from the first point in the block
    first_pt_idx = block_indices[0]
    lat, lon = points[first_pt_idx]

    # Reconstruct the grid cell indices to determine the checkerboard color
    lat_idx = int(lat // grid_size_degrees)
    lon_idx = int((lon % 360.0) // grid_size_degrees)

    # Calculate checkerboard shade (alternating 0 and 1)
    shade_index = (lat_idx + lon_idx) % 2

    # Extract all points for plotting
    block_lats = [points[i][0] for i in block_indices]
    block_lons = [points[i][1] for i in block_indices]

    ax.scatter(
        block_lons,
        block_lats,
        color=shades[shade_index],
        transform=ccrs.PlateCarree(),
        s=15,
        edgecolor="black",
        linewidth=0.3,
        zorder=5,
    )

plt.title(
    f"{n_points} Points Partitioned into {grid_size_degrees}° Checkerboard Blocks"
)
plt.tight_layout()
plt.show()
