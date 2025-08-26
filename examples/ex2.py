import numpy as np
import matplotlib.pyplot as plt
from pyslfp import FingerPrint, plot, IceModel, EarthModelParameters
import pygeoinf as inf

# Import the cartopy coordinate reference system for plotting
import cartopy.crs as ccrs

from pyslfp.operators import tide_gauge_operator


fp = FingerPrint(
    lmax=128,
    earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

A = fp.as_sobolev_linear_operator(2, 0.5)

tide_gauge_locations = A.domain.random_points(100)


B = tide_gauge_operator(A.codomain, tide_gauge_locations)


u = B.domain.random()
v = B.codomain.random()

lhs = B.codomain.inner_product(v, B(u))
rhs = B.domain.inner_product(B.adjoint(v), u)

print(lhs, rhs, np.abs(lhs - rhs) / np.abs(rhs))


u = fp.northern_hemisphere_load()
v = A(u)

sea_level_field = v[0]
w = B(v)

# Capture the figure, axes, and image artist objects
sea_level_field = v[0]
fig, ax, im = plot(sea_level_field, cmap="viridis")

# Extract longitude and latitude for plotting
lats = [loc[0] for loc in tide_gauge_locations]
lons = [loc[1] for loc in tide_gauge_locations]

# Use ax.scatter to plot the tide gauge location
ax.scatter(
    lons,
    lats,
    c=w,  # Use the calculated value 'w' to set the color
    cmap=im.get_cmap(),  # Use the same colormap as the main plot
    norm=im.norm,  # Use the same normalization (value->color scale)
    marker="o",  # A circle shows color well
    s=40,  # A large size
    edgecolor="black",  # A black edge makes the point stand out
    linewidth=1.5,
    transform=ccrs.PlateCarree(),  # Specify the coordinate system
)

# Add a title and a colorbar for context
ax.set_title("Sea Level Change with Tide Gauge Measurement", y=1.05)
fig.colorbar(
    im,
    ax=ax,
    orientation="horizontal",
    label="Sea Level Change (m)",
    pad=0.05,
    shrink=0.7,
)

plt.show()
