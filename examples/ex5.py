import geopandas as gpd
import regionmask
import numpy as np
import matplotlib.pyplot as plt

# Define the path to your shapefile

# You may need to download this file first from naturalearthdata.com
# For example, 'ne_10m_lakes.shp'
shapefile_path = "path/to/your/ne_10m_lakes.shp"

# 1. Load the shapefile into a GeoDataFrame
lakes = gpd.read_file(shapefile_path)

# 2. Select only the Caspian Sea polygon by its name
caspian_sea_polygon = lakes[lakes["name"] == "Caspian Sea"]

# 3. Create a regionmask object from the GeoDataFrame
# We use the 'name' column from the shapefile for the region's name
caspian_sea_region = regionmask.from_geopandas(caspian_sea_polygon, names="name")

# Now you can use this region object to create a mask
lon = np.arange(45, 55, 0.1)
lat = np.arange(36, 48, 0.1)

mask = caspian_sea_region.mask(lon, lat)

# The 'mask' is now an xarray DataArray where the Caspian Sea is marked.
# You can plot it to verify:
mask.plot()
plt.show()
