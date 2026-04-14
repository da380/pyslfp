# Set up the parameters and non-dimensionalisation scheme
earth_parameters = sl.EarthModelParameters(
    length_scale=1000.0e3,  # One kilometre
    density_scale=1000.0e3,  # Approx density of water
    time_scale=3600.0,  # One hour
)

# Set the truncation degree for the calculations.
LMAX = 128

# Set up the earth model using default Love number file.
earth_model = sl.EarthModel(LMAX, parameters=earth_parameters)

# For the initial state, use the simple in-built analytic model.
ice_model = sl.ice.AnalyticalIceModel(length_scale=earth_parameters.length_scale)

# Visualise the ice thickness and the sea level
ice_thickness, sea_level = ice_model.get_ice_thickness_and_sea_level(0, LMAX)

# Note that we need to provide a projection when creating figures
#  as the plot functions is built on cartopy and not matplotlib
# directly. For figures with a single axes, the convenience method
# sl.create_map_figure is provided.
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    subplot_kw={"projection": ccrs.Mercator()},
    layout="constrained",
)

# We turn off the default inclusion of coastlines for the
# Earth.
sl.plot(
    ice_thickness * earth_parameters.length_scale,
    ax=ax1,
    coasts=False,
    colorbar_kwargs={"label": "Ice thickness (m)"},
)

sl.plot(
    sea_level * earth_parameters.length_scale,
    ax=ax2,
    coasts=False,
    symmetric=True,
    colorbar_kwargs={"label": "Sea level (m)"},
)


# We can manually add in the coastlines.
lons = earth_model.lons()
lats = earth_model.lats()
coast_function = (
    earth_parameters.water_density * sea_level
    - earth_parameters.ice_density * ice_thickness
)
ax1.contour(
    lons,
    lats,
    sea_level.data,
    levels=[0],
    colors="black",
    linewidths=1.5,
    transform=ccrs.PlateCarree(),
    zorder=10,
)

ax2.contour(
    lons,
    lats,
    sea_level.data,
    levels=[0],
    colors="black",
    linewidths=1.5,
    transform=ccrs.PlateCarree(),
    zorder=10,
)


plt.show()
