from pyslfp import IceModel, IceNG
import cartopy.crs as ccrs

"""
# --- Example: Animate ICE-6G from the Last Glacial Maximum ---

# Create the model object
ice6g_model = IceNG(version=IceModel.ICE6G)


# --- Example 1: Animate ICE-7G ice thickness (global) ---
# First, create the model object
ice7g_model = IceNG(version=IceModel.ICE7G)

# Now, call the animate method on the object
ice7g_model.animate(
    "animations/ice7g_thickness_global.mp4",
    field="ice_thickness",
    cmap="Blues",
    num_frames=100,  # Fewer frames for a quick test
    vmin=0,
)

# --- Example 2: Animate ICE-5G topography over Fennoscandia ---
ice5g_model = IceNG(version=IceModel.ICE5G)

ice5g_model.animate(
    "animations/ice5g_topo_fenno.mp4",
    field="topography",
    start_date_ka=20.0,
    num_frames=201,
    fps=20,
    cmap="terrain",
    symmetric=True,
    projection=ccrs.Orthographic(central_longitude=20, central_latitude=65),
    map_extent=[-10, 50, 50, 80],
)
"""

# Create the model object
ice6g_model = IceNG(version=IceModel.ICE6G)

# Call the new joint animation method
ice6g_model.animate_joint(
    "animations/ice6g_joint_LGM.mp4",
    start_date_ka=21.0,  # Start at the Last Glacial Maximum
    end_date_ka=0.0,
    num_frames=211,  # One frame every 100 years
    fps=20,
    lmax=180,
    projection=ccrs.Orthographic(central_longitude=-80, central_latitude=55),
    # Customize the ice plot
    ice_plot_kwargs={"cmap": "cividis", "vmax": 4500},  # Set max for the color scale
    # Customize the sea level plot
    sl_plot_kwargs={
        "cmap": "RdBu_r",
        "vmin": -150,  # Set min and max for a consistent color scale
        "vmax": 150,
    },
)
