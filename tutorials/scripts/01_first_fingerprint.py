# %% [markdown]
# # 1. A first sea level fingerprint
#
# When an ice sheet loses mass, sea level does not rise uniformly. The
# lost ice no longer attracts the ocean towards it, the solid Earth
# rebounds under the lighter load, and the rotation of the Earth adjusts;
# the pattern of sea level change that results is the fingerprint of the
# melt. This script computes one, for ten per cent of the West Antarctic
# Ice Sheet, with everything at its default: PREM as the Earth model,
# present-day ICE-7G as the background state, and a truncation at degree
# 256. The datasets are downloaded on first use.

# %%
import matplotlib.pyplot as plt

import pyslfp as sl

# %% [markdown]
# ## The solver and the load
#
# `LinearSeaLevelEquation.from_defaults` builds the Earth model, the
# background state and the solver together. The state knows the ice
# basins, so a load is one call: the direct surface mass load of melting
# a fraction of the West Antarctic Ice Sheet.

# %%
sle = sl.LinearSeaLevelEquation.from_defaults(lmax=256)
direct_load = sle.state.west_antarctic_load(fraction=0.1)

# %% [markdown]
# ## Solving the equation
#
# The solver returns the sea level change, the vertical displacement of
# the solid surface, the change in gravitational potential and the change
# in the Earth's angular velocity (two components for the shift of the
# pole and one for the length of day). Only the first is used here. All are
# non-dimensional, and the parameters of the Earth model turn them back
# into metres.

# %%
sea_level_change, displacement, potential_change, angular_velocity = (
    sle.solve_sea_level_equation(direct_load)
)
metre = sle.state.model.parameters.length_scale

print(
    f"sea level change from {sea_level_change.data.min() * metre:.3f} "
    f"to {sea_level_change.data.max() * metre:.3f} m"
)

# %% [markdown]
# ## The fingerprint
#
# Sea level change is defined everywhere, but only means something over
# the oceans, so it is shown through the ocean projection. Near West
# Antarctica sea level falls; far from it, it rises by more than the
# global mean.

# %%
fig, ax = sl.create_map_figure(figsize=(12, 8))
sl.plot(
    sea_level_change * sle.state.ocean_projection() * metre,
    ax=ax,
    colorbar_kwargs={"label": "Sea level change (m)"},
)
plt.show()
