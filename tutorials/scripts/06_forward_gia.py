# %% [markdown]
# # 6. A first forward GIA simulation
#
# An ice cap of Laurentide proportions, centred near Hudson Bay, melts
# steadily over ten thousand years on a viscoelastic earth: PREM
# collapsed to isotropy, a two-layer Maxwell mantle (5 x 10^20 Pa s
# above the 670 km discontinuity, 2 x 10^21 Pa s below) and an elastic
# lithosphere. The sea level equation is solved at every step, with
# rotational feedback, under the present-day shorelines (the linear,
# frozen-shoreline equation). Three figures result: the sea level
# change and the crustal uplift fifteen thousand years in, and the
# history of the uplift at the centre of the cap, which keeps rising
# long after the melting stops -- postglacial rebound.
#
# The run takes about half a minute at degree 64.

# %%
import matplotlib.pyplot as plt
import numpy as np
from planetmodel import units

import pyslfp as sl
from pyslfp.viscoelastic import (
    SeaLevelEvolution,
    ViscoelasticOperator,
    prem_maxwell,
    viscoelastic_earth,
)
from pyslfp.ice import IceNG

LMAX = 64

# %% [markdown]
# ## The earth model and its rheology
#
# The radial model is PREM without its ocean, collapsed to its Voigt
# isotropic average, since the viscoelastic solver relaxes an isotropic
# shear modulus, and without its quality factors. `prem_maxwell` gives
# it with a Maxwell mantle: PREM's lid, the top 77 km, left elastic as
# the lithosphere, 5 x 10^20 Pa s from there down to the 670 km
# discontinuity, and 2 x 10^21 Pa s below. The viscosity is a field of
# the model, in Pa s like the rest of it; `maxwell_mantle` places other
# structures on any radial model, by radii.
#
# The library computes in one non-dimensional frame, that of the
# default Earth model parameters. `viscoelastic_earth` converts the
# model into it, builds one mesh, and reads the model twice on that
# mesh: for its elastic Love numbers, in the Earth model, and for its
# viscosity, in the rheology.

# %%
earth_model, rheology = viscoelastic_earth(prem_maxwell(), LMAX)
operator = ViscoelasticOperator(rheology, LMAX)

# %% [markdown]
# ## The sea level side
#
# The present-day ICE-7G background state on that Earth model, whose
# ocean function the linear equation freezes.

# %%
ice_model = IceNG(length_scale=earth_model.parameters.length_scale)
state = sl.EarthState.from_ice_model(ice_model, 0.0, earth_model)

# %% [markdown]
# ## The ice history
#
# A cap of 1500 m of ice, ten degrees across, melts linearly over the
# first ten thousand years and is gone thereafter; the direct load is
# ice density times the lost thickness. The times of the evolution are
# in the Earth model's non-dimensional units, so a thousand years is
# converted once. The cap's edge is softened in degree space: a sharp
# disc rings at a finite truncation, and the taper leaves degree 0,
# hence the mass, untouched.

# %%
parameters = earth_model.parameters
metre = parameters.length_scale
kyr = 1.0e3 * 365.25 * 86400.0 / parameters.scales.factor(units.TIME)

cap = state.disk_load(10.0, 60.0, -85.0, parameters.ice_density * 1500.0 / metre)
taper = np.exp(-((np.arange(LMAX + 1) / 32.0) ** 4))
cap_lm = earth_model.expand_field(cap)
cap_lm.coeffs *= taper[None, :, None]
cap = earth_model.expand_coefficient(cap_lm)
melt_time = 10.0 * kyr


def ice_load(t):
    return cap * (-min(t / melt_time, 1.0))


times = np.linspace(0.0, 15.0 * kyr, 31)
evolution = SeaLevelEvolution(operator, state)
snapshots, _ = evolution.run(times, ice_load)

# %% [markdown]
# ## Sea level and uplift at the end of the run
#
# Sea level falls by hundreds of metres beside the vanished cap, which
# no longer attracts the ocean and whose coast is rising; the colour
# scales saturate there so that the far field stays visible: the
# ocean-wide rise beyond the global mean, and the ring of subsidence
# around the uplifting dome where the peripheral bulge collapses.

# %%
final = snapshots[-1]
ocean = state.ocean_projection()

fig, ax = sl.create_map_figure(figsize=(12, 8))
sl.plot(
    final.sea_level_change * ocean * metre,
    ax=ax,
    vmin=-20.0,
    vmax=20.0,
    colorbar_kwargs={"label": "Sea level change (m)"},
)
ax.set_title("Sea level change, 15 kyr after the onset of melting")

fig, ax = sl.create_map_figure(figsize=(12, 8))
sl.plot(
    final.displacement * metre,
    ax=ax,
    vmin=-100.0,
    vmax=100.0,
    colorbar_kwargs={"label": "Vertical displacement (m)"},
)
ax.set_title("Crustal displacement, 15 kyr after the onset of melting")

# %% [markdown]
# ## The history at the centre of the cap
#
# The melting drives uplift from the start -- an elastic part instantly,
# the viscous flow behind it -- and when the melting stops at 10 kyr the
# uplift carries on as the mantle keeps flowing back: the rebound that
# is still observed today around Hudson Bay and the Gulf of Bothnia.

# %%
lat_index = int(np.argmin(np.abs(earth_model.lats() - 60.0)))
lon_index = int(np.argmin(np.abs(earth_model.lons() - (360.0 - 85.0))))
uplift = metre * np.array(
    [snap.displacement.data[lat_index, lon_index] for snap in snapshots]
)
mean_sea_level = metre * np.array(
    [
        earth_model.integrate(snap.sea_level_change * state.ocean_function)
        / state.ocean_area
        for snap in snapshots
    ]
)

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
top.plot(times / kyr, uplift)
top.axvline(melt_time / kyr, color="grey", linestyle=":")
top.set_ylabel("Uplift at the cap centre (m)")
bottom.plot(times / kyr, mean_sea_level)
bottom.axvline(melt_time / kyr, color="grey", linestyle=":")
bottom.set_ylabel("Ocean-mean sea level change (m)")
bottom.set_xlabel("Time (kyr)")
top.set_title("Melting stops at the dotted line; the rebound does not")
fig.tight_layout()

print(
    f"uplift at the cap centre: {uplift[20]:.1f} m when the melting stops, "
    f"{uplift[-1]:.1f} m at 15 kyr"
)
print(f"ocean-mean sea level change at 15 kyr: {mean_sea_level[-1]:.2f} m")

plt.show()
