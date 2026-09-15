# %% [markdown]
# # 3. Love numbers
#
# The solid Earth enters the sea level equation through its Love numbers:
# at each spherical harmonic degree, the vertical displacement, the
# tangential displacement and the change in gravitational potential at
# the surface per unit surface load, and the same per unit tidal
# potential. `pyslfp` ships a precomputed table for PREM and downloads it
# on first use. It can also compute the numbers itself, from any
# spherically layered model that `planetmodel` describes, by solving the
# loading and tidal problem degree by degree on a radial spectral-element
# mesh. This script goes through both, and ends with a look at the
# complex Love numbers of a viscoelastic body.
#
# The numbers here are the generalised Love numbers of Al-Attar et al.
# (2024): a surface load acts by pressing on the surface and by
# attracting the body, and the response to each is kept separately
# (`h_u`, `k_u` for the traction, `h_phi`, `k_phi` for the attraction),
# with `h = h_u + h_phi` the usual load number. The split is what the
# adjoint theory of the sea level equation is written in.

# %%
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from planetmodel import PREM, constant_field, frozen

import pyslfp as sl
from pyslfp.love_numbers import (
    plot_degree_solution,
    plot_love_numbers,
    solve_degree,
)

LMAX = 256

# %% [markdown]
# ## The shipped table
#
# `LoveNumbers.default` reads the precomputed PREM table, in SI units. It
# is the output of the solver below for PREM to degree 4096, and it
# records the body it was computed for: the radius of the solid
# surface, the surface gravity and the gravitational constant. Those
# three matter, because the sea level equation and its adjoint are only
# consistent when the Earth model uses the same values as its Love
# numbers, and `EarthModel` takes them from the table.

# %%
shipped = sl.LoveNumbers.default(lmax=LMAX)
print(shipped)
print(
    f"radius {shipped.radius:.0f} m, g {shipped.surface_gravity:.4f} m/s^2, "
    f"G {shipped.G:.5e}"
)

conv = shipped.conventional()
print("  l        h'          l'          k'")
for l in (1, 2, 3, 10, 100, 256):
    print(
        f"{l:4d}  {conv['h'][l]:+10.5f}  {conv['l'][l]:+10.5f}  {conv['k'][l]:+10.5f}"
    )

# %% [markdown]
# ## Computing the numbers from PREM
#
# A fluid surface cannot carry a load, so PREM is taken without its
# ocean. `LoveNumbers.from_model` builds a radial mesh, fine at the
# surface and widening with depth so that every degree is resolved where
# its solution lives, reads the model onto it once, and solves every
# degree for the four forcings on the part of the mesh where that degree
# is not negligible. Degrees 0 to 256 take about a second, and 0 to 4096
# half a minute. The result is in the model's units, SI here, and carries
# the body's radius, gravity and G, now from the model itself.
#
# Two identities hold to solver precision and are a check on every run.
# The reciprocity relation `g h_phi = k_u` links the displacement from
# the attraction of a load to the potential from its pressing, and its
# tangential counterparts do the same for the other channels. At degree
# zero the potential change from a uniform surface load is fixed by mass
# conservation, `k_0 = -4 pi G a`. Degree one is in the centre-of-mass
# frame, where `k'_1` is exactly minus one.

# %%
prem = PREM(ocean=False)
start = time.time()
love = sl.LoveNumbers.from_model(prem, LMAX)
print(f"degrees 0..{LMAX} in {time.time() - start:.1f} s:", love)
print(
    f"radius {love.radius:.0f} m, g {love.surface_gravity:.4f} m/s^2, "
    f"G {love.G:.5e}"
)

print("largest reciprocity residual:", love.reciprocity_residual().max())
print("k_0 / (-4 pi G a) =", love.k[0] / (-4.0 * np.pi * love.G * love.radius))
print("k'_1 =", love.conventional()["k"][1])

tidal = love.tidal()
print(
    f"tidal degree 2: k = {tidal['k'][2]:.4f}, h = {tidal['h'][2]:.4f}, "
    f"l = {tidal['l'][2]:.4f}"
)

# %% [markdown]
# The shipped table and the one just computed are the same numbers to
# round-off, the shipped one having been made the same way. The degree-0
# numbers are worth a look: a uniform load does deform the Earth, and
# the numbers say by how much, though no load that conserves mass, as
# every ice-and-ocean load does, has a degree-0 part.

# %%
print(
    "largest relative difference from the shipped table:",
    max(
        np.max(
            np.abs(getattr(love, n) - getattr(shipped, n))
            / np.max(np.abs(getattr(shipped, n)))
        )
        for n in ("h_u", "k_u", "h_phi", "k_phi", "l_u", "h_t", "k_t")
    ),
)
print("degree 0: h =", love.h[0], " k =", love.k[0])

# %% [markdown]
# ## The radial solution of one degree
#
# `solve_degree` solves one degree for one forcing and returns the three
# functions of radius: the radial displacement U, the tangential
# displacement V and the potential perturbation phi. Their values at the
# surface are the Love numbers of that degree. The solution is drawn
# below with radius upward and the model's boundaries marked. Through
# the outer core only phi is drawn: the quasi-static problem does not
# determine the displacement of a fluid, and the solver represents a
# fluid region by its potential alone, so U and V are undefined there
# and the lines break at the core-mantle and inner-core boundaries.

# %%
load2 = solve_degree(prem, 2)
tide2 = solve_degree(prem, 2, forcing="tide")
print("surface (U, V, phi) under a unit load:", load2.surface)
print("surface (U, V, phi) under a unit tide:", tide2.surface)

# %% [markdown]
# ## Files, and the Earth model
#
# `write` writes a plain-text file, one row per degree, with a header
# naming the columns and recording the body. `LoveNumbers.from_file`
# reads it back, and `EarthModel` takes either the table or the path.
# `EarthModel.from_planet_model` does the computation and the
# construction in one call. Whichever way, the Earth model's radius,
# surface gravity and G are the table's.

# %%
folder = Path(tempfile.mkdtemp())
path = folder / "prem_256.dat"
love.write(path)
print(path.read_text().splitlines()[1])
back = sl.LoveNumbers.from_file(path)
print("read back h_2:", back.h[2], "vs", love.h[2])

model = sl.EarthModel(LMAX, love_numbers=love)
same = sl.EarthModel(LMAX, love_numbers=path)
quick = sl.EarthModel.from_planet_model(prem, 64)
print("G in the Earth model's units:", model.parameters.gravitational_constant)
print(
    "the same model from the file:",
    np.allclose(same.love_numbers.h, model.love_numbers.h),
)
print("and computed on the fly to degree 64:", quick.love_numbers)

# %% [markdown]
# ## A viscoelastic body is a model frozen at a frequency
#
# The solver takes any model holding a density and the five elastic
# moduli, real or complex. A linear viscoelastic body forced at one
# angular frequency behaves like an elastic body with complex,
# frequency-dependent moduli, and `planetmodel.frozen` builds that body
# from the rheology each layer holds: `qmu` and `qkappa` for PREM's own
# attenuation, or a `viscosity` for a Maxwell solid. The Love numbers of
# a frozen model are complex. The sea level solver does not take them,
# but they can be computed, written down and plotted, and they are where
# a time-dependent solver would start.
#
# Below, PREM with its own Q at the semidiurnal tide, and then a Maxwell
# mantle of viscosity 1e21 Pa s across periods from minutes to hundreds
# of thousands of years: the degree-2 tidal k runs from the elastic value
# to the fluid limit, with the loss peaking in between.

# %%
semidiurnal = 2.0 * np.pi / 43200.0
attenuating = sl.LoveNumbers.from_model(frozen(prem, semidiurnal), 2)
print("PREM with its Q at 12 h:", attenuating)
print("k_2^T =", attenuating.tidal()["k"][2], "vs elastic", tidal["k"][2])

visco = prem
for layer in prem.layers:
    if layer.interval[0] >= 3480e3:
        visco = visco.with_field(
            layer.index,
            "viscosity",
            constant_field(layer.interval, 1e21, name="viscosity"),
        )
periods = np.logspace(2, 13, 40)
k2 = np.empty(periods.size, dtype=complex)
for i, period in enumerate(periods):
    k2[i] = sl.LoveNumbers.from_model(frozen(visco, 2.0 * np.pi / period), 2).tidal()[
        "k"
    ][2]
print(
    f"k_2^T at 12 h: {np.interp(43200.0, periods, k2.real):.4f}; "
    f"at 100 kyr: {np.interp(3.15e12, periods, k2.real):.4f}"
)

# %% [markdown]
# ## Figures

# %%
fig, axes = plot_love_numbers(love)
fig.suptitle("PREM, computed")

fig, ax = plot_degree_solution(load2)

fig, axes = love.plot_greens_functions()

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
years = periods / 3.15576e7
ax.semilogx(years, k2.real, label="Re $k_2^T$")
ax.semilogx(years, -k2.imag, label="$-$Im $k_2^T$")
ax.axhline(tidal["k"][2], color="0.6", lw=0.8, ls="--", label="elastic")
ax.set_xlabel("period (years)")
ax.set_title("Maxwell mantle, viscosity $10^{21}$ Pa s")
ax.legend()

plt.show()
