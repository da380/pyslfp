# PySLFP: Python Sea Level Fingerprints

[![PyPI version](https://badge.fury.io/py/pyslfp.svg)](https://badge.fury.io/py/pyslfp)
[![CI](https://github.com/da380/pyslfp/actions/workflows/ci.yml/badge.svg)](https://github.com/da380/pyslfp/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/pyslfp/badge/?version=latest)](https://pyslfp.readthedocs.io/en/latest/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

`pyslfp` computes elastic sea level fingerprints: the spatially variable pattern of
sea level change produced when mass is redistributed at the Earth's surface, for
example by the melting of an ice sheet. It solves the sea level equation, taking
account of the elastic deformation of the solid Earth, gravitational self-consistency
between ice, oceans and solid Earth, and rotational feedbacks.

The library covers both the forward problem and its use within inverse problems.
Alongside the solvers, it provides the same physics expressed as linear operators
between Hilbert spaces, together with observation models for tide gauges, satellite
altimetry and GRACE gravimetry. These build on
[`pygeoinf`](https://github.com/da380/pygeoinf) and follow the theory set out in
[Al-Attar et al. (2024)](https://academic.oup.com/gji/article/236/1/362/7338265).

Documentation is at [pyslfp.readthedocs.io](https://pyslfp.readthedocs.io).

## Installation

`pyslfp` requires Python 3.12 or later and is available from PyPI:

```bash
pip install pyslfp
```

Plotting works out of the box. `plt.show()` needs matplotlib to have an
interactive backend, which on a Python built with tkinter — the usual case — it
already has. Where tkinter is absent, or if you would rather use Qt:

```bash
pip install "pyslfp[interactive]"
```

For development, clone the repository and use Poetry:

```bash
poetry install              # runtime dependencies only
poetry install --with dev   # adds pytest, sphinx, ruff, jupyter and the hooks
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the git hooks, the documentation
build and the release process.

## Performance and threading

The spherical harmonic transforms dominate the cost of a sea level calculation.
They are performed by `pyshtools`, which uses the multi-threaded `ducc0` backend
when that package is installed; `ducc0` is a declared dependency, so this is the
default. The number of threads is read from the `OMP_NUM_THREADS` environment
variable when Python starts (all cores if it is unset).

For a single interactive calculation the default is what you want. When running
many independent solves in worker processes, for example through the `parallel`
options of the `pygeoinf` operators, set `OMP_NUM_THREADS=1` before starting
Python so that each worker runs single-threaded and the cores are shared between
workers rather than oversubscribed. The main process can still use several cores for
its serial phases by calling `pyslfp.set_num_threads(n)` after start-up; workers
started afterwards are unaffected, since they read the environment when they start.

## Data

The package needs a number of external datasets: a precomputed table of load Love
numbers, the ICE-NG ice histories, and shapefiles for the various regional
definitions. These are not
distributed with the package. They are downloaded from
[Zenodo](https://zenodo.org/records/22891291) automatically, on first use, and then
cached locally, so the first call that needs a given dataset will pause while it is
fetched and a progress bar is shown. Subsequent calls read from the cache.

By default the cache lives in `~/.pyslfp_data`. This can be changed by setting the
`PYSLFP_DATA` environment variable, which is useful on shared machines and in CI.
Datasets are fetched individually, so only what is actually used gets downloaded.
A dataset that has changed on Zenodo is not fetched again by itself, since the cache
is only checked for presence; `pyslfp.data.ensure_data("LOVE_NUMBERS", refresh=True)`
deletes the cached copy and downloads it afresh, and `LoveNumbers.default(refresh=True)`
does the same for the Love number table.

## Love numbers

The solid Earth enters the sea level equation through its elastic Love numbers. By
default `EarthModel` uses a precomputed table for PREM, downloaded as above, which
the package's own solver produced. It can compute them for any spherically layered model that
[planetmodel](https://github.com/da380/planetmodel) describes, by solving the
loading and tidal problem degree by degree on a radial spectral-element mesh:

```python
from planetmodel import PREM
from pyslfp import EarthModel, LoveNumbers

love = LoveNumbers.from_model(PREM(ocean=False), 256)   # a few seconds
love.conventional()["h"]                                 # h' by degree
love.write("prem_256.dat")                               # a file for later
model = EarthModel(256, love_numbers=love)               # or love_numbers="prem_256.dat"
```

`EarthModel.from_planet_model(PREM(ocean=False), 256)` does the same in one call.
The table carries the radius, surface gravity and gravitational constant of the
body it was computed for, and `EarthModel` takes those from the table so that the
sea level equation and its adjoint stay consistent. The numbers are the
generalised Love numbers of Al-Attar et al. (2024): the response to the traction
and the attraction of a load separately, to a tangential traction, and to a tidal
potential, with the tangential displacement numbers alongside the vertical ones.
Degree 0 also carries five axial numbers: the response to the spherical mean of
the centrifugal potential of a change in spin rate, which goes as $r^2$ rather
than being a constant, and the inertia moments of the degree-0 responses, which
the axial component of the rotational feedback needs and no surface Love number
gives. A model frozen at a frequency with `planetmodel.frozen` gives complex,
viscoelastic numbers, which can be computed and plotted but not yet used in the
sea level solver. The precomputed table is exactly what `LoveNumbers.from_model`
gives for `PREM(ocean=False)` to degree 4096.

### Definitions and conventions

The numbers the library holds are not the usual dimensionless ones, so their
definition is worth setting down. The solver works with the physical
gravitational potential, which is negative near added mass, and with
dimensional numbers: a displacement or a potential per unit of the forcing
that produced it. A surface load of density $\sigma_{lm} Y_{lm}$ acts in two
ways. It presses on the surface with the traction $-g\sigma_{lm} Y_{lm}$, and
it attracts the body as a surface mass in Poisson's equation. The generalised
Love numbers are the surface response to each acting alone. With the
displacement written as $\mathbf{u} = U Y_{lm}\hat{\mathbf{r}} + V \nabla_1 Y_{lm}$
and the potential perturbation as $\phi Y_{lm}$, both at $r = a$,

$$
U = h_l^u \zeta^u_{lm} + h_l^\phi \zeta^\phi_{lm}, \qquad
V = l_l^u \zeta^u_{lm} + l_l^\phi \zeta^\phi_{lm}, \qquad
\phi = k_l^u \zeta^u_{lm} + k_l^\phi \zeta^\phi_{lm},
$$

where $\zeta^u$ is a surface density that presses but does not attract and
$\zeta^\phi$ one that attracts but does not press. A true load does both, so
its numbers are the sums

$$
h_l = h_l^u + h_l^\phi, \qquad l_l = l_l^u + l_l^\phi, \qquad k_l = k_l^u + k_l^\phi,
$$

which are the properties `h`, `l` and `k`. In SI, $h$ and $l$ are in
m³ kg⁻¹ and $k$ in m⁴ kg⁻¹ s⁻². A third channel, the tangential traction
$-g\zeta^v_{lm} \nabla_1 Y_{lm}$, has the numbers $h^v_l$, $l^v_l$ and $k^v_l$,
and is what the adjoint problem for a functional of horizontal displacement
needs. The tidal numbers $h^t_l$, $l^t_l$ and $k^t_l$ are the response to the
unit external potential $\psi = (r/a)^l Y_{lm}$, so $h^t$ and $l^t$ are in
s² m⁻¹ and $k^t$ is dimensionless.

The conventional dimensionless load numbers $h'_l$, $l'_l$ and $k'_l$ of
Farrell (1972) refer the response to the direct potential of the load,
$4\pi G a\sigma_{lm}/(2l+1)$ in the geodetic sign convention, where the
potential is positive near mass. They follow from the numbers above by

$$
h'_l = \frac{(2l+1)\, g}{4\pi G a}\, h_l, \qquad
l'_l = \frac{(2l+1)\, g}{4\pi G a}\, l_l, \qquad
k'_l = -\frac{2l+1}{4\pi G a}\, k_l - 1,
$$

and the geodetic tidal numbers differ from the library's only by the sign of
the potential and a factor of gravity:

$$
k^T_l = k^t_l, \qquad h^T_l = -g\, h^t_l, \qquad l^T_l = -g\, l^t_l .
$$

`LoveNumbers.conventional()` and `LoveNumbers.tidal()` return these. The
problem is self-adjoint, which gives the reciprocity relations

$$
g\, h^\phi_l = k^u_l, \qquad h^v_l = l(l+1)\, l^u_l, \qquad k^v_l = g\, l(l+1)\, l^\phi_l,
$$

the first being eq. (64) of Al-Attar et al. (2024);
`LoveNumbers.reciprocity_residual()` checks all three. Degree 1 is in the
centre-of-mass frame, where the surface potential perturbation vanishes and
$k'_1 = -1$. At degree 0 the tidal numbers are zero, a uniform external
potential being a gauge, while the load numbers are not: mass conservation
fixes $k_0 = -4\pi G a$. The sea level solver uses the generalised numbers
directly, because the adjoint theory is written in them rather than in $h$
and $k$ alone.

## A first calculation

The following melts ten percent of the West Antarctic Ice Sheet and plots the
resulting sea level fingerprint. It is a condensed form of
[Tutorial 1](tutorials/tutorial1.ipynb).

```python
import matplotlib.pyplot as plt
import pyslfp as sl

# PREM Earth model with present-day ICE-7G as the background state.
sle = sl.LinearSeaLevelEquation.from_defaults(lmax=256)

# The load associated with a 10% loss of West Antarctic ice.
direct_load = sle.state.west_antarctic_load(fraction=0.1)

# Sea level change, vertical displacement, potential change, and the angular
# velocity change (polar wander and length of day).
sea_level_change, displacement, potential_change, angular_velocity_change = (
    sle.solve_sea_level_equation(direct_load)
)

# Plot the sea level change in metres, masked to the oceans.
length_scale = sle.state.model.parameters.length_scale
fig, ax = sl.create_map_figure(figsize=(12, 6))
sl.plot(
    sea_level_change * sle.state.ocean_projection() * length_scale,
    ax=ax,
    colorbar_kwargs={"label": "Sea level change (m)"},
)
plt.show()
```

`LinearSeaLevelEquation` holds the shoreline fixed, which is the usual assumption for
present-day and near-future problems. `SeaLevelEquation` provides the same linear
solver along with `solve_nonlinear_equation`, which migrates the shoreline and returns
an updated `EarthState`, and `solve_generalised_equation`, which accepts displacement,
potential and angular momentum forcings as needed in adjoint calculations.

### Units

Calculations are carried out in non-dimensional form. By default lengths, densities
and times are scaled so that the Earth's radius, mean density and surface gravity are
all equal to one. Results are returned in these units, and are converted back by
multiplying by the appropriate scale from `state.model.parameters` — `length_scale`
for sea level and displacement, `load_scale` for surface loads, and so on. The scheme
itself is set by `EarthModelParameters`, and can be replaced if a different one suits
the problem better.

## Operators and inverse problems

The same physics is also exposed as a `pygeoinf` `LinearOperator`, so that fingerprints
can be composed with observation operators, adjointed, and used within Bayesian
inversions. `FingerPrintOperator` maps a surface load to the four-component response
(sea level change, vertical displacement, potential change, angular velocity change),
and its domain and codomain may be either Lebesgue or Sobolev spaces, the latter
providing regularisation.

```python
import numpy as np
import pyslfp as sl
from pyslfp.linear_operators import FingerPrintOperator, ocean_average_operator

fingerprint = FingerPrintOperator.from_defaults(lmax=256)
response_space = fingerprint.codomain

# Compose the fingerprint with the ocean average of its sea level component.
sea_level = response_space.subspace_projection(0)
average = ocean_average_operator(fingerprint.state, response_space.subspace(0))
forward = average @ sea_level @ fingerprint

# The mean sea level change due to a given load.
datum = forward(fingerprint.state.greenland_load(fraction=0.1))

# The sensitivity kernel for that datum, obtained from the adjoint.
kernel = forward.adjoint(np.array([1.0]))
```

Built on this are the observation models in `pyslfp.linear_operators`, each pairing a
forward operator with the machinery needed to pose an inversion:

- `TideGaugeObservationModel`, using the GLOSS station network.
- `AltimetryObservationModel` and `JointAltimetryObservationModel`, for sea surface
  height over the oceans and over the ice sheets.
- `GraceObservationModel`, mapping loads to spherical harmonic coefficients of the
  potential change, with `WMBMethod` providing the purely spectral Wahr, Molenaar and
  Bryan (1998) approximation for comparison.

## Package layout

| Module | Contents |
| :--- | :--- |
| `core.py` | `EarthModelParameters` and `EarthModel`: physical constants, the non-dimensionalisation scheme (built on planetmodel's `Scales`), and the spherical harmonic discretisation. |
| `love_numbers/` | `LoveNumbers`, the generalised load and tidal Love numbers with their file and Green's functions, and the solver that computes them from any `planetmodel` model on a radial spectral-element mesh. |
| `state.py` | `EarthState`, a snapshot of ice thickness and sea level, providing the ocean function, surface integration, regional projections and ready-made loads. |
| `physics.py` | `SeaLevelEquation` and `LinearSeaLevelEquation`: the iterative solvers for the linear, non-linear and generalised forms of the equation. |
| `linear_operators/` | The physics as linear operators on Hilbert spaces, plus spatial projection and averaging operators, load mappings, and the tide gauge, altimetry and GRACE observation models. |
| `ice/` | `IceNG` for the ICE-5G, ICE-6G and ICE-7G histories, and `AnalyticalIceModel` for smooth synthetic states useful in testing. |
| `regions.py` | Regional masks and boundary plotting for the IMBIE Antarctic basins, Mouginot Greenland basins, IHO seas, HydroBASINS catchments, AR6 regions and Natural Earth oceans. |
| `plot.py` | Plotting of `pyshtools.SHGrid` fields on Cartopy projections. |
| `data/` | Location and automatic retrieval of the external datasets. |

## Tutorials

The tutorials are scripts under `tutorials/scripts`, written in cells (`# %%`) so
that they run from the command line or cell by cell in an editor:

| Script | Contents |
| :--- | :--- |
| `01_first_fingerprint.py` | Everything at its default: the fingerprint of a West Antarctic melt. |
| `02_physics.py` | The scales, the Earth model, a state from an analytic ice model, a load of one's own, the linear and non-linear solutions. |
| `03_love_numbers.py` | The shipped table, computing Love numbers from PREM, the identities, the radial solutions, files, and the complex numbers of a viscoelastic body. |
| `04_operators.py` | The fingerprint as an operator, composition, adjoints and sensitivity kernels, polar wander, Sobolev spaces and tide gauges. |
| `05_inversion.py` | A Bayesian inversion of synthetic GLOSS tide gauge data for ice thickness change. |

They read the real datasets, so the first run downloads them. Earlier versions of
the first, second, fourth and fifth exist as notebooks, which can be run locally or
in Google Colab.

| Tutorial | Colab |
| :--- | :--- |
| 1 — A first sea level fingerprint | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial1.ipynb) |
| 2 — A closer look at the physics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial2.ipynb) |
| 3 — Operators, composition and sensitivity kernels | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial3.ipynb) |
| 4 — A Bayesian inversion of tide gauge data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial4.ipynb) |

## Tests

```bash
poetry run pytest             # the fast suite, which is the default
poetry run pytest -m slow     # only the slow tests
poetry run pytest -m ""       # everything
```

The slow tests are the ones that read the real datasets, so the first run of
them downloads several hundred megabytes.

## Dependencies

`pyslfp` is built on `numpy` and `scipy`, with `pyshtools` for spherical harmonic
transforms and grids, `pygeoinf` for the Hilbert space and inference machinery,
`planetmodel` for the Earth models and radial meshes the Love numbers are computed
on, `matplotlib` and `Cartopy` for plotting, and `regionmask` with `cf-xarray` for
the regional masks.

The only optional dependency is `pyqt6`, under the `interactive` extra described
above. Nothing in the library imports it; it exists so that matplotlib has a Qt
backend to fall back on.

## Citation

If you use `pyslfp` in published work, please cite:

- Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J.X. and Lloyd, A.J., 2024.
  *Reciprocity and sensitivity kernels for sea level fingerprints*. Geophysical
  Journal International, **236(1)**, pp.362–378.

- D.A. Heathcote, T.Holland, A.M. Mag, M.E. Tamisiea, S. Coulson, S. Dangendorf, A.J. Lloyd, A. Mashayek, J.X. Mitrovica, D. Al-Attar, 2026.
  *A scalable Bayesian framework for modern sea-level inference*. arXiv, 2608.22336, https://arxiv.org/abs/2608.22336.

The datasets that `pyslfp` downloads — the ice histories, load Love numbers, tide
gauge network and regional definitions — are the work of others and are redistributed
here only for convenience. If you use them, please cite their original sources, which
are recorded on the [Zenodo record](https://zenodo.org/records/22891291).

## License

BSD-3-Clause. See [LICENSE](LICENSE).
