# PySLFP: Python Sea Level Fingerprints

[![PyPI version](https://badge.fury.io/py/pyslfp.svg)](https://badge.fury.io/py/pyslfp)
[![CI](https://github.com/da380/pyslfp/actions/workflows/ci.yml/badge.svg)](https://github.com/da380/pyslfp/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/pyslfp/badge/?version=latest)](https://pyslfp.readthedocs.io/en/latest/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

`pyslfp` computes how sea level changes when mass is redistributed at the
Earth's surface, on a spherically symmetric, self-gravitating, rotating Earth.
It solves the sea level equation with the deformation of the solid Earth,
gravitational self-consistency between ice, oceans and solid Earth, and
rotational feedbacks, for two descriptions of the solid Earth:

- **Elastic**, for sea level fingerprints: the instantaneous pattern of sea
  level change produced, for example, by the melting of an ice sheet.
- **Viscoelastic**, for glacial isostatic adjustment (GIA): sea level,
  deformation and rotation evolving through an ice history on an Earth with a
  mantle viscosity, with fixed or migrating shorelines.

Around these it provides:

- Love numbers computed for any spherically layered model described by
  [planetmodel](https://github.com/da380/planetmodel), including the generalised
  numbers the adjoint theory needs; a table for PREM is used by default.
- Sensitivity kernels by adjoint methods: for elastic fingerprints, and for a
  viscoelastic evolution with respect to the ice history, the initial sea
  level and the mantle viscosity.
- The elastic physics as linear operators between Hilbert spaces, with
  observation models for tide gauges, satellite altimetry and GRACE, for
  Bayesian inversions built on [`pygeoinf`](https://github.com/da380/pygeoinf).
- Ice histories (ICE-5G, ICE-6G and ICE-7G), regional masks, and plotting on
  map projections.

The theory follows [Al-Attar et al. (2024)](https://academic.oup.com/gji/article/236/1/362/7338265)
for fingerprints and their kernels, and Al-Attar & Tromp (2014) and Yu et al.
(2025) for viscoelastic loading and its adjoints. Documentation is at
[pyslfp.readthedocs.io](https://pyslfp.readthedocs.io).

## Installation

`pyslfp` requires Python 3.12 or later and is available from PyPI:

```bash
pip install pyslfp
```

Plotting works out of the box. `plt.show()` needs matplotlib to have an
interactive backend, which on a Python built with tkinter, the usual case, it
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

## A sea level fingerprint

The following melts ten per cent of the West Antarctic Ice Sheet and plots the
resulting sea level change. It is a condensed form of
[the first tutorial](tutorials/scripts/01_first_fingerprint.py).

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
metre = sle.state.model.parameters.length_scale
fig, ax = sl.create_map_figure(figsize=(12, 6))
sl.plot(
    sea_level_change * sle.state.ocean_projection() * metre,
    ax=ax,
    colorbar_kwargs={"label": "Sea level change (m)"},
)
plt.show()
```

`LinearSeaLevelEquation` holds the shoreline fixed, the usual assumption for
present-day and near-future problems. The engine beneath it,
`SeaLevelEquation`, also solves the nonlinear equation with migrating
shorelines, and the generalised equation that adjoint calculations need.

## A glacial isostatic adjustment run

The following runs the last deglaciation, ICE-7G from 26 ka to the present, on
PREM with a Maxwell mantle of viscosity 10²¹ Pa s beneath an elastic
lithosphere. It is a condensed form of
[the seventh tutorial](tutorials/scripts/07_ice7g_deglaciation.py), and takes
about half a minute.

```python
import numpy as np
from planetmodel import PREM, units

import pyslfp as sl
from pyslfp.ice import IceNG
from pyslfp.love_numbers import Material, graded_mesh
from pyslfp.viscoelastic import Rheology, SeaLevelEvolution, ViscoelasticOperator

lmax = 64

# PREM, made isotropic and converted into the library's units.
scales = sl.EarthModelParameters.from_defaults().scales
model = PREM(ocean=False).isotropic().converted(scales)
mesh = graded_mesh(model, lmax)

# A Maxwell mantle (PREM's layers 2 to 8), the top 77 km left elastic.
viscosity = 1.0e21 / scales.factor(units.VISCOSITY)
rheology = Rheology.maxwell(Material(mesh, model), {i: viscosity for i in range(2, 9)})
operator = ViscoelasticOperator(rheology, lmax)

# The background state that ICE-7G records at 26 ka.
earth_model = sl.EarthModel.from_planet_model(model, lmax, mesh=mesh)
ice_model = IceNG(length_scale=earth_model.parameters.length_scale)
state = sl.EarthState.from_ice_model(ice_model, 26.0, earth_model)

# Half-kiloyear steps to the present; each snapshot holds the sea level
# change, displacement, potential change and angular velocity change.
evolution = SeaLevelEvolution(operator, state)
snapshots, _ = evolution.run_ice_history(np.arange(26.0, -0.1, -0.5), ice_model)
```

`run_ice_history` takes `shorelines="migrating"` to let the shorelines move,
and `final_sea_level=` to choose the initial sea level so that the run reaches
a given present-day sea level, as an ice history without its own sea level
needs. `SeaLevelEvolution` computes sensitivity kernels of any objective built
from the snapshots, by `adjoint_ice_history` and
`continuous_adjoint_ice_history`.

## Units

Calculations are carried out in a single non-dimensional frame, planetmodel's
`Scales.geophysical` at the Earth's mean radius, in which the radius, the mean
density and the gravitational constant are one. Results are returned in these
units and converted to SI through the scales held by
`state.model.parameters`: `length_scale` for sea level and displacement,
`parameters.scales.factor(...)` for anything else. Ice histories are dated in
kiloyears before present. See the
[units page](https://pyslfp.readthedocs.io/en/latest/units.html).

## Documentation

- [Theory and methods](https://pyslfp.readthedocs.io/en/latest/theory.html):
  the sea level equation, Love numbers and their conventions, rotational
  feedback, viscoelastic loading and the adjoint kernels.
- [Numerics and performance](https://pyslfp.readthedocs.io/en/latest/numerics.html):
  discretisation, convergence, cost, and the use of threads and processes.
- [The API reference](https://pyslfp.readthedocs.io/en/latest/modules.html).
- The theory note, [`docs/theory/pyslfp_theory.pdf`](docs/theory/pyslfp_theory.pdf),
  with the full derivations.

## Tutorials

The tutorials are scripts under `tutorials/scripts`, written in cells (`# %%`) so
that they run from the command line or cell by cell in an editor. Each has a
notebook generated from it in `tutorials/notebooks`, which can be run locally or
in Google Colab. They read the real datasets, so the first run downloads them.

| Tutorial | Contents | Colab |
| :--- | :--- | :--- |
| `01_first_fingerprint` | Everything at its default: the fingerprint of a West Antarctic melt. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/01_first_fingerprint.ipynb) |
| `02_physics` | The scales, the Earth model, a state from an analytic ice model, a load of one's own, the linear and non-linear solutions. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/02_physics.ipynb) |
| `03_love_numbers` | The shipped table, computing Love numbers from PREM, the identities, the radial solutions, files, and the complex numbers of a viscoelastic body. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/03_love_numbers.ipynb) |
| `04_operators` | The fingerprint as an operator, composition, adjoints and sensitivity kernels, polar wander and length of day, Sobolev spaces and tide gauges. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/04_operators.ipynb) |
| `05_inversion` | A Bayesian inversion of synthetic GLOSS tide gauge data for ice thickness change. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/05_inversion.ipynb) |
| `06_forward_gia` | A first forward GIA run: a melting ice cap on a viscoelastic PREM, and postglacial rebound. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/06_forward_gia.ipynb) |
| `07_ice7g_deglaciation` | ICE-7G from 26 ka to the present: uplift rates, the sea level curve and a Hudson Bay emergence record. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/07_ice7g_deglaciation.ipynb) |
| `08_gia_kernels` | Ice-history sensitivity kernels by the discrete and continuous adjoints. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/08_gia_kernels.ipynb) |
| `09_decay_time_viscosity` | The decay time of postglacial emergence and its viscosity kernel. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/09_decay_time_viscosity.ipynb) |

## Data

The package needs a number of external datasets: a table of Love numbers for
PREM, the ICE-NG ice histories, the GLOSS tide gauge network, and shapefiles for
the regional definitions. They are not distributed with the package, but
downloaded from [Zenodo](https://zenodo.org/records/19494463) on first use and
cached locally, so the first call that needs a dataset pauses while it is
fetched. Datasets are fetched individually, so only what is used is downloaded.

The cache lives in `~/.pyslfp_data` unless the `PYSLFP_DATA` environment
variable says otherwise, which is useful on shared machines and in CI. The cache
is only checked for presence, so a dataset that has changed on Zenodo is not
fetched again by itself: `pyslfp.data.ensure_data("LOVE_NUMBERS", refresh=True)`
deletes the cached copy and downloads it afresh, and
`LoveNumbers.default(refresh=True)` does the same for the Love number table.

## Performance

The spherical harmonic transforms dominate the cost of a sea level
calculation; they use the multi-threaded `ducc0` backend, with the thread count
read from `OMP_NUM_THREADS` when Python starts (all cores if it is unset). When
running many independent solves in worker processes, set `OMP_NUM_THREADS=1`
before starting Python so that the workers share the cores rather than
oversubscribe them; `pyslfp.set_num_threads(n)` changes the count in the
current process. The
[numerics page](https://pyslfp.readthedocs.io/en/latest/numerics.html) gives
costs and the details.

## Package layout

| Module | Contents |
| :--- | :--- |
| `core/` | The shared engine: `EarthModelParameters` and `EarthModel` (physical constants, scales, the spherical harmonic discretisation and the Love numbers), `EarthState` (ice thickness and sea level, the ocean function, projections and ready-made loads), and `SeaLevelEquation`, the linear, generalised, adjoint and nonlinear solvers. |
| `elastic/` | `LinearSeaLevelEquation`, the fingerprint driver. |
| `viscoelastic/` | `Rheology`, `ViscoelasticOperator` (the radial time stepping), `SeaLevelEvolution` (the coupled evolution and its adjoints), the kernel and measurement types, decay-time functionals and `calibrate_initial_sea_level`. |
| `love_numbers/` | `LoveNumbers` and the radial spectral-element solver that computes them from any `planetmodel` model. |
| `linear_operators/` | The elastic physics as `pygeoinf` operators, spatial projections and averages, load mappings, and the tide gauge, altimetry and GRACE observation models. |
| `ice/` | The ice model interfaces, `IceNG` for the ICE-5G, ICE-6G and ICE-7G histories, `AnalyticalIceModel` for testing, and export to `planetmodel` surface histories. |
| `regions.py` | Regional masks: IMBIE Antarctic basins, Mouginot Greenland basins, IHO seas, HydroBASINS catchments, AR6 regions and Natural Earth oceans. |
| `plot.py` | Plotting of `pyshtools.SHGrid` fields on Cartopy projections. |
| `parallel.py` | Thread control for the transforms and BLAS. |
| `data/` | Location and retrieval of the external datasets. |

## Tests

```bash
poetry run pytest             # the fast suite, which is the default
poetry run pytest -m slow     # only the slow tests
poetry run pytest -m ""       # everything
```

The slow tests read the real datasets, so the first run of them downloads
several hundred megabytes, and include running every tutorial script.

## Dependencies

`pyslfp` is built on `numpy` and `scipy`, with `pyshtools` and `ducc0` for
spherical harmonic transforms and grids, `pygeoinf` for the Hilbert space and
inference machinery, `planetmodel` for the Earth models and radial meshes,
`matplotlib` and `Cartopy` for plotting, `regionmask` with `cf-xarray` for the
regional masks, and `threadpoolctl` for thread control.

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

The datasets that `pyslfp` downloads (the ice histories, Love numbers, tide
gauge network and regional definitions) are the work of others and are
redistributed here only for convenience. If you use them, please cite their
original sources, which are recorded on the
[Zenodo record](https://zenodo.org/records/19494463).

## License

BSD-3-Clause. See [LICENSE](LICENSE).
