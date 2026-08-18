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

For development, clone the repository and use Poetry:

```bash
poetry install              # runtime dependencies only
poetry install --with dev   # adds pytest, sphinx, ruff and jupyter
```

## Data

The package needs a number of external datasets: load Love numbers, the ICE-NG ice
histories, and shapefiles for the various regional definitions. These are not
distributed with the package. They are downloaded from
[Zenodo](https://zenodo.org/records/19555068) automatically, on first use, and then
cached locally, so the first call that needs a given dataset will pause while it is
fetched and a progress bar is shown. Subsequent calls read from the cache.

By default the cache lives in `~/.pyslfp_data`. This can be changed by setting the
`PYSLFP_DATA` environment variable, which is useful on shared machines and in CI.
Datasets are fetched individually, so only what is actually used gets downloaded.

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

# Sea level change, vertical displacement, potential change, and polar wander.
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
| `core.py` | `EarthModelParameters`, `LoveNumbers` and `EarthModel`: physical constants, the non-dimensionalisation scheme, load Love numbers, and the spherical harmonic discretisation. |
| `state.py` | `EarthState`, a snapshot of ice thickness and sea level, providing the ocean function, surface integration, regional projections and ready-made loads. |
| `physics.py` | `SeaLevelEquation` and `LinearSeaLevelEquation`: the iterative solvers for the linear, non-linear and generalised forms of the equation. |
| `linear_operators/` | The physics as linear operators on Hilbert spaces, plus spatial projection and averaging operators, load mappings, and the tide gauge, altimetry and GRACE observation models. |
| `ice/` | `IceNG` for the ICE-5G, ICE-6G and ICE-7G histories, and `AnalyticalIceModel` for smooth synthetic states useful in testing. |
| `regions.py` | Regional masks and boundary plotting for the IMBIE Antarctic basins, Mouginot Greenland basins, IHO seas, HydroBASINS catchments, AR6 regions and Natural Earth oceans. |
| `plot.py` | Plotting of `pyshtools.SHGrid` fields on Cartopy projections. |
| `data/` | Location and automatic retrieval of the external datasets. |

## Tutorials

The tutorials can be run locally or in Google Colab.

| Tutorial | Colab |
| :--- | :--- |
| 1 — A first sea level fingerprint | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial1.ipynb) |
| 2 — A closer look at the physics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial2.ipynb) |
| 3 — Operators, composition and sensitivity kernels | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial3.ipynb) |
| 4 — A Bayesian inversion of tide gauge data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial4.ipynb) |

## Tests

```bash
poetry run pytest             # the default suite
poetry run pytest -m slow     # the slower tests, excluded by default
```

## Dependencies

`pyslfp` is built on `numpy` and `scipy`, with `pyshtools` for spherical harmonic
transforms and grids, `pygeoinf` for the Hilbert space and inference machinery,
`matplotlib` and `Cartopy` for plotting, and `regionmask` with `cf-xarray` for the
regional masks.

## Citation

If you use `pyslfp` in published work, please cite:

- Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J.X. and Lloyd, A.J., 2024.
  *Reciprocity and sensitivity kernels for sea level fingerprints*. Geophysical
  Journal International, **236(1)**, pp.362–378.

The datasets that `pyslfp` downloads — the ice histories, load Love numbers, tide
gauge network and regional definitions — are the work of others and are redistributed
here only for convenience. If you use them, please cite their original sources, which
are recorded on the [Zenodo record](https://zenodo.org/records/19555068).

## License

BSD-3-Clause. See [LICENSE](LICENSE).
