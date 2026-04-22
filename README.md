# PySLFP: Python Sea Level Fingerprints 

[![PyPI version](https://badge.fury.io/py/pyslfp.svg)](https://badge.fury.io/py/pyslfp)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://img.shields.io/travis/com/your-username/pyslfp.svg)](https://travis-ci.com/your-username/pyslfp)

`pyslfp` is a Python package for computing elastic sea level "fingerprints". It provides a robust and user-friendly framework for solving the sea level equation, accounting for the Earth's elastic deformation, gravitational self-consistency between the ice, oceans, and solid Earth, and rotational feedback effects.

The core of the library is built around the `EarthState` and `FingerPrintOperator` classes, which implement an iterative solver to determine the unique pattern of sea-level change that results from a change in a surface load, such as the melting of an ice sheet.

---

## Key Features 

* **Elastic Sea Level Equation Solver:** Implements an iterative solver for the sea level equation and the generalised sea level equation needed within adjoint calculations.
* **Comprehensive Physics:** Accounts for Earth's elastic response (via load Love numbers), self-consistent gravity, and rotational feedbacks (polar wander).
* **Ice History Models:** Includes a data loader for the ICE-5G, ICE-6G, and ICE-7G global ice history models, allowing for easy setup of realistic background states.
* **Forward and Adjoint Modeling:** Provides a high-level interface for both forward calculations (predicting sea level change from a load) and adjoint modeling (for use in inverse problems), natively integrated with `pygeoinf` and based on the theory of [Al-Attar et al. (2024)](https://academic.oup.com/gji/article/236/1/362/7338265).
* **Built-in Visualization:** Comes with high-quality map plotting utilities built on `matplotlib` and `cartopy` for easy visualization of global data grids and regional boundaries (e.g., IHO Seas, AR6, IMBIE).

---

## Installation

You can install `pyslfp` directly from PyPI using pip. The package requires Python 3.11+ and its dependencies will be installed automatically.

```bash
pip install pyslfp
```

### Installation with Poetry 

Alternatively, for development purposes, you can install `pyslfp` using Poetry. First, clone the repository and then run:

```bash 
poetry install 
```

To include the development dependencies (for running tests, building documentation, etc.), use the `--with dev` flag:

```bash
poetry install --with dev
```

---

## Tutorials

You can run the interactive tutorials directly in Google Colab to get started with the core concepts of the library.

| Tutorial Name | Link to Colab |
| :--- | :--- |
| Tutorial 1 - Calculating a Basic Sea Level Fingerprint | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial1.ipynb) |
| Tutorial 2 - A Deeper Dive into the Sea Level Equation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial2.ipynb) |

---


---

## Core Components

The library is organized into several key modules designed around abstract Hilbert spaces and linear operators:

* `state.py`: Contains the `EarthState` and `EarthModel` classes for managing planetary parameters, resolutions, and static background topologies.
* `linear_operators.py`: Contains the `FingerPrintOperator`, `WMBMethod`, and associated mappings required to formulate sea level and gravimetry equations as rigorous linear inverse problems.
* `regions.py`: Provides the `Regions` mixin for automated spatial masking and boundary plotting (including IHO Seas, IMBIE, HydroBASINS, and AR6).
* `ice_ng.py`: Provides the `IceNG` class for loading and interpolating global ice history models.
* `plotting.py`: Includes plotting functions for visualizing `pyshtools.SHGrid` objects natively on Cartopy projections.

---

## Dependencies

`pyslfp` is built on top of a robust stack of scientific Python packages:

* **numpy & scipy**: For numerical operations.
* **pyshtools**: For spherical harmonic transforms and grid representations.
* **pygeoinf**: For formulating and solving associated Bayesian inverse problems.
* **Cartopy & matplotlib**: For creating high-quality map projections and plots.
* **regionmask & geopandas**: For working with geospatial vector masks and boundaries.

---

## License

This project is licensed under the BSD-3-Clause License.

--- 

## Citations

If you use `pyslfp` in your published work, please cite the following paper:

* Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J.X. and Lloyd, A.J., 2024. *Reciprocity and sensitivity kernels for sea level fingerprints*. Geophysical Journal International, **236(1)**, pp.362-378.
  
Furthermore, if you use the ice models contained in the `IceNG` class, please cite the appropriate ice history model:

* [Peltier Group Data Sets](https://www.atmosp.physics.utoronto.ca/~peltier/data.php)

---

## Contributing

Contributions are welcome! If you have a suggestion or find a bug, please open an issue. Pull requests are also encouraged.