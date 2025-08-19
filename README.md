# PySLFP: Python Sea Level Fingerprints 

[![PyPI version](https://badge.fury.io/py/pyslfp.svg)](https://badge.fury.io/py/pyslfp)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build Status](https://img.shields.io/travis/com/your-username/pyslfp.svg)](https://travis-ci.com/your-username/pyslfp)

`pyslfp` is a Python package for computing elastic sea level "fingerprints". It provides a robust and user-friendly framework for solving the sea level equation, accounting for the Earth's elastic deformation, gravitational self-consistency between the ice, oceans, and solid Earth, and rotational feedback effects.

The core of the library is the `FingerPrint` class, which implements an iterative solver to determine the unique pattern of sea-level change that results from a change in a surface load, such as the melting of an ice sheet.

---

## Key Features 

* **Elastic Sea Level Equation Solver:** Implements an iterative solver for the  sea level equation and the generalised sea level equation needed within adjoint calculations.
* **Comprehensive Physics:** Accounts for Earth's elastic response (via load Love numbers), self-consistent gravity, and rotational feedbacks (polar wander).
* **Ice History Models:** Includes a data loader for the ICE-5G, ICE-6G, and ICE-7G global ice history models, allowing for easy setup of realistic background states.
* **Forward and Adjoint Modeling:** Provides a high-level interface for both forward calculations (predicting sea level change from a load) and  adjoint modeling (for use in inverse problems), powered by `pygeoinf`, and based on the theory of [Al-Attar et al.(2024)](https://academic.oup.com/gji/article/236/1/362/7338265)
* **Built-in Visualization:** Comes with high-quality map plotting utilities built on `matplotlib` and `cartopy` for easy visualization of global data grids.


---

## Installation

You can install `pyslfp` directly from PyPI using pip. The package requires Python 3.11+ and its dependencies will be installed automatically.

```bash
pip install pyslfp
```

### Installation with Poetry 

Alternatively, for development purposes, you can install pyslfp using Poetry. First, clone the repository and then run:

```bash 
poetry install 
```

Optionally, tests and tutorials can be included using ```-with tests, tutorials```.

---

## Citation 

If you use `pyslfp` in your published work, please cite the following paper:

*   Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J.X. and Lloyd, A.J., 2024. Reciprocity and sensitivity kernels for sea level fingerprints. *Geophysical Journal International*, **236**(1), 362-378.
    
Additionally, please cite the appropriate ice history model if you use the `IceNG` class from

*   [Peltier group data sets](https://www.atmosp.physics.utoronto.ca/~peltier/data.php)

---

## Quick Start

Here's a simple example of how to compute and plot the sea level fingerprint for the melting of 10% of the Northern Hemisphere's ice sheets.

```python
import matplotlib.pyplot as plt
from pyslfp import FingerPrint, plot, IceModel

# 1. Initialise the fingerprint model
# lmax sets the spherical harmonic resolution.
fp = FingerPrint(lmax=256)

# 2. Set the background state (ice and sea level) to the present day
# This uses the built-in ICE-7G model loader.
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

# 3. Define a surface mass load
# This function calculates the load corresponding to melting 10% of
# the Northern Hemisphere's ice mass.
direct_load = fp.northern_hemisphere_load(fraction=0.1)

# 4. Solve the sea level equation for the given load
# This returns the sea level change, surface displacement, gravity change,
# and angular velocity change. In this instance, only the first of the
# returned fields is used. 
sea_level_change, _, _, _ = fp(direct_load=direct_load)

# 5. Plot the resulting sea level fingerprint,
# showing the result only over the oceans.
fig, ax, im = plot(
    sea_level_change * fp.ocean_projection(),
)

# Customize the plot
ax.set_title("Sea Level Fingerprint of Northern Hemisphere Ice Melt", y=1.1)
cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
cbar.set_label("Sea Level Change (meters)")

plt.show()
```

---
## Core Components

* The library is organized into a few key modules:

* finger_print.py: Contains the main FingerPrint class, which orchestrates the calculations.

* ice_ng.py: Provides the IceNG class for loading and interpolating global ice history models.

* plotting.py: Includes the plot function for visualizing pyshtools.SHGrid objects.

* physical_parameters.py: Defines the EarthModelParameters class, which manages physical constants and non-dimensionalization schemes.

---

## Dependencies

`pyslfp` is built on top of a robust stack of scientific Python packages:

* numpy & scipy: For numerical operations.

* pyshtools: For spherical harmonic transforms and grid representations.

* pygeoinf: For formulating and solving associated inverse problems

* Cartopy & matplotlib: For creating high-quality map projections and plots.

---

## License

This project is licensed under the BSD-3-Clause License.

--- 

## Citations

If you use `pyslfp` in your published work, please cite the following paper:

*    Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J.X. and Lloyd, A.J., 2024. *Reciprocity and sensitivity kernels for sea level fingerprints*. Geophysical Journal International, **236(1)**, pp.362-378.
  
Furthermore, if you use the ice models contained in the `IceNG` class, please cite the appropriate ice history model:

[Peltier Group Data Sets](https://www.atmosp.physics.utoronto.ca/~peltier/data.php)

## Contributing

Contributions are welcome! If you have a suggestion or find a bug, please open an issue. Pull requests are also encouraged.