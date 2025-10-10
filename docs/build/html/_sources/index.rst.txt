.. pyslfp documentation master file, created by
   sphinx-quickstart on Monday Aug 19 10:18:00 2025.

pysl:PySLFP: Python Sea Level Fingerprints 

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




.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   modules

   