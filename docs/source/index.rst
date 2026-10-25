PySLFP: Python Sea Level Fingerprints
=====================================

``pyslfp`` computes how sea level changes when mass is redistributed at the
Earth's surface, on a spherically symmetric, self-gravitating, rotating Earth.
It solves the sea level equation with the deformation of the solid Earth,
gravitational self-consistency between ice, oceans and solid Earth, and
rotational feedbacks, for two descriptions of the solid Earth:

* **Elastic**, for sea level fingerprints: the instantaneous pattern of sea
  level change produced, for example, by the melting of an ice sheet.
* **Viscoelastic**, for glacial isostatic adjustment (GIA): sea level,
  deformation and rotation evolving through an ice history on an Earth with a
  mantle viscosity, with fixed or migrating shorelines.

Around these it provides Love numbers for any spherically layered model that
planetmodel_ describes; sensitivity kernels by adjoint methods, for
fingerprints and for a viscoelastic evolution with respect to the ice history,
the initial sea level and the mantle viscosity; the elastic physics as linear
operators between Hilbert spaces, with observation models for tide gauges,
satellite altimetry and GRACE, for Bayesian inversions built on pygeoinf_; and
ice histories, regional masks and plotting.

:doc:`theory` summarises the physics and methods, :doc:`numerics` their
discretisation and cost, and :doc:`units` the non-dimensionalisation. The
source is on GitHub_, and the package is on PyPI_.

.. _planetmodel: https://github.com/da380/planetmodel
.. _pygeoinf: https://github.com/da380/pygeoinf
.. _GitHub: https://github.com/da380/pyslfp
.. _PyPI: https://pypi.org/project/pyslfp/


Installation
------------

``pyslfp`` requires Python 3.12 or later and is available from PyPI:

.. code-block:: bash

   pip install pyslfp

Plotting works out of the box. ``plt.show()`` needs matplotlib to have an
interactive backend, which on a Python built with tkinter, the usual case, it
already has. Where tkinter is absent, or if you would rather use Qt:

.. code-block:: bash

   pip install "pyslfp[interactive]"

For development, clone the repository and use Poetry:

.. code-block:: bash

   poetry install              # runtime dependencies only
   poetry install --with dev   # adds pytest, sphinx, ruff, jupyter and the hooks

The git hooks, the documentation build and the release process are described in
`CONTRIBUTING.md <https://github.com/da380/pyslfp/blob/main/CONTRIBUTING.md>`_.

The datasets the package needs (a table of Love numbers for PREM, the ICE-NG ice
histories, the GLOSS tide gauge network and the regional shapefiles) are
downloaded from Zenodo_ on first use and cached in ``~/.pyslfp_data``, or
wherever the ``PYSLFP_DATA`` environment variable points.
``pyslfp.data.ensure_data(key, refresh=True)`` fetches a dataset afresh.

.. _Zenodo: https://zenodo.org/records/19494463


A sea level fingerprint
-----------------------

The following melts ten per cent of the West Antarctic Ice Sheet and plots the
resulting sea level change:

.. code-block:: python

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

:class:`~pyslfp.elastic.fingerprint.LinearSeaLevelEquation` holds the shoreline
fixed, the usual assumption for present-day and near-future problems. The
engine beneath it, :class:`~pyslfp.core.sea_level.SeaLevelEquation`, also solves
the nonlinear equation with migrating shorelines, and the generalised equation
that adjoint calculations need. The same physics is exposed as a ``pygeoinf``
operator by :class:`~pyslfp.linear_operators.physics.FingerPrintOperator`.


A glacial isostatic adjustment run
----------------------------------

The following runs ICE-7G from 26 ka to the present on PREM with a Maxwell
mantle of viscosity :math:`10^{21}` Pa s beneath an elastic lithosphere:

.. code-block:: python

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

:meth:`~pyslfp.viscoelastic.evolution.SeaLevelEvolution.run_ice_history` takes
``shorelines="migrating"`` to let the shorelines move, and ``final_sea_level=``
to choose the initial sea level so that the run reaches a given present-day sea
level, as an ice history without its own sea level needs.
:class:`~pyslfp.viscoelastic.evolution.SeaLevelEvolution` computes sensitivity
kernels of any objective built from the snapshots.


Tutorials
---------

The tutorials are scripts in the ``tutorials/scripts`` directory of the
repository, each with a notebook generated from it in ``tutorials/notebooks``.
The notebooks can be run locally or in Google Colab:

* `1 — A first sea level fingerprint
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/01_first_fingerprint.ipynb>`_
* `2 — A closer look at the physics
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/02_physics.ipynb>`_
* `3 — Love numbers
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/03_love_numbers.ipynb>`_
* `4 — Operators, composition and sensitivity kernels
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/04_operators.ipynb>`_
* `5 — A Bayesian inversion of tide gauge data
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/05_inversion.ipynb>`_
* `6 — A first forward GIA simulation
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/06_forward_gia.ipynb>`_
* `7 — The last deglaciation
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/07_ice7g_deglaciation.ipynb>`_
* `8 — Ice-history sensitivity kernels
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/08_gia_kernels.ipynb>`_
* `9 — The decay time of postglacial emergence
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/notebooks/09_decay_time_viscosity.ipynb>`_


Citation
--------

If you use ``pyslfp`` in published work, please cite:

* Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J.X. and Lloyd, A.J., 2024.
  *Reciprocity and sensitivity kernels for sea level fingerprints*. Geophysical Journal
  International, **236(1)**, pp.362–378.
* Heathcote, D.A., Holland, T., Mag, A.M., Tamisiea, M.E., Coulson, S., Dangendorf,
  S., Lloyd, A.J., Mashayek, A., Mitrovica, J.X. and Al-Attar, D., 2026. *A scalable
  Bayesian framework for modern sea-level inference*. arXiv, 2608.22336.

The datasets that ``pyslfp`` downloads are the work of others and are redistributed
only for convenience. If you use them, please cite their original sources, which are
recorded on the `Zenodo record <https://zenodo.org/records/19494463>`_.


.. toctree::
   :maxdepth: 1
   :caption: Guide
   :hidden:

   theory
   numerics
   units

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   modules
