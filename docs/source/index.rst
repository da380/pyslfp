PySLFP: Python Sea Level Fingerprints
=====================================

``pyslfp`` computes elastic sea level fingerprints: the spatially variable pattern of
sea level change produced when mass is redistributed at the Earth's surface, for
example by the melting of an ice sheet. It solves the sea level equation, taking
account of the elastic deformation of the solid Earth, gravitational self-consistency
between ice, oceans and solid Earth, and rotational feedbacks.

The library covers both the forward problem and its use within inverse problems.
Alongside the solvers, it provides the same physics expressed as linear operators
between Hilbert spaces, together with observation models for tide gauges, satellite
altimetry and GRACE gravimetry. These build on pygeoinf_ and follow the theory set out
in `Al-Attar et al. (2024)`_.

The source is on GitHub_, and the package is on PyPI_.

.. _pygeoinf: https://github.com/da380/pygeoinf
.. _Al-Attar et al. (2024): https://academic.oup.com/gji/article/236/1/362/7338265
.. _GitHub: https://github.com/da380/pyslfp
.. _PyPI: https://pypi.org/project/pyslfp/


Installation
------------

``pyslfp`` requires Python 3.12 or later and is available from PyPI:

.. code-block:: bash

   pip install pyslfp

Plotting works out of the box. ``plt.show()`` needs matplotlib to have an
interactive backend, which on a Python built with tkinter — the usual case — it
already has. Where tkinter is absent, or if you would rather use Qt:

.. code-block:: bash

   pip install "pyslfp[interactive]"

For development, clone the repository and use Poetry:

.. code-block:: bash

   poetry install              # runtime dependencies only
   poetry install --with dev   # adds pytest, sphinx, ruff, jupyter and the hooks

The git hooks, the documentation build and the release process are described in
`CONTRIBUTING.md <https://github.com/da380/pyslfp/blob/main/CONTRIBUTING.md>`_.


Data
----

The package needs a number of external datasets: a table of load Love numbers, the ICE-NG ice
histories, and shapefiles for the various regional definitions. These are not
distributed with the package. They are downloaded from Zenodo_ automatically, on first
use, and then cached locally, so the first call that needs a given dataset will pause
while it is fetched and a progress bar is shown. Subsequent calls read from the cache.

By default the cache lives in ``~/.pyslfp_data``. This can be changed by setting the
``PYSLFP_DATA`` environment variable, which is useful on shared machines and in CI.
Datasets are fetched individually, so only what is actually used gets downloaded.
A dataset that has changed on Zenodo is picked up by
``pyslfp.data.ensure_data(key, refresh=True)``, which deletes the cached copy and
downloads it again.

.. _Zenodo: https://zenodo.org/records/22891291


Love numbers
------------

The solid Earth enters the sea level equation through its elastic Love numbers. By
default ``EarthModel`` uses the precomputed table for PREM. The package can also
compute them, from any spherically layered model that planetmodel_ describes, by
solving the loading and tidal problem degree by degree on a radial spectral-element
mesh:

.. code-block:: python

   from planetmodel import PREM
   from pyslfp import EarthModel, LoveNumbers

   love = LoveNumbers.from_model(PREM(ocean=False), 256)
   love.write("prem_256.dat")
   model = EarthModel(256, love_numbers=love)

The table carries the radius, surface gravity and gravitational constant of the body
it was computed for, and ``EarthModel`` takes those from the table so that the sea
level equation and its adjoint stay consistent. See ``pyslfp.love_numbers`` in the
API reference, and the third tutorial script.

Definitions and conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The numbers the library holds are not the usual dimensionless ones, so their
definition is worth setting down. The solver works with the physical
gravitational potential, which is negative near added mass, and with
dimensional numbers: a displacement or a potential per unit of the forcing
that produced it. A surface load of density :math:`\sigma_{lm} Y_{lm}` acts in
two ways. It presses on the surface with the traction
:math:`-g\sigma_{lm} Y_{lm}`, and it attracts the body as a surface mass in
Poisson's equation. The generalised Love numbers are the surface response to
each acting alone. With the displacement written as
:math:`\mathbf{u} = U Y_{lm}\hat{\mathbf{r}} + V \nabla_1 Y_{lm}` and the
potential perturbation as :math:`\phi Y_{lm}`, both at :math:`r = a`,

.. math::

   U = h_l^u \zeta^u_{lm} + h_l^\phi \zeta^\phi_{lm}, \qquad
   V = l_l^u \zeta^u_{lm} + l_l^\phi \zeta^\phi_{lm}, \qquad
   \phi = k_l^u \zeta^u_{lm} + k_l^\phi \zeta^\phi_{lm},

where :math:`\zeta^u` is a surface density that presses but does not attract
and :math:`\zeta^\phi` one that attracts but does not press. A true load does
both, so its numbers are the sums

.. math::

   h_l = h_l^u + h_l^\phi, \qquad l_l = l_l^u + l_l^\phi, \qquad k_l = k_l^u + k_l^\phi,

which are the properties ``h``, ``l`` and ``k``. In SI, :math:`h` and
:math:`l` are in m³ kg⁻¹ and :math:`k` in m⁴ kg⁻¹ s⁻². A third channel, the
tangential traction :math:`-g\zeta^v_{lm} \nabla_1 Y_{lm}`, has the numbers
:math:`h^v_l`, :math:`l^v_l` and :math:`k^v_l`, and is what the adjoint
problem for a functional of horizontal displacement needs. The tidal numbers
:math:`h^t_l`, :math:`l^t_l` and :math:`k^t_l` are the response to the unit
external potential :math:`\psi = (r/a)^l Y_{lm}`, so :math:`h^t` and
:math:`l^t` are in s² m⁻¹ and :math:`k^t` is dimensionless.

The conventional dimensionless load numbers :math:`h'_l`, :math:`l'_l` and
:math:`k'_l` of Farrell (1972) refer the response to the direct potential of
the load, :math:`4\pi G a\sigma_{lm}/(2l+1)` in the geodetic sign convention,
where the potential is positive near mass. They follow from the numbers above
by

.. math::

   h'_l = \frac{(2l+1)\, g}{4\pi G a}\, h_l, \qquad
   l'_l = \frac{(2l+1)\, g}{4\pi G a}\, l_l, \qquad
   k'_l = -\frac{2l+1}{4\pi G a}\, k_l - 1,

and the geodetic tidal numbers differ from the library's only by the sign of
the potential and a factor of gravity:

.. math::

   k^T_l = k^t_l, \qquad h^T_l = -g\, h^t_l, \qquad l^T_l = -g\, l^t_l .

``LoveNumbers.conventional()`` and ``LoveNumbers.tidal()`` return these. The
problem is self-adjoint, which gives the reciprocity relations

.. math::

   g\, h^\phi_l = k^u_l, \qquad h^v_l = l(l+1)\, l^u_l, \qquad k^v_l = g\, l(l+1)\, l^\phi_l,

the first being eq. (64) of Al-Attar et al. (2024);
``LoveNumbers.reciprocity_residual()`` checks all three. Degree 1 is in the
centre-of-mass frame, where the surface potential perturbation vanishes and
:math:`k'_1 = -1`. At degree 0 the tidal numbers are zero, a uniform external
potential being a gauge, while the load numbers are not: mass conservation
fixes :math:`k_0 = -4\pi G a`. Degree 0 also carries five axial numbers,
``h_c``, ``k_c``, ``m_u``, ``m_phi`` and ``m_c``: the surface response to the
spherical mean of the centrifugal potential of a change in spin rate, which
goes as :math:`r^2` rather than being a constant, and the inertia moments
:math:`\sqrt{4\pi}\int \rho\, U\, r^3\, dr` of the degree-0 responses to the two
load channels and to that potential. The component of the rotational feedback
along the rotation axis needs them, because the trace of the inertia
perturbation is not seen by the degree-2 potential; symmetry gives
:math:`m_u = \sqrt{4\pi}\, g a^4 h_c / 2`, which
``LoveNumbers.axial_reciprocity_residual()`` checks, and :math:`k_c` and
:math:`m_\phi` vanish by the shell theorem. The sea level solver uses the
generalised numbers directly, because the adjoint theory is written in them
rather than in :math:`h` and :math:`k` alone.

.. _planetmodel: https://github.com/da380/planetmodel


A first calculation
-------------------

The following melts ten percent of the West Antarctic Ice Sheet and plots the
resulting sea level fingerprint:

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
   length_scale = sle.state.model.parameters.length_scale
   fig, ax = sl.create_map_figure(figsize=(12, 6))
   sl.plot(
       sea_level_change * sle.state.ocean_projection() * length_scale,
       ax=ax,
       colorbar_kwargs={"label": "Sea level change (m)"},
   )
   plt.show()

:class:`~pyslfp.physics.LinearSeaLevelEquation` holds the shoreline fixed, which is the
usual assumption for present-day and near-future problems.
:class:`~pyslfp.physics.SeaLevelEquation` provides the same linear solver along with
``solve_nonlinear_equation``, which migrates the shoreline and returns an updated
:class:`~pyslfp.state.EarthState`, and ``solve_generalised_equation``, which accepts
displacement, potential and angular momentum forcings as needed in adjoint
calculations.


Units
-----

Calculations are carried out in non-dimensional form. By default lengths, densities and
times are scaled so that the Earth's radius, mean density and surface gravity are all
equal to one. Results are returned in these units, and are converted back by
multiplying by the appropriate scale from ``state.model.parameters`` —
``length_scale`` for sea level and displacement, ``load_scale`` for surface loads, and
so on. The scheme itself is set by
:class:`~pyslfp.core.EarthModelParameters`, and can be replaced if a different one
suits the problem better.


Operators and inverse problems
------------------------------

The same physics is also exposed as a ``pygeoinf`` ``LinearOperator``, so that
fingerprints can be composed with observation operators, adjointed, and used within
Bayesian inversions. :class:`~pyslfp.linear_operators.physics.FingerPrintOperator` maps
a surface load to the four-component response (sea level change, vertical displacement,
potential change, angular velocity change), and its domain and codomain may be either
Lebesgue or Sobolev spaces, the latter providing regularisation.

.. code-block:: python

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

Built on this are the observation models in :mod:`pyslfp.linear_operators`, each
pairing a forward operator with the machinery needed to pose an inversion:

* ``TideGaugeObservationModel``, using the GLOSS station network.
* ``AltimetryObservationModel`` and ``JointAltimetryObservationModel``, for sea surface
  height over the oceans and over the ice sheets.
* ``GraceObservationModel``, mapping loads to spherical harmonic coefficients of the
  potential change, with ``WMBMethod`` providing the purely spectral Wahr, Molenaar and
  Bryan (1998) approximation for comparison.


Tutorials
---------

Four introductory notebooks are kept in the ``tutorials`` directory of the
repository, and can be run locally or in Google Colab:

* `Tutorial 1 — A first sea level fingerprint
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial1.ipynb>`_
* `Tutorial 2 — A closer look at the physics
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial2.ipynb>`_
* `Tutorial 3 — Operators, composition and sensitivity kernels
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial3.ipynb>`_
* `Tutorial 4 — A Bayesian inversion of tide gauge data
  <https://colab.research.google.com/github/da380/pyslfp/blob/main/tutorials/tutorial4.ipynb>`_


Citation
--------

If you use ``pyslfp`` in published work, please cite:

* Al-Attar, D., Syvret, F., Crawford, O., Mitrovica, J.X. and Lloyd, A.J., 2024.
  *Reciprocity and sensitivity kernels for sea level fingerprints*. Geophysical Journal
  International, **236(1)**, pp.362–378.

The datasets that ``pyslfp`` downloads are the work of others and are redistributed
only for convenience. If you use them, please cite their original sources, which are
recorded on the `Zenodo record <https://zenodo.org/records/22891291>`_.


.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   modules
