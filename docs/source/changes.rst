Changes in 4.2
==============

One change of results: the sea level that :class:`~pyslfp.ice.IceNG`
builds under floating ice now uses the Earth model's densities. No
signature changes; ``IceNG`` gains two optional arguments, and
``export_ice_history`` writes the landlocked depressions with the
histories.


Changed results
---------------

**ICE-NG sea levels under floating ice.** :class:`~pyslfp.ice.IceNG`
built its sea level from the topography with a flotation ratio of
917/1028, while the Earth model's ocean function tests flotation with
its own densities, 917/1000 by default: a shelf was built to float in
water of one density and tested in water of another. ``IceNG`` now takes
``ice_density`` and ``water_density``, which default to the default
model's (``pyslfp.core.model.ICE_DENSITY``, ``WATER_DENSITY``), and
exposes ``flotation_ratio``. ``EarthState.from_defaults`` passes its
model's densities.

The sea level returned by ``get_ice_thickness_and_sea_level`` changes
wherever there is ice over the sea floor, by the ice thickness times
(917/1000 - 917/1028), about 2.5 % of it; the ice thickness and the
topography are unchanged, as are files written by
``export_ice_history``, which hold the topography. With an Earth model
of other densities, pass them:

.. code-block:: python

   parameters = earth_model.parameters
   ice_model = IceNG(
       length_scale=parameters.length_scale,
       ice_density=parameters.ice_density,
       water_density=parameters.water_density,
   )

The old sea level is ``IceNG(water_density=1028.0)``.


Additions
---------

- ``IceNG(ice_density=..., water_density=...)`` and
  ``IceNG.flotation_ratio``.
- ``export_ice_history`` writes a third field, ``land``: one at the
  nodes of the dataset's grid held as land whatever the flotation (the
  present-day landlocked depressions, as ``IceNG.get_land`` gives them
  on a model grid) and zero elsewhere, with no time axis. A consumer of
  the file can then hold the Caspian and the other basins below sea
  level as land, as an ``EarthState`` built from ICE-NG does.
  ``land=False`` leaves it out.


Fixes
-----

- ``export_ice_history`` described the exported topography as "bedrock
  where ice-covered". It is the ICE-NG topography as stored: the ice
  surface under grounded ice and the sea floor under floating ice, which
  is what the sea level's construction assumes. The values written are
  unchanged; only the variable's ``meaning`` text is corrected.


Changes in 4.1
==============

Two changes to the viscoelastic derivatives: the sensitivity kernels are
now functions per unit volume, and the decay-time measurement carries its
Hessian. The first changes the meaning of existing arrays without
changing any signature.


Changed meaning
---------------

**Kernels are densities per unit volume.**
:attr:`IceKernels.itau`, :meth:`IceKernels.viscosity` and
:attr:`KernelDerivatives.itau` hold samples of a function ``K(r)`` at the
nodes of the radial mesh, with

.. math::

   \delta J = \int K\,\delta\,\mathrm{d}V
            = 4\pi \int K(r)\,\delta(r)\,r^2\,\mathrm{d}r .

Neither the :math:`r^2` nor the :math:`4\pi` is in the kernel. In 4.0 each
entry was the kernel multiplied by its node's volume weight, and paired
with a perturbation by a plain sum.

.. code-block:: python

   # old
   dJ = np.sum(kernels.itau[k] * d_itau)
   # new
   dJ = material.volume_pairing(kernels.itau[k], d_itau)

``LogViscositySpace.from_itau_kernels`` expects the new kernels and
applies the weights itself, so derivatives and gradients on the
log-viscosity space are unchanged. Code that paired, integrated or
plotted the raw arrays must use the pairing or multiply by
``Material.volume_weights``.

Kernels computed on different radial meshes now agree as functions; the
old entries depended on the mesh.


Additions
---------

- ``Material.volume_weights``, ``Material.volume_pairing`` and
  ``Material.kernel_samples``: the nodal volume weights
  :math:`4\pi r^2 w`, the pairing :math:`\sum W K \delta`, and the map
  from a nodal gradient to the kernel's samples.
- ``DecayTimeFunctional``: the decay time of a relative sea level series
  as a functional on the series, with its gradient and exact Hessian, for
  use in Newton-type inversions. ``DecayTimeFit`` gains ``hessian``, and
  ``fit_decay_time`` finishes with a Newton step.
- Theory documentation: the kernels' convention, and the second
  derivative of the decay-time fit.


Other
-----

- Tutorial 9 (decay time and viscosity) plots and integrates the kernels
  under the new convention; its results are unchanged.
- The planning notes are no longer kept in the repository.


Changes in 4.0
==============

One release carries the port to pygeoinf 2 and the work planned with it. The
spaces, operators and measures ``pyslfp`` returns are pygeoinf 2 objects, and
``pyslfp``'s own interface changes with them, so the major number changes.
The minimum versions are ``pygeoinf>=2.2.0`` and
``planetmodel>=1.5.0``. Each breaking change below gives the
old form and the new; the additions follow by subpackage.


Breaking changes
----------------

**The fingerprint operator's response.** ``FingerPrintOperator`` maps a load
to a :class:`~pyslfp.core.response.GRDResponse` of a
:class:`~pyslfp.core.response.GRDSpace`, not to a tuple of a ``DirectSum``.

.. code-block:: python

   # old
   slc, disp, gpc, omega = fingerprint(load)
   fingerprint.codomain.projection(0)
   # new
   response = fingerprint(load)
   response.field("relative_sea_level_change")
   fingerprint.codomain.field_operator("relative_sea_level_change")
   fingerprint.codomain.projection("angular_velocity_change")

The angular velocity change is a member only with rotational feedback.
``lebesgue_response_space`` and ``sobolev_response_space`` return a
``GRDSpace``, and ``check_response_space`` checks for one.

**Terms.** The names follow Gregory et al. (2019), with no aliases for the
old ones: ``sea_surface_height_operator`` is ``geocentric_sea_level_operator``,
and the relative sea level change is ``relative_sea_level_change`` wherever it
is named, the response's member, ``RELATIVE_SEA_LEVEL_CHANGE`` and the
viscoelastic snapshots' field (old: ``sea_level_change``).

**The land mask.** ``EarthState(exclude_caspian=...)`` is
``EarthState(land=...)``, a mask held as land whatever the flotation, none by
default; the ``exclude_caspian`` property is ``land``.
``EarthState.from_ice_model`` and ``calibrate_initial_sea_level`` take
``land="model"`` by default, the ice model's own: ICE-NG holds the present-day
landlocked depressions, an analytical model nothing.
``caspian_sea_projection`` is ``landlocked_projection(names=...)``, and the
region "Caspian Sea" is the depression from the topography, not a rectangle.

**Ice and glaciers.** ``ice_projection`` is all the state's ice, and
``exclude_glaciers`` is gone. The ice sheets are ``ice_sheet_projection``, by
rule (all ice south of 60S, all ice in GTN-G region 5), with
``ice_sheet_projection_operator``; ``ice_projection_operator`` and
``ice_average_operator`` follow ``ice_projection``. ``ice_altimetry_points``
takes ``sheets=`` in place of ``exclude_glaciers``. ``glacier_projection`` and
the region "Glaciers" are the RGI 7.0 glaciers outside the ice sheets.

**Viscoelastic snapshots.** A ``SeaLevelSnapshot`` holds the response as a
``GRDResponse``, ``response``, in place of separate fields.

.. code-block:: python

   # old
   snap.relative_sea_level_change
   snap.displacement
   snap.gravity_potential_change
   snap.angular_velocity_change
   # new
   snap.response.field("relative_sea_level_change")
   snap.response.field("displacement")        # or snap.response.displacement
   snap.response.field("potential_change")
   snap.response.angular_velocity_change     # None without rotational feedback

The runs take ``space=``, a ``GRDSpace``; with the horizontal part the
response holds the horizontal displacement, the memory field's included.

**Viscoelastic measurements.** ``Measurement``'s fields take the response's
member names: ``sea_level`` is ``relative_sea_level_change``,
``gravity_potential`` is ``potential_change`` and ``angular_velocity`` is
``angular_velocity_change``; ``displacement`` is unchanged. The adjoints also
take a ``GRDResponse`` as a measurement, its inner product with the response
in its own space, which observes the horizontal displacement as well.

**Module paths.** ``pyslfp.plot`` is ``pyslfp.plotting``, ``pyslfp.regions``
is ``pyslfp.core.regions`` and ``pyslfp.parallel`` is ``pyslfp.core.parallel``;
the top-level names (``pyslfp.plot``, ``pyslfp.set_num_threads``) are
unchanged.

**``pyslfp.linear_operators`` is gone**, its operators beside the physics they
serve:

* ``pyslfp.core``: the space builders (``lebesgue_load_space``,
  ``sobolev_load_space``, ``lebesgue_response_space``,
  ``sobolev_response_space``), ``underlying_space``, ``check_load_space``,
  ``check_response_space``, ``centrifugal_potential_operator``,
  ``geocentric_sea_level_operator``, and the projections and averages
  (``ocean_projection_operator``, ``ocean_average_operator``,
  ``ice_sheet_projection_operator``, ``region_average_operator``,
  ``averaging_operator``, ...);
* ``pyslfp.elastic``: ``FingerPrintOperator``;
* ``pyslfp.observations``: the tide gauge, altimetry and GRACE operators and
  models;
* ``pyslfp.models``: the load operators
  (``ice_thickness_change_to_load_operator``,
  ``sea_level_change_to_load_operator``, ``steric_sea_level_operator``, ...).

**Observation models take their observable.** ``TideGaugeObservationModel``,
``AltimetryObservationModel`` and ``GraceObservationModel`` take, in place of
a ``FingerPrintOperator``, the observable as an operator from the model
space, with the physics and any direct terms already in it, and add the
sampling, the stations and an optional noise
(:class:`~pyslfp.observations.base.ObservationModel`).

.. code-block:: python

   # old
   TideGaugeObservationModel(fingerprint, points, names=names)
   TideGaugeObservationModel.from_gloss_network(fingerprint)
   GraceObservationModel(fingerprint, degree)
   # new
   TideGaugeObservationModel.from_fingerprint(fingerprint, points, names=names)
   TideGaugeObservationModel(setup.relative_sea_level_change(fingerprint), points)
   TideGaugeObservationModel.from_gloss_network(observable)
   GraceObservationModel.from_fingerprint(fingerprint, degree)
   GraceObservationModel(observable, model, degree)

The models' ``response_to_data_operator`` is ``sampling_operator``, and
``fingerprint_operator`` is gone.

**Altimetry models.** ``AltimetryObservationModel(fingerprint, points)``
observed an ice thickness space and ``JointAltimetryObservationModel`` ice and
the sterodynamic sea level; the model spaces are the set-ups' now:
``AltimetryObservationModel(setup.geocentric_sea_level(fingerprint), points)``,
with ``ice_and_ocean`` for the joint case. ``JointAltimetryObservationModel``
and ``joint_ice_ocean_to_load_operator`` are removed.

**Removed wrappers.** The deprecated ``l2_products_operator`` and
``spatial_multiplication_operator`` are removed; the pygeoinf space methods of
the same names serve (``space.l2_products_operator``,
``space.multiplication_operator``).

**GRACE noise.** ``WMBMethod.load_measure_to_observation_measure`` takes a
coefficient's variance as the spectral variance over the metric, which the
earlier version did not; results built on it change.


Additions
---------

**Core.** The response classes, ``GRDResponse`` and ``GRDSpace`` with
``SurfaceDisplacement`` and its space, the horizontal displacement held
spectrally and evaluated at points with its exact adjoint, and
``plot_displacement``. The solver's response methods take every generalised
channel: ``solve_response`` the load channels, the ready responses with
``horizontal_response`` and the tangential traction ``tangential_load``;
``solve_nonlinear_response`` returns the new state and a ``GRDResponse``;
``adjoint_generalised_solve`` takes ``horizontal_dual``. The water bodies
(``pyslfp.core.water_bodies``), the GTN-G regions and the RGI 7.0 glaciers.
``EarthModel.integration_weights`` and ``integrate`` on arrays. Fields on the
sphere through time: ``FieldHistorySpace`` (``pyslfp.core.history``), the
pygeoinf tensor product of an interval space and a sphere space, with the
samplers ``at_time``, ``time_average``, ``difference_quotient`` and
``sample_at_times``; its time coordinate is the model's non-dimensional time,
with ``EarthModel.times_from_dates`` and ``dates_from_times`` and
``EarthModelParameters.kiloyear`` for dates in kiloyears before present.
``EarthState.flotation_width``: the flotation condition smoothed over a
height, under which the ocean function depends differentiably on the sea
level and the ice, with ``effective_ocean_function`` and
``flotation_curvature`` the fields the first and second derivatives of the
loads see. The package exposes ``__version__``.

**Plotting.** Radial profiles on a material's mesh
(``pyslfp.plotting.profiles``), drawn through planetmodel's profile plots:
``plot_profile`` a nodal array, a viscosity kernel say, one segment per layer
with joiners across the discontinuities and guides at the boundaries;
``plot_viscosity`` a rheology's viscosity in pascal seconds on a logarithmic
axis; ``plot_log_viscosity`` a vector of a ``LogViscositySpace``.

**Models.** ``pyslfp.models``: named load components (ice sheets, glaciers,
land water storage, firn, sterodynamic sea level, ocean density),
``ModelSetup`` assembling them with their loads, observables, constraints and
GMSL functionals, the ready-made set-ups, the diagnostics through the sea
level equation (``response``, ``relative_sea_level_change``,
``geocentric_sea_level``, ``gmsl``, ``regional_decomposition``), and
``HistorySetup``, a set-up through time on a tensor product with a time
space, every derived operator lifted to act at each time and the scalar
diagnostics as functions of time.

**Observations.** ``ObservationModel``, the shape the models share, with the
noise as a measure or standard deviations and ``noise_measure``;
``GnssObservationModel``, the displacement or its rate at stations in east,
north and up, with the frame of the data as Helmert nuisance parameters
(``frame="nuisance"``, with ``joint_prior``), reduced away
(``frame="invariant"``) or not at all, and ``HelmertTransformation`` with
``frame_reduction``; GRACE reading the potential change's coefficients from
the response's components with no transform, and Stokes coefficients
(``convention="stokes"``).

**Viscoelastic.** ``prem_maxwell``, ``maxwell_mantle`` and
``viscoelastic_earth``. ``IceHistoryOperator``, the frozen-shoreline map from
an ice thickness history to the responses at observation dates, whose adjoint
gives the Sobolev kernels. ``LogViscositySpace``, the log viscosity of every
branch as a function of radius, one Sobolev space of radial profiles per
relaxing layer. ``SeaLevelEvolutionOperator``, the evolution as a pygeoinf
operator on the ice history, the initial sea level and the log viscosity,
under frozen or migrating shorelines, with its derivative by the discrete or
the continuous method (``tangent_linear_ice_history`` forwards, the adjoints
backwards), its second derivative (``second_order_tangent_linear_ice_history``
and ``second_order_adjoint_ice_history``, the second-order adjoint equations
of Yu et al. 2025), and ``fixing`` to hold arguments fixed, giving
``IceHistoryOperator`` in the frozen case. Under migrating shorelines the
derivatives are complete when the flotation condition is smoothed; under the
sharp rule they hold the trajectory's ocean functions, and
``tools/shoreline_hessian.py`` measures what that omits.


After this release
------------------

A priors layer, physically motivated measures on a set-up's space and its
histories, follows; every operator and set-up takes its prior from the caller.
