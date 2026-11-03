# pyslfp 4.0.0

One release carries the port to pygeoinf 2 and all the planned work below.
pyslfp's own interface breaks, so the major number changes. This file keeps
the list of breaking changes current; docs/source/changes.rst is the
"Changes in 4.0" page built from it, each entry giving the old form and the new.

## Breaking changes so far

- **pygeoinf 2.** The spaces, operators and measures pyslfp returns are
  pygeoinf 2 objects; the minimum versions are `pygeoinf>=2.2.0` and
  `planetmodel>=1.5.0`. The name map is in
  [pygeoinf_v2_port.md](pygeoinf_v2_port.md).
- **The fingerprint operator's response.** `FingerPrintOperator` maps a load
  to a `GRDResponse` of a `GRDSpace` (`pyslfp.core.response`), not to the
  tuple (sea level, displacement, potential, angular velocity) of a
  `DirectSum`. Old: `slc, disp, gpc, omega = fingerprint(load)`,
  `codomain.projection(0)`, `codomain.subspace(0)`. New:
  `response.field("relative_sea_level_change")`, `codomain.field_operator(...)`,
  `codomain.projection("angular_velocity_change")`, `codomain.field_space`.
  The angular velocity change is a member only with rotational feedback.
  `lebesgue_response_space` and `sobolev_response_space` return a `GRDSpace`,
  and `check_response_space` checks for one.
- **Terms.** The names follow Gregory et al. (2019), with no aliases for the
  old ones. `sea_surface_height_operator` is `geocentric_sea_level_operator`.
  The relative sea level change is `relative_sea_level_change` wherever it is
  named: the response's member and `RELATIVE_SEA_LEVEL_CHANGE`, and the
  viscoelastic snapshots' field (old: `sea_level_change`).
- **The Caspian.** `EarthState(exclude_caspian=...)` becomes
  `EarthState(land=...)`, a mask held as land whatever the flotation, default
  none; the `exclude_caspian` property becomes `land`.
  `EarthState.from_ice_model` and `calibrate_initial_sea_level` take
  `land="model"` by default, the ice model's own: ICE-NG holds the present-day
  landlocked depressions, an analytical model nothing.
  `caspian_sea_projection` becomes `landlocked_projection(names=...)`, and the
  region "Caspian Sea" is the depression from the topography, not a rectangle.
- **Ice and glaciers.** `ice_projection` is all the state's ice;
  `exclude_glaciers` is gone. The ice sheets are `ice_sheet_projection`, by
  rule (all ice south of 60S, all ice in GTN-G region 5), with
  `ice_sheet_projection_operator`; `ice_projection_operator` and
  `ice_average_operator` follow `ice_projection`. `ice_altimetry_points` takes
  `sheets=` in place of `exclude_glaciers`. `glacier_projection` and the region
  "Glaciers" are the RGI 7.0 glaciers outside the ice sheets, not a box.
- **Viscoelastic snapshots.** A `SeaLevelSnapshot` holds the response as a
  `GRDResponse`, `response`, in place of the fields
  `relative_sea_level_change`, `displacement`, `gravity_potential_change` and
  `angular_velocity_change`. Old: `snap.relative_sea_level_change`,
  `snap.displacement`, `snap.gravity_potential_change`. New:
  `snap.response.field("relative_sea_level_change")`,
  `snap.response.field("displacement")` (or `snap.response.displacement`, the
  whole `SurfaceDisplacement`), `snap.response.field("potential_change")`.
  `snap.angular_velocity_change` is `snap.response.angular_velocity_change`,
  None without rotational feedback where it was zeros. The runs take
  `space=`, a `GRDSpace`; with the horizontal part the response holds the
  horizontal displacement, the memory field's included.
- **Viscoelastic measurements.** `Measurement`'s fields take the response's
  member names: `sea_level` is `relative_sea_level_change`,
  `gravity_potential` is `potential_change` and `angular_velocity` is
  `angular_velocity_change`; `displacement` is unchanged. The adjoints also
  take a `GRDResponse` as a measurement, its inner product with the response
  in its own space, which observes the horizontal displacement as well.
- **Module paths.** `pyslfp.plot` is `pyslfp.plotting`, `pyslfp.regions` is
  `pyslfp.core.regions`, `pyslfp.parallel` is `pyslfp.core.parallel`; the
  top-level names (`pyslfp.plot`, `pyslfp.set_num_threads`, ...) are unchanged.
- **`pyslfp.linear_operators` is gone**, its operators beside the physics they
  serve. Old: `from pyslfp.linear_operators import X`. New:
  - `pyslfp.core`: the space builders (`lebesgue_load_space`,
    `sobolev_load_space`, `lebesgue_response_space`,
    `sobolev_response_space`), `underlying_space`, `check_load_space`,
    `check_response_space`, `centrifugal_potential_operator`,
    `geocentric_sea_level_operator`, and the projections and averages
    (`ocean_projection_operator`, `ocean_average_operator`,
    `ice_projection_operator`, `ice_sheet_projection_operator`,
    `region_average_operator`, `averaging_operator`, ...);
  - `pyslfp.elastic`: `FingerPrintOperator`;
  - `pyslfp.observations`: the tide gauge, altimetry and GRACE operators and
    models (`TideGaugeObservationModel`, `GraceObservationModel`, `WMBMethod`,
    `altimetry_point_operator`, `ocean_altimetry_points`, ...);
  - `pyslfp.models`: the load operators (`ice_thickness_change_to_load_operator`,
    `sea_level_change_to_load_operator`, `steric_sea_level_operator`, ...).
- **Observation models take their observable.** `TideGaugeObservationModel`,
  `AltimetryObservationModel` and `GraceObservationModel` take, in place of a
  `FingerPrintOperator`, the observable as an operator from the model space,
  with the physics and any direct terms already in it, and add the sampling,
  the stations and an optional noise (`ObservationModel`). Old:
  `TideGaugeObservationModel(fingerprint, points, names=...)`,
  `TideGaugeObservationModel.from_gloss_network(fingerprint)`,
  `GraceObservationModel(fingerprint, degree, ...)`. New:
  `TideGaugeObservationModel.from_fingerprint(fingerprint, points, names=...)`
  or `TideGaugeObservationModel(setup.relative_sea_level_change(fingerprint),
  points)`, `TideGaugeObservationModel.from_gloss_network(observable)`,
  `GraceObservationModel.from_fingerprint(fingerprint, degree, ...)` or
  `GraceObservationModel(observable, model, degree, ...)`. The models'
  `response_to_data_operator` is `sampling_operator`, and
  `fingerprint_operator` is gone.
- **Altimetry models.** `AltimetryObservationModel(fingerprint, points)` observed
  an ice thickness space and `JointAltimetryObservationModel(fingerprint,
  points)` ice and the sterodynamic sea level; the model spaces are the
  set-ups' now. New: `AltimetryObservationModel(setup.geocentric_sea_level(
  fingerprint), points)`, with `ice_and_ocean` for the joint case.
  `JointAltimetryObservationModel` and `joint_ice_ocean_to_load_operator` are
  removed.
- **Removed wrappers.** The deprecated `l2_products_operator` and
  `spatial_multiplication_operator` are removed; use the pygeoinf space
  methods of the same names (`space.l2_products_operator`,
  `space.multiplication_operator`).
- **GRACE noise.** `WMBMethod.load_measure_to_observation_measure` takes a
  coefficient's variance as the spectral variance over the metric, which v1
  did not; results built on it change.

## Additions so far

The models submodule (`pyslfp.models`), with its ready-made set-ups and the
diagnostics through the sea level equation (`gmsl`, `regional_decomposition`);
horizontal displacement and the
response classes, with `plot_displacement` (arrows over colour); the water bodies (`pyslfp.core.water_bodies`) and the GTN-G
regions; the RGI 7.0 glaciers; Stokes coefficients for GRACE
(`convention="stokes"`); `prem_maxwell`, `maxwell_mantle` and
`viscoelastic_earth`; `EarthModel.integration_weights`. The solver's response
methods take every generalised channel: `solve_response` the load channels,
the ready responses (with `horizontal_response`, a ready V) and the
tangential traction `tangential_load`; `solve_nonlinear_response` returns the
new state and a `GRDResponse`; and `adjoint_generalised_solve` takes
`horizontal_dual`. Fields on the sphere through time: `FieldHistorySpace`
(`pyslfp.core.history`), the pygeoinf tensor product of an interval space
and a sphere space, whose vectors are (K, N) arrays of components. The
load and response space builders (`lebesgue_load_space`,
`sobolev_load_space`, `lebesgue_response_space`, `sobolev_response_space`) in
`pyslfp.core`. `ModelSetup.geocentric_sea_level`, the observable of
altimetry, and `ModelSetup.response`, the response to the set-up's load.
`EarthModel.times_from_dates` and `dates_from_times`, with
`EarthModelParameters.kiloyear`: the model's non-dimensional time from a
date in kiloyears before present and back, the time coordinate of the
history spaces being the model's, like every other coordinate.
`ObservationModel`, the shape the observation models share, with their
noise, and `noise_measure`, which turns standard deviations into a measure.
`GnssObservationModel` (`pyslfp.observations`): the displacement, or its
rate, at stations in east, north and up, with the frame of the data taken as
Helmert nuisance parameters joined to the model space (`frame="nuisance"`,
with `joint_prior`), by reducing the data to the combinations the frame
cannot change (`frame="invariant"`), or not at all; `HelmertTransformation`,
the design at the stations, any of the translation, rotation and scale, with
`frame_reduction`. `grace_observation_operator` and `GraceObservationModel`
read the potential change's coefficients from the response's components, with
no transform. `IceHistoryOperator` (`pyslfp.viscoelastic`): the
frozen-shoreline map from an ice thickness history to the responses at
observation dates, whose adjoint gives the Sobolev kernels.
`LogViscositySpace` (`pyslfp.viscoelastic`): the log viscosity of every
branch of a rheology as a function of radius, a pygeoinf layered radial
space per branch with one Sobolev space per relaxing layer, with its values
at the mesh's nodes, their transpose, the rheology a vector names and the
functional the inverse-relaxation-time kernels define on it.
`SeaLevelEvolution.tangent_linear_ice_history`: the derivative of a run's
responses along perturbations of the ice history, the initial sea level and
the inverse relaxation times, about a recorded forward run, frozen or
migrating, by the forward mode of the time-stepped map (`method="discrete"`,
the transpose of `adjoint_ice_history`) or by the continuous tangent-linear
problem (`method="continuous"`). `SeaLevelEvolutionOperator`
(`pyslfp.viscoelastic`): the evolution as a pygeoinf `Operator` from the ice
history, the initial sea level and the log viscosity to the responses at
observation dates, under frozen or migrating shorelines, with its derivative
by either method, one run forwards and one sweep backwards; `fixing` holds
arguments fixed by composition, and gives `IceHistoryOperator`, the linear
case, under frozen shorelines with the other two fixed. Its second derivative,
by `SeaLevelEvolution.second_order_tangent_linear_ice_history` forwards and
`second_order_adjoint_ice_history` backwards (the second-order adjoint
equations of Yu et al. 2025, with the shoreline terms omitted), gives pygeoinf
exact Hessians; `KernelDerivatives` holds the derivative of the kernels along
a direction, and the scheme weights gain their second derivatives.
`EarthState.flotation_width`: the flotation condition smoothed over a
height, under which the ocean function is differentiable, with
`effective_ocean_function` and `flotation_curvature` the fields the first and
second derivatives of the loads see; the derivatives and the Hessian of a
migrating run are then complete, the shorelines' terms included, where the
sharp rule holds the ocean functions. The package exposes `__version__`. Radial profiles on a material's mesh
(`pyslfp.plotting.profiles`): `plot_profile` draws a nodal array, a kernel
say, through planetmodel 1.5's profile plots, one segment per layer with the
joiners and boundary guides of the model's own fields; `plot_viscosity` a
rheology's viscosity in pascal seconds on a logarithmic axis, and
`plot_log_viscosity` a vector of a `LogViscositySpace`.

## Planned work still to land

The operators of the release's modules carry typed generics
(`LinearOperator[SHGrid, GRDResponse]`, `Operator[tuple, tuple]`), with
`Any` where a space is the caller's.

- The priors layer, after the release.

