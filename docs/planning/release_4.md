# pyslfp 4.0.0

One release carries the port to pygeoinf 2 and all the planned work below.
pyslfp's own interface breaks, so the major number changes. This file keeps
the list of breaking changes current; it becomes the "Changes in 4.0" page of
the documentation, each entry giving the old form and the new.

## Breaking changes so far

- **pygeoinf 2.** The spaces, operators and measures pyslfp returns are
  pygeoinf 2 objects; the minimum versions are `pygeoinf>=2.1.2` and
  `planetmodel>=1.4.1`. The name map is in
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
- **Removed:** `AltimetryObservationModel` and
  `JointAltimetryObservationModel`, whose model spaces the set-ups now
  hold, and `joint_ice_ocean_to_load_operator` with them. Old:
  `AltimetryObservationModel(fingerprint, points).forward_operator`. New:
  `fingerprint.field_space.point_evaluation_operator(points) @
  setup.geocentric_sea_level(fingerprint)` for a set-up `setup` on the
  fingerprint operator's load space. The deprecated wrappers
  `l2_products_operator` and `spatial_multiplication_operator` are removed;
  use the pygeoinf space methods of the same names
  (`space.l2_products_operator`, `space.multiplication_operator`).
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
(`pyslfp.core.history`), the product of a pygeoinf interval space and a
sphere space with the product metric, and its vectors, `FieldHistory`. The
load and response space builders (`lebesgue_load_space`,
`sobolev_load_space`, `lebesgue_response_space`, `sobolev_response_space`) in
`pyslfp.core`. `ModelSetup.geocentric_sea_level`, the observable of
altimetry. `IceHistoryOperator` (`pyslfp.viscoelastic`): the
frozen-shoreline map from an ice thickness history to the responses at
observation dates, whose adjoint gives the Sobolev kernels.

## Planned work still to land

- GNSS observation models with the frame of the data, and the observation
  models reshaped with them: each takes its observable as an operator from
  the model space (a set-up's, or a viscoelastic operator's) and adds the
  sampling, the stations' data and the noise
  ([response_plan.md](response_plan.md), package 6).
- The rest of the scope set with the port: the viscoelastic operators beyond
  `IceHistoryOperator` (the migrating-shoreline map, which is nonlinear, and
  the viscosity kernels), planetmodel 1.4's plots, the priors layer, and
  time-dependent model spaces for the set-ups' components, on
  `FieldHistorySpace`, which moves onto pygeoinf's tensor products of
  coordinate spaces when they land.
- Typed generics (`DiagonalMetricSpace[GRDResponse]`,
  `LinearOperator[SHGrid, GRDResponse]`) in code written or touched for the
  release.
