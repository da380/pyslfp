# pyslfp 4.0.0

One release carries the port to pygeoinf 2 and all the planned work below.
pyslfp's own interface breaks, so the major number changes. This file keeps
the list of breaking changes current; it becomes the "Changes in 4.0" page of
the documentation, each entry giving the old form and the new.

## Breaking changes so far

- **pygeoinf 2.** The spaces, operators and measures pyslfp returns are
  pygeoinf 2 objects; the minimum versions are `pygeoinf>=2.1.1` and
  `planetmodel>=1.4.1`. The name map is in
  [pygeoinf_v2_port.md](pygeoinf_v2_port.md).
- **The fingerprint operator's response.** `FingerPrintOperator` maps a load
  to a `GRDResponse` of a `GRDSpace` (`pyslfp.core.response`), not to the
  tuple (sea level, displacement, potential, angular velocity) of a
  `DirectSum`. Old: `slc, disp, gpc, omega = fingerprint(load)`,
  `codomain.projection(0)`, `codomain.subspace(0)`. New:
  `response.field("sea_level_change")`, `codomain.field_operator(...)`,
  `codomain.projection("angular_velocity_change")`, `codomain.field_space`.
  The angular velocity change is a member only with rotational feedback.
  `lebesgue_response_space` and `sobolev_response_space` return a `GRDSpace`,
  and `check_response_space` checks for one.
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
- **Module paths.** `pyslfp.plot` is `pyslfp.plotting`, `pyslfp.regions` is
  `pyslfp.core.regions`, `pyslfp.parallel` is `pyslfp.core.parallel`; the
  top-level names (`pyslfp.plot`, `pyslfp.set_num_threads`, ...) are unchanged.
- **GRACE noise.** `WMBMethod.load_measure_to_observation_measure` takes a
  coefficient's variance as the spectral variance over the metric, which v1
  did not; results built on it change.

## Additions so far

The models submodule (`pyslfp.models`); horizontal displacement and the
response classes; the water bodies (`pyslfp.core.water_bodies`) and the GTN-G
regions; the RGI 7.0 glaciers; Stokes coefficients for GRACE
(`convention="stokes"`); `prem_maxwell`, `maxwell_mantle` and
`viscoelastic_earth`; `EarthModel.integration_weights`.

## Planned work still to land

- The viscoelastic side of the response: snapshots holding a `GRDResponse`,
  the horizontal memory field, the adjoints; GNSS observation models with the
  frame of the data ([response_plan.md](response_plan.md), packages 5 and 6).
- WP3 of the models submodule: ready-made set-ups, the diagnostics that take
  the fingerprint operator, new simplified scripts in place of
  `Heathcote2026/` ([models_plan.md](models_plan.md)).
- The deprecations: the wrappers in `linear_operators/utils.py`
  (`l2_products_operator`, `spatial_multiplication_operator`) and the
  altimetry observation models the models plan replaces, removed or kept as
  deprecated aliases, to be decided.
- The rest of the scope set with the port: the viscoelastic adjoints wrapped
  as pygeoinf operators, planetmodel 1.4's plots, the priors layer, and
  time-dependent model spaces, which wait on a tensor product of spaces in
  pygeoinf.
- Typed generics (`DiagonalMetricSpace[GRDResponse]`,
  `LinearOperator[SHGrid, GRDResponse]`) in code written or touched for the
  release.
