# The version-3 package structure

Decisions settled with David on 14 October 2026; this records the
target and the reasoning so the restructuring commit needs no
narrative.

## The shape

One sea-level engine, two drivers.  The viscoelastic time step drives
the same equilibrium solve as the elastic fingerprint, with per-step
effective Love numbers and the memory field through the generalised
channels, so the engine lives in a common core and the two physical
approximations are thin drivers over it:

    pyslfp/
      core/                 what both sides share
        model.py            EarthModel, EarthModelParameters
        state.py            EarthState
        sea_level.py        SeaLevelEquation: the linear, generalised
                            and nonlinear solvers (physics.py retires)
      love_numbers/         the radial degree solves, serving both sides
      ice/                  ice histories
      elastic/              the fingerprint driver (LinearSeaLevelEquation,
                            name kept)
      viscoelastic/         rheology, operator, evolution; the
                            ice-history adjoint lands here next
      linear_operators/     unchanged: bound to pygeoinf v1, reworked
                            only when pygeoinf 2 is live
      plot.py, parallel.py, regions.py, data/    shared utilities,
                            top level
      __init__.py           re-exports everything currently public

## The reasoning, briefly

- Subpackages are named by PHYSICAL APPROXIMATION (elastic,
  viscoelastic), not by application or instrument; "GIA" names a use
  of the viscoelastic driver, not the physics.
- The two drivers are not symmetric -- one is a stateless map from a
  load to fields, the other a time integrator -- so they do not carry
  mirrored class names; `SeaLevelEquation` keeps its name as the
  engine, which is what it always was (nothing in it is elastic but
  the table it is handed).
- Relative imports throughout the new subpackages, so later renames
  are directory renames.
- Compatibility: the public import surface of v2 (`import pyslfp as
  sl`, top-level names, `pyslfp.linear_operators`) is preserved by
  re-exports; deep module paths (`pyslfp.physics`,
  `pyslfp.core` as a module) are not, which is acceptable because the
  only production consumer (the Heathcote2026 scripts) is pinned to a
  v2 release in its own repository.
- No pygeoinf integration of the viscoelastic side at the outset; it
  waits for pygeoinf 2.
