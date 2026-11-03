# Open work

What is planned or open, with enough context to pick each item up. A design
that needs more than a few paragraphs has its own file in this directory.

## The models submodule

A vocabulary of physically named load components (grounded ice, glaciers, land
water storage, firn, dynamic topography, ocean density) and an assembler that
turns a list of them into an inversion space, with the load operator,
constraints and GMSL diagnostics derived from each component's role. The design
is in [models_plan.md](models_plan.md). The component record and the factories
(WP1), glaciers included, and the assembler, `ModelSetup` (WP2), are built.
WP3, the ready-made set-ups and the rest of the harvest, remains; it follows
the response classes and horizontal displacement (see below).

It also completes the harvest of physics from the Heathcote2026 scripts: the
barystatic and steric GMSL split and the regional decomposition after Gregory
et al. (2019) are to land as diagnostics of a set-up. The steric and
mass-conservation operators are in `models/loads.py`. The
scripts are then replaced by new, simplified ones built on set-ups, tested
against a local record of the current scripts kept outside the repository;
the current scripts leave the repository once the new ones reproduce them.

## Water bodies

The ocean function is decided by flotation, with no geography assumed; regions
held as land whatever the flotation are an `EarthState`'s `land` mask, which
the non-linear solver and the viscoelastic evolution carry into the states
they create. The present-day landlocked depressions (`landlocked_projection`)
come from `pyslfp.core.water_bodies`, built from the ETOPO 2022 bedrock at two
arc-minutes by `tools/build_landlocked.py` and shipped as
`pyslfp/data/landlocked.npz`; ICE-NG supplies them as its default land.

- **Connectivity through time.** The present-day mask is held at every date.
  At low stands seas were cut off that are joined today (the Black Sea behind
  the Bosporus at the glacial maximum), and the reverse. Giving the stated
  channels sill depths, and the resolved straits theirs from the bedrock,
  would let a state decide which are open at its own sea level: the basins
  come from one labelling, and each state is then a check over the channels.
- **Several water bodies.** The extended sea level equation of Coulson,
  Al-Attar and Mitrovica treats several water bodies with levels and mass
  budgets of their own, and their exchanges. It needs the bodies and their
  connections found automatically, which `label_water_bodies` begins.
- **Near the coast** the two-arc-minute raster cannot tell a narrow tidal
  channel from a dyke, so bodies within 15 km of the open ocean are joined to
  it; a finer bedrock, or more stated channels, would settle them.

## The viscoelastic operator

`SeaLevelEvolutionOperator` is built, with its derivative and second
derivative by the discrete and the continuous methods, as set out in
[viscoelastic_operator_plan.md](viscoelastic_operator_plan.md); the kernel
drivers (`adjoint_ice_history`, `continuous_adjoint_ice_history`,
`IceKernels`, the decay-time functionals) are the computational layer under
it. Under migrating shorelines the derivatives are complete when the
flotation condition is smoothed (`EarthState.flotation_width`); under the
sharp rule they hold the trajectory's ocean functions and omit the
shorelines' terms, whose size `tools/shoreline_hessian.py` measures. What
remains is the 3D viscosity kernels about a radial model, the time integral
of the pointwise contraction of the adjoint internal variables with the
forward strain, which needs the forward and adjoint fields at every radius
multiplied across degrees; they are kernels only, no forward map leaving the
radial models.

## Glacial-cycle spin-up

A run starts with its internal variables zero, as if the initial load had sat
on an undisturbed elastic Earth, so its first few kiloyears carry a spin-up
bias. Production runs start a glacial cycle earlier or repeat the deglacial
sawtooth; a helper for either is wanted.

## SI entry points for the rheology

A viscosity placed on the radial model (`maxwell_mantle`, `prem_maxwell`) is
given in Pa s and converts with the model, and `viscoelastic_earth` and
`Rheology.from_model` read it in the library's frame, so the common path needs
no conversion. The `Rheology` constructors that take values directly
(`maxwell`, `from_relaxation_time`) still take them in the library's frame, so
user code converts from Pa s and years there. Constructors that accept SI
would remove that step, but changing the meaning of the existing arguments
would break callers silently, so they need new keyword names (`eta_si=`,
`tau_years=`, say) or planetmodel-style tagged quantities. The API is
undecided.

A GIA deck (planetmodel's `GIA` format: `r rho vpv vsv vph vsh eta viscosity`
in SI) has no model type to read it into; planetmodel has one only for mineos
decks. A `GIAModel` there, and `to_deck` for models not read from a deck,
would let published viscosity profiles be loaded and `prem_maxwell` be written
out.

## Horizontal displacement

Horizontal displacement, the response to a load as one spectral object
(`GRDResponse`, `SurfaceDisplacement`, `pyslfp.core.response`), elastic and
viscoelastic, and the observation models with GNSS and the frame of its data
are built, as set out in [response_plan.md](response_plan.md). A projection of
the displacement onto a line of sight, for InSAR, remains.

## Continuous-in-time measurements

Measurements are taken at points in time. A continuous kind, with duals per
unit time, would be folded into the node duals by quadrature in the discrete
adjoint (and excluded from the singular lumps, since a density produces none),
and carried as in-step forcings without a jump in the continuous adjoint. It is
straightforward, but has no known use: decay-time functionals are fits to
discrete samples.

## Optional checks

- Viscoelastic Love numbers against gia1d, the one-dimensional GIA code of
  Crawford (2018), itself benchmarked against Mitrovica's viscoelastic
  Love-number code.
- The time-stepped Love-number histories against the frequency-domain numbers
  of a frozen model by numerical inverse Laplace transform (Talbot), beyond the
  existing steady-periodic check.

## Minor

- `ViscoelasticOperator.effective_love_numbers` assembles its own degree
  systems for each step size rather than sharing those of the flavour cache.

## Other repositories

- sl3dpp: regenerate the stage-5 kernel fixture with the continuous-adjoint
  relaxation-time kernels enabled, and drop the SI anchors for the potential
  and angular-velocity duals from the stage fixtures, now that both codes
  compute in planetmodel's geophysical frame.
