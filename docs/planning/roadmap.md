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
WP3, the ready-made set-ups and the rest of the harvest, remains, and starts
once the Caspian below is settled.

It also completes the harvest of physics from the Heathcote2026 scripts: the
barystatic and steric GMSL split and the regional decomposition after Gregory
et al. (2019) are to land as diagnostics of a set-up. The steric and
mass-conservation operators are already in `linear_operators/loads.py`. The
scripts are then replaced by new, simplified ones built on set-ups, tested
against a local record of the current scripts kept outside the repository;
the current scripts leave the repository once the new ones reproduce them.

## Hard-coded regions

The ice sheets are selected by rule (`EarthState.ice_sheet_projection`) and the
glaciers come from the Randolph Glacier Inventory 7.0
(`EarthState.glacier_area_density`). One rectangular mask in `regions.py` still
stands in for a physical definition, and is to be replaced.

**The Caspian box** (36-49.5N, 45.5-55E), excluded from the ocean function by
`EarthState(exclude_caspian=True)`, the default, and carried through the
non-linear solver and the viscoelastic evolution. The underlying fault is that
the ocean function is decided by flotation alone, so every depression below
sea level counts as ocean once resolved: the Caspian, the Dead Sea, Qattara,
Turpan, Death Valley. Two routes were checked and neither is ready as it
stands:

- HydroBASINS, which pyslfp ships, flags endorheic basins, but HydroSHEDS
  treats the Caspian as coastline, so the basins around it are not flagged.
- Connectivity to the open ocean, decided on high-resolution topography
  rather than the model grid (where Gibraltar and the Bosporus close), would
  generalise. The shipped ETOPO 2022 bedrock is at 2', which resolves
  Gibraltar but not the Bosporus, so the Black Sea would need its connection
  stated.

Changing either default changes the Heathcote2026 results. The clean-up
comes first, before WP3 of the models submodule and the paper's revision
runs, so that both use the physical definitions.

## Operator wrappers for the viscoelastic adjoints

The viscoelastic kernel drivers (`adjoint_ice_history`,
`continuous_adjoint_ice_history`, `IceKernels`, `Measurement`,
`ForwardTrajectory`, the decay-time functionals) are a computational layer. The
inversion-facing surface is to be pygeoinf 2 `Operator` and `LinearOperator`
objects wrapping them, as `linear_operators` wraps the elastic adjoint. Until
then the drivers should be kept working and documented, but not grown into a
public API that the wrappers would replace. The port of `linear_operators`
itself is planned in `pygeoinf_v2_port.md`.

## Glacial-cycle spin-up

A run starts with its internal variables zero, as if the initial load had sat
on an undisturbed elastic Earth, so its first few kiloyears carry a spin-up
bias. Production runs start a glacial cycle earlier or repeat the deglacial
sawtooth; a helper for either is wanted.

## SI entry points for the rheology

The `Rheology` constructors take viscosities and relaxation times in the
library's frame, so user code converts from Pa s and years. Constructors that
accept SI would remove that step, but changing the meaning of the existing
arguments would break callers silently, so they need new keyword names
(`eta_si=`, `tau_years=`, say) or planetmodel-style tagged quantities. The API
is undecided.

## Horizontal displacement

The Love number tables carry the tangential numbers (`l_u`, `l_phi`, `l_t`)
and the tangential-traction channel whose adjoint a functional of horizontal
displacement needs, but the sea level solvers return the vertical displacement
only. Horizontal displacement is wanted as an optional output, with its adjoint,
and the repackaging of the solver outputs this implies is best designed
together with the pygeoinf operators.

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
