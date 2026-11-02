# Open work

What is planned or open, with enough context to pick each item up. A design
that needs more than a few paragraphs has its own file in this directory.

## The models submodule

A vocabulary of physically named load components (grounded ice, glaciers, land
water storage, firn, dynamic topography, ocean density) and an assembler that
turns a list of them into an inversion space, with the load operator,
constraints and GMSL diagnostics derived from each component's role. The design
is in [models_plan.md](models_plan.md). The component record and the factories
(WP1) are built; the glacier factory waits for the glacier inventory below.

It also completes the harvest of physics from the Heathcote2026 scripts: the
barystatic and steric GMSL split and the regional decomposition after Gregory
et al. (2019) are to land as diagnostics of a set-up. The steric and
mass-conservation operators are already in `linear_operators/loads.py`. The
scripts stay in the repository until the harvest is complete.

## Hard-coded regions

Two rectangular masks in `regions.py` stand in for physical definitions, and
are to be replaced.

**The glacier box** (30-70N, 180-270E). The ice-ng models carry present-day
ice outside the ice sheets, and a synthetic inversion whose prior has a
constant standard deviation over the ice gives those small regions an
unphysically large signal, so they were cut out. The aim was right but the box
is not: it is the default of `EarthState.ice_projection` (`exclude_glaciers`),
and through that reaches `ice_projection_operator`, the ice altimetry points,
the "Glaciers" region name, the Heathcote2026 ice priors and the models
`ice_sheets` component. The replacement has two parts:

- **The ice sheets, defined by rule.** The IMBIE (Antarctic) and Mouginot
  (Greenland) basin outlines alone do not do: at lmax 64, 802 of 2186 ice
  cells fall outside them, 686 south of 60S (ice shelves, which the IMBIE
  basins leave out, and margin cells whose centres fall outside the
  outlines), 59 at Greenland's margins and 57 elsewhere (24 of them in the
  box). A rule along the lines of all ice south of 60S plus the ice of
  Greenland, margins included, is needed, and is to be settled.
- **The glaciers, from the Randolph Glacier Inventory 7.0** (NSIDC-0770),
  packaged for the pyslfp Zenodo record like the other datasets, as
  simplified glacier-complex outlines or a fine gridded glacier fraction,
  with the first-order regions for grouping. Whether the Greenland and
  Antarctic peripheral glaciers count as glaciers, as in RGI, or as ice
  sheet is to be decided with the ice-sheet rule.

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

Changing either default changes the Heathcote2026 results, so the timing is
tied to the paper's revision runs.

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
