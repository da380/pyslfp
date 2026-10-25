# Open work

What is planned or open, with enough context to pick each item up. A design
that needs more than a few paragraphs has its own file in this directory.

## The models submodule

A vocabulary of physically named load components (grounded ice, glaciers, land
water storage, firn, dynamic topography, ocean density) and an assembler that
turns a list of them into an inversion space, with the load operator,
constraints and GMSL diagnostics derived from each component's role. The design
is in [models_plan.md](models_plan.md); nothing is built yet.

It also completes the harvest of physics from the Heathcote2026 scripts: the
barystatic and steric GMSL split and the regional decomposition after Gregory
et al. (2019) are to land as diagnostics of a set-up. The steric and
mass-conservation operators are already in `linear_operators/loads.py`. The
scripts stay in the repository until the harvest is complete.

## Operator wrappers for the viscoelastic adjoints

The viscoelastic kernel drivers (`adjoint_ice_history`,
`continuous_adjoint_ice_history`, `IceKernels`, `Measurement`,
`ForwardTrajectory`, the decay-time functionals) are a computational layer. The
inversion-facing surface is to be `pygeoinf` linear and nonlinear operators
wrapping them, as `linear_operators` wraps the elastic adjoint, built once
pygeoinf 2 is released; `linear_operators` itself is reworked for pygeoinf 2 at
the same time. Until then the drivers should be kept working and documented,
but not grown into a public API that the wrappers would replace.

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
