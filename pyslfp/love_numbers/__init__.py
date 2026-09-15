"""Love numbers: the quasi-static loading and tidal problem of a spherically
layered body, solved degree by degree on a planetmodel radial mesh.

A spherically symmetric, self-gravitating, hydrostatically pre-stressed
body responds to a surface load or an external tidal potential by a
displacement and a perturbation of its gravitational potential.  This
sub-package solves that problem by the reduced weak form of Al-Attar &
Tromp (2014), Appendix D, with transversely isotropic elasticity, fluid
regions characterised by the potential alone, and the exterior closed by
Dirichlet-to-Neumann terms; `assembly` states the form.  The results are
the generalised Love numbers of Al-Attar et al. (2024), the conventional
load and tidal numbers derived from them, the radial solutions
themselves, and the file the sea level solver reads; `table` states the
conventions.

The material enters through `Material`, which reads a model on a mesh
once: density, its gradient, gravity, fluidity and the five moduli at
the nodes.  The model is any planetmodel model holding density and what
`planetmodel.moduli` reads the five moduli from, real or complex: a
viscoelastic body is a model frozen at a frequency by `planetmodel.frozen`,
and its Love numbers are complex.  Everything is in the model's units
with the model's G; `LoveNumbers.converted` or `write` converts at the
end.

    love = LoveNumbers.from_model(PREM(ocean=False), 256)
    love.conventional()["h"]                  # h' by degree
    love.write("love.dat")                    # for EarthModel(love_number_file=...)
    LoveNumbers.default()                     # the shipped PREM table
    solve_degree(model, 2, forcing="tide").evaluate(radii)
    plot_love_numbers(love); plot_degree_solution(solve_degree(model, 2))
"""

from .assembly import DegreeSystem
from .material import Material, NodalModuli, nodal_moduli
from .plot import plot_degree_solution, plot_love_numbers
from .solver import FORCINGS, DegreeSolution, graded_mesh, love_numbers, solve_degree
from .table import LEGACY_COLUMNS, NAMES, LoveNumbers, read_love_numbers

__all__ = [
    "Material",
    "NodalModuli",
    "nodal_moduli",
    "DegreeSystem",
    "FORCINGS",
    "DegreeSolution",
    "solve_degree",
    "graded_mesh",
    "LoveNumbers",
    "love_numbers",
    "read_love_numbers",
    "NAMES",
    "LEGACY_COLUMNS",
    "plot_love_numbers",
    "plot_degree_solution",
]
