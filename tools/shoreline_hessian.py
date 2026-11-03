"""The size of the shoreline terms the Hessian omits under migrating
shorelines, measured on a marine cap: a 12 km cap on a 3 km ocean, melting
over ten kiloyears on a slowly relaxing homogeneous body.

The second derivative the library computes holds the trajectory's ocean
functions; the terms it omits are those of the shorelines' own displacement,
which on the grid are the jumps of the gradient when a cell flips between
land and ocean. Along a direction in each argument this compares the held-C
Hessian-vector product with central differences of the gradient and second
differences of an objective across ranges of the perturbation that flip
cells, counting the flips, and then the omitted part against the
Gauss-Newton term of a misfit with a residual of ten per cent of the
response. Run with the surface degree as the argument (16 by default):

    python tools/shoreline_hessian.py 16
"""

import sys
import warnings
import numpy as np
from planetmodel import LayeredIsotropicElastic, RadialMesh, units
from pyshtools import SHGrid, SHCoeffs
from pyslfp.core import EarthModel, EarthState
from pyslfp.ice import BaseIceModel
from pyslfp.love_numbers import Material, fluid_tidal_number, love_numbers
from pyslfp.viscoelastic import Rheology, SeaLevelEvolution, ViscoelasticOperator
from pyslfp.viscoelastic.ice_history import _NodalIceModel

warnings.simplefilter("ignore")

LMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SCALES = units.Scales.geophysical(6.371e6)
TAU = 6.0e6
model = LayeredIsotropicElastic(
    [0.0, 1.0], rho=[1.0], vp=[3.1], vs=[1.3], scales=SCALES
)
mesh = RadialMesh(model, ngll=5, lmax=LMAX)
material = Material(mesh, model)
base_model = EarthModel.from_planet_model(model, LMAX, mesh=mesh)
parameters = base_model.parameters.with_hydrostatic_moments(
    fluid_tidal_number(material, 2)
)
earth = EarthModel(
    LMAX, parameters=parameters, love_numbers=love_numbers(material, LMAX)
)
metre = earth.parameters.length_scale


class Cap(BaseIceModel):
    START, THICKNESS = 10.0, 12000.0

    def get_ice_thickness(self, date, lmax, /, *, grid="DH", sampling=1, extend=True):
        grid, sampling = self.resolve_grid(grid, sampling)
        cap = SHGrid.from_cap(
            15.0, 60.0, -85.0, lmax=lmax, grid=grid, sampling=sampling, extend=extend
        )
        return (
            (self.THICKNESS / self.length_scale)
            * np.clip(date / self.START, 0.0, 1.0)
            * cap
        )


ice_model = Cap(length_scale=metre)
dates = np.linspace(10.0, 0.0, 6)
base_ice = [
    ice_model.get_ice_thickness(
        float(d), LMAX, grid=earth.grid_name, extend=earth.extend
    )
    for d in dates
]
sea0 = earth.constant_grid(3000.0 / metre)
base_rheology = Rheology.from_relaxation_time(material, 10.0 * TAU)


def smooth(seed, amplitude=1.0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(2, LMAX + 1, LMAX + 1))
    for degree in range(LMAX + 1):
        c[:, degree, degree + 1 :] = 0.0
    c[1, :, 0] = 0.0
    return amplitude * earth.expand_coefficient(
        SHCoeffs.from_array(c, normalization=earth.normalization, csphase=earth.csphase)
    )


g = smooth(1)


def pair(a, b):
    return float(earth.integrate(SHGrid.from_array(a.data * b.data, grid=earth.grid)))


def run(eps, u):
    sea = (
        sea0
        if u["sea_level"] is None
        else SHGrid.from_array(sea0.data + eps * u["sea_level"].data, grid=earth.grid)
    )
    ice = (
        base_ice
        if u["ice"] is None
        else [
            SHGrid.from_array(b.data + eps * d.data, grid=earth.grid)
            for b, d in zip(base_ice, u["ice"])
        ]
    )
    rheology = (
        base_rheology
        if u["itau"] is None
        else Rheology(
            material,
            [(base_rheology.mu[0], base_rheology.itau[0] + eps * u["itau"][0])],
        )
    )
    state = EarthState(ice[0], sea, earth, land=None)
    evolution = SeaLevelEvolution(ViscoelasticOperator(rheology, LMAX), state)
    snapshots, _ = evolution.run_ice_history(
        dates,
        _NodalIceModel(dates, ice, length_scale=metre),
        shorelines="migrating",
        record=True,
        rtol=1e-11,
    )
    return evolution, snapshots


def pairing(kernels, u):
    total = 0.0
    if u["ice"] is not None:
        total += sum(pair(kernels.thickness[n], u["ice"][n]) for n in range(len(dates)))
    if u["sea_level"] is not None:
        total += pair(kernels.initial_sea_level, u["sea_level"])
    if u["itau"] is not None:
        total += float(np.sum(kernels.itau[0] * u["itau"][0]))
    return total


def gradient(evolution, snapshots, u):
    kernels = evolution.adjoint_ice_history(
        dates,
        {0.0: g},
        ocean_functions=[s.ocean_function for s in snapshots],
        internal_states=snapshots,
        rtol=1e-11,
    )
    value = pair(g, snapshots[-1].response.field("relative_sea_level_change"))
    return value, pairing(kernels, u)


def oceans(snapshots):
    return np.stack([s.ocean_function.data for s in snapshots])


r = mesh.r
directions = {
    "initial sea level, uniform 100 m": dict(
        ice=None, sea_level=earth.constant_grid(100.0 / metre), itau=None
    ),
    "initial sea level, smooth 100 m": dict(
        ice=None, sea_level=smooth(7, 100.0 / metre), itau=None
    ),
    "ice thickness, smooth 100 m at every date": dict(
        ice=[smooth(8, 100.0 / metre)] * len(dates), sea_level=None, itau=None
    ),
    "viscosity, itau +10% (eta -10%)": dict(
        ice=None, sea_level=None, itau=[0.1 * base_rheology.itau[0]]
    ),
    "mixed: sea level 100 m and itau +10%": dict(
        ice=None,
        sea_level=earth.constant_grid(100.0 / metre),
        itau=[0.1 * base_rheology.itau[0]],
    ),
}
evolution, snapshots = run(0.0, directions["initial sea level, uniform 100 m"])
C_base = oceans(snapshots)
print(
    f"lmax {LMAX}: grid {C_base.shape[1]}x{C_base.shape[2]}; cells flipping during the base run {int((C_base[0] != C_base[-1]).sum())}"
)
for label, u in directions.items():
    J0, G0u = gradient(evolution, snapshots, u)
    tangent, _ = evolution.tangent_linear_ice_history(
        dates,
        snapshots,
        ice_thickness=u["ice"],
        initial_sea_level=u["sea_level"],
        itau=u["itau"],
        record=True,
        rtol=1e-11,
    )
    _, derivatives = evolution.second_order_adjoint_ice_history(
        dates,
        {0.0: g},
        tangent,
        itau=u["itau"],
        ocean_functions=[s.ocean_function for s in snapshots],
        internal_states=snapshots,
        rtol=1e-11,
    )
    Huu = pairing(derivatives, u)
    print(f"\n{label}:  <g, F'u> = {G0u:.4e}   <g, F''[u,u]> held-C = {Huu:.4e}")
    print(
        "   eps   flips(C0) flips(all)   secant of <G,u>   2nd diff of J   (2nd diff - heldC)/|<g,F'u>|   quad/lin at eps"
    )
    for eps in (0.05, 0.1, 0.2, 0.5, 1.0):
        out = []
        for sign in (1.0, -1.0):
            evo, snaps = run(sign * eps, u)
            J, Gu = gradient(evo, snaps, u)
            C = oceans(snaps)
            out.append(
                (J, Gu, int((C[0] != C_base[0]).sum()), int((C != C_base).sum()))
            )
        secant = (out[0][1] - out[1][1]) / (2 * eps)
        second = (out[0][0] - 2 * J0 + out[1][0]) / eps**2
        print(
            f"  {eps:5.2f}   {max(o[2] for o in out):4d}      {max(o[3] for o in out):5d}      {secant: .4e}      {second: .4e}        {(second - Huu)/abs(G0u): .3e}            {abs(0.5 * eps * second / G0u):.3e}"
        )

# ---- against the Gauss-Newton term of a misfit --------------------------
print(
    "\nAgainst the Gauss-Newton term: residual r = 0.1 xi(t_f), misfit |F - d|^2 / 2 in the sea level at the present"
)
xi_base = snapshots[-1].response.field("relative_sea_level_change")
r_grid = SHGrid.from_array(0.1 * xi_base.data, grid=earth.grid)
g = r_grid  # the functional <r, F(.)> now stands for the residual's pairing
print(
    "   direction                                   eps   flips   GN |F'u|^2      <r, F''[u,u]> held-C   <r,F''[u,u]> measured   omitted/GN"
)
for label in (
    "initial sea level, uniform 100 m",
    "ice thickness, smooth 100 m at every date",
    "mixed: sea level 100 m and itau +10%",
):
    u = directions[label]
    tangent, _ = evolution.tangent_linear_ice_history(
        dates,
        snapshots,
        ice_thickness=u["ice"],
        initial_sea_level=u["sea_level"],
        itau=u["itau"],
        record=True,
        rtol=1e-11,
    )
    Fu = tangent[-1].response.field("relative_sea_level_change")
    gn = pair(Fu, Fu)
    _, derivatives = evolution.second_order_adjoint_ice_history(
        dates,
        {0.0: g},
        tangent,
        itau=u["itau"],
        ocean_functions=[s.ocean_function for s in snapshots],
        internal_states=snapshots,
        rtol=1e-11,
    )
    held = pairing(derivatives, u)
    J0 = pair(g, xi_base)
    for eps in (0.5, 1.0):
        vals, flips = [], []
        for sign in (1.0, -1.0):
            evo, snaps = run(sign * eps, u)
            vals.append(pair(g, snaps[-1].response.field("relative_sea_level_change")))
            flips.append(int((oceans(snaps) != C_base).sum()))
        second = (vals[0] - 2 * J0 + vals[1]) / eps**2
        print(
            f"   {label:42s} {eps:4.2f}   {max(flips):3d}    {gn: .4e}      {held: .4e}            {second: .4e}         {abs(second - held)/gn:.3e}"
        )
