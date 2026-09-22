"""Love numbers by degree, their units, the file that holds them, and the
Green's functions built from them.

The generalised Love numbers of Al-Attar et al. (2024), eq. (62), split
the surface load into a traction-like term and a potential-source term:

    u_lm   = h_l sigma_lm + h^u_l zeta^u_lm + h^phi_l zeta^phi_lm,
    phi_lm = k_l sigma_lm + k^u_l zeta^u_lm + k^phi_l zeta^phi_lm,

a true surface load acting through both channels, so h = h^u + h^phi
and k = k^u + k^phi; the tangential numbers l^u, l^phi follow V in the
same way, and h^t, l^t, k^t are the response to the unit external
potential psi = (r/a)^l; the load sums are the properties `h`, `l` and
`k`.  A third load channel, absent from the paper, is a tangential
traction -g zeta^v grad_1 Y_lm, the tangential counterpart of the
traction piece of a surface load, with h^v, l^v, k^v its numbers; it is
what a functional of horizontal displacement needs of the adjoint
problem.  The bilinear form is symmetric, so per degree the surface
response (U, V, phi) to the three load channels is a symmetric matrix
up to the factors of g and of the norm l(l + 1) of grad_1 Y_lm:

    g h^phi = k^u,   h^v = l(l + 1) l^u,   k^v = g l(l + 1) l^phi,

the first being eq. (64) there; `reciprocity_residual` checks all
three.  All numbers are in the units of `scales`, with phi the physical
potential perturbation, negative near added mass; the conventional
dimensionless load numbers are

    h' = h g (2l + 1) / (4 pi G a),   l' = l g (2l + 1) / (4 pi G a),
    k' = -k (2l + 1) / (4 pi G a) - 1,

and the geodetic tidal numbers k^T = k^t, h^T = -g h^t, l^T = -g l^t.
Degree 1 is in the centre-of-mass frame, where the potential channel
vanishes and k' = -1; at degree 0 the tidal numbers vanish, a uniform
external potential being a gauge, while the load numbers do not:
k_0 = -4 pi G a by mass conservation.  The split matters: the adjoint
theory of the sea level equation is written in the generalised numbers,
not in h and k alone.

Five scalars, the axial numbers, describe degree 0 further, for the
component of the rotational feedback along the rotation axis.  A change
in spin rate has a centrifugal potential whose spherical mean is
proportional to r^2, not a constant, and the trace of the inertia
tensor changes with any uniform radial deformation; neither is in the
surface response to a load or to a harmonic tide.  `h_c` and `k_c` are
the surface displacement and potential per unit centrifugal potential
(r/a)^2 at degree 0 (k_c vanishes by the shell theorem), and `m_u`,
`m_phi` and `m_c` are the inertia moments sqrt(4 pi) int rho U r^3 dr,
which is int rho u . x dV, of the degree-0 response to the two load
channels and to the centrifugal potential.  Symmetry of the bilinear
form gives

    m_u = sqrt(4 pi) g a^4 h_c / 2,   m_phi = sqrt(4 pi) a^4 k_c / 2,

checked by `axial_reciprocity_residual`; m_phi vanishes with k_c, a
surface shell attracting nothing inside it.  A table from a file that
predates them has them NaN, and the sea level equation then neglects
the deformation terms of the axial feedback.

The file is plain text, one row per degree from 0, in SI: h per unit
surface density in m^3 kg^-1, k likewise in m^4 kg^-1 s^-2, h_t in
s^2 m^-1 and k_t dimensionless.  Its comment lines name the columns
(`columns: l h_u l_u k_u ...`), record the body's radius, surface
gravity and G, and give the axial numbers (`axial h_c ... k_c ...`),
all of which the reader takes back; a file with no column line is read
in the seven-column layout of the original table,
`l h_u k_u h_phi k_phi h_t k_t`, with the tangential numbers NaN.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import pyshtools as sh
from planetmodel.units import FREQUENCY, Dimensions, Scales

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from planetmodel import Model, RadialMesh

__all__ = [
    "NAMES",
    "AXIAL_NAMES",
    "LEGACY_COLUMNS",
    "LoveNumbers",
    "read_love_numbers",
]

#: The twelve Love numbers, in the order the file writes them.
NAMES = (
    "h_u", "l_u", "k_u",
    "h_phi", "l_phi", "k_phi",
    "h_v", "l_v", "k_v",
    "h_t", "l_t", "k_t",
)  # fmt: skip

#: The five axial numbers of degree 0, in the order the file writes them.
AXIAL_NAMES = ("h_c", "k_c", "m_u", "m_phi", "m_c")

#: The column layout of a file with no line naming its columns.
LEGACY_COLUMNS = ("l", "h_u", "k_u", "h_phi", "k_phi", "h_t", "k_t")

#: Dimensions of each Love number: displacement or potential per unit
#: surface density, or per unit potential.
_PER_LOAD_LENGTH = Dimensions(mass=-1, length=3)
_PER_LOAD_POTENTIAL = Dimensions(mass=-1, length=4, time=-2)
_PER_POTENTIAL_LENGTH = Dimensions(length=-1, time=2)
_DIMENSIONS = {
    "h_u": _PER_LOAD_LENGTH,
    "l_u": _PER_LOAD_LENGTH,
    "k_u": _PER_LOAD_POTENTIAL,
    "h_phi": _PER_LOAD_LENGTH,
    "l_phi": _PER_LOAD_LENGTH,
    "k_phi": _PER_LOAD_POTENTIAL,
    "h_v": _PER_LOAD_LENGTH,
    "l_v": _PER_LOAD_LENGTH,
    "k_v": _PER_LOAD_POTENTIAL,
    "h_t": _PER_POTENTIAL_LENGTH,
    "l_t": _PER_POTENTIAL_LENGTH,
    "k_t": Dimensions(),
    # the axial numbers: h_c like h_t, k_c like k_t, and the moments an
    # inertia (mass length^2) per unit of their forcing
    "h_c": _PER_POTENTIAL_LENGTH,
    "k_c": Dimensions(),
    "m_u": Dimensions(length=4),
    "m_phi": Dimensions(length=4),
    "m_c": Dimensions(mass=1, time=2),
}
_LENGTH = Dimensions(length=1)
_ACCELERATION = Dimensions(length=1, time=-2)
_GRAVITATIONAL_CONSTANT = Dimensions(mass=-1, length=3, time=-2)

#: The comment line recording the body the numbers belong to, in SI.
_BODY_LINE = "body radius {radius:.15e} surface_gravity {g:.15e} G {G:.15e}"

#: The comment line recording the axial numbers, in SI.
_AXIAL_LINE = "axial " + " ".join(f"{name} {{{name}:+.15e}}" for name in AXIAL_NAMES)


@dataclass(frozen=True)
class LoveNumbers:
    """Love numbers by degree, in the units of `scales`; see the module
    docstring for what each is.

    `radius`, `surface_gravity` and `G` are the body's, in the same
    units; `omega` is the frequency of a frozen viscoelastic model, whose
    numbers are complex, and None for an elastic one.  The five axial
    numbers of degree 0 are NaN when unknown.  Built by `from_model`,
    `from_file` or `default`, never by hand in normal use.
    """

    degree: np.ndarray
    h_u: np.ndarray
    l_u: np.ndarray
    k_u: np.ndarray
    h_phi: np.ndarray
    l_phi: np.ndarray
    k_phi: np.ndarray
    h_v: np.ndarray
    l_v: np.ndarray
    k_v: np.ndarray
    h_t: np.ndarray
    l_t: np.ndarray
    k_t: np.ndarray
    radius: float
    surface_gravity: float
    G: float
    _: KW_ONLY
    scales: Scales = Scales.SI
    omega: float | None = None
    h_c: complex = np.nan
    k_c: complex = np.nan
    m_u: complex = np.nan
    m_phi: complex = np.nan
    m_c: complex = np.nan

    def __post_init__(self) -> None:
        n = len(self.degree)
        for name in ("degree",) + NAMES:
            a = np.array(getattr(self, name))
            if a.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: Model,
        lmax: int,
        /,
        *,
        mesh: RadialMesh | None = None,
        ngll: int = 5,
        eps: float = 1e-8,
    ) -> LoveNumbers:
        """The Love numbers of a planetmodel model for every degree from 0
        to `lmax`, in the model's units; see `love_numbers`."""
        from .solver import love_numbers

        return love_numbers(model, lmax, mesh=mesh, ngll=ngll, eps=eps)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        /,
        *,
        lmax: Optional[int] = None,
        radius: Optional[float] = None,
        surface_gravity: Optional[float] = None,
        G: Optional[float] = None,
    ) -> LoveNumbers:
        """A `LoveNumbers` in SI from a file written by `write`, or from a
        seven-column file with no column line.

        Columns the file does not hold are NaN.  The body's radius,
        surface gravity and G are taken from the file's `body` line;
        each keyword given here overrides it, and is NaN where neither
        is available.  `lmax` truncates the table.
        """
        names, body = _read_header(path)
        data = np.loadtxt(path, ndmin=2)
        if names is None:
            if data.shape[1] != len(LEGACY_COLUMNS):
                raise ValueError(
                    f"a file with no 'columns:' line must have "
                    f"{len(LEGACY_COLUMNS)} columns, got {data.shape[1]}"
                )
            names = list(LEGACY_COLUMNS)
        if len(names) != data.shape[1]:
            raise ValueError(
                f"the column line names {len(names)} columns but the file "
                f"has {data.shape[1]}"
            )
        if "l" not in names:
            raise ValueError("the column line must name the degree column 'l'")
        columns = {name: data[:, j] for j, name in enumerate(names)}
        n = data.shape[0]
        values = {name: columns.get(name, np.full(n, np.nan)).copy() for name in NAMES}
        for key, given in (
            ("radius", radius),
            ("surface_gravity", surface_gravity),
            ("G", G),
        ):
            if given is not None:
                body[key] = float(given)
        axial = {name: body.pop(name) for name in AXIAL_NAMES}
        out = cls(columns["l"].astype(int), scales=Scales.SI, **body, **values, **axial)
        return out if lmax is None else out.truncated(lmax)

    @classmethod
    def default(
        cls, *, lmax: Optional[int] = None, refresh: bool = False
    ) -> LoveNumbers:
        """The shipped table, downloaded on first use, in SI: PREM without
        its ocean, degrees 0 to 4096, computed by `from_model` on the
        graded mesh and written by `write`, so it carries every column
        and its body line.

        Args:
            lmax (Optional[int]): Truncate the table at this degree.
            refresh (bool): Delete the cached copy and download it again,
                which is how a table that has changed on Zenodo is picked up.
        """
        from pyslfp.data import ensure_data

        folder = ensure_data("LOVE_NUMBERS", refresh=refresh)
        return cls.from_file(folder / "PREM_4096.dat", lmax=lmax)

    # -- shape --------------------------------------------------------------

    @property
    def lmax(self) -> int:
        """The highest degree held."""
        return int(self.degree[-1])

    @property
    def is_complex(self) -> bool:
        return self.h_u.dtype.kind == "c"

    @property
    def has_axial(self) -> bool:
        """Whether the five axial numbers of degree 0 are known."""
        return all(np.isfinite(getattr(self, name)) for name in AXIAL_NAMES)

    def truncated(self, lmax: int, /) -> LoveNumbers:
        """The same numbers for degrees up to `lmax`."""
        if lmax > self.lmax:
            raise ValueError(
                f"lmax ({lmax}) exceeds the maximum degree held ({self.lmax})"
            )
        keep = self.degree <= lmax
        return replace(
            self, **{name: getattr(self, name)[keep] for name in ("degree",) + NAMES}
        )

    # -- the load sums and the conventional forms ---------------------------

    @property
    def h(self) -> np.ndarray:
        """h = h^u + h^phi, the radial response to a unit surface density."""
        return self.h_u + self.h_phi

    @property
    def l(self) -> np.ndarray:
        """l = l^u + l^phi, the tangential response to a unit surface density."""
        return self.l_u + self.l_phi

    @property
    def k(self) -> np.ndarray:
        """k = k^u + k^phi, the potential response to a unit surface density."""
        return self.k_u + self.k_phi

    def conventional(self) -> dict[str, np.ndarray]:
        """The dimensionless load Love numbers h', l', k' by degree."""
        fac = (2.0 * self.degree + 1.0) / (4.0 * np.pi * self.G * self.radius)
        g = self.surface_gravity
        return {"h": self.h * g * fac, "l": self.l * g * fac, "k": -self.k * fac - 1.0}

    def tidal(self) -> dict[str, np.ndarray]:
        """The geodetic tidal Love numbers k^T, h^T, l^T by degree."""
        g = self.surface_gravity
        return {"k": self.k_t.copy(), "h": -g * self.h_t, "l": -g * self.l_t}

    def reciprocity_residual(self) -> np.ndarray:
        """The largest per degree of the three reciprocity residuals,
        ``abs(k_u - g h_phi)``, ``abs(h_v - l(l+1) l_u)`` and
        ``abs(k_v - g l(l+1) l_phi)``, each relative to the size of the
        numbers in it; zero where all vanish (degree 1 for the first,
        degree 0 for the others), and NaN for a table that lacks the
        tangential columns."""
        g = self.surface_gravity
        k2 = self.degree * (self.degree + 1.0)
        pairs = (
            (self.k_u, g * self.h_phi, self.k_phi),
            (self.h_v, k2 * self.l_u, self.h_u),
            (self.k_v, g * k2 * self.l_phi, self.k_phi),
        )
        worst = np.zeros(len(self.degree))
        for x, y, z in pairs:
            scale = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))
            safe = np.where(scale > 0.0, scale, 1.0)
            worst = np.maximum(worst, np.where(scale > 0.0, np.abs(x - y) / safe, 0.0))
        return worst

    def axial_reciprocity_residual(self) -> float:
        """The larger of the two relative residuals of the axial numbers,
        ``abs(m_u - sqrt(4 pi) g a^4 h_c / 2)`` against the size of m_u
        and ``abs(m_phi - sqrt(4 pi) a^4 k_c / 2)`` against the size of
        m_u again, m_phi and k_c being zero in exact arithmetic; NaN for
        a table without the axial numbers."""
        if not self.has_axial:
            return np.nan
        half = 0.5 * np.sqrt(4.0 * np.pi) * self.radius**4
        scale = abs(self.m_u)
        return max(
            abs(self.m_u - half * self.surface_gravity * self.h_c) / scale,
            abs(self.m_phi - half * self.k_c) / scale,
        )

    # -- units and files ----------------------------------------------------

    def converted(self, scales: Scales, /) -> LoveNumbers:
        """The same numbers under other scales."""
        if scales == self.scales:
            return self
        changes = {}
        for name, dims in _DIMENSIONS.items():
            ratio = self.scales.factor(dims) / scales.factor(dims)
            changes[name] = getattr(self, name) * ratio
        for name, dims in (
            ("radius", _LENGTH),
            ("surface_gravity", _ACCELERATION),
            ("G", _GRAVITATIONAL_CONSTANT),
        ):
            changes[name] = (
                getattr(self, name) * self.scales.factor(dims) / scales.factor(dims)
            )
        if self.omega is not None:
            changes["omega"] = (
                self.omega * self.scales.factor(FREQUENCY) / scales.factor(FREQUENCY)
            )
        return replace(self, scales=scales, **changes)

    def in_si(self) -> LoveNumbers:
        return self.converted(Scales.SI)

    def write(self, path: str | Path, /) -> None:
        """Write the file: rows from degree 0 to lmax, the twelve numbers in
        SI with a column line naming them and a body line.  Refused for
        complex numbers and for a table not starting at degree 0."""
        if self.is_complex:
            raise ValueError("the file holds real Love numbers")
        if not np.array_equal(self.degree, np.arange(self.lmax + 1)):
            raise ValueError("the file needs every degree from 0 to lmax")
        si = self.in_si()
        cols = np.column_stack([si.degree] + [getattr(si, n) for n in NAMES])
        header = (
            "Love numbers, SI, physical-potential sign convention\n"
            "columns: l " + " ".join(NAMES) + "\n"
            "load columns per unit surface density, tidal columns per unit "
            "external potential (r/a)^l, degree 1 in the centre-of-mass frame\n"
            + _BODY_LINE.format(radius=si.radius, g=si.surface_gravity, G=si.G)
        )
        if si.has_axial:
            header += "\n" + _AXIAL_LINE.format(
                **{name: getattr(si, name) for name in AXIAL_NAMES}
            )
        np.savetxt(path, cols, fmt=["%6d"] + ["%+.15e"] * len(NAMES), header=header)

    # -- Green's functions --------------------------------------------------

    def displacement_greens_function(
        self, angle: float, /, *, lmax: Optional[int] = None
    ) -> float:
        """The vertical displacement per unit point mass at angular
        separation `angle` (radians), summed to `lmax` with a Gaussian
        taper, in the units of `scales`."""
        return self._greens_function(angle, lmax=lmax, displacement=True)

    def potential_greens_function(
        self, angle: float, /, *, lmax: Optional[int] = None
    ) -> float:
        """The potential perturbation per unit point mass at angular
        separation `angle` (radians), summed to `lmax` with a Gaussian
        taper, in the units of `scales`."""
        return self._greens_function(angle, lmax=lmax, displacement=False)

    def _greens_function(
        self, angle: float, /, *, lmax: Optional[int] = None, displacement: bool = True
    ) -> float:
        calc_lmax = lmax if lmax is not None else self.lmax
        x = np.cos(angle)
        ps = sh.legendre.PLegendre(calc_lmax, x)
        degrees = np.arange(calc_lmax + 1)
        numbers = self.h[: calc_lmax + 1] if displacement else self.k[: calc_lmax + 1]
        smoothing = np.exp(-10 * (degrees**2) / calc_lmax**2)
        terms = (
            (2 * degrees + 1) * numbers * smoothing * ps / (4 * np.pi * self.radius**2)
        )
        return float(np.sum(terms))

    def plot_greens_functions(
        self, /, *, lmax: Optional[int] = None, n_points: int = 181
    ) -> Tuple[Figure, np.ndarray]:
        """The displacement and geoid Green's functions against angular
        separation, on two axes."""
        import matplotlib.pyplot as plt

        calc_lmax = lmax if lmax is not None else self.lmax
        angles_deg = np.linspace(1e-4, 180, n_points)
        angles_rad = np.deg2rad(angles_deg)
        g_disp = [
            self.displacement_greens_function(a, lmax=calc_lmax) for a in angles_rad
        ]
        g_pot = [
            self.potential_greens_function(a, lmax=calc_lmax) / self.surface_gravity
            for a in angles_rad
        ]

        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True, layout="tight")
        axes[0].plot(angles_deg, g_disp, "b-")
        axes[0].set_title("Displacement Green's function", fontsize=20)
        axes[0].set_ylabel("Length per unit mass", fontsize=20)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[1].plot(angles_deg, g_pot, "r-")
        axes[1].set_title("Potential Green's function", fontsize=20)
        axes[1].set_ylabel("Length per unit mass", fontsize=20)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].set_xlabel("Angular separation (degrees)", fontsize=20)
        axes[1].set_xlim(0, 180)
        return fig, axes

    def plot_greens_functions_split(
        self,
        /,
        *,
        split_angle: float = 20.0,
        lmax: Optional[int] = None,
        n_points: int = 300,
    ) -> Tuple[Figure, np.ndarray]:
        """The Green's functions on a broken axis: the near field to
        `split_angle` degrees on the left, the far field on the right."""
        import matplotlib.pyplot as plt

        calc_lmax = lmax if lmax is not None else self.lmax
        angles_deg = np.linspace(1e-4, 180, n_points)
        angles_rad = np.deg2rad(angles_deg)
        g_disp = np.array(
            [self.displacement_greens_function(a, lmax=calc_lmax) for a in angles_rad]
        )
        g_geoid = np.array(
            [
                -self.potential_greens_function(a, lmax=calc_lmax)
                / self.surface_gravity
                for a in angles_rad
            ]
        )

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            gridspec_kw={"width_ratios": [1, 3], "wspace": 0.05},
            constrained_layout=True,
        )
        fig.supxlabel("Angular separation (degrees)", fontsize=20)
        axes[0, 0].set_ylabel("Displacement", fontsize=20)
        axes[1, 0].set_ylabel("Geoid anomaly", fontsize=20)
        near = angles_deg < split_angle
        far = ~near
        axes[0, 0].plot(angles_deg[near], g_disp[near], "b-")
        axes[0, 1].plot(angles_deg[far], g_disp[far], "b-")
        axes[0, 0].set_title("Near field", fontsize=20)
        axes[0, 1].set_title("Far field", fontsize=20)
        axes[1, 0].plot(angles_deg[near], g_geoid[near], "r-")
        axes[1, 1].plot(angles_deg[far], g_geoid[far], "r-")
        for i in range(2):
            axes[i, 0].set_xlim(0, split_angle)
            axes[i, 1].set_xlim(split_angle, 180)
            for j in range(2):
                axes[i, j].grid(True, linestyle=":", alpha=0.6)
        for ax in axes[0, :]:
            ax.tick_params(axis="x", labelbottom=False)
        return fig, axes

    def __repr__(self) -> str:
        kind = "complex " if self.is_complex else ""
        at = "" if self.omega is None else f", omega={self.omega:g}"
        return (
            f"LoveNumbers({kind}degrees {int(self.degree[0])}..{self.lmax}, "
            f"{self.scales!r}{at})"
        )


def _read_header(path: str | Path) -> tuple[list[str] | None, dict[str, float]]:
    """The column names of a file's `columns:` line (None without one), the
    radius, surface gravity and G of its `body` line and the axial numbers
    of its `axial` line (NaN without)."""
    names = None
    body = {"radius": np.nan, "surface_gravity": np.nan, "G": np.nan}
    body.update({name: np.nan for name in AXIAL_NAMES})
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            tokens = line[1:].split()
            if tokens[:1] == ["columns:"]:
                names = tokens[1:]
            elif tokens[:1] == ["body"] and len(tokens) == 7:
                body["radius"] = float(tokens[2])
                body["surface_gravity"] = float(tokens[4])
                body["G"] = float(tokens[6])
            elif tokens[:1] == ["axial"]:
                pairs = dict(zip(tokens[1::2], tokens[2::2]))
                for name in AXIAL_NAMES:
                    if name in pairs:
                        body[name] = float(pairs[name])
    return names, body


def read_love_numbers(path: str | Path, /) -> LoveNumbers:
    """`LoveNumbers.from_file(path)`."""
    return LoveNumbers.from_file(path)
