"""The degree-l system of the loading problem: local matrices, dofs, band.

For one spherical-harmonic degree l, with k^2 = l(l + 1), the unknowns
are the radial and tangential displacement scalars U, V and the
potential perturbation phi on solid nodes, and phi alone on fluid nodes,
where the deformation is characterised by the potential with the
stratification term g^-1 (d rho / dr) phi phi' r^2 and explicit
fluid-solid interface couplings.  The bilinear form is eq. (D76) of
Al-Attar & Tromp (2014) with its elastic block written for a
transversely isotropic medium: with X = r d_r U, Y = 2U - k^2 V and
S = U + r d_r V - V,

    C X X' + (A - N) Y Y' + F (X Y' + Y X') + k^2 L S S'
      + k^2 (k^2 - 2) N V V',

which is the isotropic form under A = C = kappa + 4 mu / 3,
F = kappa - 2 mu / 3, L = N = mu.  The exterior is closed by the
Dirichlet-to-Neumann term (l + 1) a phi phi' / 4 pi G at the surface
and, where the domain starts above the centre, by the interior
continuation l r phi phi' / 4 pi G at the bottom.  For l = 1 the surface
potential dof is removed (the centre-of-mass frame); for l = 0 there is
no tangential dof, fluid regions are mu = 0 elastic media, and U(0) = 0
is imposed at a centre.  Degrees above zero are solved on the sub-mesh
where the solution is not negligible, from `RadialMesh.start_element`.

The system is symmetric and stored as an upper band; it is real for
real moduli and complex symmetric otherwise.  Everything is in the
material's units.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import cho_solve_banded, cholesky_banded, solve_banded

from .material import Material, NodalModuli

__all__ = ["DegreeSystem"]

COMPONENTS = ("U", "V", "phi")


def _solid_local(
    l: int,
    r: np.ndarray,
    rho: np.ndarray,
    g: np.ndarray,
    m: NodalModuli,
    w: np.ndarray,
    jac: np.ndarray,
    D: np.ndarray,
    G: float,
) -> np.ndarray:
    """Local matrices of solid elements, shape (ne, 3n, 3n); dofs (U, V, phi)
    per node, node-major."""
    ne, n = r.shape
    k2 = l * (l + 1.0)
    iU = 3 * np.arange(n)
    iV = iU + 1
    iP = iU + 2
    dtype = np.result_type(m.A, float)
    B = r[:, :, None] * D[None, :, :] / jac[:, None, None]  # r d_r of node values
    I = np.broadcast_to(np.eye(n), (ne, n, n))
    Wq = w[None, :] * jac[:, None]
    L = np.zeros((ne, 3 * n, 3 * n), dtype=dtype)

    def quad(P: np.ndarray, coef: np.ndarray) -> np.ndarray:
        return np.einsum("eqi,eq,eqj->eij", P, coef * Wq, P)

    X = np.zeros((ne, n, 3 * n))
    X[:, :, iU] = B
    Y = np.zeros((ne, n, 3 * n))
    Y[:, :, iU] = 2.0 * I
    Y[:, :, iV] = -k2 * I
    S = np.zeros((ne, n, 3 * n))
    S[:, :, iU] = I
    S[:, :, iV] = B - I
    L += quad(X, m.C)
    L += quad(Y, m.A - m.N)
    XF = np.einsum("eqi,eq,eqj->eij", X, m.F * Wq, Y)
    L += XF + np.transpose(XF, (0, 2, 1))
    L += k2 * quad(S, m.L)
    L[:, iV, iV] += k2 * (k2 - 2.0) * m.N * Wq
    L[:, iU, iU] += 4.0 * rho * (np.pi * G * rho * r - g) * r * Wq
    cUV = k2 * rho * g * r * Wq
    L[:, iU, iV] += cUV
    L[:, iV, iU] += cUV
    cVP = k2 * rho * r * Wq
    L[:, iV, iP] += cVP
    L[:, iP, iV] += cVP
    M = (rho * r * r * w[None, :])[:, :, None] * D[None, :, :]
    L[:, iU[:, None], iP[None, :]] += M
    L[:, iP[:, None], iU[None, :]] += np.transpose(M, (0, 2, 1))
    ifpg = 1.0 / (4.0 * np.pi * G)
    KP = np.einsum("qi,eq,qj->eij", D, w[None, :] * r * r, D) / jac[:, None, None]
    L[:, iP[:, None], iP[None, :]] += ifpg * KP
    L[:, iP, iP] += ifpg * k2 * Wq
    return L


def _fluid_local(
    l: int,
    r: np.ndarray,
    drho: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
    jac: np.ndarray,
    D: np.ndarray,
    G: float,
) -> np.ndarray:
    """Local matrices of fluid elements, shape (ne, 3n, 3n); only phi is live."""
    ne, n = r.shape
    k2 = l * (l + 1.0)
    iP = 3 * np.arange(n) + 2
    ifpg = 1.0 / (4.0 * np.pi * G)
    Wq = w[None, :] * jac[:, None]
    L = np.zeros((ne, 3 * n, 3 * n))
    KP = np.einsum("qi,eq,qj->eij", D, w[None, :] * r * r, D) / jac[:, None, None]
    L[:, iP[:, None], iP[None, :]] += ifpg * KP
    L[:, iP, iP] += ifpg * k2 * Wq
    L[:, iP, iP] += _stratification(r, drho, g) * Wq
    return L


def _stratification(r: np.ndarray, drho: np.ndarray, g: np.ndarray) -> np.ndarray:
    """r^2 (d rho / dr) / g, zero where g vanishes (the centre)."""
    out = np.zeros(np.shape(r))
    ok = g > 0.0
    np.divide(r * r * drho, g, out=out, where=ok)
    return out


class DegreeSystem:
    """The assembled degree-l system of a `Material`, ready to solve.

    `dof[node, component]` is the global dof of each component (0, 1, 2)
    = (U, V, phi) at each global node, -1 where absent; `ndof` their
    number; `first_element` the start of the sub-mesh; `band` the upper
    band storage with `band[kd + i - j, j] = A[i, j]` for i <= j.
    `solve` takes one or several right-hand sides; `load`, `tangential`
    and `tide` build them.
    """

    def __init__(self, material: Material, l: int, *, eps: float = 1e-8) -> None:
        if not isinstance(material, Material):
            raise TypeError(f"expected a Material, got {type(material).__name__}")
        if l < 0:
            raise ValueError("degree must be non-negative")
        self.material = material
        self.l = int(l)
        mesh = material.mesh
        self.mesh = mesh
        self.first_element = mesh.start_element(l, eps=eps) if l > 0 else 0
        self.dof, self.ndof = self._dof_map()
        self.kd = 3 * (mesh.ngll - 1) + 2
        self.band = self._assemble()
        self._cholesky = None
        self._cholesky_failed = False

    # -- numbering ------------------------------------------------------------

    def _dof_map(self) -> tuple[np.ndarray, int]:
        mesh, l, e0 = self.mesh, self.l, self.first_element
        nglob = mesh.nglob
        g0 = int(mesh.gmap[e0, 0])
        solid = np.zeros(nglob, dtype=bool)
        for e in range(e0, mesh.nspec):
            if l == 0 or not self.material.fluid[e]:
                solid[mesh.gmap[e]] = True
        comps = (0, 2) if l == 0 else (0, 1, 2)
        dof = -np.ones((nglob, 3), dtype=int)
        nd = 0
        for gn in range(g0, nglob):
            for c in comps if solid[gn] else (2,):
                dof[gn, c] = nd
                nd += 1

        def drop(index: int) -> None:
            nonlocal nd
            dof[dof > index] -= 1
            nd -= 1

        if l == 1:
            k = dof[nglob - 1, 2]
            dof[nglob - 1, 2] = -1
            drop(k)
        if l == 0 and float(mesh.left[e0]) == 0.0:
            k = dof[g0, 0]
            dof[g0, 0] = -1
            drop(k)
        return dof, nd

    # -- assembly ---------------------------------------------------------------

    def _assemble(self) -> np.ndarray:
        mat, mesh, l, kd = self.material, self.mesh, self.l, self.kd
        e0 = self.first_element
        G = mat.G
        D, w = mesh.deriv, mesh.w
        dtype = complex if mat.is_complex else float
        band = np.zeros((kd + 1, self.ndof), dtype=dtype)
        elements = np.arange(e0, mesh.nspec)
        fluid = mat.fluid[elements] & (l > 0)
        rows, cols, vals = [], [], []

        def scatter(es: np.ndarray, local: np.ndarray) -> None:
            gd = self.dof[mesh.gmap[es]].reshape(len(es), -1)
            GA, GB = gd[:, :, None], gd[:, None, :]
            sel = (GA >= 0) & (GB >= 0) & (GA <= GB) & (local != 0.0)
            rows.append(np.broadcast_to(kd + GA - GB, local.shape)[sel])
            cols.append(np.broadcast_to(GB, local.shape)[sel])
            vals.append(local[sel])

        es = elements[~fluid]
        if es.size:
            m = mat.moduli
            sub = type(m)(m.A[es], m.C[es], m.F[es], m.L[es], m.N[es])
            scatter(
                es,
                _solid_local(
                    l, mesh.r[es], mat.rho[es], mat.g[es], sub, w, mesh.jac[es], D, G
                ),
            )
        es = elements[fluid]
        if es.size:
            scatter(
                es,
                _fluid_local(
                    l, mesh.r[es], mat.drho[es], mat.g[es], w, mesh.jac[es], D, G
                ),
            )

        def point(i: int, j: int, v: float) -> None:
            if i < 0 or j < 0:
                return
            if i > j:
                i, j = j, i
            rows.append(np.array([kd + i - j]))
            cols.append(np.array([j]))
            vals.append(np.array([v]))

        # fluid-solid interfaces: +/- rho_f (g U U' + phi U' + U phi') r^2
        for e, gn, rt, gt, rf, sgn in self._interfaces():
            point(self.dof[gn, 0], self.dof[gn, 0], sgn * rf * gt * rt * rt)
            point(self.dof[gn, 0], self.dof[gn, 2], sgn * rf * rt * rt)

        # Dirichlet-to-Neumann closures
        ifpg = 1.0 / (4.0 * np.pi * G)
        top = int(mesh.gmap[-1, -1])
        if l != 1:
            point(self.dof[top, 2], self.dof[top, 2], (l + 1) * mat.radius * ifpg)
        if mesh.left[e0] > 0.0:
            bot = int(mesh.gmap[e0, 0])
            point(self.dof[bot, 2], self.dof[bot, 2], l * float(mesh.left[e0]) * ifpg)

        np.add.at(
            band, (np.concatenate(rows), np.concatenate(cols)), np.concatenate(vals)
        )
        return band

    def _interfaces(self) -> Iterator[tuple[int, int, float, float, float, float]]:
        """(element below, node, r, g, rho_fluid, sign) at each fluid-solid
        interface of the active sub-mesh; the sign is + with the fluid below."""
        mat, mesh, e0 = self.material, self.mesh, self.first_element
        if self.l == 0:
            return
        flu = mat.fluid
        for e in range(e0, mesh.nspec - 1):
            if flu[e] == flu[e + 1]:
                continue
            gn = int(mesh.gmap[e, -1])
            rt, gt = float(mesh.r[e, -1]), float(mat.g[e, -1])
            if flu[e]:
                yield e, gn, rt, gt, float(mat.rho[e, -1]), +1.0
            else:
                yield e, gn, rt, gt, float(mat.rho[e + 1, 0]), -1.0

    # -- right-hand sides ---------------------------------------------------

    def load(self, *, part: str = "both") -> np.ndarray:
        """The surface-load force of a unit surface density: the traction
        piece -g a^2 on U'(a) (`part="force"`), the attraction piece -a^2
        on phi'(a) (`part="potential"`), or both.  At l = 1 the potential
        piece vanishes with the removed dof."""
        if part not in ("force", "potential", "both"):
            raise ValueError(
                f"part must be 'force', 'potential' or 'both', " f"got {part!r}"
            )
        b = np.zeros(self.ndof)
        top = int(self.mesh.gmap[-1, -1])
        a = self.material.radius
        if part in ("force", "both"):
            b[self.dof[top, 0]] = -self.material.surface_gravity * a * a
        if part in ("potential", "both") and self.dof[top, 2] >= 0:
            b[self.dof[top, 2]] = -a * a
        return b

    def tangential(self) -> np.ndarray:
        """The force of a unit tangential traction at the surface, -g grad_1
        Y_lm per unit amplitude in the units of a surface density, so that
        it stands beside the traction piece of `load`: the V dof takes
        -g a^2 l(l + 1), the l(l + 1) being the norm of grad_1 Y_lm.  Zero
        at l = 0, where there is no tangential dof."""
        b = np.zeros(self.ndof)
        top = int(self.mesh.gmap[-1, -1])
        i = self.dof[top, 1]
        if i >= 0:
            a = self.material.radius
            b[i] = -self.material.surface_gravity * a * a * self.l * (self.l + 1.0)
        return b

    def tide(self, *, power: int | None = None) -> np.ndarray:
        """The force of a unit external potential psi = (r/a)^power, the
        power defaulting to l, the harmonic tide: the known psi moved to
        the right-hand side, term by term of the bilinear form, with the
        fluid density perturbation responding to the total potential.  A
        constant potential (power 0) is a gauge and gives zero; at l = 1
        the centre-of-mass constraint leaves a non-trivial response.  The
        one non-harmonic case in use is power 2 at l = 0, the spherical
        mean of the centrifugal potential of a change in spin rate, which
        is a body force alone: at l = 0 there is no tangential dof and a
        fluid region is a mu = 0 elastic medium."""
        mat, mesh, l, e0 = self.material, self.mesh, self.l, self.first_element
        p = l if power is None else int(power)
        if p < 0:
            raise ValueError("power must be non-negative")
        b = np.zeros(self.ndof)
        if p == 0:
            return b
        a = mat.radius
        k2 = l * (l + 1.0)
        for e in range(e0, mesh.nspec):
            r = mesh.r[e]
            Wq = mesh.w * float(mesh.jac[e])
            psi = (r / a) ** p
            dpsi = (p / a) * (r / a) ** (p - 1)
            gd = self.dof[mesh.gmap[e]]
            if mat.fluid[e] and l > 0:
                b[gd[:, 2]] -= Wq * _stratification(r, mat.drho[e], mat.g[e]) * psi
            else:
                # at l = 0 the centre has no U dof
                live = gd[:, 0] >= 0
                b[gd[live, 0]] -= (Wq * mat.rho[e] * r * r * dpsi)[live]
                if l > 0:
                    b[gd[:, 1]] -= k2 * Wq * mat.rho[e] * r * psi
        for e, gn, rt, gt, rf, sgn in self._interfaces():
            b[self.dof[gn, 0]] -= sgn * rf * rt * rt * (rt / a) ** p
        return b

    # -- solving ----------------------------------------------------------------

    def solve(self, b: ArrayLike) -> np.ndarray:
        """x with A x = b, for b of shape (ndof,) or (ndof, k).

        The system is scaled symmetrically by its diagonal first, which
        equalises the units of the displacement and potential dofs.  A
        real system is then solved by banded Cholesky, falling back to
        banded LU should that fail; a complex system by banded LU.
        """
        b = np.asarray(b)
        if b.shape[0] != self.ndof:
            raise ValueError(f"expected {self.ndof} rows, got {b.shape[0]}")
        b2 = b if b.ndim == 2 else b[:, None]
        x = None
        if not self.material.is_complex and not self._cholesky_failed:
            x = self._solve_cholesky(b2)
        if x is None:
            x = self._solve_lu(b2)
        return x if b.ndim == 2 else x[:, 0]

    def _solve_cholesky(self, b2: np.ndarray) -> np.ndarray | None:
        """The Cholesky solve, or None when the factorisation is not available."""
        kd, band = self.kd, self.band
        if self._cholesky is None:
            d = band[kd].real
            if not np.all(d > 0.0):
                self._cholesky_failed = True
                return None
            s = 1.0 / np.sqrt(d)
            scaled = _scale_band(band, s)
            try:
                c = cholesky_banded(scaled, lower=False)
            except np.linalg.LinAlgError:
                self._cholesky_failed = True
                return None
            self._cholesky = (c, s)
        c, s = self._cholesky
        return s[:, None] * cho_solve_banded((c, False), s[:, None] * b2)

    def _solve_lu(self, b2: np.ndarray) -> np.ndarray:
        kd, band = self.kd, self.band
        n = band.shape[1]
        d = np.abs(band[kd])
        s = 1.0 / np.sqrt(np.where(d > 0.0, d, 1.0))
        scaled = _scale_band(band, s)
        full = np.zeros((2 * kd + 1, n), dtype=band.dtype)
        full[: kd + 1] = scaled
        for m in range(kd):
            u = kd - m
            if u < n:
                full[kd + u, : n - u] = scaled[m, u:]
        return s[:, None] * solve_banded((kd, kd), full, s[:, None] * b2)

    # -- views --------------------------------------------------------------------

    def surface_dof(self, component: str) -> int:
        """The dof of U, V or phi at the surface node, -1 when absent."""
        top = int(self.mesh.gmap[-1, -1])
        return int(self.dof[top, COMPONENTS.index(component)])

    def expand(self, x: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nodal arrays (U, V, phi) of shape (nspec, ngll) from a solution
        vector: absent components and the sub-mesh below the first
        element are zero.  Above degree zero the displacement of a fluid
        region is not determined, so U and V are NaN throughout its
        elements, their boundary nodes included; the solid side of a
        fluid-solid boundary keeps its values in the solid element."""
        x = np.asarray(x)
        out = []
        for c in range(3):
            full = np.zeros(self.mesh.nglob, dtype=x.dtype)
            live = self.dof[:, c] >= 0
            full[live] = x[self.dof[live, c]]
            nodal = full[self.mesh.gmap]
            if c < 2 and self.l > 0:
                nodal[self.material.fluid] = np.nan
            out.append(nodal)
        return tuple(out)

    def __repr__(self) -> str:
        return (
            f"DegreeSystem(l={self.l}, {self.ndof} dofs from element "
            f"{self.first_element} of {self.mesh.nspec})"
        )


def _scale_band(band: np.ndarray, s: np.ndarray) -> np.ndarray:
    """S[i, j] = s_i band[i, j] s_j in the same upper storage."""
    kd, n = band.shape[0] - 1, band.shape[1]
    out = np.zeros_like(band)
    for m in range(kd + 1):
        u = kd - m
        if u < n:
            js = np.arange(u, n)
            out[m, u:] = band[m, u:] * s[js] * s[js - u]
    return out
