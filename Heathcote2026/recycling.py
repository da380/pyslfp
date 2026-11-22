"""
Reuse of Krylov information between conjugate-gradient solves (Heathcote et
al., 2026): a prototype.

The inversions solve the data-space normal equations, N x = d with N = A Q
A* + R, many times with the same N: for the posterior expectation, for each
row of a quantity of interest, for each posterior sample. A conjugate-
gradient solve produces search directions p_1, ..., p_k and their images N
p_i at no cost beyond the solve itself. They hold what the solve learnt of
N, and above all of the directions in which a first-level preconditioner P
is poor, since those are the ones the solve spent its iterations on.

`RecycledSolver` keeps the directions of every solve it makes, as the
columns of S, with Y = N S, and uses them in the next in two ways:

- the first iterate is the Galerkin solution on their span, x_0 = S (S* N
  S)^-1 S* d, so the first residual is orthogonal to S;
- the preconditioner is the limited-memory one of Gratton, Sartenaer and
  Tshimanga (2011; after Morales and Nocedal, 2000, and Tshimanga et al.,
  2008),

      H = (I - S G^-1 Y*) P (I - Y G^-1 S*) + S G^-1 S*,   G = S* N S,

  which is the inverse of N on the span of S and P, projected, off it.

A solve whose right-hand side lies in the span costs nothing; one that does
not is left with the part of N's spectrum the earlier solves did not meet.
Nothing is assumed of N or P but that both are self-adjoint and positive
definite, and no application of N is made beyond those of the solves.

The vectors are held as components, 2 k n numbers for k directions on n
data. The data space of these experiments is small (tens of thousands), so
hundreds of directions are cheap to keep; `max_vectors` bounds them, the
oldest being dropped.

The space must be one whose inner product is the dot product of its
components, as a Euclidean data space and its direct sums are; this is
checked.

    python recycling.py        a self-test on a random matrix
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import pygeoinf as inf


class RecycledSolver:
    """Conjugate gradients on `normal` with the first-level preconditioner
    `base` (an operator on the data space, or None), recycling the search
    directions of each solve into the next.

    `rcond` is the relative size below which a direction is taken to add
    nothing to those already kept.
    """

    def __init__(
        self,
        normal,
        base=None,
        /,
        *,
        max_vectors: Optional[int] = None,
        rcond: float = 1.0e-8,
    ):
        self.normal, self.base = normal, base
        self.space = normal.domain
        self.max_vectors, self.rcond = max_vectors, rcond
        dim = self.space.dim
        self._directions = np.zeros((dim, 0))
        self._images = np.zeros((dim, 0))
        # The kept directions recombined so that S* N S is the identity.
        self._s = np.zeros((dim, 0))
        self._y = np.zeros((dim, 0))
        self.applications = 0
        self._check_space()

    # ------------------------------------------------------------- #
    #                          The space                            #
    # ------------------------------------------------------------- #

    def _check_space(self) -> None:
        rng = np.random.default_rng(0)
        a, b = rng.standard_normal((2, self.space.dim))
        inner = self.space.inner_product(
            self.space.from_components(a), self.space.from_components(b)
        )
        if not np.isclose(inner, a @ b, rtol=1.0e-10):
            raise ValueError(
                "RecycledSolver needs a space whose inner product is the dot "
                "product of the components."
            )

    def _apply(self, operator, components: np.ndarray) -> np.ndarray:
        space = self.space
        return np.asarray(
            space.to_components(operator(space.from_components(components)))
        )

    def _normal(self, components: np.ndarray) -> np.ndarray:
        self.applications += 1
        return self._apply(self.normal, components)

    def _base(self, components: np.ndarray) -> np.ndarray:
        if self.base is None:
            return components.copy()
        return self._apply(self.base, components)

    # ------------------------------------------------------------- #
    #                      The preconditioner                       #
    # ------------------------------------------------------------- #

    @property
    def rank(self) -> int:
        """The number of directions in use."""
        return self._s.shape[1]

    def precondition(self, residual: np.ndarray) -> np.ndarray:
        """H applied to the components of a residual."""
        if self.rank == 0:
            return self._base(residual)
        s, y = self._s, self._y
        coefficients = s.T @ residual
        inner = self._base(residual - y @ coefficients)
        return inner - s @ (y.T @ inner) + s @ coefficients

    def operator(self):
        """H as an operator on the data space, for use as the
        preconditioner of another solver. It is the H of the directions
        kept when this is called."""
        space = self.space
        s, y = self._s.copy(), self._y.copy()
        frozen = RecycledSolver.__new__(RecycledSolver)
        frozen.__dict__.update(self.__dict__)
        frozen._s, frozen._y = s, y

        def mapping(vector):
            return space.from_components(
                frozen.precondition(np.asarray(space.to_components(vector)))
            )

        operator = inf.LinearOperator.from_callables(
            space, space, mapping, adjoint=mapping
        )
        return _definite(operator)

    def _absorb(self, directions: np.ndarray, images: np.ndarray) -> None:
        """Adds the directions of a solve and recombines all those kept so
        that S* N S is the identity, dropping what is dependent."""
        self._directions = np.hstack([self._directions, directions])
        self._images = np.hstack([self._images, images])
        if self.max_vectors is not None:
            self._directions = self._directions[:, -self.max_vectors :]
            self._images = self._images[:, -self.max_vectors :]
        gram = self._directions.T @ self._images
        gram = 0.5 * (gram + gram.T)
        values, vectors = np.linalg.eigh(gram)
        keep = values > self.rcond * values[-1]
        mix = vectors[:, keep] / np.sqrt(values[keep])
        self._s = self._directions @ mix
        self._y = self._images @ mix

    def seed(self, count: int, /, *, power: int = 1, rng=None) -> None:
        """Before any solve, takes `count` directions from random vectors:
        S = (P N)^power P W, with W white noise on the data, at the cost of
        `count` x (`power` + 1) applications of N. This is a randomised
        low-rank correction of the preconditioned operator, in the form of
        the same limited-memory preconditioner; each power of P N turns the
        directions towards those in which P is poorest."""
        rng = np.random.default_rng() if rng is None else rng
        directions = []
        for _ in range(count):
            vector = self._base(rng.standard_normal(self.space.dim))
            for _ in range(power):
                vector = self._base(self._normal(vector))
                vector /= np.linalg.norm(vector)
            directions.append(vector)
        images = [self._normal(vector) for vector in directions]
        self._absorb(np.array(directions).T, np.array(images).T)

    # ------------------------------------------------------------- #
    #                           The solve                           #
    # ------------------------------------------------------------- #

    def solve(
        self,
        rhs,
        /,
        *,
        rtol: float = 1.0e-6,
        max_iterations: int = 1000,
        recycle: bool = True,
    ):
        """Solves N x = rhs to a relative residual `rtol`.

        Returns (x, residuals): the solution, a vector of the space, and
        the norm of the residual at the start and after each iteration,
        relative to that of the right-hand side. With `recycle`, the
        directions of this solve are kept for the next.
        """
        space = self.space
        b = np.asarray(space.to_components(rhs), dtype=float)
        scale = np.linalg.norm(b)
        if scale == 0.0:
            return space.from_components(np.zeros_like(b)), np.zeros(1)
        if self.rank:
            x = self._s @ (self._s.T @ b)
            r = b - self._y @ (self._s.T @ b)
        else:
            x, r = np.zeros_like(b), b.copy()
        residuals = [np.linalg.norm(r) / scale]
        directions, images = [], []
        z = self.precondition(r)
        p = z.copy()
        rz = r @ z
        while residuals[-1] > rtol and len(residuals) <= max_iterations:
            image = self._normal(p)
            curvature = p @ image
            if curvature <= 0.0:
                raise RuntimeError("The operator is not positive definite.")
            step = rz / curvature
            x += step * p
            r -= step * image
            directions.append(p / np.sqrt(curvature))
            images.append(image / np.sqrt(curvature))
            residuals.append(np.linalg.norm(r) / scale)
            z = self.precondition(r)
            rz, previous = r @ z, rz
            p = z + (rz / previous) * p
        if recycle and directions:
            self._absorb(np.array(directions).T, np.array(images).T)
        return space.from_components(x), np.array(residuals)


def _definite(operator):
    """The operator, claimed self-adjoint and positive definite."""
    return operator.with_traits(
        operator.traits | inf.Traits.SELF_ADJOINT | inf.Traits.POSITIVE_DEFINITE
    )


def iterations_to(residuals: np.ndarray, tolerances) -> list:
    """The iteration at which each tolerance is first met, or None."""
    counts = []
    for tolerance in tolerances:
        reached = np.flatnonzero(residuals <= tolerance)
        counts.append(int(reached[0]) if reached.size else None)
    return counts


def _self_test() -> None:
    """A normal matrix of the kind the experiments have: a noise term, a
    signal term whose spectrum falls slowly, and a first-level
    preconditioner that inverts the noise alone."""
    rng = np.random.default_rng(1)
    n = 2000
    basis, _ = np.linalg.qr(rng.standard_normal((n, n)))
    signal = 1.0e3 * (1.0 + np.arange(n)) ** -1.5
    noise = np.exp(rng.uniform(-1.0, 1.0, n))
    matrix = (basis * signal) @ basis.T + np.diag(noise)
    space = inf.EuclideanSpace(n)
    normal = _definite(
        inf.LinearOperator.from_matrix(space, space, matrix, form="components")
    )
    base = inf.LinearOperator.from_matrix(
        space, space, np.diag(1.0 / noise), form="components"
    )
    tolerances = (1.0e-2, 1.0e-4, 1.0e-6)
    solver = RecycledSolver(normal, base)
    plain = RecycledSolver(normal, base)
    print(f"{'solve':<8}{'kept':>6}{'recycled':>22}{'not recycled':>22}")
    for index in range(6):
        rhs = space.from_components(rng.standard_normal(n))
        kept = solver.rank
        x, residuals = solver.solve(rhs)
        _, reference = plain.solve(rhs, recycle=False)
        error = np.linalg.norm(
            matrix @ np.asarray(space.to_components(x))
            - np.asarray(space.to_components(rhs))
        ) / np.linalg.norm(np.asarray(space.to_components(rhs)))
        assert error < 2.0e-6, error
        print(
            f"{index + 1:<8}{kept:>6}"
            f"{str(iterations_to(residuals, tolerances)):>22}"
            f"{str(iterations_to(reference, tolerances)):>22}"
        )
    seeded = RecycledSolver(normal, base)
    seeded.seed(40, power=2, rng=rng)
    rhs = space.from_components(rng.standard_normal(n))
    _, residuals = seeded.solve(rhs, recycle=False)
    print(
        f"seeded with 40 random directions, two powers ({seeded.applications} "
        f"applications): {iterations_to(residuals, tolerances)}"
    )
    frozen = solver.operator()
    progress = inf.ProgressCallback()
    rhs = space.from_components(rng.standard_normal(n))
    inf.CGSolver(rtol=1.0e-6, preconditioner=frozen, callback=progress)(normal)(rhs)
    print(f"pygeoinf's CGSolver with the kept directions: {progress.iterations}")


if __name__ == "__main__":
    _self_test()
