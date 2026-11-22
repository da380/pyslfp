"""
A sparse approximate inverse of a covariance matrix between points on a
sphere, from nearest neighbours (Heathcote et al., 2026): a prototype.

The altimetry block of the data-space normal matrix is the covariance of a
field between the points plus a diagonal noise, T = K + D: the matrix of a
Gaussian-process regression. Nothing dense can be formed on the points,
whose number is not bounded. Tapering K to a sparse matrix and factorising
it is one answer; it needs a cut-off distance, distorts a smooth kernel
within it, and its factor fills in.

Vecchia's approximation (Vecchia, 1988; Katzfuss and Guinness, 2021; in the
form of a factor of the inverse, Schafer, Katzfuss and Owhadi, 2021) is
another. The points are put in an order, and each is regressed on the k
nearest of those before it:

    x_i = sum_j B_ij x_j + e_i ,   var(e_i) = f_i ,

with the coefficients and the variance from the k x k block of T on those
neighbours alone. Then

    T^-1  ~  (1 - B)^T F^-1 (1 - B) ,

with B sparse and strictly lower triangular (k entries a row) and F
diagonal. It is positive definite by construction, has no cut-off, is the
best approximation with that sparsity in the Kullback-Leibler sense, costs
n small dense solves to build, and applies in two sparse products. The
order is the maximin one, each point the farthest from all before it, so
that the early points cover the sphere coarsely and carry the long
wavelengths.

Nothing here is specific to sea level: the kernel is a function of the
angle between two points.

    python vecchia.py        a self-test on random points
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular


def unit_vectors(points) -> np.ndarray:
    """The unit vectors of points given as (latitude, longitude) in degrees."""
    lat, lon = np.deg2rad(np.asarray(points, dtype=float)).T
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def maximin_order(xyz: np.ndarray) -> np.ndarray:
    """The maximin order of points on the unit sphere: the first is that
    nearest their mean direction, and each after is the one farthest from
    all those before it."""
    count = len(xyz)
    order = np.empty(count, dtype=int)
    order[0] = int(np.argmax(xyz @ xyz.mean(axis=0)))
    # The largest cosine to a chosen point: the smallest is the farthest.
    nearest = xyz @ xyz[order[0]]
    nearest[order[0]] = np.inf
    for index in range(1, count):
        chosen = int(np.argmin(nearest))
        order[index] = chosen
        np.maximum(nearest, xyz @ xyz[chosen], out=nearest)
        nearest[chosen] = np.inf
    return order


def earlier_neighbours(xyz: np.ndarray, count: int, block: int = 512) -> np.ndarray:
    """For points in their order, the indices of the `count` nearest of
    those before each, nearest first; -1 where there are fewer. Returns an
    array (n, count)."""
    total = len(xyz)
    result = np.full((total, count), -1, dtype=int)
    for start in range(1, total, block):
        stop = min(start + block, total)
        cosines = xyz[start:stop] @ xyz[:stop].T
        rows = np.arange(start, stop)
        # A point's neighbours come before it.
        cosines[np.arange(stop)[None, :] >= rows[:, None]] = -np.inf
        keep = min(count, stop - 1)
        nearest = np.argpartition(-cosines, keep - 1, axis=1)[:, :keep]
        values = np.take_along_axis(cosines, nearest, axis=1)
        inner = np.argsort(-values, axis=1)
        nearest = np.take_along_axis(nearest, inner, axis=1)
        values = np.take_along_axis(values, inner, axis=1)
        nearest[~np.isfinite(values)] = -1
        result[start:stop, :keep] = nearest
    return result


class VecchiaInverse:
    """The approximate inverse of T = K + D between points on a sphere.

    Args:
        xyz: The unit vectors of the points, (n, 3).
        kernel: The covariance K as a function of an array of angles in
            radians.
        nugget: The diagonal D, one value a point, or a single value.
        neighbours: k, the points each is regressed on.
    """

    def __init__(
        self,
        xyz: np.ndarray,
        kernel: Callable[[np.ndarray], np.ndarray],
        nugget,
        /,
        *,
        neighbours: int = 30,
        batch: int = 2048,
    ):
        count = len(xyz)
        self.order = maximin_order(xyz)
        ordered = xyz[self.order]
        nugget = np.broadcast_to(np.asarray(nugget, dtype=float), (count,))[self.order]
        sets = earlier_neighbours(ordered, min(neighbours, count - 1))
        variance = float(kernel(np.zeros(1))[0])

        rows, columns, values = [], [], []
        conditional = np.empty(count)
        conditional[0] = variance + nugget[0]
        sizes = (sets >= 0).sum(axis=1)
        # Points with the same number of neighbours are solved together.
        # and in batches, to bound the memory of their small matrices.
        groups = []
        for size in np.unique(sizes[1:]):
            group = np.flatnonzero(sizes == size)
            group = group[group > 0]
            groups += [
                (size, group[start : start + batch])
                for start in range(0, len(group), batch)
            ]
        for size, members in groups:
            index = sets[members, :size]
            near = ordered[index]
            among = self._kernel(kernel, np.einsum("nik,njk->nij", near, near))
            among[:, np.arange(size), np.arange(size)] += nugget[index]
            cross = self._kernel(
                kernel, np.einsum("nik,nk->ni", near, ordered[members])
            )
            weights = np.linalg.solve(among, cross[:, :, None])[:, :, 0]
            conditional[members] = (
                variance + nugget[members] - np.einsum("ni,ni->n", cross, weights)
            )
            rows.append(np.repeat(members, size))
            columns.append(index.ravel())
            values.append(weights.ravel())
        if np.any(conditional <= 0.0):
            raise ValueError("A conditional variance is not positive.")
        regression = sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(count, count),
        )
        self._residual = (sparse.identity(count, format="csr") - regression).tocsr()
        self._residual_t = self._residual.T.tocsr()
        self._conditional = conditional
        self.entries_per_point = regression.nnz / count

    @staticmethod
    def _kernel(kernel, cosines: np.ndarray) -> np.ndarray:
        angles = np.arccos(np.clip(cosines, -1.0, 1.0))
        return np.asarray(kernel(angles.ravel())).reshape(angles.shape)

    def inverse(self, vector: np.ndarray) -> np.ndarray:
        """The approximate T^-1 applied to a vector on the points."""
        residual = self._residual @ vector[self.order]
        result = np.empty_like(residual)
        result[self.order] = self._residual_t @ (residual / self._conditional)
        return result

    def forward(self, vector: np.ndarray) -> np.ndarray:
        """The inverse of `inverse`: the covariance the approximation has."""
        half = spsolve_triangular(self._residual_t, vector[self.order], lower=False)
        full = spsolve_triangular(self._residual, half * self._conditional, lower=True)
        result = np.empty_like(full)
        result[self.order] = full
        return result


def _self_test() -> None:
    """Conjugate gradients on K + D for random points and a smooth kernel,
    with and without the approximate inverse."""
    rng = np.random.default_rng(2)
    count = 4000
    xyz = rng.standard_normal((count, 3))
    xyz /= np.linalg.norm(xyz, axis=1)[:, None]
    scale = 0.05  # radians: about 300 km on the Earth

    def kernel(angle):
        # Matern of smoothness 5/2.
        r = np.sqrt(5.0) * angle / scale
        return (1.0 + r + r**2 / 3.0) * np.exp(-r)

    nugget = np.full(count, (6.0 / 46.0) ** 2)
    matrix = kernel(np.arccos(np.clip(xyz @ xyz.T, -1.0, 1.0))) + np.diag(nugget)
    rhs = rng.standard_normal(count)

    def solve(precondition):
        x, r = np.zeros(count), rhs.copy()
        z = precondition(r)
        p, rz, iterations = z.copy(), r @ z, 0
        while np.linalg.norm(r) > 1.0e-6 * np.linalg.norm(rhs) and iterations < 2000:
            image = matrix @ p
            step = rz / (p @ image)
            x += step * p
            r -= step * image
            z = precondition(r)
            rz, previous = r @ z, rz
            p = z + (rz / previous) * p
            iterations += 1
        return iterations

    print(f"{count} points; no preconditioner: {solve(lambda r: r.copy())} iterations")
    print(f"noise alone: {solve(lambda r: r / nugget)} iterations")
    for neighbours in (10, 20, 40, 80):
        approximate = VecchiaInverse(xyz, kernel, nugget, neighbours=neighbours)
        check = rng.standard_normal(count)
        round_trip = np.linalg.norm(
            approximate.forward(approximate.inverse(check)) - check
        ) / np.linalg.norm(check)
        print(
            f"{neighbours:>3} neighbours: {solve(approximate.inverse):>4} iterations "
            f"(round trip {round_trip:.1e})"
        )


if __name__ == "__main__":
    _self_test()
