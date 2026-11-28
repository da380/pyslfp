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
from scipy.sparse.linalg import spilu, spsolve_triangular


def unit_vectors(points) -> np.ndarray:
    """The unit vectors of points given as (latitude, longitude) in degrees."""
    lat, lon = np.deg2rad(np.asarray(points, dtype=float)).T
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def maximin_order(xyz: np.ndarray, switch: int = 1024) -> np.ndarray:
    """The maximin order of points on the unit sphere: the first is that
    nearest their mean direction, and each after is the one farthest from
    all those before it.

    A point's distance to those chosen can only fall when a point is chosen
    within the distance it now has, and no point's is more than the chosen
    one's own. So once `switch` points are in, and that distance is small,
    each choice updates only the points in a ball about it, found with a
    tree, and the next is read from a heap: the cost is then nearly linear
    in the number of points, where updating every point each time is
    quadratic."""
    import heapq

    from scipy.spatial import cKDTree

    count = len(xyz)
    order = np.empty(count, dtype=int)
    order[0] = int(np.argmax(xyz @ xyz.mean(axis=0)))
    # The largest cosine to a chosen point: the smallest is the farthest.
    nearest = xyz @ xyz[order[0]]
    nearest[order[0]] = np.inf
    direct = min(count, max(switch, 1))
    for index in range(1, direct):
        chosen = int(np.argmin(nearest))
        order[index] = chosen
        np.maximum(nearest, xyz @ xyz[chosen], out=nearest)
        nearest[chosen] = np.inf
    if direct == count:
        return order
    tree = cKDTree(xyz)
    left = np.flatnonzero(np.isfinite(nearest))
    heap = list(zip(nearest[left].tolist(), left.tolist()))
    heapq.heapify(heap)
    for index in range(direct, count):
        while True:
            value, chosen = heapq.heappop(heap)
            if nearest[chosen] == value:
                break
        order[index] = chosen
        nearest[chosen] = np.inf
        # Those nearer the chosen point than it was to the rest: the chord
        # of the angle whose cosine is `value`, with a margin for rounding.
        radius = np.sqrt(max(2.0 - 2.0 * value, 0.0)) * (1.0 + 1.0e-9) + 1.0e-12
        ball = np.asarray(tree.query_ball_point(xyz[chosen], radius), dtype=int)
        cosines = xyz[ball] @ xyz[chosen]
        closer = cosines > nearest[ball]
        for member, cosine in zip(ball[closer].tolist(), cosines[closer].tolist()):
            nearest[member] = cosine
            heapq.heappush(heap, (cosine, member))
    return order


def _earlier_neighbours_direct(xyz, count, result, first, last, block=512):
    """`earlier_neighbours` for the points first to last, by comparing each
    with every point before it."""
    for start in range(first, last, block):
        stop = min(start + block, last)
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


def earlier_neighbours(xyz: np.ndarray, count: int, direct: int = 4096) -> np.ndarray:
    """For points in their order, the indices of the `count` nearest of
    those before each, nearest first; -1 where there are fewer. Returns an
    array (n, count).

    The first `direct` points are compared with every point before them.
    The rest are taken in stretches that double the number of points so
    far: a tree of the points to the end of a stretch gives each of its
    points more of its nearest than are wanted, and those that come after
    it are dropped; a point left short is asked for more. The cost is then
    nearly linear in the number of points."""
    from scipy.spatial import cKDTree

    total = len(xyz)
    result = np.full((total, count), -1, dtype=int)
    first = min(total, max(direct, count + 1))
    _earlier_neighbours_direct(xyz, count, result, 1, first)
    start = first
    while start < total:
        stop = min(2 * start, total)
        tree = cKDTree(xyz[:stop])
        rows = np.arange(start, stop)
        asked = min(stop, 3 * count + 1)
        while len(rows) > 0:
            _, found = tree.query(xyz[rows], k=asked, workers=-1)
            found = found.reshape(len(rows), -1)
            earlier = found < rows[:, None]
            enough = (earlier.sum(axis=1) >= count) | (asked == stop)
            # The first `count` of those before the point, in the order
            # found, which is nearest first.
            rank = np.cumsum(earlier, axis=1)
            keep = earlier & (rank <= count)
            done = np.flatnonzero(enough)
            for row in done[keep[done].sum(axis=1) < count]:
                # Fewer than `count` before it: cannot arise after `direct`.
                result[rows[row], : keep[row].sum()] = found[row][keep[row]]
            full = done[keep[done].sum(axis=1) == count]
            result[rows[full]] = found[full][keep[full]].reshape(len(full), count)
            rows = rows[~enough]
            asked = min(stop, 2 * asked)
        start = stop
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
        workers: int | None = None,
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

        # The batches are independent, and their work is in compiled code
        # that releases the interpreter, so they are shared among threads.
        def regress(group):
            size, members = group
            index = sets[members, :size]
            near = ordered[index]
            among = self._kernel(kernel, near @ near.transpose(0, 2, 1))
            among[:, np.arange(size), np.arange(size)] += nugget[index]
            cross = self._kernel(
                kernel, np.einsum("nik,nk->ni", near, ordered[members])
            )
            weights = np.linalg.solve(among, cross[:, :, None])[:, :, 0]
            return (
                members,
                variance + nugget[members] - np.einsum("ni,ni->n", cross, weights),
                np.repeat(members, size),
                index.ravel(),
                weights.ravel(),
            )

        import os
        from concurrent.futures import ThreadPoolExecutor

        if workers is None:
            workers = int(os.environ.get("VECCHIA_WORKERS", "0")) or min(
                16, os.cpu_count() or 1
            )
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for members, variances_, row, column, value in pool.map(regress, groups):
                conditional[members] = variances_
                rows.append(row)
                columns.append(column)
                values.append(value)
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


class NoiseFreeVecchiaInverse:
    """The approximate inverse of T = K + D with the nearest-neighbour
    approximation applied to K alone (Schafer, Katzfuss and Owhadi, 2021,
    section 4.1).

    With the noise on the diagonal a point is screened less well by its
    nearest neighbours: what the others add once the nearest are known does
    not fall away with distance as it does for K, and `VecchiaInverse` on T
    needs more neighbours the denser the points. So the sparse factor is
    made for K, whose inverse is then U^T U with U = F^-1/2 (1 - B), and
    the noise is put back exactly:

        T^-1 = U^T (1 + U D U^T)^-1 U .

    The matrix in the middle is sparse and positive definite, and equal to
    one where the prior's conditional variance far exceeds the noise. An
    incomplete factorisation of it in the maximin order
    (`scipy.sparse.linalg.spilu`, values below `drop_tol` of a column's
    largest dropped, the factors held to `fill_factor` times the matrix)
    stands for its inverse; the mean of its action and its transpose's keeps
    the result symmetric.

    K alone can be too smooth for the small solves to be stable, so a part
    `jitter` of the smallest noise variance is left with K and taken from D.

    Args as for `VecchiaInverse`.
    """

    def __init__(
        self,
        xyz: np.ndarray,
        kernel: Callable[[np.ndarray], np.ndarray],
        nugget,
        /,
        *,
        neighbours: int = 30,
        jitter: float = 1.0e-2,
        drop_tol: float = 1.0e-4,
        fill_factor: float = 3.0,
    ):
        count = len(xyz)
        nugget = np.broadcast_to(np.asarray(nugget, dtype=float), (count,))
        kept = jitter * float(nugget.min())
        smooth = VecchiaInverse(xyz, kernel, kept, neighbours=neighbours)
        self.order = smooth.order
        # In the maximin order K^-1 = U^T U with U = F^-1/2 (1 - B), lower
        # triangular, so that T = U^-1 (1 + U D U^T) U^-T and
        # T^-1 = U^T (1 + U D U^T)^-1 U: a congruence, which any positive
        # definite stand-in for the middle inverse keeps positive definite,
        # and which loses nothing to cancellation where K far exceeds D.
        self._whiten = (
            sparse.diags(1.0 / np.sqrt(smooth._conditional)) @ smooth._residual
        ).tocsr()
        self._whiten_t = self._whiten.T.tocsr()
        noise = sparse.diags(nugget[self.order] - kept)
        matrix = (
            sparse.identity(count, format="csc")
            + (self._whiten @ noise @ self._whiten_t)
        ).tocsc()
        self._factor = spilu(
            matrix,
            drop_tol=drop_tol,
            fill_factor=fill_factor,
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )
        self.entries_per_point = smooth.entries_per_point
        self.matrix_entries_per_point = matrix.nnz / count
        self.factor_entries_per_point = (
            self._factor.L.nnz + self._factor.U.nnz
        ) / count

    def inverse(self, vector: np.ndarray) -> np.ndarray:
        """The approximate T^-1 applied to a vector on the points."""
        white = self._whiten @ vector[self.order]
        solved = 0.5 * (self._factor.solve(white) + self._factor.solve(white, trans="T"))
        result = np.empty_like(white)
        result[self.order] = self._whiten_t @ solved
        return result


def _self_test() -> None:
    """Conjugate gradients on K + D for random points and a smooth kernel,
    with and without the approximate inverse."""
    import sys
    import time

    rng = np.random.default_rng(2)
    # Points in a cap, dense beside the kernel's scale if asked for: it is
    # there that the noise weakens the screening.
    dense = "dense" in sys.argv
    count = 12000 if dense else 4000
    xyz = rng.standard_normal((count, 3))
    if dense:
        xyz[:, 2] = np.abs(xyz[:, 2]) + 5.0
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
    for neighbours in (10, 20, 40):
        start = time.perf_counter()
        approximate = NoiseFreeVecchiaInverse(xyz, kernel, nugget, neighbours=neighbours)
        seconds = time.perf_counter() - start
        print(
            f"{neighbours:>3} neighbours, noise-free form: "
            f"{solve(approximate.inverse):>4} iterations (build {seconds:.1f} s; "
            f"entries a point: matrix {approximate.matrix_entries_per_point:.0f}, "
            f"factors {approximate.factor_entries_per_point:.0f})"
        )


if __name__ == "__main__":
    _self_test()
