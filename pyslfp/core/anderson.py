"""Anderson acceleration for fixed-point iterations.

A fixed point x = G(x) solved by plain iteration converges at the
rate of G's contraction; Anderson mixing combines the recent iterates
in the least-squares sense, which is a multisecant quasi-Newton
method built from the iterates themselves -- no Jacobian is formed or
solved.  The sea level equation's load iteration and the initial
sea-level calibration both use it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["AndersonMixer"]


class AndersonMixer:
    """
    Anderson acceleration for a fixed-point iteration x = G(x).

    The last ``memory`` differences of iterates and residuals are kept in
    ring buffers and combined in the least-squares sense (Walker and Ni,
    2011). The small Gram matrix is updated incrementally, so each update
    costs a handful of passes over the vector. Dot products are formed with
    ``numpy.einsum`` rather than BLAS: BLAS worker threads spin-wait after a
    call and slow down single-threaded transforms that follow.
    """

    def __init__(self, memory: int) -> None:
        self._memory = memory
        self._x_prev: Optional[np.ndarray] = None
        self._f_prev: Optional[np.ndarray] = None
        self._d_x: Optional[np.ndarray] = None  # (memory, n) ring buffer
        self._d_f: Optional[np.ndarray] = None  # (memory, n) ring buffer
        self._gram: Optional[np.ndarray] = None  # (memory, memory)
        self._count = 0  # differences stored so far (capped at memory)
        self._slot = 0  # next ring-buffer slot to overwrite

    def update(self, x: np.ndarray, gx: np.ndarray) -> np.ndarray:
        """Returns the next iterate given the current one and G applied to it."""
        if self._memory <= 0:
            return gx

        x_flat = x.ravel()
        f_flat = gx.ravel() - x_flat

        if self._x_prev is None:
            n = x_flat.size
            self._d_x = np.empty((self._memory, n))
            self._d_f = np.empty((self._memory, n))
            self._gram = np.zeros((self._memory, self._memory))
            self._x_prev = x_flat.copy()
            self._f_prev = f_flat
            return gx

        slot = self._slot
        np.subtract(x_flat, self._x_prev, out=self._d_x[slot])
        np.subtract(f_flat, self._f_prev, out=self._d_f[slot])
        self._x_prev = x_flat.copy()
        self._f_prev = f_flat

        self._count = min(self._count + 1, self._memory)
        self._slot = (slot + 1) % self._memory
        m = self._count

        # Refresh the row and column of the Gram matrix for the new entry
        gram_row = np.einsum("ij,j->i", self._d_f[:m], self._d_f[slot])
        self._gram[slot, :m] = gram_row
        self._gram[:m, slot] = gram_row

        rhs = np.einsum("ij,j->i", self._d_f[:m], f_flat)
        gamma = np.linalg.lstsq(self._gram[:m, :m], rhs, rcond=None)[0]

        x_new = gx.ravel() - np.einsum("i,ij->j", gamma, self._d_x[:m])
        x_new -= np.einsum("i,ij->j", gamma, self._d_f[:m])
        return x_new.reshape(x.shape)
