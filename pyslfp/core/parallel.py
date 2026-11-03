"""
Thread control for the numerical backends used by pyslfp.

Two libraries do the heavy lifting in a sea level calculation: the spherical
harmonic transforms (pyshtools, using ducc0 when it is installed) and BLAS
(through numpy and scipy). Both read the ``OMP_NUM_THREADS`` environment
variable when the process starts, and ducc0 additionally sizes its thread
pool from it. The functions here change those settings for the *current*
process at run time, which is useful when a script alternates between
serial phases that should use several cores and parallel phases that fan
out over single-threaded worker processes: the workers are separate
interpreters and keep taking their thread count from the environment.
"""

from __future__ import annotations
import os
from typing import Optional

_blas_limits = None  # keeps the threadpoolctl limiter alive so limits persist


def physical_core_count() -> int:
    """
    Returns the number of physical cores available to this process.

    Simultaneous multithreading siblings are not counted, since they give
    no benefit for the transforms. Falls back to the logical CPU count if
    the physical count cannot be determined.
    """
    try:
        from joblib import cpu_count

        return max(int(cpu_count(only_physical_cores=True)), 1)
    except Exception:
        return max(os.cpu_count() or 1, 1)


def set_num_threads(n: Optional[int] = None) -> int:
    """
    Sets the number of threads used by the transforms and BLAS in this process.

    Args:
        n: Thread count. ``None`` or a value below one selects the number of
            physical cores.

    Returns:
        The thread count that was applied.

    Notes:
        Only the current process is affected. Worker processes started later
        (for example by joblib) read ``OMP_NUM_THREADS`` from the environment
        when they start, so a script can give its serial phases the whole
        core budget while keeping its workers single-threaded by exporting
        ``OMP_NUM_THREADS=1`` and calling this function with the budget.
    """
    global _blas_limits

    if n is None or n < 1:
        n = physical_core_count()

    from threadpoolctl import threadpool_limits

    _blas_limits = threadpool_limits(limits=n)

    try:
        import ducc0
        from pyshtools.backends import preferred_backend, select_preferred_backend
    except ImportError:
        return n

    if ducc0.misc.thread_pool_size() != n:
        ducc0.misc.resize_thread_pool(n)
    if preferred_backend() == "ducc":
        select_preferred_backend("ducc", nthreads=n)

    return n
